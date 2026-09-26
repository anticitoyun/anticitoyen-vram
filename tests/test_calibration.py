"""Calibration AWQ : elle doit relever de vraies statistiques et aider réellement."""

import hashlib
import json
import os

import pytest
import torch

from acvram.engine.config import load_model_spec
from acvram.quant.calibrate import ActStats
from acvram.quant.collect import DEFAULT_CALIB_FILE, collect_activation_stats, load_calib_ids


def test_le_corpus_de_calibration_integre_est_bien_bras_a():
    """poste7-priorite-apres-campagne-17-09 SS4, poste7-calibration-verdict-17-09 :
    le corpus par défaut doit être le bras A (prose anglaise ordinaire,
    Gutenberg #1342) -- pas les six phrases mêlées (prose+code+SQL+français)
    qui favorisaient wikitext et inversaient le classement privé/public.
    Épingle le sha256, pas seulement le chemin : `test_calib_source_
    manifeste.py` vérifie déjà que le manifeste NOMME `DEFAULT_CALIB_FILE`,
    pas que son CONTENU est le bon -- un remplacement silencieux du fichier
    passerait ce test-là sans être vu."""
    sha = hashlib.sha256(open(DEFAULT_CALIB_FILE, "rb").read()).hexdigest()
    assert sha == "cb7c0d9af2041d31e655fafe79becb0877c3fc775690c7a558eb8707035e5138", (
        f"DEFAULT_CALIB_FILE ({DEFAULT_CALIB_FILE}) n'est plus le bras A "
        f"(sha256 {sha}) -- corpus de calibration intégré change en silence")


@pytest.fixture(scope="module")
def tokenized_checkpoint(tiny_checkpoint):
    """Donne un tokeniseur au point de contrôle source pour que la calibration tourne."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    vocab = {f"tok{i}": i for i in range(1024)}
    for i, w in enumerate(["the", "a", "of", "and", "quick", "brown", "fox",
                           "models", "text", "de", "la", "sur"]):
        vocab[w] = 900 + i
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="tok0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.WordPiece(prefix="")
    tok.save(os.path.join(tiny_checkpoint, "tokenizer.json"))
    json.dump({"eos_token": "tok0"},
              open(os.path.join(tiny_checkpoint, "tokenizer_config.json"), "w"))
    return tiny_checkpoint


def test_calibration_refuses_to_pretend_without_a_tokenizer():
    """Une inaction silencieuse serait pire qu'une erreur."""
    with pytest.raises(ValueError, match="tokeniseur"):
        load_calib_ids(None, None, 4, 128, 1024)


def test_collects_one_entry_per_linear(tokenized_checkpoint):
    from acvram.server.chat import load_tokenizer

    spec = load_model_spec(tokenized_checkpoint, "tiny")
    tok = load_tokenizer(tokenized_checkpoint)
    calib = load_calib_ids(tok, None, 4, 64, spec.vocab_size)
    stats = collect_activation_stats(tokenized_checkpoint, spec, calib,
                                     device="cpu", dtype=torch.float32)
    # 7 couches linéaires par bloc : q, k, v, o, gate, up, down
    assert len(stats) == spec.num_layers * 7
    for name, st in stats.items():
        assert isinstance(st, ActStats)
        assert st.n_samples > 0
        assert torch.isfinite(st.mean_abs).all()
        assert st.mean_abs.min() >= 0


def test_statistics_are_not_uniform(tokenized_checkpoint):
    """Si tous les canaux se ressemblaient, AWQ n'aurait rien à exploiter."""
    from acvram.server.chat import load_tokenizer

    spec = load_model_spec(tokenized_checkpoint, "tiny")
    tok = load_tokenizer(tokenized_checkpoint)
    calib = load_calib_ids(tok, None, 4, 64, spec.vocab_size)
    stats = collect_activation_stats(tokenized_checkpoint, spec, calib,
                                     device="cpu", dtype=torch.float32)
    st = stats["model.layers.0.mlp.down_proj.weight"]
    spread = (st.mean_abs.max() / st.mean_abs.min().clamp(min=1e-9)).item()
    assert spread > 5.0


def test_awq_improves_the_objective_it_optimises(tokenized_checkpoint):
    """AWQ minimise l'erreur en *sortie de couche* : c'est donc elle qui doit
    s'améliorer.

    Mesurer le cosinus des logits de bout en bout sur un modèle de quatre
    couches à poids aléatoires place l'effet sous le plancher de bruit — la
    version précédente de ce test faisait exactement cela et oscillait. Le
    rapport signal/bruit en sortie de couche est l'objectif propre d'AWQ et il
    est déterministe une fois les statistiques données : c'est donc lui qu'on
    affirme.
    """
    from safetensors import safe_open

    from acvram.quant.calibrate import quantize_with_calibration
    from acvram.server.chat import load_tokenizer

    spec = load_model_spec(tokenized_checkpoint, "tiny")
    tok = load_tokenizer(tokenized_checkpoint)
    calib = load_calib_ids(tok, None, 6, 128, spec.vocab_size)
    stats = collect_activation_stats(tokenized_checkpoint, spec, calib,
                                     device="cpu", dtype=torch.float32)

    with safe_open(os.path.join(tokenized_checkpoint, "model.safetensors"),
                   framework="pt", device="cpu") as fh:
        weights = {k: fh.get_tensor(k) for k in fh.keys()
                   if k.endswith("proj.weight")}

    wins = {"nvfp4": 0, "int4_awq": 0}
    total = 0
    for name, w in weights.items():
        st = stats.get(name)
        if st is None:
            continue
        total += 1
        for fmt in ("nvfp4", "int4_awq"):
            _, _, rtn = quantize_with_calibration(
                w.to(torch.float32), fmt, st, use_hadamard=False, use_awq=False)
            _, _, awq = quantize_with_calibration(
                w.to(torch.float32), fmt, st, use_hadamard=False, use_awq=True)
            # La grille contient alpha = 0, c'est-à-dire aucune mise à
            # l'échelle : une recherche calibrée ne peut donc jamais faire pire
            # que l'arrondi au plus proche sur son propre objectif.
            assert awq["out_snr_db"] >= rtn["out_snr_db"] - 1e-6, (
                f"{name} {fmt}: {awq['out_snr_db']:.3f} < {rtn['out_snr_db']:.3f}")
            if awq["out_snr_db"] > rtn["out_snr_db"] + 0.01:
                wins[fmt] += 1

    assert total > 0
    # Ce doit être une amélioration réelle sur une majorité réelle de tenseurs,
    # pas un artefact d'arrondi sur l'un d'eux.
    assert wins["nvfp4"] > total / 2, wins
    assert wins["int4_awq"] > total / 2, wins


def test_awq_does_not_degrade_the_model_end_to_end(tokenized_checkpoint,
                                                   target_rig, tmp_path):
    """Le contrôle de bout en bout est un garde-fou, pas une démonstration."""
    from acvram.engine.loader import load_model
    from acvram.engine.model import ForwardBatch
    from acvram.memory.kvcache import BLOCK_SIZE, BlockAllocator
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint
    from acvram.server.chat import load_tokenizer

    spec = load_model_spec(tokenized_checkpoint, "tiny")
    tok = load_tokenizer(tokenized_checkpoint)
    calib = load_calib_ids(tok, None, 6, 128, spec.vocab_size)
    stats = collect_activation_stats(tokenized_checkpoint, spec, calib,
                                     device="cpu", dtype=torch.float32)
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=512))

    def prefill(model, prompt):
        n = len(prompt)
        alloc = BlockAllocator(model.caches[0].cfg.num_blocks)
        blocks = alloc.allocate((n + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)
        slots = torch.tensor([blocks[i // BLOCK_SIZE] * BLOCK_SIZE + i % BLOCK_SIZE
                              for i in range(n)])
        return model(ForwardBatch(torch.tensor(prompt), torch.arange(n), [n], [n],
                                  [torch.tensor(blocks)], slots, True))[0]

    def build(tag, fmt, st, awq):
        out = tmp_path / tag
        for lp in plan.layers:
            lp.fmt = fmt
        convert_checkpoint(tokenized_checkpoint, plan,
                           ConversionOptions(out_dir=str(out), awq=awq,
                                             use_hadamard="never",
                                             lm_head_format=fmt),
                           spec=spec, stats=st)
        return load_model(str(out), dtype=torch.float32,
                          device_override="cpu").model

    prompt = calib[0][:16]
    ref = prefill(build("ref", "bf16", None, False), prompt)
    for fmt in ("nvfp4", "int4_awq"):
        rtn = prefill(build(f"{fmt}-rtn", fmt, None, False), prompt)
        awq = prefill(build(f"{fmt}-awq", fmt, stats, True), prompt)
        cos_rtn = torch.nn.functional.cosine_similarity(ref, rtn, dim=0).item()
        cos_awq = torch.nn.functional.cosine_similarity(ref, awq, dim=0).item()
        # Un garde-fou, pas une démonstration : sur quatre couches de poids
        # aléatoires, l'écart de bout en bout entre AWQ et l'arrondi au plus
        # proche est dans le bruit. On affirme donc seulement que la calibration
        # n'aggrave pas nettement les choses. La démonstration est le test en
        # sortie de couche ci-dessus.
        assert cos_awq > cos_rtn - 0.03, f"{fmt}: AWQ {cos_awq:.5f} vs RTN {cos_rtn:.5f}"


@pytest.fixture(scope="module")
def multimodal_checkpoint(tokenized_checkpoint):
    """Le même point de contrôle, sous l'enrobage multimodal HF (Qwen3.8-27B,
    `qwen3_5`) : le modèle de langue vit sous `model.language_model.`, plus
    une tour visuelle non servie — même règle que `convert.py::_adapt_hf`.
    Trouvé le 17/09 en reconvertissant Qwen3.8-27B : `get("model.embed_
    tokens.weight")` levait KeyError (clé réelle `model.language_model.
    embed_tokens.weight`), capturé par le `except Exception` générique du
    CLI et rapporté comme « calibration indisponible » — un faux repli sur
    l'arrondi au plus proche, jamais annoncé comme tel."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(os.path.join(tokenized_checkpoint, "model.safetensors"),
                   framework="pt", device="cpu") as fh:
        sd = {f"model.language_model.{k[len('model.'):]}" if k.startswith("model.") else k: fh.get_tensor(k)
              for k in fh.keys()}
    sd["model.visual.blocks.0.weight"] = torch.randn(4, 4, dtype=torch.bfloat16)
    d = os.path.dirname(tokenized_checkpoint) + "-mm"
    os.makedirs(d, exist_ok=True)
    save_file(sd, os.path.join(d, "model.safetensors"))
    for fn in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        import shutil
        shutil.copy(os.path.join(tokenized_checkpoint, fn), os.path.join(d, fn))
    return d


def test_collecte_normalise_le_prefixe_multimodal(tokenized_checkpoint,
                                                   multimodal_checkpoint):
    """Témoin : les statistiques relevées sous l'enrobage `model.language_
    model.` doivent être EXACTEMENT celles du même point de contrôle sans
    enrobage — sinon la normalisation ne fait rien d'utile, ou fait autre
    chose. La tour visuelle (`model.visual.`) ne doit lever aucune erreur
    ni entrer dans les statistiques."""
    from acvram.server.chat import load_tokenizer

    spec = load_model_spec(tokenized_checkpoint, "tiny")
    tok = load_tokenizer(tokenized_checkpoint)
    calib = load_calib_ids(tok, None, 4, 64, spec.vocab_size)

    plat = collect_activation_stats(tokenized_checkpoint, spec, calib,
                                    device="cpu", dtype=torch.float32)
    mm = collect_activation_stats(multimodal_checkpoint, spec, calib,
                                  device="cpu", dtype=torch.float32)

    assert set(mm) == set(plat), "les jeux de tenseurs statistiqués diffèrent"
    for name, st in plat.items():
        assert torch.equal(st.mean_abs, mm[name].mean_abs), \
            f"{name} : statistiques différentes sous l'enrobage multimodal"
        assert st.n_samples == mm[name].n_samples


def test_calibration_survit_a_un_echec_de_passe_sur_une_couche(tokenized_checkpoint,
                                                                monkeypatch):
    """Un échec de PASSE AVANT (RuntimeError), pas seulement de construction
    (KeyError), sur UNE couche ne doit éteindre AWQ que pour CETTE couche —
    même motif que le KeyError de construction (MLA à q_lora, GLM-4.7-Flash,
    15/09), étendu à l'échec de passe. Trouvé sur Qwen3.8-27B (`qwen3_5`,
    attention pleine avec `attn_output_gate`) : la construction de la couche
    réussissait, la passe avant échouait plus loin (vue de forme incompatible
    avec l'attention que cette passe suppose) — non rattrapé avant cette
    correction, l'échec remontait jusqu'à cli.py et désactivait AWQ pour le
    modèle entier pour UNE couche sur 64."""
    from acvram.engine.model import DecoderLayer
    from acvram.server.chat import load_tokenizer

    original_forward = DecoderLayer.forward

    def echoue_apres_les_hooks(self, *a, **kw):
        sortie = original_forward(self, *a, **kw)
        if self.index == 1:
            raise RuntimeError("shape '[512, 24, 256]' is invalid for input of size 6291456")
        return sortie

    monkeypatch.setattr(DecoderLayer, "forward", echoue_apres_les_hooks)

    spec = load_model_spec(tokenized_checkpoint, "tiny")
    tok = load_tokenizer(tokenized_checkpoint)
    calib = load_calib_ids(tok, None, 4, 64, spec.vocab_size)
    stats = collect_activation_stats(tokenized_checkpoint, spec, calib,
                                     device="cpu", dtype=torch.float32)

    assert not any(name.startswith("model.layers.1.") for name in stats), \
        "la couche en échec a laissé des statistiques partielles au lieu de les retirer"
    for i in (0, 2, 3):
        assert any(name.startswith(f"model.layers.{i}.") for name in stats), \
            f"couche {i} : ses statistiques ont été perdues par l'échec d'une autre couche"


class _TokGabarit:
    """Tokeniseur factice : encode = codes des caractères ; gabarit = « § » + user + « ¶ » + assistant."""
    template = "{{ bos_token }}…"      # non vide : le modèle « a » un gabarit

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 1024 for c in text]

    def apply_chat_template(self, messages, add_generation_prompt, extra=None):
        assert [m["role"] for m in messages] == ["user", "assistant"]
        return "§" + messages[0]["content"] + "¶" + messages[1]["content"]


def test_calib_gabarit_applique_le_gabarit_et_casse_sans(tmp_path):
    """pièce 55 : avec --calib-gabarit chaque séquence PORTE le gabarit (tête § et balise ¶),
    sans lui aucune ne le porte ; un tokeniseur sans gabarit fait refuser, jamais un repli."""
    corpus = tmp_path / "c.txt"
    corpus.write_text("abcdefgh" * 40, encoding="utf-8")
    tok = _TokGabarit()
    avec = load_calib_ids(tok, str(corpus), 4, 64, 1024, gabarit=True)
    sans = load_calib_ids(tok, str(corpus), 4, 64, 1024)
    assert len(avec) == 4 and len(sans) == 4
    assert all(seq[0] == ord("§") % 1024 and ord("¶") % 1024 in seq for seq in avec)
    assert not any(ord("§") % 1024 in seq or ord("¶") % 1024 in seq for seq in sans)

    class _SansGabarit(_TokGabarit):
        template = None
    with pytest.raises(ValueError, match="gabarit"):
        load_calib_ids(_SansGabarit(), str(corpus), 4, 64, 1024, gabarit=True)
