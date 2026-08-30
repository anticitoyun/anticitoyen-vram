"""Calibration AWQ : elle doit relever de vraies statistiques et aider réellement."""

import json
import os

import pytest
import torch

from acvram.engine.config import load_model_spec
from acvram.quant.calibrate import ActStats
from acvram.quant.collect import collect_activation_stats, load_calib_ids


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
