"""Statistiques d'activation relevées couche par couche, pour AWQ.

La mise à l'échelle guidée par les activations a besoin de savoir quels canaux
d'entrée portent de grandes activations, et cela ne se lit pas dans les poids :
c'est une propriété des données. Le relever naïvement supposerait de tenir tout
le modèle en bf16, ce qui n'a pas de sens sur une machine incapable de tenir le
modèle en bf16 en premier lieu.

Le relevé parcourt donc le modèle un bloc à la fois :

    caché <- plongement(jetons de calibration)
    pour chaque bloc :
        matérialiser le bloc en bf16 depuis le point de contrôle source
        y faire passer `caché`, en notant les magnitudes d'entrée de chaque linéaire
        caché <- la sortie du bloc
        libérer le bloc

Le pic de mémoire est d'un bloc, pas du modèle. C'est la structure qu'emploient
AWQ et GPTQ, et c'est elle qui rend possible ici la calibration d'un modèle de
70 milliards de paramètres.

Sans cela, ``--awq`` n'a rien sur quoi travailler et le convertisseur retombe en
silence sur l'arrondi au plus proche — le CLI traite donc « AWQ demandé, aucune
statistique relevée » comme une erreur, plutôt que d'en faire discrètement moins
qu'annoncé.
"""

from __future__ import annotations

import os
from typing import Callable, Iterator, Optional

import torch

from ..engine.config import ModelSpec
from ..engine.gdn import GatedDeltaNet
from ..engine.layers import QuantLinear, RMSNorm, RotaryEmbedding
from ..engine.mla import MLAttention
from ..engine.model import (Attention, DecoderLayer, DecoderLayerGDN,
                           ForwardBatch, MLP, MLP2, MoEBlock)
from ..quant.formats import PlainTensor
from .calibrate import ActStats
from .convert import (_NORMES_ZERO_CENTREES, _QWEN35_HF, _QWEN35_RENOMMAGE,
                      _nemotron_h_rename, _nemotron_h_valeur)

__all__ = ["collect_activation_stats", "DEFAULT_CALIB_FILE", "default_calib_path", "load_calib_ids"]

# Corpus de calibration intégré : Gutenberg #1342 (Orgueil et Préjugés, domaine
# public, 738 Ko, `acvram/data/calibration-anglais.txt`) — Sage,
# `sage-calibration-verdict-17-09` : les six phrases mêlées d'avant (prose,
# code, SQL, français) favorisaient wikitext et inversaient le classement
# privé/public ; un texte anglais long et ordinaire est le bras A des mesures.
DEFAULT_CALIB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "data", "calibration-anglais.txt")


def default_calib_path() -> str:
    """Chemin du corpus intégré ; erreur claire s'il manque du paquet."""
    if not os.path.isfile(DEFAULT_CALIB_FILE):
        raise FileNotFoundError(f"corpus de calibration intégré absent : {DEFAULT_CALIB_FILE}")
    return DEFAULT_CALIB_FILE


def load_calib_ids(tokenizer, path: Optional[str], n_seqs: int,
                   seq_len: int, vocab_size: int) -> list[list[int]]:
    """Tokenise le corpus de calibration, ou échoue plutôt que de rendre du bruit."""
    texts: list[str] = []
    if path and not os.path.isfile(path):
        raise FileNotFoundError(f"fichier de calibration introuvable : {path}")
    path = path or default_calib_path()
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        blob = fh.read()
    step = max(1, len(blob) // max(1, n_seqs))
    texts = [blob[i:i + step] for i in range(0, len(blob), step)][:n_seqs]

    if tokenizer is None:
        # Des identifiants aléatoires donnent des statistiques de canaux
        # uniformes, ce qui rend AWQ inopérant. Les rendre quand même serait
        # pire que de le dire.
        raise ValueError(
            "la calibration exige un tokeniseur ; aucun n'a été trouvé dans le "
            "répertoire du modèle. Passez --no-awq pour convertir sans lui.")

    out = []
    for text in texts:
        ids = tokenizer.encode(text)[:seq_len]
        if len(ids) >= 8:
            out.append(ids)
    if not out:
        raise ValueError("le corpus de calibration n'a produit aucune séquence utilisable")
    return out


class _StatCollector:
    """Crochets d'avant-passe accumulant les magnitudes d'entrée par canal."""

    def __init__(self) -> None:
        self.stats: dict[str, ActStats] = {}
        self._handles: list = []

    def attach(self, module: torch.nn.Module, prefix: str) -> None:
        for name, sub in module.named_modules():
            if not isinstance(sub, QuantLinear):
                continue
            key = f"{prefix}{name}.weight" if name else f"{prefix}weight"
            self.attach_named(sub, key)

    def attach_named(self, sub: torch.nn.Module, key: str) -> None:
        """Un crochet sur UN module, sous une clé choisie plutôt que dérivée
        de son attribut Python — nécessaire quand le nom d'attribut interne
        (`kv_a_proj`, `linear_attn`) diffère du nom du tenseur source
        (`kv_a_proj_with_mqa`, `self_attn`) : aliaser l'attribut ne suffit
        pas, `named_modules()` déduplique par identité et ne revisite jamais
        un même sous-module sous un second nom."""
        if not isinstance(sub, QuantLinear):
            return

        def hook(_mod, args, key=key):
            if not args:
                return
            x = args[0]
            if not isinstance(x, torch.Tensor) or x.dim() < 2:
                return
            new = ActStats.from_inputs(x.detach().to(torch.float32).cpu())
            prev = self.stats.get(key)
            self.stats[key] = prev.merge(new) if prev else new

        self._handles.append(sub.register_forward_pre_hook(hook))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _plain(tensor: torch.Tensor, device, dtype) -> QuantLinear:
    t = tensor.to(dtype).to(device)
    return QuantLinear(PlainTensor(t, tuple(tensor.shape), "bf16"),
                       out_features=tensor.shape[0], in_features=tensor.shape[1])


def collect_activation_stats(
    model_path: str,
    spec: ModelSpec,
    calib_ids: list[list[int]],
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict[str, ActStats]:
    """Parcourt le point de contrôle bloc par bloc, en notant les statistiques d'entrée."""
    from safetensors import safe_open

    dev = torch.device(device if torch.cuda.is_available()
                       or device == "cpu" else "cpu")
    files = _shard_files(model_path)
    handles = {fn: safe_open(os.path.join(model_path, fn), framework="pt",
                             device="cpu") for fn in files}
    # Enrobages multimodaux HF (Sage-kv-lm4 non lié ; convert.py::_adapt_hf
    # applique la même règle au flux principal) : le modèle de langue vit
    # sous `model.language_model.`, la tour visuelle n'est pas servie ici.
    # Sans cette normalisation, `get("model.embed_tokens.weight")` levait
    # KeyError sur Qwen3.8-27B (clé réelle `model.language_model.embed_
    # tokens.weight`) — capturé par le `except Exception` générique du
    # CLI et rapporté comme « calibration indisponible », un faux repli
    # sur l'arrondi au plus proche qui n'annonçait jamais avoir moins fait.
    # Qwen3.5/GDN (chantier calibration-hybrides-gdn-17-09) : même
    # renommage/normes/repli que `convert.py::_adapt_hf` sur le flux
    # principal (linear_attn.in_proj_* -> qkv/gate/alpha/beta, A_log/
    # dt_bias -> *.weight, conv1d [d,1,L] -> [d,L], normes centrées à
    # zéro -> (1 + w)) — sans cette normalisation, `_build_bf16_layer`
    # ne trouverait ni les tenseurs GDN sous leur nom canonique, ni la
    # bonne valeur des normes.
    mt = str(getattr(spec, "model_type", "") or spec.raw.get("model_type", ""))
    qwen35 = mt in _QWEN35_HF and not spec.raw.get("gdn_a_log_negexp")
    nemotron_h = mt == "nemotron_h"

    location: dict[str, tuple[str, str]] = {}
    for fn, h in handles.items():
        for k in h.keys():
            if k.startswith(("model.visual.", "visual.",
                             "model.vision_tower.", "model.audio_tower.")):
                continue
            if nemotron_h:
                # sage-hybrides-etape1-close-gemm-dense-17-09 : le point de
                # contrôle brut nomme ses tenseurs `backbone.layers.N.mixer.*`
                # (pas `model.layers.N.*`) -- même table que `_adapt_hf`
                # (flux principal), sinon `get("model.embed_tokens.weight")`
                # lève KeyError avant même d'atteindre la boucle par couche
                # et la calibration entière se replie sur l'arrondi au plus
                # proche (message vu le 17/09, avant ce correctif).
                nom = _nemotron_h_rename(k, spec.num_layers)
                if nom is None:
                    continue
            else:
                nom = k.replace("model.language_model.", "model.")
                if qwen35:
                    for src, dst in _QWEN35_RENOMMAGE.items():
                        if src in nom:
                            nom = nom.replace(src, dst)
                            break
            location[nom] = (fn, k)

    def get(key: str) -> torch.Tensor:
        fn, reelle = location[key]
        t = handles[fn].get_tensor(reelle)
        if qwen35:
            if key.endswith("linear_attn.conv1d.weight") and t.dim() == 3:
                t = t.reshape(t.shape[0], t.shape[-1])
            if key.endswith(_NORMES_ZERO_CENTREES):
                t = t.to(torch.float32) + 1.0
        if nemotron_h:
            t = _nemotron_h_valeur(key, t)
        return t

    collector = _StatCollector()
    rope = RotaryEmbedding(spec.head_dim, spec.max_position_embeddings,
                           spec.rope_theta, spec.rope_scaling)
    rope_mla = (RotaryEmbedding(spec.qk_rope_head_dim, spec.max_position_embeddings,
                                spec.rope_theta, spec.rope_scaling)
               if spec.est_mla and spec.mla_rope else None)
    embed = get("model.embed_tokens.weight").to(dtype).to(dev)

    # Un tenseur d'état caché par séquence de calibration, transporté d'un bloc à l'autre.
    hiddens = [torch.nn.functional.embedding(
        torch.tensor(ids, device=dev), embed).to(dtype) for ids in calib_ids]

    with torch.inference_mode():
        for i in range(spec.num_layers):
            p = f"model.layers.{i}."
            avant = set(collector.stats)
            layer = None
            try:
                layer = _build_bf16_layer(spec, p, get, dev, dtype, rope, i, rope_mla)
                if isinstance(layer, DecoderLayerGDN) and isinstance(layer.linear_attn, GatedDeltaNet):
                    # Attributs Python (qwen/gate/alpha/beta_proj/out_proj)
                    # ne portent pas les noms canoniques du manifeste
                    # (linear_attn.{qkv,gate,alpha,beta,out}.weight) — chaque
                    # linéaire attachée sous son nom exact, comme kv_a_proj
                    # ci-dessous pour la MLA.
                    collector.attach_named(layer.linear_attn.qkv, p + "linear_attn.qkv.weight")
                    collector.attach_named(layer.linear_attn.gate, p + "linear_attn.gate.weight")
                    collector.attach_named(layer.linear_attn.alpha, p + "linear_attn.alpha.weight")
                    collector.attach_named(layer.linear_attn.beta_proj, p + "linear_attn.beta.weight")
                    collector.attach_named(layer.linear_attn.out_proj, p + "linear_attn.out.weight")
                    collector.attach(layer.mlp, p + "mlp.")
                elif spec.est_mla:
                    # `linear_attn` (nom générique de DecoderLayerGDN) et
                    # `kv_a_proj` (attribut de MLAttention) ne portent pas les
                    # noms du manifeste ("self_attn", "kv_a_proj_with_mqa") —
                    # attacher les sous-arbres séparément, sous le bon préfixe,
                    # plutôt que de dépendre du chemin d'attribut par défaut.
                    collector.attach(layer.linear_attn, p + "self_attn.")
                    collector.attach_named(
                        layer.linear_attn.kv_a_proj,
                        p + "self_attn.kv_a_proj_with_mqa.weight")
                    collector.attach(layer.mlp, p + "mlp.")
                else:
                    collector.attach(layer, p)
                if isinstance(layer.mlp, MoEBlock) and layer.mlp.shared is not None:
                    # `MoEBlock.shared` (attribut Python) != `mlp.shared_
                    # expert.*` (clé du manifeste) -- même piège que
                    # `linear_attn`/`kv_a_proj` ci-dessus, trouvé le 17/09 en
                    # calibrant nemotron_h (couche MoE, `self_attn=None`,
                    # attach générique) mais préexistant sur TOUT modèle à
                    # expert partagé qui n'emprunte pas les branches GDN/MLA
                    # : le collecteur générique notait les stats sous
                    # `mlp.shared.up_proj.weight`, jamais lu par `calibrate.
                    # py` qui cherche `mlp.shared_expert.up_proj.weight` --
                    # l'expert partagé retombait en silence sur l'arrondi au
                    # plus proche. `MLP2` (nemotron_h) n'a pas de gate_proj.
                    collector.attach_named(layer.mlp.shared.up_proj,
                                           p + "mlp.shared_expert.up_proj.weight")
                    collector.attach_named(layer.mlp.shared.down_proj,
                                           p + "mlp.shared_expert.down_proj.weight")
                    if hasattr(layer.mlp.shared, "gate_proj"):
                        collector.attach_named(layer.mlp.shared.gate_proj,
                                               p + "mlp.shared_expert.gate_proj.weight")
                sorties = []
                for h in hiddens:
                    n = h.shape[0]
                    batch = ForwardBatch(
                        tokens=torch.zeros(n, dtype=torch.long),
                        positions=torch.arange(n, device=dev),
                        seq_lens=[n], query_lens=[n], block_tables=[],
                        slot_mapping=torch.zeros(n, dtype=torch.long),
                        is_prefill=True)
                    sorties.append(layer(h, batch, None))
                hiddens[:] = sorties
            except (KeyError, RuntimeError) as exc:
                # Une couche a une structure inattendue : ELLE seule perd ses
                # statistiques (et les partielles qu'un hook aurait déjà
                # notées avant l'échec, retirées ci-dessous), pas tout le
                # modèle. cli.py ne désactivait AWQ qu'en cas d'exception non
                # rattrapée ici — le 15/09, un q_proj absent (MLA à q_lora,
                # GLM-4.7-Flash, KeyError DE CONSTRUCTION) éteignait AWQ pour
                # les 223 tenseurs du modèle entier pour UN nom manquant sur
                # UNE couche. Étendu ici à l'échec DE PASSE (RuntimeError) :
                # trouvé sur Qwen3.8-27B (`qwen3_5`, attention pleine avec
                # `attn_output_gate`) — la construction réussissait, la passe
                # avant échouait plus loin (vue/repli de forme incompatible
                # avec l'attention que cette passe suppose), et l'échec
                # n'était pas rattrapé ici avant cette correction : il
                # remontait jusqu'à cli.py et désactivait AWQ pour les 866
                # tenseurs du modèle entier pour UNE couche sur 64.
                for k in set(collector.stats) - avant:
                    del collector.stats[k]
                print(f"  [avertissement] calibration couche {i} indisponible "
                     f"({exc}) ; ses tenseurs se replient sur l'arrondi au "
                     f"plus proche, les autres couches restent calibrees")
            finally:
                collector.detach()
                if layer is not None:
                    del layer
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                if progress:
                    progress(i + 1, spec.num_layers)

    for h in handles.values():
        h.__exit__(None, None, None) if hasattr(h, "__exit__") else None
    return collector.stats


def _has(get, key: str) -> bool:
    try:
        get(key)
        return True
    except KeyError:
        return False


def _build_bf16_layer(spec: ModelSpec, prefix: str, get, dev, dtype, rope,
                      index: int, rope_mla=None):
    if (spec.model_type == "nemotron_h" and spec.layer_types
            and index < len(spec.layer_types)):
        # Chaque couche nemotron_h est à USAGE UNIQUE (`layers_block_type` :
        # "mamba" XOR "moe" XOR "attention", jamais deux ensemble) --
        # contrairement au dispatch générique ci-dessous, qui suppose
        # toujours une attention ET un mlp sur la même couche. Sans ces
        # branches, la construction levait KeyError sur le premier tenseur
        # absent (`self_attn.q_proj.weight` sur une couche mamba/moe,
        # `mlp.experts.0.gate_proj.weight` sur une couche moe -- nemotron_h
        # n'a pas de porte, `MLP2` à 2 projections + ReLU², pas la `MLP` à
        # 3 projections du dispatch générique) et la couche entière
        # retombait en IDENTITÉ : les couches EN AVAL calibraient sur un état
        # caché faux, silencieusement — 23+23 couches sur 52 du vrai modèle.
        kind = spec.layer_types[index]
        in_norm = RMSNorm(get(prefix + "input_layernorm.weight").to(dtype).to(dev),
                          spec.rms_norm_eps)
        if kind == "mamba":
            # Même construction que `loader.py:669-682`. `mlp=None` accepté
            # par `DecoderLayerGDN.forward` (`if self.mlp is None: return x`).
            # Les statistiques d'entrée de `mamba.in_proj`/`out_proj` que le
            # collecteur générique attache sous une mauvaise clé
            # (`linear_attn.in_proj`, pas `mamba.in_proj`) sont sans
            # conséquence : ces deux tenseurs restent en bf16 depuis
            # `sage-hybrides-etape1-close-gemm-dense-17-09`
            # (`convert.py::TensorRouter.format_for`), jamais lus par AWQ.
            from ..engine.mamba2 import Mamba2Mixer
            pm = prefix + "mamba."
            petit = lambda suffix: get(pm + suffix).to(torch.float32).to(dev)
            mamba = Mamba2Mixer(
                in_proj=_plain(get(pm + "in_proj.weight"), dev, dtype),
                out_proj=_plain(get(pm + "out_proj.weight"), dev, dtype),
                conv_weight=petit("conv1d.weight"),
                conv_bias=petit("conv1d.bias") if _has(get, pm + "conv1d.bias") else None,
                dt_bias=petit("dt_bias.weight"), A=petit("A.weight"), D=petit("D.weight"),
                norm_weight=petit("norm.weight"),
                num_heads=spec.mamba_num_heads, head_dim=spec.mamba_head_dim,
                n_groups=spec.mamba_n_groups, state_size=spec.mamba_state_size,
                eps=spec.rms_norm_eps).to(dev)
            return DecoderLayerGDN(index, mamba, None, in_norm, None, dev)
        if kind in ("mlp", "moe"):
            # `MLP2` (up/down, ReLU²) -- pas la `MLP` gate/up/down du
            # dispatch générique, qui n'existe pas sur ce modèle. Même
            # structure que `loader.py:684-707` : routeur nommé `mlp.gate.*`
            # (pas `mlp.router.*`, vérifié le 17/09 sur le point de contrôle
            # réel), expert partagé optionnel sans porte propre.
            if kind == "mlp":
                mlp: torch.nn.Module = MLP2(
                    _plain(get(prefix + "mlp.up_proj.weight"), dev, dtype),
                    _plain(get(prefix + "mlp.down_proj.weight"), dev, dtype), "relu2")
            else:
                router = _plain(get(prefix + "mlp.gate.weight"), dev, torch.float32)
                experts, e = [], 0
                while _has(get, prefix + f"mlp.experts.{e}.up_proj.weight"):
                    experts.append(MLP2(
                        _plain(get(prefix + f"mlp.experts.{e}.up_proj.weight"), dev, dtype),
                        _plain(get(prefix + f"mlp.experts.{e}.down_proj.weight"), dev, dtype),
                        "relu2"))
                    e += 1
                shared = None
                if _has(get, prefix + "mlp.shared_expert.up_proj.weight"):
                    shared = MLP2(
                        _plain(get(prefix + "mlp.shared_expert.up_proj.weight"), dev, dtype),
                        _plain(get(prefix + "mlp.shared_expert.down_proj.weight"), dev, dtype),
                        "relu2")
                score_bias = (get(prefix + "mlp.gate.e_score_correction_bias").float().to(dev)
                             if _has(get, prefix + "mlp.gate.e_score_correction_bias") else None)
                mlp = MoEBlock(
                    router, experts, spec.num_experts_per_tok or 2, shared,
                    norm_topk_prob=bool(spec.raw.get("norm_topk_prob", True)),
                    scoring=spec.router_scoring, score_bias=score_bias,
                    routed_scale=spec.routed_scaling_factor)
            return DecoderLayer(index, None, mlp, in_norm, None, dev)
        # attention (sans RoPE, `spec.attention_rope` déjà mis à faux par
        # `load_model_spec`), même construction que `loader.py:709-716`.
        pa = prefix + "self_attn."
        attn = Attention(
            spec,
            _plain(get(pa + "q_proj.weight"), dev, dtype),
            _plain(get(pa + "k_proj.weight"), dev, dtype),
            _plain(get(pa + "v_proj.weight"), dev, dtype),
            _plain(get(pa + "o_proj.weight"), dev, dtype),
            None if not spec.attention_rope else rope)
        return DecoderLayer(index, attn, None, in_norm, None, dev)
    pa = prefix + "self_attn."
    # GDN pur (Qwen3.5, `qwen3_5_text`) : layer_types[index] tranche, jamais
    # une liste de model_type — même critère que loader.py:791 (`est_gdn`).
    # kimi_linear et MLA partagent le même NOM d'attribut (`linear_attn`,
    # historique) mais un chemin de chargement distinct (loader.py:719) ;
    # ce chantier ne couvre que le cas simple GDN, pas kimi_linear.
    est_gdn = (bool(spec.layer_types) and index < len(spec.layer_types)
              and spec.layer_types[index] == "linear_attention"
              and not spec.est_mla and spec.model_type != "kimi_linear")
    if est_gdn:
        pla = prefix + "linear_attn."
        petit = lambda suffix: get(pla + suffix).to(torch.float32).to(dev)
        attn = GatedDeltaNet(
            qkv=_plain(get(pla + "qkv.weight"), dev, dtype),
            gate=_plain(get(pla + "gate.weight"), dev, dtype),
            alpha=_plain(get(pla + "alpha.weight"), dev, dtype),
            beta=_plain(get(pla + "beta.weight"), dev, dtype),
            out=_plain(get(pla + "out.weight"), dev, dtype),
            conv_weight=petit("conv1d.weight"),
            dt_bias=petit("dt_bias.weight"),
            # gdn_a_log_negexp (GGUF) : hors de ce chemin, `collect_
            # activation_stats` ne l'active jamais côté qwen35 (voir
            # `qwen35` ci-dessus) — a_log lu tel quel, comme loader.py:814.
            a_log=petit("a_log.weight"),
            norm_weight=petit("norm.weight"),
            num_k_heads=spec.linear_num_key_heads,
            num_v_heads=spec.linear_num_value_heads,
            head_k_dim=spec.linear_key_head_dim,
            head_v_dim=spec.linear_value_head_dim,
            eps=spec.rms_norm_eps).to(dev)
    elif spec.est_mla:
        # Meme critere que loader.py (bead anticitoyen-vram-992) : la
        # structure (q_lora ou non) decide, jamais une liste de noms. GLM-
        # 4.7-Flash a q_a_proj/q_b_proj (pas de q_proj plat) — le manquer
        # levait un KeyError qui remontait jusqu'a desactiver AWQ pour tout
        # le modele (cli.py:469, 15/09).
        q_lora = _has(get, pa + "q_a_proj.weight")
        attn = MLAttention(
            q_proj=None if q_lora else _plain(get(pa + "q_proj.weight"), dev, dtype),
            q_a_proj=_plain(get(pa + "q_a_proj.weight"), dev, dtype) if q_lora else None,
            q_a_norm=(get(pa + "q_a_layernorm.weight").to(dtype).to(dev)
                     if q_lora else None),
            q_b_proj=_plain(get(pa + "q_b_proj.weight"), dev, dtype) if q_lora else None,
            rope=rope_mla,
            kv_a_proj=_plain(get(pa + "kv_a_proj_with_mqa.weight"), dev, dtype),
            o_proj=_plain(get(pa + "o_proj.weight"), dev, dtype),
            kv_a_norm=get(pa + "kv_a_layernorm.weight").to(dtype).to(dev),
            # k_b/v_b : jamais quantifies par AWQ (SENSITIVE_SUFFIXES,
            # convert.py:219), donc jamais lus ici — ce sont des tenseurs
            # d'ABSORPTION deja scindes a la conversion, pas des lineaires
            # a calibrer. Une paire de zeros de la bonne forme suffit : le
            # collecteur ne les hooke pas (ce ne sont pas des QuantLinear),
            # ils ne participent qu'au calcul de l'attention en aval, dont
            # les statistiques ne sont pas ce que cette passe releve.
            k_b=torch.zeros(spec.num_attention_heads, spec.kv_lora_rank,
                            spec.qk_nope_head_dim, dtype=dtype, device=dev),
            v_b=torch.zeros(spec.num_attention_heads, spec.v_head_dim,
                            spec.kv_lora_rank, dtype=dtype, device=dev),
            num_heads=spec.num_attention_heads,
            qk_nope=spec.qk_nope_head_dim, qk_rope=spec.qk_rope_head_dim,
            kv_lora_rank=spec.kv_lora_rank, v_dim=spec.v_head_dim,
            eps=spec.rms_norm_eps).to(dev)
        attn.fuse_projections()
    else:
        # attn_output_gate (Qwen3.5 attention pleine, `attn_output_gate:
        # true`) : q_proj sort [q_h | porte_h] par tête — deux fois
        # `num_attention_heads * head_dim`, pas une fois. Sans
        # `output_gate=True`, `Attention._proj` (engine/model.py:297)
        # tentait de reformer `q_proj(x)` en [t, num_attention_heads,
        # head_dim] alors que sa dernière dimension vaut le double :
        # `modeling_qwen3_5.py:761-763` (q_proj HF), `:787-790` (chunk en
        # query/porte) — RuntimeError shape, pas une KeyError, non
        # rattrapée avant le correctif du 17/09 (69b6d9e). q_norm/k_norm
        # (`:773-774`) ajoutés pour la même fidélité que loader.py:837-839.
        attn = Attention(
            spec,
            _plain(get(pa + "q_proj.weight"), dev, dtype),
            _plain(get(pa + "k_proj.weight"), dev, dtype),
            _plain(get(pa + "v_proj.weight"), dev, dtype),
            _plain(get(pa + "o_proj.weight"), dev, dtype),
            rope,
            q_norm=(RMSNorm(get(pa + "q_norm.weight").to(dtype).to(dev), spec.rms_norm_eps)
                   if _has(get, pa + "q_norm.weight") else None),
            k_norm=(RMSNorm(get(pa + "k_norm.weight").to(dtype).to(dev), spec.rms_norm_eps)
                   if _has(get, pa + "k_norm.weight") else None),
            output_gate=spec.attn_output_gate)

    try:
        router = _plain(get(prefix + "mlp.gate.weight"), dev, torch.float32)
        experts, e = [], 0
        while True:
            try:
                experts.append(MLP(
                    _plain(get(prefix + f"mlp.experts.{e}.gate_proj.weight"), dev, dtype),
                    _plain(get(prefix + f"mlp.experts.{e}.up_proj.weight"), dev, dtype),
                    _plain(get(prefix + f"mlp.experts.{e}.down_proj.weight"), dev, dtype)))
                e += 1
            except KeyError:
                break
        # Meme structure que loader.py:441-456 : expert partage, biais de
        # correction et fonction de score sont optionnels PAR MODELE, jamais
        # devines. Sans eux, la calibration de GLM-4.7-Flash routait en
        # softmax sans biais (le defaut de MoEBlock) alors que le modele
        # route en sigmoid+biais — des experts differents de la production,
        # donc des statistiques d'activation pour les mauvais tenseurs.
        shared, shared_gate = None, None
        if _has(get, prefix + "mlp.shared_expert.gate_proj.weight"):
            shared = MLP(
                _plain(get(prefix + "mlp.shared_expert.gate_proj.weight"), dev, dtype),
                _plain(get(prefix + "mlp.shared_expert.up_proj.weight"), dev, dtype),
                _plain(get(prefix + "mlp.shared_expert.down_proj.weight"), dev, dtype))
            if _has(get, prefix + "mlp.shared_expert_gate.weight"):
                shared_gate = get(prefix + "mlp.shared_expert_gate.weight").to(dtype).to(dev)
        score_bias = (get(prefix + "mlp.gate.e_score_correction_bias").float().to(dev)
                     if _has(get, prefix + "mlp.gate.e_score_correction_bias") else None)
        mlp: torch.nn.Module = MoEBlock(
            router, experts, spec.num_experts_per_tok or 2, shared,
            shared_gate=shared_gate,
            norm_topk_prob=bool(spec.raw.get("norm_topk_prob", True)),
            scoring=spec.router_scoring, score_bias=score_bias,
            routed_scale=spec.routed_scaling_factor)
    except KeyError:
        mlp = MLP(_plain(get(prefix + "mlp.gate_proj.weight"), dev, dtype),
                  _plain(get(prefix + "mlp.up_proj.weight"), dev, dtype),
                  _plain(get(prefix + "mlp.down_proj.weight"), dev, dtype))

    in_norm = RMSNorm(get(prefix + "input_layernorm.weight").to(dtype).to(dev),
                      spec.rms_norm_eps)
    post_norm = RMSNorm(
        get(prefix + "post_attention_layernorm.weight").to(dtype).to(dev),
        spec.rms_norm_eps)
    if est_gdn or spec.est_mla:
        # GatedDeltaNet.forward et MLAttention.forward rendent (y, etat) sur
        # UNE sequence et non (x, batch, cache) -> seule DecoderLayerGDN sait
        # les appeler (meme chemin que loader.py:787/827, ou "linear_attn"
        # designe aussi bien une recurrence lineaire qu'une MLA — le nom est
        # historique).
        return DecoderLayerGDN(index, attn, mlp, in_norm, post_norm, dev)
    return DecoderLayer(index, attn, mlp, in_norm, post_norm, dev)


def _shard_files(path: str) -> list[str]:
    import json
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index, "r", encoding="utf-8") as fh:
            return sorted(set(json.load(fh)["weight_map"].values()))
    return [f for f in sorted(os.listdir(path)) if f.endswith(".safetensors")]
