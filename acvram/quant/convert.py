"""Convertit un point de contrôle Hugging Face en fragments acvram.

Le résultat n'est pas un modèle quantifié mais un modèle *par classe
d'appareil* : un tenseur destiné à la RTX 5090 est écrit en NVFP4, le même
tenseur destiné à la RTX 3080 Ti est écrit en INT4. C'est le plan de placement
qui tranche, si bien que la conversion et le chargement s'accordent par
construction — il n'existe aucune vérification à l'exécution du type « ce GPU
sait-il lire ce format ? » que l'on puisse rater.

La calibration, quand elle est activée, procède couche par couche à la manière
d'AWQ : ne tenir qu'un seul bloc en bf16, y faire passer les états cachés de
calibration pour relever les magnitudes d'activation par canal, quantifier ce
bloc, le libérer, passer au suivant. Le pic de mémoire est d'un bloc et non du
modèle entier, ce qui rend simplement possible la calibration d'un modèle de
70 milliards de paramètres sur cette machine.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator, Optional

import torch

from ..engine.config import ModelSpec, load_model_spec
from ..memory.tiering import Plan
from . import formats
from .calibrate import ActStats, ChannelScaler, quantize_with_calibration

__all__ = ["ConversionOptions", "ConversionReport", "convert_checkpoint",
           "TensorRouter"]

SHARD_TARGET_BYTES = 4 * 1024 ** 3


@dataclass
class ConversionOptions:
    out_dir: str
    calibrate: bool = False
    calib_tokens: int = 128
    calib_seqs: int = 16
    use_hadamard: str = "auto"        # auto | always | never
    awq: bool = True
    group_size: int = 128
    keep_sensitive_16bit: bool = True  # normalisations, routeur, plongements
    lm_head_format: Optional[str] = None
    n_grid: int = 20
    device: str = "cuda:0"
    # Appareil sur lequel se fait la recherche AWQ et la quantification. Elle
    # est dominee par des produits matriciels sur la grille de recherche : un
    # GPU la rend une dizaine de fois plus rapide qu'un i9. "auto" prend le
    # premier GPU disponible, "cpu" force l'ancien chemin.
    quant_device: str = "auto"
    dry_run: bool = False
    mixed_precision: str = "auto"     # auto | off
    # dB de rapport signal/bruit en sortie de couche sous lequel un tenseur est
    # promu d'un barreau. Zéro, donc rien : le décodage est limité par la bande
    # passante, et sur un 27B les promotions coûtaient 13,4 % de mémoire et
    # 10,6 % de débit pour 2,0 % de perplexité. Le mécanisme reste entier,
    # `--snr-floor 25` rétablit l'ancien comportement.
    snr_floor: float = 0.0
    # Autorise une conversion qui produit un modèle PLUS GROS que sa source.
    # Refusée par défaut depuis le 8/09/2026 : convertir un GGUF de 3,4 bits
    # par poids vers NVFP4 (4,5) a fait grossir Qwen3-Coder-Next d'un tiers,
    # créé 14 Gio d'exil en RAM hôte et coûté un facteur quinze au décodage.
    autoriser_grossissement: bool = False
    max_promotions: float = 0.15      # part maximale de tenseurs promus
    # Prix plafond d'une promotion, en mébioctets ajoutés (0 = pas de plafond).
    # Le quota ci-dessus compte des tenseurs ; or une porte de 0,1 Mio et une
    # projection MLP de 39 Mio gagnent le même nombre de décibels en montant
    # d'un barreau. À quota par tenseurs, l'ordre de rencontre décide, et les
    # projections épuisent le budget avant que les portes soient vues. Un
    # plafond de prix trie par ce qui compte vraiment : les octets relus à
    # chaque jeton.
    promotion_cout_max_mib: float = 0.0
    # Budget d'octets pour l'affectation par sac à dos (0 = mécanisme classique
    # de plancher SNR). Avec un budget, chaque tenseur promouvable est mesuré
    # dans les deux formats, puis les promotions sont choisies par gain de SNR
    # par octet dépensé, jusqu'à épuisement : « le meilleur modèle qui tient
    # dans N gibioctets », au lieu d'un seuil arbitraire.
    bits_budget_gib: float = 0.0


@dataclass
class ConversionReport:
    model: str = ""
    tensors: int = 0
    in_bytes: int = 0
    out_bytes: int = 0
    per_format: dict[str, int] = field(default_factory=dict)
    worst_layers: list[dict] = field(default_factory=list)
    promotions: list[dict] = field(default_factory=list)
    mean_out_snr_db: float = 0.0
    seconds: float = 0.0

    @property
    def ratio(self) -> float:
        return self.in_bytes / max(1, self.out_bytes)

    def render(self) -> str:
        lines = [f"converti : {self.model}", ""]
        lines.append(f"  tenseurs         {self.tensors}")
        lines.append(f"  entree           {_h(self.in_bytes)}")
        lines.append(f"  sortie           {_h(self.out_bytes)}  "
                     f"(x{self.ratio:.2f} plus petit)")
        for fmt, n in sorted(self.per_format.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {fmt:<12} {_h(n)}")
        lines.append(f"  SNR sortie moyen {self.mean_out_snr_db:.1f} dB")
        if self.promotions:
            lines.append(f"  promus           {len(self.promotions)} tenseurs vers un "
                         f"format plus large (sous le plancher de SNR) :")
            for p in self.promotions[:5]:
                lines.append(f"    {p['name']:<46} {p['from']} -> {p['to']}  "
                             f"{p['before']:.1f} -> {p['after']:.1f} dB")
            if len(self.promotions) > 5:
                lines.append(f"    ... et {len(self.promotions) - 5} autres")
        if self.worst_layers:
            lines.append("  pires tenseurs :")
            for w in self.worst_layers[:5]:
                lines.append(f"    {w['name']:<52} {w['out_snr_db']:6.1f} dB")
        lines.append(f"  duree            {self.seconds:.1f} s")
        return "\n".join(lines)


def _h(n: float) -> str:
    for unite in ("o", "Kio", "Mio", "Gio", "Tio"):
        if abs(n) < 1024 or unite == "Tio":
            return f"{int(n)} o" if unite == "o" else f"{n:.1f} {unite}"
        n /= 1024
    return f"{n:.1f} Tio"


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------


# Échelles de promotion. Un tenseur qui se quantifie mal monte d'un barreau
# plutôt que d'entraîner tout le modèle vers un format plus large : dépenser
# 8 bits sur les quelques pour cent de tenseurs qui en ont besoin coûte une
# fraction de bit par poids sur l'ensemble.
PROMOTE = {"int4_awq": "int8", "nvfp4": "int8", "int8": "bf16"}

# Largeur nominale de chaque format, bits par poids échelles comprises. Sert à
# chiffrer le prix d'une promotion avant de la calculer : la mesurer d'abord
# reviendrait à quantifier deux fois tous les tenseurs du modèle.
BPW_NOMINAL = {"bf16": 16.0, "fp16": 16.0, "int8": 8.25,
               "nvfp4": 4.5, "int4_awq": 4.25}


def cout_promotion_mib(numel: int, base: str, cible: str) -> float:
    """Mébioctets qu'ajoute le passage de ``base`` à ``cible``."""
    ecart = BPW_NOMINAL.get(cible, 16.0) - BPW_NOMINAL.get(base, 16.0)
    return numel * ecart / 8 / 1048576

SENSITIVE_SUFFIXES = (
    "layernorm.weight", "norm.weight", "_norm.weight",
    "conv1d.weight", "a_log.weight", "dt_bias.weight",
    "conv1d_q.weight", "conv1d_k.weight", "conv1d_v.weight",
    "conv.conv.weight",           # noyau court LFM2 [d, L]
    "mamba.conv1d.weight", "mamba.conv1d.bias", "mamba.A.weight",
    "mamba.D.weight", "mamba.dt_bias.weight", "mamba.norm.weight",
    ".a.weight",                  # -exp(A_log) de KDA (kimi-linear)
    "e_score_correction_bias",
    "layernorm.bias", "norm.bias",
    "router_scale.weight", "per_expert_scale.weight",
    "k_b_proj.weight", "v_b_proj.weight",   # absorptions MLA [H, r, d] : 3D, petits
    "mlp.gate.weight",            # routeur MoE : minuscule et décisif
    "shared_expert_gate.weight",  # porte de l'expert partagé : 1 ligne
    "embed_tokens.weight",
)


class TensorRouter:
    """Décide du format de stockage de chaque tenseur, à partir du plan de placement."""

    def __init__(self, spec: ModelSpec, plan: Plan, opts: ConversionOptions) -> None:
        self.spec = spec
        self.plan = plan
        self.opts = opts
        self._layer_fmt = {lp.index: lp.fmt for lp in plan.layers}

    def layer_index(self, name: str) -> Optional[int]:
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    return None
        return None

    def format_for(self, name: str) -> str:
        """Le format d'un tenseur est celui de l'appareil où sa couche s'exécute."""
        if self.opts.keep_sensitive_16bit and name.endswith(SENSITIVE_SUFFIXES):
            return "bf16"
        if name.endswith(".bias"):
            return "bf16"
        if name.startswith("lm_head"):
            return (self.opts.lm_head_format
                    or self._layer_fmt.get(self.spec.num_layers - 1, "int4_awq"))
        if ".mtp." in name or name.startswith("model.mtp."):
            # La tête de prédiction multi-jetons est un bloc de transformeur de
            # plus : elle suit le format de la dernière couche, pas le bf16 des
            # tenseurs hors couches.
            return self._layer_fmt.get(self.spec.num_layers - 1, "int4_awq")
        idx = self.layer_index(name)
        if idx is None:
            return "bf16"
        return self._layer_fmt.get(idx, "int4_awq")

    def wants_hadamard(self, name: str, fmt: str) -> bool:
        mode = self.opts.use_hadamard
        if mode == "never":
            return False
        if mode == "always":
            return True
        # auto : une rotation mérite sa place quand le groupe d'échelle est
        # large. Les groupes de 128 de l'INT4 ne peuvent pas absorber un canal
        # aberrant isolé, si bien qu'étaler les valeurs extrêmes aide de façon
        # mesurable. Les blocs de 16 du NVFP4 portent déjà leur propre échelle
        # et la rotation n'apporte que peu, au prix d'une transformée en
        # n log n sur chaque activation.
        return fmt == "int4_awq" and not name.endswith(SENSITIVE_SUFFIXES)


def _quantize_on(dev: torch.device, tensor: torch.Tensor, fmt: str,
                 st: Optional[ActStats], **kw):
    """Quantifie sur ``dev``, en retombant sur le processeur si la VRAM manque.

    La recherche AWQ garde plusieurs copies en float32 du tenseur ; sur un
    ``lm_head`` de 150 000 lignes cela depasse ce que laisse une carte deja
    occupee. Un tenseur trop gros n'est pas une erreur : il se quantifie plus
    lentement, ailleurs.
    """
    if dev.type != "cpu":
        try:
            st_dev = None if st is None else ActStats(
                st.mean_abs.to(dev),
                None if st.max_abs is None else st.max_abs.to(dev),
                st.n_samples)
            return quantize_with_calibration(
                tensor.to(torch.float32).to(dev), fmt, st_dev, **kw)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
    return quantize_with_calibration(tensor.to(torch.float32), fmt, st, **kw)


def _resolve_quant_device(choice: str) -> torch.device:
    """Ou quantifier. ``auto`` prend un GPU s'il y en a un, sinon le processeur."""
    if choice not in ("auto", ""):
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# checkpoint reading
# --------------------------------------------------------------------------


# Familles HF dont les couches à récurrence portent d'autres noms que notre
# manifeste (celui-ci a été fixé sur le GGUF de Qwen3.5) : renommage, normes
# zéro-centrées remises en (1 + w), conv1d aplatie, tours visuelle et MTP
# ignorées. Les conversions GGUF n'y passent pas (déjà dans nos conventions).
_QWEN35_HF = ("qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text",
              "qwen3_next")
_QWEN35_RENOMMAGE = {
    "linear_attn.in_proj_qkv": "linear_attn.qkv",
    "linear_attn.in_proj_z": "linear_attn.gate",
    "linear_attn.in_proj_a": "linear_attn.alpha",
    "linear_attn.in_proj_b": "linear_attn.beta",
    "linear_attn.out_proj": "linear_attn.out",
    "linear_attn.A_log": "linear_attn.a_log.weight",
    "linear_attn.dt_bias": "linear_attn.dt_bias.weight",
}
_NORMES_ZERO_CENTREES = ("input_layernorm.weight", "post_attention_layernorm.weight",
                         "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                         "model.norm.weight")


def _adapt_muse(source: Iterator[tuple[str, torch.Tensor]], spec
                ) -> Iterator[tuple[str, torch.Tensor]]:
    """Muse-Glimmer : normes centrées (+1), plongement normalisé par ligne,
    gate_proj fusionné par tête dans q_proj ([q_h | porte_h]), q_norm/k_norm
    synthétisés (sans poids, facteur qk_scale_factor replié dans q_norm)."""
    nh, hd = spec.num_attention_heads, spec.head_dim
    qk = float(spec.raw.get("qk_scale_factor") or 3.87)
    eps = float(spec.rms_norm_eps)
    en_attente: dict[str, dict[str, torch.Tensor]] = {}
    for name, t in source:
        name = name.replace("model.language_model.", "model.")
        if name.endswith(("input_layernorm.weight", "post_attention_layernorm.weight",
                          "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight")):
            yield name, (t.to(torch.float32) + 1.0).to(t.dtype)
            continue
        if name == "model.embed_tokens.weight":
            e = t.to(torch.float32)
            yield name, (e * torch.rsqrt(e.pow(2).mean(-1, keepdim=True) + eps)).to(t.dtype)
            continue
        if name.endswith(("self_attn.q_proj.weight", "self_attn.gate_proj.weight")):
            pref = name.rsplit("self_attn.", 1)[0] + "self_attn."
            lot = en_attente.setdefault(pref, {})
            lot["q" if name.endswith("q_proj.weight") else "g"] = t
            if len(lot) == 2:
                q, g = lot.pop("q"), lot.pop("g"); del en_attente[pref]
                fused = torch.cat((q.view(nh, hd, -1), g.view(nh, hd, -1)), dim=1)
                yield pref + "q_proj.weight", fused.reshape(nh * 2 * hd, -1).contiguous()
                yield pref + "q_norm.weight", torch.full((hd,), qk, dtype=t.dtype)
                yield pref + "k_norm.weight", torch.ones(hd, dtype=t.dtype)
            continue
        yield name, t
    assert not en_attente, f"q_proj/gate_proj dépareillés : {list(en_attente)}"


def _adapt_hf(source: Iterator[tuple[str, torch.Tensor]], spec
              ) -> Iterator[tuple[str, torch.Tensor]]:
    mt = str(getattr(spec, "model_type", "") or spec.raw.get("model_type", ""))

    def _texte_seul(src):
        # enrobages multimodaux HF : le modèle de langue vit sous
        # model.language_model., la tour visuelle n'est pas servie
        for n, t in src:
            if n.startswith(("model.visual.", "visual.", "model.vision_tower.", "model.audio_tower.")):
                continue
            yield n.replace("model.language_model.", "model."), t
    source = _texte_seul(source)
    if mt == "muse_glimmer":
        yield from _adapt_muse(source, spec)
        return
    if mt in ("gemma4", "gemma4_text"):
        # HF/EXL3 : Gemma4RMSNorm multiplie par w tel quel (pas de 1 + w,
        # contrairement à Gemma 3) et le convertisseur llama.cpp ne décale
        # rien non plus (norm_shift = 0) : les normes passent intactes
        for name, t in source:
            if name.endswith(".layer_scalar"):        # HF : sans suffixe .weight
                name += ".weight"
            yield name, t
        return
    if mt == "nemotron_h":
        # noms HF (backbone.layers.N.mixer.*) → noms acvram ; A = −exp(A_log)
        # (convention GGUF ssm_a), conv1d [d,1,L] → [d,L]. Les GGUF passent
        # ici sans être touchés (déjà nommés).
        mamba = ("in_proj", "out_proj", "conv1d", "A_log", "D", "dt_bias", "norm")
        n_layers = int(spec.num_layers)
        ignores = 0
        # EXL3 rembourre les deux dimensions à un multiple de 128 (in_proj
        # 10304 → 10368, experts 1856 → 1920) : on rogne aux tailles du modèle
        inner = int(spec.mamba_num_heads) * int(spec.mamba_head_dim)
        n_in_proj = 2 * inner + 2 * int(spec.mamba_n_groups) * int(spec.mamba_state_size) \
            + int(spec.mamba_num_heads)
        moe_i = int(spec.moe_intermediate_size or spec.intermediate_size or 0)
        shared_i = int(spec.shared_expert_intermediate_size or 0)
        dense_i = int(spec.intermediate_size or 0)

        def rogner(t: torch.Tensor, rows: int = 0, cols: int = 0) -> torch.Tensor:
            if rows and t.shape[0] > rows:
                t = t[:rows]
            if cols and t.dim() == 2 and t.shape[1] > cols:
                t = t[:, :cols]
            return t.contiguous()

        for name, t in source:
            if name.endswith(".weight") and t.dim() == 2:
                if name.endswith("mixer.in_proj.weight"):
                    t = rogner(t, rows=n_in_proj)
                elif name.endswith("mixer.out_proj.weight"):
                    t = rogner(t, cols=inner)
                elif ".mixer.experts." in name:
                    t = rogner(t, rows=moe_i) if name.endswith("up_proj.weight") else rogner(t, cols=moe_i)
                elif ".mixer.shared_experts." in name:
                    t = rogner(t, rows=shared_i) if name.endswith("up_proj.weight") else rogner(t, cols=shared_i)
                elif name.endswith(("mixer.up_proj.weight", "mixer.down_proj.weight")):
                    t = rogner(t, rows=dense_i) if name.endswith("up_proj.weight") else rogner(t, cols=dense_i)
            if name.startswith("backbone.layers."):
                _, _, idx, rest = name.split(".", 3)
                if int(idx) >= n_layers:
                    ignores += 1
                    continue
                pre = f"model.layers.{idx}."
                if rest == "norm.weight":
                    yield pre + "input_layernorm.weight", t
                elif rest.startswith("mixer."):
                    sub = rest[len("mixer."):]
                    tete = sub.split(".")[0]
                    if tete in mamba:
                        if tete == "A_log":
                            yield pre + "mamba.A.weight", (-torch.exp(t.to(torch.float32)))
                        elif tete == "conv1d" and sub.endswith("weight"):
                            yield pre + "mamba.conv1d.weight", t.reshape(t.shape[0], -1)
                        elif tete in ("D", "dt_bias"):
                            yield pre + f"mamba.{tete}.weight", t.reshape(-1)
                        else:
                            yield pre + "mamba." + sub, t
                    elif tete in ("q_proj", "k_proj", "v_proj", "o_proj"):
                        yield pre + "self_attn." + sub, t
                    else:
                        yield pre + "mlp." + sub.replace("shared_experts.", "shared_expert."), t
                else:
                    yield pre + rest, t
            elif name == "backbone.embeddings.weight":
                yield "model.embed_tokens.weight", t
            elif name == "backbone.norm_f.weight":
                yield "model.norm.weight", t
            elif name.startswith("mtp."):
                ignores += 1
            else:
                yield name, t
        if ignores:
            print(f"  nemotron_h : {ignores} tenseurs MTP ignorés")
        return
    if mt in ("lfm2", "lfm2_moe"):
        # noms HF Lfm2 : operator_norm/ffn_norm, feed_forward.w1/w3/w2,
        # self_attn.out_proj, q/k_layernorm, embedding_norm, expert_bias
        renames = (("feed_forward.experts.", "mlp.experts."), ("feed_forward.gate.weight", "mlp.gate.weight"),
                   ("feed_forward.expert_bias", "mlp.gate.e_score_correction_bias"),
                   ("feed_forward.w1.", "mlp.gate_proj."), ("feed_forward.w3.", "mlp.up_proj."),
                   ("feed_forward.w2.", "mlp.down_proj."), ("operator_norm.", "input_layernorm."),
                   ("ffn_norm.", "post_attention_layernorm."), ("self_attn.out_proj.", "self_attn.o_proj."),
                   ("self_attn.q_layernorm.", "self_attn.q_norm."), ("self_attn.k_layernorm.", "self_attn.k_norm."),
                   ("model.embedding_norm.", "model.norm."))
        for name, t in source:
            for a, b in renames:
                name = name.replace(a, b)
            if ".mlp.experts." in name:
                name = name.replace(".w1.", ".gate_proj.").replace(".w3.", ".up_proj.").replace(".w2.", ".down_proj.")
            if name.endswith("conv.conv.weight") and t.dim() == 3:
                t = t.reshape(t.shape[0], -1)
            yield name, t
        return
    if mt in ("deepseek_v2", "deepseek_v3", "glm4_moe") and int(getattr(spec, "kv_lora_rank", 0) or 0):
        # HF : kv_b_proj [nh·(nope+v), rank] scindé en k_b [nh, rank, nope] et
        # v_b [nh, v, rank] (convention du convertisseur llama.cpp, attendue
        # par le loader) ; experts partagés sous mlp.shared_expert
        nh, nope, vd, rank = (int(spec.num_attention_heads), int(spec.qk_nope_head_dim),
                              int(spec.v_head_dim), int(spec.kv_lora_rank))
        for name, t in source:
            if name.endswith("self_attn.kv_b_proj.weight"):
                kv = t.reshape(nh, nope + vd, rank)
                base = name[: -len("kv_b_proj.weight")]
                yield base + "k_b_proj.weight", kv[:, :nope, :].transpose(1, 2).contiguous()
                yield base + "v_b_proj.weight", kv[:, nope:, :].contiguous()
            else:
                yield name.replace("mlp.shared_experts.", "mlp.shared_expert."), t
        return
    if mt == "ernie4_5_moe":
        nh, nkv = spec.num_attention_heads, spec.num_key_value_heads

        def _depermute(t: torch.Tensor, n_head: int) -> torch.Tensor:
            # lignes par tête [p0a, p0b, p1a, p1b, …] (RoPE entrelacé HF)
            # → [a…, b…] (moitiés, notre RotaryEmbedding)
            r, c = t.shape
            return t.reshape(n_head, r // n_head // 2, 2, c).transpose(1, 2).reshape(r, c)

        for name, t in source:
            if name.endswith("mlp.moe_statics.e_score_correction_bias"):
                yield name.replace("moe_statics.", "gate."), t.reshape(-1)
            elif ".mlp.shared_experts." in name:
                yield name.replace("mlp.shared_experts.", "mlp.shared_expert."), t
            elif name.endswith("self_attn.q_proj.weight"):
                yield name, _depermute(t, nh)
            elif name.endswith("self_attn.k_proj.weight"):
                yield name, _depermute(t, nkv)
            else:
                yield name, t
        return
    if mt == "starcoder2":
        for name, t in source:
            yield name.replace("mlp.c_fc.", "mlp.up_proj.").replace("mlp.c_proj.", "mlp.down_proj."), t
        return
    if mt not in _QWEN35_HF or spec.raw.get("gdn_a_log_negexp"):
        yield from source
        return
    for name, t in source:
        name = name.replace("model.language_model.", "model.")
        if name.startswith(("model.visual", "visual.")):
            continue
        for src, dst in _QWEN35_RENOMMAGE.items():
            if src in name:
                name = name.replace(src, dst)
                break
        if name.endswith("linear_attn.conv1d.weight") and t.dim() == 3:
            t = t.reshape(t.shape[0], t.shape[-1])
        if name.endswith(_NORMES_ZERO_CENTREES):
            t = t.to(torch.float32) + 1.0        # (1 + w) de la référence
        yield name, t


def _iter_checkpoint(path: str) -> Iterator[tuple[str, torch.Tensor]]:
    """Lit les tenseurs d'un point de contrôle safetensors ou GGUF."""
    from safetensors import safe_open

    from .gguf import GGUFFile, is_gguf
    if is_gguf(path):
        g = GGUFFile(path)
        g.check_executable()
        yield from g.iter_tensors()
        return

    from .exl3 import EXL3Checkpoint, is_exl3
    if is_exl3(path):
        yield from EXL3Checkpoint(path).iter_tensors()
        return

    from .hfquant import HFQuantCheckpoint, is_hfquant
    if is_hfquant(path):
        yield from HFQuantCheckpoint(path).iter_tensors()
        return

    index_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as fh:
            index = json.load(fh)
        files = sorted(set(index["weight_map"].values()))
    else:
        files = [f for f in sorted(os.listdir(path)) if f.endswith(".safetensors")]
    if not files:
        raise FileNotFoundError(f"aucun fichier .safetensors dans {path}")

    for fn in files:
        full = os.path.join(path, fn)
        with safe_open(full, framework="pt", device="cpu") as fh:
            for key in fh.keys():
                yield key, fh.get_tensor(key)


class ShardWriter:
    """Accumule des tenseurs et les vide en fragments safetensors d'environ 4 Gio."""

    def __init__(self, out_dir: str, target: int = SHARD_TARGET_BYTES) -> None:
        self.out_dir = out_dir
        self.target = target
        self._buf: dict[str, torch.Tensor] = {}
        self._bytes = 0
        self._shard = 0
        self.weight_map: dict[str, str] = {}
        os.makedirs(out_dir, exist_ok=True)

    def add(self, key: str, tensor: torch.Tensor) -> None:
        t = tensor.detach().cpu().contiguous()
        self._buf[key] = t
        self._bytes += t.numel() * t.element_size()
        if self._bytes >= self.target:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        from safetensors.torch import save_file

        name = f"acvram-{self._shard:05d}.safetensors"
        save_file(self._buf, os.path.join(self.out_dir, name))
        for key in self._buf:
            self.weight_map[key] = name
        self._buf.clear()
        self._bytes = 0
        self._shard += 1

    @property
    def total_bytes(self) -> int:
        total = 0
        for fn in os.listdir(self.out_dir):
            if fn.endswith(".safetensors"):
                total += os.path.getsize(os.path.join(self.out_dir, fn))
        return total


# --------------------------------------------------------------------------
# conversion
# --------------------------------------------------------------------------


def octets_du_modele(model_path: str) -> int:
    """Octets des poids de la source : fichiers gguf ou safetensors du dossier."""
    import glob
    p = model_path if os.path.isdir(model_path) else os.path.dirname(model_path) or "."
    fichiers = [f for motif in ("*.gguf", "*.safetensors", "*.bin")
                for f in glob.glob(os.path.join(p, motif))]
    return sum(os.path.getsize(f) for f in fichiers)


def garde_grossissement(octets_source: int, total_params: int,
                        bpw_cible: float, autorise: bool) -> Optional[str]:
    """Refuse une conversion qui ferait grossir le modèle.

    Rend un message d'avertissement à journaliser si la conversion grossit
    mais est autorisée ; rend None si elle ne grossit pas ; lève sinon.
    Le critère n'est pas « le format est plus large » mais « le résultat sera
    plus gros que la source » — payer de la place pour des instructions
    natives est un bon échange tant qu'on en a (fiche 17 du relevé biblio) ;
    ici on chiffre précisément ce qui sera payé.
    """
    if not octets_source or not total_params:
        return None
    bpw_source = octets_source * 8 / total_params
    if bpw_cible <= bpw_source * 1.02:      # 2 % de jeu : en-têtes, échelles
        return None
    octets_cible = int(total_params * bpw_cible / 8)
    msg = (f"la conversion ferait GROSSIR le modèle : source "
           f"{octets_source / 2**30:.1f} Gio ({bpw_source:.2f} bits/poids), "
           f"cible {octets_cible / 2**30:.1f} Gio ({bpw_cible:.2f} bits/poids), "
           f"+{(octets_cible / octets_source - 1) * 100:.0f} %")
    if autorise:
        return msg + " — autorisée par --autoriser-grossissement"
    raise ValueError(
        msg + ". Refusée : le surplus serait exilé en RAM hôte et coûterait "
        "plus cher que les instructions natives ne rapportent (Coder-Next, "
        "7-8/09/2026 : facteur 15 au décodage). Aucun format d'acvram ne "
        "descend aujourd'hui sous 4,25 bits/poids ; servez la source par un "
        "moteur GGUF, ou passez --autoriser-grossissement en connaissance "
        "de cause.")

def convert_checkpoint(model_path: str, plan: Plan, opts: ConversionOptions,
                       spec: Optional[ModelSpec] = None,
                       stats: Optional[dict[str, ActStats]] = None,
                       progress: Optional[Callable[[str, int, int], None]] = None
                       ) -> ConversionReport:
    """Quantifie chaque tenseur dans le format qu'attend son appareil de destination."""
    t0 = time.time()
    spec = spec or load_model_spec(model_path)
    bpw_cible = max((BPW_NOMINAL.get(t.weight_format, 4.5)
                     for t in plan.tiers if t.kind == "gpu"), default=4.5)
    avert = garde_grossissement(octets_du_modele(model_path),
                                getattr(spec, "total_params", 0) or 0,
                                bpw_cible, opts.autoriser_grossissement)
    if avert:
        print(f"[acvram] {avert}", flush=True)
    router = TensorRouter(spec, plan, opts)
    report = ConversionReport(model=spec.name)
    writer = ShardWriter(opts.out_dir)
    manifest: dict[str, Any] = {
        "acvram_version": 1,
        "model": spec.to_dict(),
        "plan": plan.to_dict(),
        "options": asdict(opts),
        "tensors": {},
    }

    qdev = _resolve_quant_device(opts.quant_device)

    snrs: list[float] = []
    per_layer: list[dict] = []
    keys = []
    budget_candidats: list[dict] = []

    for name, tensor in _adapt_hf(_iter_checkpoint(model_path), spec):
        report.tensors += 1
        report.in_bytes += tensor.numel() * tensor.element_size()
        fmt = router.format_for(name)
        entry: dict[str, Any] = {"format": fmt, "shape": list(tensor.shape)}

        if progress and report.tensors % 25 == 0:
            progress(name, report.tensors, 0)

        if fmt in ("bf16", "fp16") or tensor.dim() != 2:
            out = tensor.to(torch.bfloat16 if fmt == "bf16" else torch.float16)
            if not opts.dry_run:
                writer.add(f"{name}", out)
            entry["keys"] = [name]
            report.per_format[fmt] = report.per_format.get(fmt, 0) + \
                out.numel() * out.element_size()
            manifest["tensors"][name] = entry
            keys.append(name)
            continue

        st = stats.get(name) if stats else None
        qt, scaler, metrics = _quantize_on(
            qdev, tensor, fmt, st,
            group_size=opts.group_size,
            use_hadamard=router.wants_hadamard(name, fmt),
            use_awq=opts.awq,
            n_grid=opts.n_grid,
        )

        # Précision mixte, deux régimes : plancher SNR classique (défaut,
        # plafonné), ou budget global (bits_budget_gib > 0) où les deux
        # formats sont mesurés et la décision revient au sac à dos de fin de
        # passe. Les experts d'un bloc restent exclus dans les deux cas : le
        # chemin de décodage groupé exige leur pile homogène, et leur SNR
        # individuel pèse peu.
        est_expert = ".mlp.experts." in name
        candidat_budget = (opts.bits_budget_gib > 0 and not est_expert
                           and fmt in PROMOTE
                           and metrics["out_snr_db"] < 40.0)
        if candidat_budget:
            wider = PROMOTE[fmt]
            q2, s2, m2 = _quantize_on(
                qdev, tensor, wider, st,
                group_size=opts.group_size,
                use_hadamard=router.wants_hadamard(name, wider),
                use_awq=opts.awq, n_grid=opts.n_grid)
            if m2["out_snr_db"] > metrics["out_snr_db"] + 0.5:
                sd2 = q2.state_dict(prefix=f"{name}.")
                sd2.update(s2.state_dict(prefix=f"{name}."))
                sd2 = {k: v.cpu() for k, v in sd2.items()}
                base_octets = sum(v.numel() * v.element_size()
                                  for v in qt.state_dict().values())
                larges_octets = sum(v.numel() * v.element_size()
                                    for v in sd2.values())
                budget_candidats.append({
                    "name": name, "from": fmt, "to": wider,
                    "gain_db": m2["out_snr_db"] - metrics["out_snr_db"],
                    "cout": max(1, larges_octets - base_octets),
                    "sd": sd2, "metrics": m2,
                })
            else:
                candidat_budget = False
        elif (not est_expert
                and opts.mixed_precision != "off"
                and metrics["out_snr_db"] < opts.snr_floor
                and fmt in PROMOTE
                and (not opts.promotion_cout_max_mib
                     or cout_promotion_mib(tensor.numel(), fmt, PROMOTE[fmt])
                     <= opts.promotion_cout_max_mib)
                and len(report.promotions) < opts.max_promotions * max(1, len(keys) + 1)):
            wider = PROMOTE[fmt]
            q2, s2, m2 = _quantize_on(
                qdev, tensor, wider, st,
                group_size=opts.group_size,
                use_hadamard=router.wants_hadamard(name, wider),
                use_awq=opts.awq, n_grid=opts.n_grid)
            if m2["out_snr_db"] > metrics["out_snr_db"] + 1.0:
                report.promotions.append({
                    "name": name, "from": fmt, "to": wider,
                    "before": round(metrics["out_snr_db"], 2),
                    "after": round(m2["out_snr_db"], 2)})
                fmt, qt, scaler, metrics = wider, q2, s2, m2
                entry["format"] = fmt
                entry["promoted_from"] = report.promotions[-1]["from"]
        sd = qt.state_dict(prefix=f"{name}.")
        sd.update(scaler.state_dict(prefix=f"{name}."))
        # Les fragments s'ecrivent depuis la memoire hote : on redescend ce que
        # la quantification a produit sur le GPU.
        sd = {k: v.cpu() for k, v in sd.items()}

        if candidat_budget and budget_candidats and \
                budget_candidats[-1]["name"] == name:
            # decision differee au sac a dos : rien n'est ecrit maintenant
            budget_candidats[-1].update({
                "sd_base": sd, "entry": entry, "fmt_base": fmt,
                "metrics_base": metrics, "nbytes_base": qt.nbytes,
                "hadamard_block": scaler.hadamard_block,
                "has_act_scale": scaler.scale is not None,
            })
            keys.append(name)
            continue

        snrs.append(metrics["out_snr_db"])
        per_layer.append({"name": name, **{k: round(v, 2) if isinstance(v, float)
                                           else v for k, v in metrics.items()}})
        if not opts.dry_run:
            for k, v in sd.items():
                writer.add(k, v)
        entry.update({
            "keys": list(sd.keys()),
            "group_size": opts.group_size,
            "hadamard_block": scaler.hadamard_block,
            "has_act_scale": scaler.scale is not None,
            "bpw": round(metrics["bpw"], 3),
            "out_snr_db": round(metrics["out_snr_db"], 2),
        })
        report.per_format[fmt] = report.per_format.get(fmt, 0) + qt.nbytes
        manifest["tensors"][name] = entry
        keys.append(name)

    # ---- sac à dos : depenser le budget la ou chaque octet paie le plus ----
    if budget_candidats:
        deja = sum(report.per_format.values()) + \
            sum(c["nbytes_base"] for c in budget_candidats)
        reste = opts.bits_budget_gib * 1024 ** 3 - deja
        # par gain de SNR par octet, decroissant — le glouton du sac a dos
        # fractionnaire, optimal a un tenseur pres
        ordre = sorted(budget_candidats, key=lambda c: -c["gain_db"] / c["cout"])
        promus = set()
        for c in ordre:
            if c["cout"] <= reste:
                promus.add(c["name"])
                reste -= c["cout"]
        for c in budget_candidats:
            name = c["name"]
            large = name in promus
            sd = c["sd"] if large else c["sd_base"]
            met = c["metrics"] if large else c["metrics_base"]
            fmt = c["to"] if large else c["fmt_base"]
            entry = c["entry"]
            snrs.append(met["out_snr_db"])
            per_layer.append({"name": name,
                              **{k: round(v, 2) if isinstance(v, float) else v
                                 for k, v in met.items()}})
            if not opts.dry_run:
                for k, v in sd.items():
                    writer.add(k, v)
            entry.update({
                "format": fmt,
                "keys": list(sd.keys()),
                "group_size": opts.group_size,
                "hadamard_block": c["hadamard_block"],
                "has_act_scale": c["has_act_scale"],
                "bpw": round(met["bpw"], 3),
                "out_snr_db": round(met["out_snr_db"], 2),
            })
            octets = sum(v.numel() * v.element_size() for v in sd.values())
            report.per_format[fmt] = report.per_format.get(fmt, 0) + octets
            manifest["tensors"][name] = entry
            if large:
                report.promotions.append({
                    "name": name, "from": c["from"], "to": c["to"],
                    "before": round(c["metrics_base"]["out_snr_db"], 2),
                    "after": round(met["out_snr_db"], 2)})

    # Un point de contrôle dont l'architecture n'est pas vraiment comprise
    # (noms de tenseurs non traduits) produirait un modèle mutilé qui échoue
    # au chargement — ou pire, qui répond du charabia. Vérifier que chaque
    # couche attendue par la spécification a bien ses projections.
    attendus = []
    for i in range(spec.num_layers):
        if spec.layer_types and spec.layer_types[i] == "conv":
            attendus.append(f"model.layers.{i}.conv.in_proj.weight")
        elif spec.layer_types and spec.layer_types[i] == "parallel":
            attendus.append(f"model.layers.{i}.mamba.in_proj.weight")
            attendus.append(f"model.layers.{i}.self_attn.q_proj.weight")
            attendus.append(f"model.layers.{i}.mlp.gate_proj.weight")
            continue
        elif spec.layer_types and spec.layer_types[i] in ("mamba", "mlp", "moe"):
            attendus.append(f"model.layers.{i}." + {
                "mamba": "mamba.in_proj.weight", "mlp": "mlp.up_proj.weight",
                "moe": "mlp.experts.0.up_proj.weight"}[spec.layer_types[i]])
            continue
        elif spec.layer_types and spec.layer_types[i] == "linear_attention":
            if spec.model_type == "kimi_linear":
                attendus.append(f"model.layers.{i}.linear_attn.q_proj.weight")
                attendus.append(f"model.layers.{i}.linear_attn.conv1d_q.weight")
            else:
                attendus.append(f"model.layers.{i}.linear_attn.qkv.weight")
                attendus.append(f"model.layers.{i}.linear_attn.conv1d.weight")
        elif spec.q_lora_rank:
            attendus.append(f"model.layers.{i}.self_attn.q_a_proj.weight")
        else:
            attendus.append(f"model.layers.{i}.self_attn.q_proj.weight")
        if spec.model_type == "nemotron_h":
            continue                        # couche d'attention seule, sans MLP
        if not spec.mlp_gated:
            attendus.append(f"model.layers.{i}.mlp.up_proj.weight")
            continue
        if spec.is_moe and i >= spec.first_k_dense_replace:
            attendus.append(f"model.layers.{i}.mlp.experts.0.gate_proj.weight")
        else:
            attendus.append(f"model.layers.{i}.mlp.gate_proj.weight")
    manquants = [n for n in attendus if n not in manifest["tensors"]]
    if manquants:
        raise ValueError(
            f"conversion incomplète : {len(manquants)} tenseurs attendus "
            f"absents (premier : {manquants[0]}). L'architecture de la source "
            f"n'est probablement pas prise en charge — rien n'est écrit.")

    if not opts.dry_run:
        writer.flush()
        manifest["weight_map"] = writer.weight_map
        with open(os.path.join(opts.out_dir, "acvram_manifest.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        _copy_tokenizer(model_path, opts.out_dir)
        from .gguf import GGUFFile, is_gguf
        if is_gguf(model_path) and not opts.dry_run:
            GGUFFile(model_path).export_sidecars(opts.out_dir)
        from .exl3 import EXL3Checkpoint, is_exl3
        if is_exl3(model_path) and not opts.dry_run:
            EXL3Checkpoint(model_path).export_sidecars(opts.out_dir)
        report.out_bytes = writer.total_bytes
    else:
        report.out_bytes = sum(report.per_format.values())

    report.mean_out_snr_db = sum(snrs) / len(snrs) if snrs else 0.0
    report.worst_layers = sorted(per_layer, key=lambda d: d["out_snr_db"])
    report.seconds = time.time() - t0
    return report


_TEKKEN_CHAT = (
    "{{ bos_token }}{% for m in messages %}{% if m['role'] == 'system' %}"
    "[SYSTEM_PROMPT]{{ m['content'] }}[/SYSTEM_PROMPT]{% elif m['role'] == 'user' %}"
    "[INST]{{ m['content'] }}[/INST]{% else %}{{ m['content'] }}{{ eos_token }}"
    "{% endif %}{% endfor %}")


def _copy_tokenizer(src: str, dst: str) -> None:
    import shutil
    for fn in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
               "special_tokens_map.json", "generation_config.json", "config.json",
               "chat_template.jinja"):
        p = os.path.join(src, fn)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(dst, fn))
    tekken = os.path.join(src, "tekken.json")
    if os.path.isfile(tekken) and not os.path.isfile(os.path.join(src, "tokenizer.json")):
        # Mistral « tekken » seul (Devstral, Small 3.x) : tokenizer HF
        # reconstruit, gabarit v7-tekken ([SYSTEM_PROMPT]/[INST])
        from transformers.integrations.mistral import convert_tekken_tokenizer
        tok = convert_tekken_tokenizer(tekken)
        tok.chat_template = _TEKKEN_CHAT
        tok.save_pretrained(dst)
        try:                                    # regex de pré-tokenisation corrigée
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(dst, fix_mistral_regex=True).save_pretrained(dst)
        except TypeError:
            pass
        print("  tokenizer reconstruit depuis tekken.json")
