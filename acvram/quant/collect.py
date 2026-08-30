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
from ..engine.layers import QuantLinear, RMSNorm, RotaryEmbedding
from ..engine.model import Attention, DecoderLayer, ForwardBatch, MLP, MoEBlock
from ..quant.formats import PlainTensor
from .calibrate import ActStats

__all__ = ["collect_activation_stats", "DEFAULT_CALIB_TEXT", "load_calib_ids"]

# Un échantillon délibérément mêlé : prose, code et texte non anglais, parce
# que les canaux qui comptent diffèrent d'un registre à l'autre et qu'un jeu de
# calibration monolingue biaise les échelles vers ce qu'il contenait.
DEFAULT_CALIB_TEXT = [
    "The quick brown fox jumps over the lazy dog. "
    "Machine learning models are trained on large corpora of text.",
    "def quicksort(xs):\n    if len(xs) <= 1:\n        return xs\n"
    "    pivot = xs[len(xs) // 2]\n"
    "    return quicksort([x for x in xs if x < pivot]) + "
    "[x for x in xs if x == pivot] + quicksort([x for x in xs if x > pivot])",
    "La quantification sur quatre bits reduit la taille des poids d'un facteur "
    "proche de quatre, au prix d'une erreur de reconstruction qu'il faut "
    "compenser par une mise a l'echelle par canal.",
    "In a distributed system, consistency, availability and partition "
    "tolerance cannot all be guaranteed simultaneously.",
    "SELECT customer_id, SUM(amount) AS total FROM orders "
    "WHERE created_at >= '2024-01-01' GROUP BY customer_id HAVING total > 1000;",
    "Les modeles a melange d'experts n'activent qu'une fraction de leurs "
    "parametres par jeton, ce qui change completement le calcul de placement.",
]


def load_calib_ids(tokenizer, path: Optional[str], n_seqs: int,
                   seq_len: int, vocab_size: int) -> list[list[int]]:
    """Tokenise le corpus de calibration, ou échoue plutôt que de rendre du bruit."""
    texts: list[str] = []
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            blob = fh.read()
        step = max(1, len(blob) // max(1, n_seqs))
        texts = [blob[i:i + step] for i in range(0, len(blob), step)][:n_seqs]
    else:
        texts = (DEFAULT_CALIB_TEXT * ((n_seqs // len(DEFAULT_CALIB_TEXT)) + 1))[:n_seqs]

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
    location: dict[str, str] = {}
    for fn, h in handles.items():
        for k in h.keys():
            location[k] = fn

    def get(key: str) -> torch.Tensor:
        return handles[location[key]].get_tensor(key)

    collector = _StatCollector()
    rope = RotaryEmbedding(spec.head_dim, spec.max_position_embeddings,
                           spec.rope_theta, spec.rope_scaling)
    embed = get("model.embed_tokens.weight").to(dtype).to(dev)

    # Un tenseur d'état caché par séquence de calibration, transporté d'un bloc à l'autre.
    hiddens = [torch.nn.functional.embedding(
        torch.tensor(ids, device=dev), embed).to(dtype) for ids in calib_ids]

    with torch.inference_mode():
        for i in range(spec.num_layers):
            p = f"model.layers.{i}."
            layer = _build_bf16_layer(spec, p, get, dev, dtype, rope, i)
            collector.attach(layer, p)
            for j, h in enumerate(hiddens):
                n = h.shape[0]
                batch = ForwardBatch(
                    tokens=torch.zeros(n, dtype=torch.long),
                    positions=torch.arange(n, device=dev),
                    seq_lens=[n], query_lens=[n], block_tables=[],
                    slot_mapping=torch.zeros(n, dtype=torch.long),
                    is_prefill=True)
                hiddens[j] = layer(h, batch, None)
            collector.detach()
            del layer
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            if progress:
                progress(i + 1, spec.num_layers)

    for h in handles.values():
        h.__exit__(None, None, None) if hasattr(h, "__exit__") else None
    return collector.stats


def _build_bf16_layer(spec: ModelSpec, prefix: str, get, dev, dtype, rope,
                      index: int) -> DecoderLayer:
    attn = Attention(
        spec,
        _plain(get(prefix + "self_attn.q_proj.weight"), dev, dtype),
        _plain(get(prefix + "self_attn.k_proj.weight"), dev, dtype),
        _plain(get(prefix + "self_attn.v_proj.weight"), dev, dtype),
        _plain(get(prefix + "self_attn.o_proj.weight"), dev, dtype),
        rope)

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
        mlp: torch.nn.Module = MoEBlock(router, experts,
                                        spec.num_experts_per_tok or 2)
    except KeyError:
        mlp = MLP(_plain(get(prefix + "mlp.gate_proj.weight"), dev, dtype),
                  _plain(get(prefix + "mlp.up_proj.weight"), dev, dtype),
                  _plain(get(prefix + "mlp.down_proj.weight"), dev, dtype))

    in_norm = RMSNorm(get(prefix + "input_layernorm.weight").to(dtype).to(dev),
                      spec.rms_norm_eps)
    post_norm = RMSNorm(
        get(prefix + "post_attention_layernorm.weight").to(dtype).to(dev),
        spec.rms_norm_eps)
    return DecoderLayer(index, attn, mlp, in_norm, post_norm, dev)


def _shard_files(path: str) -> list[str]:
    import json
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index, "r", encoding="utf-8") as fh:
            return sorted(set(json.load(fh)["weight_map"].values()))
    return [f for f in sorted(os.listdir(path)) if f.endswith(".safetensors")]
