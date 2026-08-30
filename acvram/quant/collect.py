"""Layer-sequential activation statistics, for AWQ.

Activation-aware scaling needs to know which input channels carry large
activations, and that cannot be read off the weights -- it is a property of
the data. Collecting it naively would mean holding the whole model in bf16,
which defeats the purpose on a machine that cannot hold the model in bf16 in
the first place.

So the collection walks the model one block at a time:

    hidden <- embed(calibration tokens)
    for each block:
        materialise the block in bf16 from the source checkpoint
        run `hidden` through it, recording every linear's input magnitudes
        hidden <- the block's output
        free the block

Peak memory is one block, not the model. This is the same structure AWQ and
GPTQ use, and it is what makes calibrating a 70B checkpoint possible here.

Without this, ``--awq`` has nothing to work from and the converter silently
degrades to round-to-nearest -- so the CLI treats "AWQ requested, no stats
collected" as an error rather than quietly doing less than it claims.
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

# A deliberately mixed sample: prose, code, and non-English text, because the
# channels that matter differ between them and a monolingual calibration set
# biases the scales toward whatever it contained.
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
    """Tokenize the calibration corpus, or fall back to noise with a warning."""
    texts: list[str] = []
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            blob = fh.read()
        step = max(1, len(blob) // max(1, n_seqs))
        texts = [blob[i:i + step] for i in range(0, len(blob), step)][:n_seqs]
    else:
        texts = (DEFAULT_CALIB_TEXT * ((n_seqs // len(DEFAULT_CALIB_TEXT)) + 1))[:n_seqs]

    if tokenizer is None:
        # Random ids give uniform channel statistics, which makes AWQ a no-op.
        # Returning them anyway would be worse than saying so.
        raise ValueError(
            "calibration needs a tokenizer; none was found in the model "
            "directory. Pass --no-awq to convert without it.")

    out = []
    for text in texts:
        ids = tokenizer.encode(text)[:seq_len]
        if len(ids) >= 8:
            out.append(ids)
    if not out:
        raise ValueError("calibration corpus produced no usable sequences")
    return out


class _StatCollector:
    """Forward pre-hooks that accumulate per-channel input magnitudes."""

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
    """Walk the checkpoint block by block, recording linear input statistics."""
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

    # One hidden-state tensor per calibration sequence, carried forward.
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
