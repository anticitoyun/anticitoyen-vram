"""Convert a Hugging Face checkpoint into acvram shards.

The output is not one quantized model but one *per device class*: a tensor
destined for the RTX 5090 is written as NVFP4, the same tensor destined for
the RTX 3080 Ti is written as INT4. The placement plan decides which, so the
conversion and the eventual load agree by construction -- there is no run-time
"can this GPU read this format" check to get wrong.

Calibration, when enabled, is layer-sequential in the AWQ manner: hold exactly
one block in bf16, push the calibration hidden states through it to collect
per-channel activation magnitudes, quantize that block, free it, move to the
next. Peak memory is one block, not the whole model, which is what makes it
possible to calibrate a 70B checkpoint on this machine at all.
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
    keep_sensitive_16bit: bool = True  # norms, router, embeddings
    lm_head_format: Optional[str] = None
    n_grid: int = 20
    device: str = "cuda:0"
    dry_run: bool = False
    mixed_precision: str = "auto"     # auto | off
    snr_floor: float = 25.0           # dB of layer-output SNR below which a
                                      # tensor is promoted to a wider format
    max_promotions: float = 0.15      # fraction of tensors allowed to promote


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
        lines = [f"converted {self.model}", ""]
        lines.append(f"  tensors          {self.tensors}")
        lines.append(f"  input            {_h(self.in_bytes)}")
        lines.append(f"  output           {_h(self.out_bytes)}  "
                     f"({self.ratio:.2f}x smaller)")
        for fmt, n in sorted(self.per_format.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {fmt:<12} {_h(n)}")
        lines.append(f"  mean output SNR  {self.mean_out_snr_db:.1f} dB")
        if self.promotions:
            lines.append(f"  promoted         {len(self.promotions)} tensors to a "
                         f"wider format (below the SNR floor):")
            for p in self.promotions[:5]:
                lines.append(f"    {p['name']:<46} {p['from']} -> {p['to']}  "
                             f"{p['before']:.1f} -> {p['after']:.1f} dB")
            if len(self.promotions) > 5:
                lines.append(f"    ... and {len(self.promotions) - 5} more")
        if self.worst_layers:
            lines.append("  worst layers:")
            for w in self.worst_layers[:5]:
                lines.append(f"    {w['name']:<52} {w['out_snr_db']:6.1f} dB")
        lines.append(f"  took             {self.seconds:.1f} s")
        return "\n".join(lines)


def _h(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------


# Promotion ladders. A tensor that quantizes badly moves one rung up rather
# than dragging the whole model to a wider format: spending 8 bits on the few
# per cent of tensors that need them costs a fraction of a bit per weight
# overall and recovers most of the loss.
PROMOTE = {"int4_awq": "int8", "nvfp4": "int8", "int8": "bf16"}

SENSITIVE_SUFFIXES = (
    "layernorm.weight", "norm.weight", "_norm.weight",
    "mlp.gate.weight",            # MoE router: 1 x n_experts, tiny and decisive
    "embed_tokens.weight",
)


class TensorRouter:
    """Decides the storage format of every tensor, from the placement plan."""

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
        """A tensor's format is the format of the device its layer runs on."""
        if self.opts.keep_sensitive_16bit and name.endswith(SENSITIVE_SUFFIXES):
            return "bf16"
        if name.endswith(".bias"):
            return "bf16"
        if name.startswith("lm_head"):
            return (self.opts.lm_head_format
                    or self._layer_fmt.get(self.spec.num_layers - 1, "int4_awq"))
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
        # auto: a rotation earns its keep when the scale group is wide. INT4's
        # 128-wide groups cannot absorb a single outlier channel, so spreading
        # the outliers helps measurably. NVFP4's 16-wide blocks already carry
        # their own scale and the rotation buys little, at the cost of an
        # n log n transform on every activation.
        return fmt == "int4_awq" and not name.endswith(SENSITIVE_SUFFIXES)


# --------------------------------------------------------------------------
# checkpoint reading
# --------------------------------------------------------------------------


def _iter_checkpoint(path: str) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream tensors from a safetensors checkpoint without loading it whole."""
    from safetensors import safe_open

    index_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as fh:
            index = json.load(fh)
        files = sorted(set(index["weight_map"].values()))
    else:
        files = [f for f in sorted(os.listdir(path)) if f.endswith(".safetensors")]
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {path}")

    for fn in files:
        full = os.path.join(path, fn)
        with safe_open(full, framework="pt", device="cpu") as fh:
            for key in fh.keys():
                yield key, fh.get_tensor(key)


class ShardWriter:
    """Accumulates tensors and flushes them into ~4 GiB safetensors shards."""

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


def convert_checkpoint(model_path: str, plan: Plan, opts: ConversionOptions,
                       spec: Optional[ModelSpec] = None,
                       stats: Optional[dict[str, ActStats]] = None,
                       progress: Optional[Callable[[str, int, int], None]] = None
                       ) -> ConversionReport:
    """Quantize every tensor into the format its destination device wants."""
    t0 = time.time()
    spec = spec or load_model_spec(model_path)
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

    snrs: list[float] = []
    per_layer: list[dict] = []
    keys = []

    for name, tensor in _iter_checkpoint(model_path):
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
        qt, scaler, metrics = quantize_with_calibration(
            tensor.to(torch.float32), fmt, st,
            group_size=opts.group_size,
            use_hadamard=router.wants_hadamard(name, fmt),
            use_awq=opts.awq,
            n_grid=opts.n_grid,
        )

        # Mixed precision: a tensor that lands below the floor is worth more
        # bits. Capped, so a badly calibrated run cannot quietly inflate the
        # whole model back to 8 bits.
        if (opts.mixed_precision != "off"
                and metrics["out_snr_db"] < opts.snr_floor
                and fmt in PROMOTE
                and len(report.promotions) < opts.max_promotions * max(1, len(keys) + 1)):
            wider = PROMOTE[fmt]
            q2, s2, m2 = quantize_with_calibration(
                tensor.to(torch.float32), wider, st,
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
        snrs.append(metrics["out_snr_db"])
        per_layer.append({"name": name, **{k: round(v, 2) if isinstance(v, float)
                                           else v for k, v in metrics.items()}})

        sd = qt.state_dict(prefix=f"{name}.")
        sd.update(scaler.state_dict(prefix=f"{name}."))
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

    if not opts.dry_run:
        writer.flush()
        manifest["weight_map"] = writer.weight_map
        with open(os.path.join(opts.out_dir, "acvram_manifest.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        _copy_tokenizer(model_path, opts.out_dir)
        report.out_bytes = writer.total_bytes
    else:
        report.out_bytes = sum(report.per_format.values())

    report.mean_out_snr_db = sum(snrs) / len(snrs) if snrs else 0.0
    report.worst_layers = sorted(per_layer, key=lambda d: d["out_snr_db"])
    report.seconds = time.time() - t0
    return report


def _copy_tokenizer(src: str, dst: str) -> None:
    import shutil
    for fn in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
               "special_tokens_map.json", "generation_config.json", "config.json",
               "chat_template.jinja"):
        p = os.path.join(src, fn)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(dst, fn))
