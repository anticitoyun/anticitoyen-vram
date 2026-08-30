"""Weight format registry.

One entry per storage format, with the bookkeeping the placement planner
needs (bits per weight) and the minimum compute capability that can *use* it
natively. This is what makes "a different format per GPU" a first-class idea
rather than a special case scattered through the loader.

    format      bpw     native on        notes
    --------    -----   --------------   ----------------------------------
    nvfp4       4.50    sm_100+          FP4 tensor cores (RTX 5090)
    int4_awq    4.16    sm_75+           weight-only, dequant to FP16 in-kernel
    int8        8.13    sm_75+           per-group symmetric, fallback
    bf16       16.00    sm_80+           untouched, for norms / embeddings
    fp16       16.00    any              untouched
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from .int4 import INT4Tensor, dequantize_int4, quantize_int4
from .nvfp4 import NVFP4Tensor, dequantize_nvfp4, quantize_nvfp4

__all__ = ["FormatSpec", "FORMATS", "get_format", "quantize", "dequantize",
           "bits_per_weight", "estimate_bytes", "PlainTensor"]


@dataclass
class PlainTensor:
    """Uncompressed passthrough, so every code path can assume the same API."""

    weight: torch.Tensor
    shape: tuple[int, ...]
    format: str = "bf16"

    @property
    def nbytes(self) -> int:
        return self.weight.numel() * self.weight.element_size()

    @property
    def bits_per_weight(self) -> float:
        return self.weight.element_size() * 8.0

    def to(self, device, non_blocking: bool = False) -> "PlainTensor":
        return PlainTensor(self.weight.to(device, non_blocking=non_blocking),
                           self.shape, self.format)

    def state_dict(self, prefix: str = "") -> dict[str, torch.Tensor]:
        return {f"{prefix}weight": self.weight}

    @staticmethod
    def from_state_dict(sd, prefix, shape, dtype="bf16"):
        return PlainTensor(sd[f"{prefix}weight"], tuple(shape), dtype)


def _q_int8(w: torch.Tensor, group_size: int = 128, **_: Any) -> INT4Tensor:
    """INT8 reuses the grouped-affine machinery with a wider range."""
    raise NotImplementedError("int8 path is handled by quantize() directly")


@dataclass(frozen=True)
class FormatSpec:
    name: str
    bpw: float                       # nominal, at the default group size
    min_sm: int                      # lowest cc that can run it natively
    quantize: Optional[Callable[..., Any]]
    dequantize: Optional[Callable[..., torch.Tensor]]
    compute_dtype: torch.dtype
    description: str

    def bits_per_weight(self, in_features: int, group_size: Optional[int] = None) -> float:
        if self.name == "nvfp4":
            return 4.0 + 8.0 / 16.0
        if self.name == "int4_awq":
            g = group_size or 128
            return 4.0 + 16.0 / g + 4.0 / g
        if self.name == "int8":
            g = group_size or 128
            return 8.0 + 16.0 / g + 8.0 / g
        return self.bpw


FORMATS: dict[str, FormatSpec] = {
    "nvfp4": FormatSpec(
        name="nvfp4", bpw=4.5, min_sm=100,
        quantize=quantize_nvfp4, dequantize=dequantize_nvfp4,
        compute_dtype=torch.bfloat16,
        description="FP4 E2M1 + FP8 E4M3 block scale (16), Blackwell tensor cores",
    ),
    "int4_awq": FormatSpec(
        name="int4_awq", bpw=4.15625, min_sm=75,
        quantize=quantize_int4, dequantize=dequantize_int4,
        compute_dtype=torch.float16,
        description="uint4 group-128 affine, weight-only, dequant to FP16 in-kernel",
    ),
    "int8": FormatSpec(
        name="int8", bpw=8.1875, min_sm=75,
        quantize=None, dequantize=None,
        compute_dtype=torch.float16,
        description="uint8 group-128 affine, fallback for sensitive layers",
    ),
    "bf16": FormatSpec(
        name="bf16", bpw=16.0, min_sm=80,
        quantize=None, dequantize=None, compute_dtype=torch.bfloat16,
        description="uncompressed bfloat16",
    ),
    "fp16": FormatSpec(
        name="fp16", bpw=16.0, min_sm=0,
        quantize=None, dequantize=None, compute_dtype=torch.float16,
        description="uncompressed float16",
    ),
}


def get_format(name: str) -> FormatSpec:
    try:
        return FORMATS[name]
    except KeyError:
        raise KeyError(f"unknown weight format {name!r}; "
                       f"known: {', '.join(FORMATS)}") from None


def bits_per_weight(name: str, in_features: int = 4096,
                    group_size: Optional[int] = None) -> float:
    return get_format(name).bits_per_weight(in_features, group_size)


def estimate_bytes(n_params: int, name: str, group_size: Optional[int] = None) -> int:
    return int(n_params * bits_per_weight(name, group_size=group_size) / 8)


def quantize(weight: torch.Tensor, fmt: str, group_size: Optional[int] = None,
             **kwargs: Any):
    """Quantize a 2-D weight into ``fmt``."""
    spec = get_format(fmt)
    if fmt == "nvfp4":
        return spec.quantize(weight, **kwargs)
    if fmt == "int4_awq":
        return spec.quantize(weight, group_size=group_size or 128, **kwargs)
    if fmt == "int8":
        return _quantize_int8(weight, group_size or 128)
    if fmt in ("bf16", "fp16"):
        dtype = torch.bfloat16 if fmt == "bf16" else torch.float16
        return PlainTensor(weight.detach().to(dtype), tuple(weight.shape), fmt)
    raise KeyError(fmt)


def dequantize(t: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    fmt = getattr(t, "format", None)
    if fmt == "nvfp4":
        return dequantize_nvfp4(t, dtype or torch.bfloat16)
    if fmt == "int4_awq":
        return dequantize_int4(t, dtype or torch.float16)
    if fmt == "int8":
        return _dequantize_int8(t, dtype or torch.float16)
    if isinstance(t, PlainTensor):
        return t.weight.to(dtype) if dtype else t.weight
    if isinstance(t, torch.Tensor):
        return t.to(dtype) if dtype else t
    raise TypeError(f"cannot dequantize {type(t)!r}")


# --------------------------------------------------------------------------
# INT8 grouped affine -- kept here because it shares INT4Tensor's container
# --------------------------------------------------------------------------


@dataclass
class INT8Tensor:
    qweight: torch.Tensor        # uint8 [out, in]
    scales: torch.Tensor         # fp16  [out, ng]
    zeros: torch.Tensor          # uint8 [out, ng]
    group_size: int
    shape: tuple[int, ...]
    format: str = "int8"

    @property
    def nbytes(self) -> int:
        return self.qweight.numel() + self.scales.numel() * 2 + self.zeros.numel()

    @property
    def bits_per_weight(self) -> float:
        n = 1
        for d in self.shape:
            n *= d
        return self.nbytes * 8 / max(1, n)

    def to(self, device, non_blocking: bool = False) -> "INT8Tensor":
        return INT8Tensor(self.qweight.to(device, non_blocking=non_blocking),
                          self.scales.to(device, non_blocking=non_blocking),
                          self.zeros.to(device, non_blocking=non_blocking),
                          self.group_size, self.shape)

    def state_dict(self, prefix: str = "") -> dict[str, torch.Tensor]:
        return {f"{prefix}qweight": self.qweight, f"{prefix}scales": self.scales,
                f"{prefix}zeros": self.zeros}


def _quantize_int8(weight: torch.Tensor, group_size: int) -> INT8Tensor:
    w = weight.detach().to(torch.float32)
    out_f, k = w.shape
    if k % group_size:
        w = torch.nn.functional.pad(w, (0, group_size - k % group_size))
    ng = w.shape[1] // group_size
    wg = w.view(out_f, ng, group_size)
    wmax, wmin = wg.amax(-1, keepdim=True), wg.amin(-1, keepdim=True)
    scale = ((wmax - wmin) / 255.0).clamp(min=1e-9)
    zero = (-wmin / scale).round().clamp(0, 255)
    q = (wg / scale + zero).round().clamp(0, 255).to(torch.uint8).reshape(out_f, -1)
    return INT8Tensor(q, scale.squeeze(-1).to(torch.float16),
                      zero.squeeze(-1).to(torch.uint8), group_size,
                      tuple(weight.shape))


def _dequantize_int8(t: INT8Tensor, dtype: torch.dtype) -> torch.Tensor:
    out_f, k_pad = t.qweight.shape
    ng = k_pad // t.group_size
    q = t.qweight.to(torch.float32).view(out_f, ng, t.group_size)
    out = (q - t.zeros.to(torch.float32).unsqueeze(-1)) * \
          t.scales.to(torch.float32).unsqueeze(-1)
    out = out.reshape(out_f, k_pad)[:, : t.shape[-1]]
    return out.to(dtype)


FORMATS["int8"] = FormatSpec(
    name="int8", bpw=8.1875, min_sm=75,
    quantize=_quantize_int8, dequantize=_dequantize_int8,
    compute_dtype=torch.float16,
    description="uint8 group-128 affine, fallback for sensitive layers",
)
