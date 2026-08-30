"""INT4 group-wise weight-only quantization (Ampere / sm_86 path).

The RTX 3080 Ti is GA102: its tensor cores do FP16/BF16/TF32/INT8, but there
is no FP8 and no FP4 datapath, so a 4-bit *compute* format is not available.
The way to still get a 4x smaller footprint is weight-only quantization --
store 4 bits, dequantize a tile to FP16 inside the kernel, and feed the
ordinary FP16 tensor cores. Memory traffic (the actual bottleneck during
decode) drops 4x; arithmetic stays FP16.

Layout, asymmetric, AWQ-compatible:

    q[i]   uint4                   group of 128 along K
    scale  fp16   per group
    zero   uint4  per group

    w[i] ~= (q[i] - zero) * scale

Storage cost per weight:

    4 + 16/128 + 4/128 = 4.156 bpw   (3.85x smaller than FP16)

The "AWQ" part proper -- activation-aware channel scaling -- lives in
:mod:`acvram.quant.calibrate`; this module is the pure integer codec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

__all__ = ["INT4Tensor", "quantize_int4", "dequantize_int4", "pack_uint4", "unpack_uint4"]

GROUP = 128


def pack_uint4(x: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit values, two per byte, low nibble first."""
    if x.shape[-1] % 2:
        x = torch.nn.functional.pad(x, (0, 1))
    x = x.to(torch.uint8)
    return (x[..., 0::2] & 0x0F) | ((x[..., 1::2] & 0x0F) << 4)


def unpack_uint4(packed: torch.Tensor) -> torch.Tensor:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    return torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1],
                                                 packed.shape[-1] * 2)


@dataclass
class INT4Tensor:
    qweight: torch.Tensor        # uint8 [out, in//2]
    scales: torch.Tensor         # fp16  [out, in//group]
    zeros: torch.Tensor          # uint8 [out, ceil(in/group/2)] packed uint4
    group_size: int
    shape: tuple[int, ...]
    padded_in: int

    format = "int4_awq"

    @property
    def nbytes(self) -> int:
        return self.qweight.numel() + self.scales.numel() * 2 + self.zeros.numel()

    @property
    def bits_per_weight(self) -> float:
        n = 1
        for d in self.shape:
            n *= d
        return self.nbytes * 8 / max(1, n)

    def to(self, device, non_blocking: bool = False) -> "INT4Tensor":
        return INT4Tensor(
            self.qweight.to(device, non_blocking=non_blocking),
            self.scales.to(device, non_blocking=non_blocking),
            self.zeros.to(device, non_blocking=non_blocking),
            self.group_size, self.shape, self.padded_in,
        )

    def state_dict(self, prefix: str = "") -> dict[str, torch.Tensor]:
        return {
            f"{prefix}qweight": self.qweight,
            f"{prefix}scales": self.scales,
            f"{prefix}zeros": self.zeros,
        }

    @staticmethod
    def from_state_dict(sd: dict[str, torch.Tensor], prefix: str,
                        shape: tuple[int, ...], group_size: int = GROUP) -> "INT4Tensor":
        q = sd[f"{prefix}qweight"]
        return INT4Tensor(q, sd[f"{prefix}scales"], sd[f"{prefix}zeros"],
                          group_size, tuple(shape), q.shape[-1] * 2)


def quantize_int4(weight: torch.Tensor, group_size: int = GROUP,
                  symmetric: bool = False) -> INT4Tensor:
    """Quantize ``[out_features, in_features]`` to grouped uint4.

    Asymmetric by default: LLM weight groups are rarely centred on zero, and
    spending the zero-point buys roughly half a bit of effective precision
    for 4 bits of metadata per 128 weights.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got {tuple(weight.shape)}")
    orig_shape = tuple(weight.shape)
    w = weight.detach().to(torch.float32)
    out_f, k = w.shape
    if k % group_size:
        w = torch.nn.functional.pad(w, (0, group_size - k % group_size))
    k_pad = w.shape[1]
    ng = k_pad // group_size
    wg = w.view(out_f, ng, group_size)

    if symmetric:
        amax = wg.abs().amax(dim=-1, keepdim=True)
        scale = (amax / 7.0).clamp(min=1e-8)
        zero = torch.full_like(scale, 8.0)
    else:
        wmax = wg.amax(dim=-1, keepdim=True)
        wmin = wg.amin(dim=-1, keepdim=True)
        # Never let a constant group collapse the scale to zero.
        scale = ((wmax - wmin) / 15.0).clamp(min=1e-8)
        zero = (-wmin / scale).round().clamp(0, 15)

    q = (wg / scale + zero).round().clamp(0, 15).to(torch.uint8)
    q = q.reshape(out_f, k_pad)

    scales = scale.squeeze(-1).to(torch.float16)          # [out, ng]
    zeros = pack_uint4(zero.squeeze(-1).to(torch.uint8))  # [out, ceil(ng/2)]

    return INT4Tensor(pack_uint4(q), scales, zeros, group_size,
                      orig_shape, k_pad)


def dequantize_int4(t: INT4Tensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Reference dequantization; the fused CUDA kernel must match this."""
    q = unpack_uint4(t.qweight).to(torch.float32)         # [out, k_pad]
    out_f, k_pad = q.shape
    ng = k_pad // t.group_size
    zeros = unpack_uint4(t.zeros)[:, :ng].to(torch.float32)
    scales = t.scales[:, :ng].to(torch.float32)
    qg = q.view(out_f, ng, t.group_size)
    out = (qg - zeros.unsqueeze(-1)) * scales.unsqueeze(-1)
    out = out.reshape(out_f, k_pad)
    if k_pad != t.shape[-1]:
        out = out[:, : t.shape[-1]]
    return out.to(dtype)
