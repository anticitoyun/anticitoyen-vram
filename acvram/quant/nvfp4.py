"""NVFP4 -- 4-bit block-scaled floating point (Blackwell / sm_120).

Layout, matching the NVIDIA/OCP definition that Blackwell tensor cores and
CUTLASS consume:

    element      FP4 E2M1     1 sign + 2 exponent + 1 mantissa
                              magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}
    block scale  FP8 E4M3     one per 16 consecutive elements along K
    global scale FP32         one per tensor

    w[i] ~= code_level(q[i]) * e4m3(block_scale[i//16]) * global_scale

Storage cost per weight:

    4 bits (element) + 8/16 bits (block scale) = 4.5 bpw

i.e. 3.56x smaller than BF16. On 32 GB of VRAM that is ~56 G parameters of
weight, against ~16 G in BF16.

Why a *global* scale on top of the block scale: E4M3 tops out at 448, so a
per-tensor divisor is what keeps every block scale inside the representable
range regardless of the tensor's dynamic range.

This module is pure PyTorch and runs on CPU, which makes it testable without
a GPU; the fused CUDA path in ``acvram.kernels`` must reproduce it bit for
bit (see tests/test_quant_roundtrip.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

__all__ = [
    "E2M1_LEVELS",
    "E4M3_MAX",
    "E2M1_MAX",
    "NVFP4Tensor",
    "quantize_nvfp4",
    "dequantize_nvfp4",
    "pack_e2m1",
    "unpack_e2m1",
    "round_to_e2m1",
]

BLOCK = 16
E2M1_MAX = 6.0
E4M3_MAX = 448.0

# code -> magnitude, index is the 3-bit magnitude field of E2M1
E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# Midpoints between consecutive levels. Round-to-nearest-even on the *code*
# index means alternating strict / non-strict comparisons: a tie must land on
# an even code, so 0.25 rounds down to code 0 while 0.75 rounds up to code 2.
_TIE_DOWN = (0.25, 1.25, 2.5, 5.0)     # ties resolve downward (even code below)
_TIE_UP = (0.75, 1.75, 3.5)            # ties resolve upward   (even code above)


def round_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round magnitudes to the E2M1 grid, returning the 3-bit code 0..7.

    Saturating: anything above 6 clamps to code 7 (E2M1 has no infinity).
    """
    m = x.abs()
    code = (
        (m > _TIE_DOWN[0]).to(torch.uint8)
        + (m >= _TIE_UP[0]).to(torch.uint8)
        + (m > _TIE_DOWN[1]).to(torch.uint8)
        + (m >= _TIE_UP[1]).to(torch.uint8)
        + (m > _TIE_DOWN[2]).to(torch.uint8)
        + (m >= _TIE_UP[2]).to(torch.uint8)
        + (m > _TIE_DOWN[3]).to(torch.uint8)
    )
    return code


def _levels_tensor(device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(E2M1_LEVELS, device=device, dtype=dtype)


def pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit codes (last dim, even length) two per byte.

    Nibble order is low-first: element 2k goes in bits 0..3, element 2k+1 in
    bits 4..7. This is the ``e2m1_x2`` convention CUTLASS and TensorRT use, so
    a packed buffer can be handed to a Blackwell GEMM unchanged.
    """
    if codes.shape[-1] % 2:
        raise ValueError("e2m1 packing needs an even number of elements")
    codes = codes.to(torch.uint8)
    lo = codes[..., 0::2]
    hi = codes[..., 1::2]
    return (lo & 0x0F) | ((hi & 0x0F) << 4)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_e2m1`."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.stack((lo, hi), dim=-1)
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


@dataclass
class NVFP4Tensor:
    """A weight matrix stored in NVFP4.

    ``qweight``     uint8            [out, in//2]     packed E2M1 pairs
    ``block_scale`` float8_e4m3fn    [out, in//16]
    ``global_scale`` float32         scalar
    ``shape``       original logical shape (before any K padding)
    """

    qweight: torch.Tensor
    block_scale: torch.Tensor
    global_scale: torch.Tensor
    shape: tuple[int, ...]
    padded_in: int

    format = "nvfp4"

    @property
    def nbytes(self) -> int:
        return (
            self.qweight.numel()
            + self.block_scale.numel()          # 1 byte per E4M3
            + 4
        )

    @property
    def bits_per_weight(self) -> float:
        n = 1
        for d in self.shape:
            n *= d
        return self.nbytes * 8 / max(1, n)

    def to(self, device, non_blocking: bool = False) -> "NVFP4Tensor":
        return NVFP4Tensor(
            qweight=self.qweight.to(device, non_blocking=non_blocking),
            block_scale=self.block_scale.to(device, non_blocking=non_blocking),
            global_scale=self.global_scale.to(device, non_blocking=non_blocking),
            shape=self.shape,
            padded_in=self.padded_in,
        )

    def state_dict(self, prefix: str = "") -> dict[str, torch.Tensor]:
        return {
            f"{prefix}qweight": self.qweight,
            # safetensors cannot store float8 metadata portably in every
            # version, so scales travel as raw bytes and are reinterpreted.
            f"{prefix}block_scale": self.block_scale.view(torch.uint8),
            f"{prefix}global_scale": self.global_scale,
        }

    @staticmethod
    def from_state_dict(sd: dict[str, torch.Tensor], prefix: str,
                        shape: tuple[int, ...]) -> "NVFP4Tensor":
        q = sd[f"{prefix}qweight"]
        bs = sd[f"{prefix}block_scale"].view(torch.float8_e4m3fn)
        gs = sd[f"{prefix}global_scale"]
        return NVFP4Tensor(q, bs, gs, tuple(shape), q.shape[-1] * 2)


def _pad_k(w: torch.Tensor, block: int) -> tuple[torch.Tensor, int]:
    k = w.shape[-1]
    rem = k % block
    if rem == 0:
        return w, k
    pad = block - rem
    return torch.nn.functional.pad(w, (0, pad)), k


def quantize_nvfp4(
    weight: torch.Tensor,
    block: int = BLOCK,
    global_scale: Optional[torch.Tensor] = None,
) -> NVFP4Tensor:
    """Quantize a 2-D weight ``[out_features, in_features]`` to NVFP4.

    Blocks run along ``in_features`` (the reduction dimension), which is what
    a K-major GEMM wants: every 16-wide slice of K carries its own scale, so a
    single outlier channel cannot flatten a whole row.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got {tuple(weight.shape)}")
    orig_shape = tuple(weight.shape)
    w = weight.detach().to(torch.float32)
    w, orig_k = _pad_k(w, block)
    out_f, k = w.shape
    wb = w.view(out_f, k // block, block)

    if global_scale is None:
        amax = wb.abs().amax()
        # Pick g so the largest block scale lands exactly on E4M3's ceiling.
        g = amax / (E2M1_MAX * E4M3_MAX)
        if not torch.isfinite(g) or g <= 0:
            g = torch.tensor(1.0)
        global_scale = g.reshape(())
    gs = global_scale.to(torch.float32).reshape(())

    block_amax = wb.abs().amax(dim=-1)                    # [out, k/block]
    ideal = block_amax / E2M1_MAX                         # exact per-block scale
    # Represent the block scale in E4M3 -- this is a real, lossy rounding and
    # the kernel must use the *rounded* value, never `ideal`.
    bs_e4m3 = (ideal / gs).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    bs = bs_e4m3.to(torch.float32) * gs                   # effective scale

    safe = bs.clamp(min=torch.finfo(torch.float32).tiny)
    normed = wb / safe.unsqueeze(-1)
    codes = round_to_e2m1(normed)
    codes = torch.where((bs.unsqueeze(-1) > 0), codes, torch.zeros_like(codes))
    sign = (wb < 0).to(torch.uint8) << 3
    codes = codes | sign
    codes = codes.reshape(out_f, k)

    return NVFP4Tensor(
        qweight=pack_e2m1(codes),
        block_scale=bs_e4m3,
        global_scale=gs.clone(),
        shape=orig_shape,
        padded_in=k,
    )


def dequantize_nvfp4(t: NVFP4Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Reference dequantization. The CUDA kernel must match this exactly."""
    codes = unpack_e2m1(t.qweight)                        # [out, k]
    out_f, k = codes.shape
    mag = codes & 0x07
    neg = (codes & 0x08) != 0
    levels = _levels_tensor(codes.device)
    vals = levels[mag.long()]
    vals = torch.where(neg, -vals, vals)
    block = k // t.block_scale.shape[-1]
    vals = vals.view(out_f, k // block, block)
    scale = t.block_scale.to(torch.float32) * t.global_scale.to(torch.float32)
    out = vals * scale.unsqueeze(-1)
    out = out.reshape(out_f, k)
    if k != t.shape[-1]:
        out = out[:, : t.shape[-1]]
    return out.to(dtype)
