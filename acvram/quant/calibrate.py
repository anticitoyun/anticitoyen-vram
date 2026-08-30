"""Making 4 bits actually usable: AWQ channel scaling + Hadamard rotation.

Naive 4-bit round-to-nearest costs roughly 20 dB SNR on a weight matrix,
which is enough to visibly damage a model. Two cheap, orthogonal techniques
recover most of it, and both are *storage-compatible* with the codecs in
:mod:`acvram.quant.nvfp4` and :mod:`acvram.quant.int4` -- they only change
what gets quantized, never how it is packed.

1. Activation-aware channel scaling (AWQ)
   Salient input channels -- the ones activations are large on -- deserve
   more of the 4-bit budget. Scale weight column j up by s_j before
   quantizing and divide the activation by s_j at run time; the product is
   unchanged but the quantization grid now lands where it matters.
   s = mean|x_j| ** alpha, with alpha found by grid search on layer output
   error.

2. Random Hadamard rotation (QuaRot / SpinQuant family)
   Multiplying by an orthogonal Hadamard matrix spreads outliers across
   channels, turning a heavy-tailed distribution into a near-Gaussian one
   that a uniform 4-bit grid fits far better. Applied to both sides it
   cancels exactly:  y = x W^T = (x H)(W H)^T  because H H^T = I.
   This is what makes NVFP4 viable on attention projections.

Both produce a per-layer ``ChannelScaler`` that the runtime applies to the
input activation. The cost at decode time is one elementwise multiply (AWQ)
and one n log n transform (Hadamard) per linear -- negligible against the
GEMM itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from . import formats

__all__ = ["ChannelScaler", "search_channel_scales", "hadamard_transform",
           "largest_pow2_divisor", "apply_hadamard_weight", "ActStats",
           "quantize_with_calibration"]


# --------------------------------------------------------------------------
# Hadamard
# --------------------------------------------------------------------------


def largest_pow2_divisor(n: int, cap: int = 8192) -> int:
    """Largest power of two that divides ``n`` (bounded, for block-diagonal use)."""
    p = 1
    while p * 2 <= min(n, cap) and n % (p * 2) == 0:
        p *= 2
    return p


def hadamard_transform(x: torch.Tensor, block: Optional[int] = None,
                       normalize: bool = True) -> torch.Tensor:
    """Fast Walsh-Hadamard transform over the last dimension.

    When the last dim is not a power of two the transform is applied
    block-diagonally over the largest power-of-two divisor, which is still an
    exact orthogonal map (a direct sum of Hadamards) and still decorrelates
    within each block.
    """
    n = x.shape[-1]
    b = block or largest_pow2_divisor(n)
    if b < 2:
        return x
    if n % b:
        raise ValueError(f"hadamard block {b} does not divide {n}")
    orig_shape = x.shape
    y = x.reshape(-1, n // b, b).clone()
    h = 1
    while h < b:
        y = y.view(y.shape[0], y.shape[1], b // (2 * h), 2, h)
        a = y[..., 0, :]
        c = y[..., 1, :]
        y = torch.stack((a + c, a - c), dim=-2)
        h *= 2
    y = y.reshape(-1, n // b, b)
    if normalize:
        y = y / math.sqrt(b)
    return y.reshape(orig_shape)


def apply_hadamard_weight(weight: torch.Tensor, block: Optional[int] = None) -> torch.Tensor:
    """Rotate a ``[out, in]`` weight along its input dimension: ``W <- W H``."""
    return hadamard_transform(weight, block=block)


# --------------------------------------------------------------------------
# activation statistics
# --------------------------------------------------------------------------


@dataclass
class ActStats:
    """Per-input-channel activation magnitude, collected on calibration data."""

    mean_abs: torch.Tensor          # [in_features]
    max_abs: Optional[torch.Tensor] = None
    n_samples: int = 0

    @staticmethod
    def from_inputs(x: torch.Tensor) -> "ActStats":
        flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
        return ActStats(flat.abs().mean(0), flat.abs().amax(0), flat.shape[0])

    def merge(self, other: "ActStats") -> "ActStats":
        n = self.n_samples + other.n_samples
        if n == 0:
            return self
        w1, w2 = self.n_samples / n, other.n_samples / n
        return ActStats(
            self.mean_abs * w1 + other.mean_abs * w2,
            torch.maximum(self.max_abs, other.max_abs)
            if self.max_abs is not None and other.max_abs is not None else None,
            n,
        )


@dataclass
class ChannelScaler:
    """What the runtime must apply to the activation before the GEMM.

    ``x_eff = hadamard(x) / scale``  when ``hadamard`` is set, else
    ``x_eff = x / scale``.
    """

    scale: Optional[torch.Tensor]      # [in_features], fp16/bf16
    hadamard_block: int = 0

    @property
    def is_identity(self) -> bool:
        return self.scale is None and self.hadamard_block == 0

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if self.hadamard_block:
            x = hadamard_transform(x, block=self.hadamard_block)
        if self.scale is not None:
            x = x / self.scale.to(x.dtype)
        return x

    def to(self, device, non_blocking: bool = False) -> "ChannelScaler":
        return ChannelScaler(
            self.scale.to(device, non_blocking=non_blocking)
            if self.scale is not None else None,
            self.hadamard_block,
        )

    def state_dict(self, prefix: str = "") -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        if self.scale is not None:
            out[f"{prefix}act_scale"] = self.scale
        return out


# --------------------------------------------------------------------------
# AWQ grid search
# --------------------------------------------------------------------------


def _quant_dequant(w: torch.Tensor, fmt: str, group_size: Optional[int]) -> torch.Tensor:
    t = formats.quantize(w, fmt, group_size=group_size)
    return formats.dequantize(t, torch.float32)


def search_channel_scales(
    weight: torch.Tensor,
    stats: Optional[ActStats],
    fmt: str,
    group_size: Optional[int] = None,
    n_grid: int = 20,
    calib_x: Optional[torch.Tensor] = None,
) -> tuple[ChannelScaler, float]:
    """AWQ grid search for the per-channel scale.

    Returns the chosen scaler and the relative output error it achieves. When
    ``calib_x`` is given the objective is the true layer output error on real
    activations; otherwise the activation is approximated by its per-channel
    mean magnitude, which is what AWQ's cheap mode does.
    """
    w = weight.detach().to(torch.float32)
    device = w.device
    k = w.shape[1]

    if stats is None:
        act = torch.ones(k, device=device)
    else:
        act = stats.mean_abs.to(device).to(torch.float32).clamp(min=1e-6)

    if calib_x is not None:
        x = calib_x.reshape(-1, k).to(torch.float32).to(device)
    else:
        # Surrogate: a diagonal probe weighted by channel magnitude reproduces
        # AWQ's per-channel error weighting without storing activations.
        x = torch.diag(act)

    y_ref = x @ w.t()
    ref_norm = y_ref.norm().clamp(min=1e-12)

    best_err = float("inf")
    best_scale: Optional[torch.Tensor] = None

    for i in range(n_grid + 1):
        alpha = i / n_grid
        s = act.pow(alpha)
        s = s / s.mean().clamp(min=1e-12)            # keep the scale centred
        s = s.clamp(min=1e-4, max=1e4)
        wq = _quant_dequant(w * s.unsqueeze(0), fmt, group_size)
        y = (x / s) @ wq.t()
        err = ((y - y_ref).norm() / ref_norm).item()
        if err < best_err:
            best_err, best_scale = err, s.clone()

    assert best_scale is not None
    identity = torch.ones_like(best_scale)
    if torch.allclose(best_scale, identity, atol=1e-3):
        best_scale = None
    return ChannelScaler(
        best_scale.to(torch.float16) if best_scale is not None else None
    ), best_err


def quantize_with_calibration(
    weight: torch.Tensor,
    fmt: str,
    stats: Optional[ActStats] = None,
    group_size: Optional[int] = None,
    use_hadamard: bool = False,
    use_awq: bool = True,
    n_grid: int = 20,
) -> tuple[Any, ChannelScaler, dict]:
    """Full per-layer pipeline: rotate, scale, quantize.

    Order matters. The Hadamard rotation goes first because it changes the
    channel statistics the AWQ search operates on; searching before rotating
    would optimise a scale for a distribution that no longer exists.
    """
    w = weight.detach().to(torch.float32)
    had_block = 0
    if use_hadamard:
        had_block = largest_pow2_divisor(w.shape[1])
        if had_block >= 8:
            w = apply_hadamard_weight(w, had_block)
            if stats is not None:
                rotated = hadamard_transform(stats.mean_abs.reshape(1, -1).to(torch.float32),
                                             block=had_block).abs().reshape(-1)
                stats = ActStats(rotated.clamp(min=1e-6), None, stats.n_samples)
        else:
            had_block = 0

    scaler = ChannelScaler(None, had_block)
    if use_awq:
        found, _ = search_channel_scales(w, stats, fmt, group_size, n_grid)
        scaler = ChannelScaler(found.scale, had_block)

    w_eff = w * scaler.scale.to(torch.float32).unsqueeze(0) \
        if scaler.scale is not None else w
    qt = formats.quantize(w_eff, fmt, group_size=group_size)

    deq = formats.dequantize(qt, torch.float32)
    if scaler.scale is not None:
        deq = deq / scaler.scale.to(torch.float32).unsqueeze(0)

    # Two different errors, and they do not move together.
    #
    #   w_err   how far the reconstructed weights are from the originals
    #   out_err how far the *layer output* is, on activations that look like
    #           the calibration set
    #
    # AWQ deliberately makes w_err worse to make out_err better: it spends
    # grid resolution on the channels the activations are actually large on.
    # Judging AWQ by w_err would reject it every time, so out_err is the
    # number the converter reports and ranks on.
    w_err = ((deq - w).norm() / w.norm().clamp(min=1e-12)).item()

    probe = (stats.mean_abs.to(torch.float32).clamp(min=1e-6)
             if stats is not None else torch.ones(w.shape[1]))
    x = torch.diag(probe.to(w.device))
    y_ref = x @ w.t()
    y_q = x @ deq.t()
    out_err = ((y_q - y_ref).norm() / y_ref.norm().clamp(min=1e-12)).item()

    metrics = {
        "w_rel_err": w_err,
        "w_snr_db": 20 * math.log10(1.0 / max(w_err, 1e-12)),
        "out_rel_err": out_err,
        "out_snr_db": 20 * math.log10(1.0 / max(out_err, 1e-12)),
        "hadamard_block": had_block,
        "awq": scaler.scale is not None,
        "bpw": getattr(qt, "bits_per_weight", 16.0),
    }
    return qt, scaler, metrics
