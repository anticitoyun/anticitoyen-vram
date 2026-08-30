"""Transformer building blocks that know about quantization and tiering.

Every linear layer in the model is a :class:`QuantLinear`. It holds its weight
in whatever format its device chose, applies the calibration scaler the
converter produced, and dispatches to the fused kernel when one is available.
A layer whose weights live in host RAM wraps them in :class:`StreamedWeight`,
which copies them to the GPU on a side stream so the transfer for layer i+1
overlaps the compute of layer i.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import kernels
from ..quant.calibrate import ChannelScaler
from ..quant.formats import PlainTensor, dequantize
from ..quant.int4 import INT4Tensor
from ..quant.nvfp4 import NVFP4Tensor

__all__ = ["QuantLinear", "StreamedWeight", "RMSNorm", "RotaryEmbedding",
           "apply_rope", "repeat_kv", "repeat_kv_batched", "attention",
           "batched_decode_attention", "causal_mask"]


class StreamedWeight:
    """A weight that lives in pinned host memory and visits the GPU on demand.

    Pinned memory is what makes the copy asynchronous; a pageable source would
    force the driver to stage it synchronously and the overlap would vanish.
    The double buffer means layer i+1's transfer is already in flight while
    layer i computes, so a streamed layer costs ``max(copy, compute)`` rather
    than their sum -- which is exactly what the planner's cost model assumes.
    """

    def __init__(self, host_tensors: dict[str, torch.Tensor], device: torch.device,
                 n_buffers: int = 2) -> None:
        self.host = {k: (v.pin_memory() if not v.is_pinned() else v)
                     for k, v in host_tensors.items()}
        self.device = device
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self._buffers: list[dict[str, torch.Tensor]] = []
        self._events: list[Any] = []
        self._slot = 0
        self.n_buffers = n_buffers

    def _ensure(self) -> None:
        if self._buffers:
            return
        for _ in range(self.n_buffers):
            self._buffers.append({
                k: torch.empty_like(v, device=self.device)
                for k, v in self.host.items()})
            self._events.append(
                torch.cuda.Event() if self.device.type == "cuda" else None)

    def prefetch(self) -> int:
        """Start the copy into the next buffer; returns its slot."""
        if self.device.type != "cuda":
            return 0
        self._ensure()
        slot = self._slot
        self._slot = (self._slot + 1) % self.n_buffers
        with torch.cuda.stream(self.stream):
            for k, dst in self._buffers[slot].items():
                dst.copy_(self.host[k], non_blocking=True)
            self._events[slot].record(self.stream)
        return slot

    def wait(self, slot: int) -> dict[str, torch.Tensor]:
        if self.device.type != "cuda":
            return self.host
        self._events[slot].wait(torch.cuda.current_stream(self.device))
        return self._buffers[slot]

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.host.values())


class QuantLinear(nn.Module):
    """``y = x @ W.T (+ b)`` where W is stored quantized.

    The scaler is applied to the *input*, never folded into the weight: the
    converter chose it precisely so the weight quantizes well after scaling,
    and folding it back would undo that.
    """

    def __init__(self, qweight: Any, bias: Optional[torch.Tensor] = None,
                 scaler: Optional[ChannelScaler] = None,
                 out_features: Optional[int] = None,
                 in_features: Optional[int] = None) -> None:
        super().__init__()
        self.qweight = qweight
        self.scaler = scaler
        self.bias = bias
        shape = getattr(qweight, "shape", None)
        self.out_features = out_features or (shape[0] if shape else 0)
        self.in_features = in_features or (shape[1] if shape else 0)
        self.streamed: Optional[StreamedWeight] = None
        self._pending_slot: Optional[int] = None

    # -- placement -------------------------------------------------------
    def to_device(self, device: torch.device, streamed: bool = False) -> "QuantLinear":
        if streamed:
            self.streamed = StreamedWeight(
                dict(self.qweight.state_dict()), device)
        else:
            self.qweight = self.qweight.to(device)
            if self.bias is not None:
                self.bias = self.bias.to(device)
        if self.scaler is not None:
            self.scaler = self.scaler.to(device)
        return self

    def prefetch(self) -> None:
        if self.streamed is not None:
            self._pending_slot = self.streamed.prefetch()

    # -- forward ---------------------------------------------------------
    def _resolved_weight(self) -> Any:
        if self.streamed is None:
            return self.qweight
        slot = self._pending_slot if self._pending_slot is not None \
            else self.streamed.prefetch()
        tensors = self.streamed.wait(slot)
        self._pending_slot = None
        return _rehydrate(self.qweight, tensors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scaler is not None and not self.scaler.is_identity:
            x = self.scaler.apply(x)
        w = self._resolved_weight()
        fmt = getattr(w, "format", None)
        if fmt == "nvfp4":
            y = kernels.nvfp4_matmul(x, w)
        elif fmt == "int4_awq":
            y = kernels.int4_matmul(x, w)
        elif isinstance(w, PlainTensor):
            y = F.linear(x, w.weight.to(x.dtype))
        else:
            y = F.linear(x, dequantize(w, x.dtype))
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y

    @property
    def nbytes(self) -> int:
        return getattr(self.qweight, "nbytes", 0)

    def extra_repr(self) -> str:
        fmt = getattr(self.qweight, "format", "?")
        return (f"in={self.in_features}, out={self.out_features}, fmt={fmt}"
                f"{', streamed' if self.streamed else ''}")


def _rehydrate(template: Any, tensors: dict[str, torch.Tensor]) -> Any:
    """Rebuild a quantized tensor object around freshly copied GPU buffers."""
    if isinstance(template, NVFP4Tensor):
        return NVFP4Tensor(
            tensors["qweight"], tensors["block_scale"].view(torch.float8_e4m3fn),
            tensors["global_scale"], template.shape, template.padded_in)
    if isinstance(template, INT4Tensor):
        return INT4Tensor(tensors["qweight"], tensors["scales"], tensors["zeros"],
                          template.group_size, template.shape, template.padded_in)
    if isinstance(template, PlainTensor):
        return PlainTensor(tensors["weight"], template.shape, template.format)
    raise TypeError(f"cannot rehydrate {type(template)!r}")


class RMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        # Accumulate the variance in fp32: at 4-bit weights the activations are
        # already noisy, and a half-precision reduction over 8192 channels adds
        # error for no speed worth having.
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return (x32.to(dtype) * self.weight.to(dtype))


class RotaryEmbedding(nn.Module):
    """RoPE with the scaling variants current models actually ship."""

    def __init__(self, head_dim: int, max_position: int, base: float = 10000.0,
                 scaling: Optional[dict] = None, device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_position = max_position
        self.base = base
        self.scaling = scaling or {}
        inv_freq = self._build_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache_len = 0
        self._cos: Optional[torch.Tensor] = None
        self._sin: Optional[torch.Tensor] = None
        self._dtype = dtype

    def _build_inv_freq(self, device) -> torch.Tensor:
        dim = self.head_dim
        inv = 1.0 / (self.base ** (torch.arange(0, dim, 2, device=device,
                                                dtype=torch.float32) / dim))
        rtype = str(self.scaling.get("rope_type") or self.scaling.get("type") or "")
        factor = float(self.scaling.get("factor", 1.0) or 1.0)
        if rtype in ("linear",):
            return inv / factor
        if rtype in ("dynamic", "ntk"):
            base = self.base * (factor ** (dim / (dim - 2)))
            return 1.0 / (base ** (torch.arange(0, dim, 2, device=device,
                                                dtype=torch.float32) / dim))
        if rtype in ("llama3",):
            low = float(self.scaling.get("low_freq_factor", 1.0))
            high = float(self.scaling.get("high_freq_factor", 4.0))
            orig = float(self.scaling.get("original_max_position_embeddings", 8192))
            wavelen = 2 * math.pi / inv
            low_wl, high_wl = orig / low, orig / high
            smooth = ((orig / wavelen) - low) / max(1e-6, (high - low))
            smoothed = (1 - smooth) * (inv / factor) + smooth * inv
            inv = torch.where(wavelen > low_wl, inv / factor, inv)
            inv = torch.where((wavelen <= low_wl) & (wavelen >= high_wl), smoothed, inv)
            return inv
        return inv

    def _ensure(self, seq_len: int, device, dtype) -> None:
        if self._cos is not None and seq_len <= self._cache_len \
                and self._cos.device == device:
            return
        n = max(seq_len, 1024)
        t = torch.arange(n, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos = emb.cos().to(dtype)
        self._sin = emb.sin().to(dtype)
        self._cache_len = n

    def forward(self, positions: torch.Tensor, device, dtype
                ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure(int(positions.max().item()) + 1 if positions.numel() else 1,
                     device, dtype)
        return self._cos[positions], self._sin[positions]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``q``/``k`` are [tokens, heads, dim]; cos/sin are [tokens, dim]."""
    cos = cos.unsqueeze(1).to(q.dtype)
    sin = sin.unsqueeze(1).to(q.dtype)
    return (q * cos + _rotate_half(q) * sin,
            k * cos + _rotate_half(k) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand GQA key/value heads to match the query head count."""
    if n_rep == 1:
        return x
    t, h, d = x.shape
    return x.unsqueeze(2).expand(t, h, n_rep, d).reshape(t, h * n_rep, d)


def causal_mask(q_len: int, kv_len: int, q_offset: int, device,
                dtype: torch.dtype) -> Optional[torch.Tensor]:
    """Mask for a query block that starts at absolute position ``q_offset``.

    ``F.scaled_dot_product_attention(is_causal=True)`` aligns the triangle to
    the *top left*, which is only correct when the query covers the whole
    sequence. The moment a prefill is chunked -- or a prefix is served from
    cache and only the tail is prefilled -- the query block starts partway
    through the sequence and the built-in flag silently masks the wrong
    cells. Prefix caching and chunked prefill both depend on getting this
    right, so the mask is built explicitly whenever the query is offset.
    """
    if q_len == 1:
        return None                       # decode attends to everything
    if q_offset == 0 and q_len == kv_len:
        return None                       # the built-in causal flag is correct
    rows = torch.arange(q_offset, q_offset + q_len, device=device).unsqueeze(1)
    cols = torch.arange(kv_len, device=device).unsqueeze(0)
    allowed = cols <= rows
    mask = torch.zeros(q_len, kv_len, device=device, dtype=dtype)
    return mask.masked_fill(~allowed, float("-inf"))


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              causal: bool = True, scale: Optional[float] = None,
              q_offset: int = 0) -> torch.Tensor:
    """Scaled dot-product attention over ``[tokens, heads, dim]`` tensors.

    Delegates to PyTorch's SDPA, which picks FlashAttention on any GPU that
    supports it. The transposes are views, not copies.
    """
    qh = q.transpose(0, 1).unsqueeze(0)          # [1, heads, tq, dim]
    kh = k.transpose(0, 1).unsqueeze(0)
    vh = v.transpose(0, 1).unsqueeze(0)
    q_len, kv_len = q.shape[0], k.shape[0]
    mask = causal_mask(q_len, kv_len, q_offset, q.device, q.dtype) if causal else None
    if mask is not None:
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=scale)
    else:
        use_causal = bool(causal and q_len > 1 and q_offset == 0
                          and q_len == kv_len)
        out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=use_causal,
                                             scale=scale)
    return out.squeeze(0).transpose(0, 1).contiguous()


def batched_decode_attention(q: torch.Tensor, keys: list[torch.Tensor],
                             values: list[torch.Tensor], n_rep: int,
                             scale: float) -> torch.Tensor:
    """One SDPA call for a whole decode batch, instead of one per sequence.

    Sequences have different context lengths, so the keys are right-padded to
    the longest and the padding is masked out. That costs
    ``batch x (max_len - len)`` wasted key slots; against it, the Python loop
    disappears and the GPU sees one launch instead of ``batch``. At batch 16
    the launch overhead alone was the larger cost.
    """
    b = len(keys)
    lens = [kk.shape[0] for kk in keys]
    max_len = max(lens)
    hq, d = q.shape[1], q.shape[2]
    hkv = keys[0].shape[1]
    device, dtype = q.device, q.dtype

    kpad = torch.zeros(b, max_len, hkv, d, device=device, dtype=dtype)
    vpad = torch.zeros(b, max_len, hkv, d, device=device, dtype=dtype)
    for i, (kk, vv) in enumerate(zip(keys, values)):
        kpad[i, : lens[i]] = kk
        vpad[i, : lens[i]] = vv

    kh = repeat_kv_batched(kpad, n_rep).permute(0, 2, 1, 3)   # [b, hq, s, d]
    vh = repeat_kv_batched(vpad, n_rep).permute(0, 2, 1, 3)
    qh = q.unsqueeze(2)                                       # [b, hq, 1, d]

    valid = torch.arange(max_len, device=device).unsqueeze(0) < \
        torch.tensor(lens, device=device).unsqueeze(1)        # [b, s]
    mask = torch.zeros(b, 1, 1, max_len, device=device, dtype=dtype)
    mask = mask.masked_fill(~valid[:, None, None, :], float("-inf"))

    out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=scale)
    return out.squeeze(2)                                     # [b, hq, d]


def repeat_kv_batched(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """``[b, s, h_kv, d]`` -> ``[b, s, h_kv * n_rep, d]``."""
    if n_rep == 1:
        return x
    b, s, h, d = x.shape
    return x.unsqueeze(3).expand(b, s, h, n_rep, d).reshape(b, s, h * n_rep, d)
