"""Blackwell FP4 tensor-core GEMM, when the installed torch exposes one.

The fused GEMV in ``acvram_kernels.cu`` solves the decode problem: it reads
4-bit weights and never expands them. It does not solve the *prefill* problem,
because prefill is compute bound, and there the interesting number is not how
few bytes were read but how many FLOPs the tensor cores can retire. On
Blackwell the FP4 datapath is roughly four times the BF16 one, and the current
prefill path throws that away by dequantizing to BF16 and calling cuBLAS.

Reaching it needs a block-scaled FP4 GEMM. Writing one from scratch means
CUTLASS; borrowing one means whatever the installed PyTorch exposes, which is
moving quickly and differs between versions. So this module *probes* rather
than assumes: it tries the operation once on a small matrix, remembers whether
it worked, and reports the reason if it did not. ``acvram doctor`` prints that
reason, so the answer to "am I getting tensor-core FP4?" is always a fact
rather than a hope.

The fallback is not a failure mode -- it is the same dequantize-and-cuBLAS
path that was there before, with identical numerics.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from ..quant.nvfp4 import BLOCK, NVFP4Tensor

__all__ = ["fp4_mm_available", "fp4_mm_info", "nvfp4_mm_tensorcore"]

_PROBED = False
_OK = False
_REASON = "not probed"
_IMPL = ""


def _swizzle_scales(bs: torch.Tensor) -> torch.Tensor:
    """Block scales as the GEMM wants them: contiguous E4M3 bytes, row major.

    Kept as its own step because every implementation that has appeared so far
    wants a different layout, and this is the one place to change it.
    """
    return bs.view(torch.float8_e4m3fn).contiguous()


def _probe() -> None:
    global _PROBED, _OK, _REASON, _IMPL
    if _PROBED:
        return
    _PROBED = True

    if os.environ.get("ACVRAM_DISABLE_FP4_GEMM"):
        _REASON = "disabled by ACVRAM_DISABLE_FP4_GEMM"
        return
    if not torch.cuda.is_available():
        _REASON = "no CUDA device"
        return
    caps = {torch.cuda.get_device_capability(i)
            for i in range(torch.cuda.device_count())}
    if not any(c >= (10, 0) for c in caps):
        _REASON = (f"no Blackwell device (found "
                   f"{', '.join(f'sm_{a}{b}' for a, b in sorted(caps))}); "
                   f"FP4 tensor cores need sm_100 or newer")
        return
    if not hasattr(torch, "float4_e2m1fn_x2"):
        _REASON = (f"torch {torch.__version__} has no float4_e2m1fn_x2 dtype; "
                   f"2.8 or newer is needed")
        return
    if not hasattr(torch, "_scaled_mm"):
        _REASON = "torch._scaled_mm is missing"
        return

    dev = next(torch.device(f"cuda:{i}")
               for i in range(torch.cuda.device_count())
               if torch.cuda.get_device_capability(i) >= (10, 0))
    try:
        m = n = k = 128
        a = torch.zeros(m, k // 2, dtype=torch.uint8, device=dev)
        b = torch.zeros(n, k // 2, dtype=torch.uint8, device=dev)
        sa = torch.ones(m, k // BLOCK, dtype=torch.float8_e4m3fn, device=dev)
        sb = torch.ones(n, k // BLOCK, dtype=torch.float8_e4m3fn, device=dev)
        af = a.view(torch.float4_e2m1fn_x2)
        bf = b.view(torch.float4_e2m1fn_x2)
        torch._scaled_mm(af, bf.t(), sa, sb, out_dtype=torch.bfloat16)
        _OK = True
        _IMPL = "torch._scaled_mm(float4_e2m1fn_x2)"
        _REASON = ""
    except Exception as exc:                          # noqa: BLE001
        _REASON = (f"torch._scaled_mm rejected the FP4 operands "
                   f"({type(exc).__name__}: {str(exc)[:180]})")
        _OK = False


def fp4_mm_available() -> bool:
    _probe()
    return _OK


def fp4_mm_info() -> dict:
    _probe()
    return {"available": _OK, "impl": _IMPL, "reason": _REASON,
            "torch": torch.__version__}


def nvfp4_mm_tensorcore(x: torch.Tensor, t: NVFP4Tensor) -> Optional[torch.Tensor]:
    """``x @ W.T`` on the FP4 tensor cores, or None if that path is unavailable.

    The activation is quantized to NVFP4 on the fly with per-16 block scales,
    which is what makes the operation a *tensor-core* FP4 GEMM rather than a
    weight-only one. That is only sound for prefill: at 4 bits the activation
    quantization is a real error source, and it is amortised over a large
    batch but not over a single decoded token. The caller decides.
    """
    if not fp4_mm_available():
        return None
    from ..quant.nvfp4 import quantize_nvfp4

    orig = x.shape
    xf = x.reshape(-1, x.shape[-1])
    if xf.shape[-1] != t.padded_in:
        xf = torch.nn.functional.pad(xf, (0, t.padded_in - xf.shape[-1]))
    try:
        xq = quantize_nvfp4(xf.to(torch.float32))
        out = torch._scaled_mm(
            xq.qweight.view(torch.float4_e2m1fn_x2),
            t.qweight.view(torch.float4_e2m1fn_x2).t(),
            _swizzle_scales(xq.block_scale),
            _swizzle_scales(t.block_scale),
            out_dtype=torch.bfloat16)
        out = out * (xq.global_scale.to(out.device) * t.global_scale.to(out.device))
        return out.to(x.dtype).reshape(*orig[:-1], t.shape[0])
    except Exception:                                 # noqa: BLE001
        global _OK, _REASON
        _OK = False
        _REASON = "the FP4 GEMM probe succeeded but a real call failed"
        return None
