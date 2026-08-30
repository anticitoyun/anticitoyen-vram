"""Loading the fused kernels, with a working fallback when they are absent.

The CUDA extension is compiled on first use with ``torch.utils.cpp_extension``
and cached under ``~/.cache/acvram/kernels``. If nvcc is missing, the arch is
unsupported, or the build fails, every entry point falls back to the pure
PyTorch reference implementation. That fallback is slow -- it materialises a
16-bit copy of the weights -- but it is numerically identical, so the engine
runs correctly on a machine with no compiler, and the test suite can check the
kernels against it.
"""

from __future__ import annotations

import functools
import os
import sys
import warnings
from typing import Any, Optional

import torch

from ..quant.int4 import INT4Tensor, dequantize_int4
from ..quant.nvfp4 import NVFP4Tensor, dequantize_nvfp4

from .cpu import (cpu_build_info, cpu_kernels_available, int4_matmul_cpu,
                  nvfp4_matmul_cpu)
from .fp4_gemm import fp4_mm_available, fp4_mm_info, nvfp4_mm_tensorcore

__all__ = ["get_extension", "kernels_available", "build_info",
           "nvfp4_dequant", "nvfp4_matmul", "int4_dequant", "int4_matmul",
           "cpu_kernels_available", "cpu_build_info",
           "fp4_mm_available", "fp4_mm_info"]

_EXT: Optional[Any] = None
_TRIED = False
_ERROR: str = ""

# Blackwell needs CUDA 12.8 or newer; anything older cannot emit sm_120 at all.
_MIN_CUDA_FOR_SM120 = (12, 8)


def _arch_flags() -> list[str]:
    """Emit code for exactly the architectures present, plus a PTX fallback."""
    archs: set[tuple[int, int]] = set()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            archs.add(torch.cuda.get_device_capability(i))
    if not archs:
        archs = {(8, 6), (12, 0)}
    flags: list[str] = []
    for major, minor in sorted(archs):
        cc = f"{major}{minor}"
        flags += [f"-gencode=arch=compute_{cc},code=sm_{cc}"]
    highest = max(archs)
    flags += [f"-gencode=arch=compute_{highest[0]}{highest[1]},"
              f"code=compute_{highest[0]}{highest[1]}"]
    return flags


def _cuda_version() -> tuple[int, int]:
    v = torch.version.cuda or "0.0"
    try:
        parts = v.split(".")
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return (0, 0)


def build_info() -> dict:
    caps = []
    if torch.cuda.is_available():
        caps = [f"sm_{a}{b}" for a, b in
                (torch.cuda.get_device_capability(i)
                 for i in range(torch.cuda.device_count()))]
    return {
        "available": kernels_available(),
        "error": _ERROR,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_caps": caps,
        "arch_flags": _arch_flags(),
        "cpu": cpu_build_info(),
        "fp4_tensorcore": fp4_mm_info(),
    }


def get_extension():
    """Compile (once) and return the extension module, or None."""
    global _EXT, _TRIED, _ERROR
    if _TRIED:
        return _EXT
    _TRIED = True

    if os.environ.get("ACVRAM_DISABLE_KERNELS"):
        _ERROR = "disabled by ACVRAM_DISABLE_KERNELS"
        return None
    if not torch.cuda.is_available():
        _ERROR = "no CUDA device"
        return None

    caps = {torch.cuda.get_device_capability(i)
            for i in range(torch.cuda.device_count())}
    if any(c >= (12, 0) for c in caps) and _cuda_version() < _MIN_CUDA_FOR_SM120:
        _ERROR = (f"a Blackwell device is present but torch was built against "
                  f"CUDA {torch.version.cuda}; sm_120 needs 12.8 or newer. "
                  f"Install a cu128/cu130 torch build.")
        warnings.warn(_ERROR)
        return None

    try:
        from torch.utils.cpp_extension import load
        here = os.path.dirname(os.path.abspath(__file__))
        cache = os.path.expanduser("~/.cache/acvram/kernels")
        os.makedirs(cache, exist_ok=True)
        _EXT = load(
            name="acvram_kernels",
            sources=[os.path.join(here, "acvram_kernels.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"] + _arch_flags(),
            extra_cflags=["-O3"],
            build_directory=cache,
            verbose=bool(os.environ.get("ACVRAM_VERBOSE_BUILD")),
        )
    except Exception as exc:                      # noqa: BLE001 - report, don't crash
        _ERROR = f"{type(exc).__name__}: {exc}"
        _EXT = None
        warnings.warn(f"acvram: falling back to reference kernels ({_ERROR})")
    return _EXT


def kernels_available() -> bool:
    return get_extension() is not None


# --------------------------------------------------------------------------
# NVFP4
# --------------------------------------------------------------------------


def nvfp4_dequant(t: NVFP4Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    ext = get_extension()
    if ext is None or not t.qweight.is_cuda:
        return dequantize_nvfp4(t, dtype)
    out = ext.nvfp4_dequant(
        t.qweight.contiguous(),
        t.block_scale.view(torch.uint8).contiguous(),
        float(t.global_scale.item()),
        t.padded_in, dtype)
    return out[:, : t.shape[-1]] if t.padded_in != t.shape[-1] else out


def nvfp4_matmul(x: torch.Tensor, t: NVFP4Tensor,
                 gemv_threshold: int = 8) -> torch.Tensor:
    """``x @ W.T`` with W stored in NVFP4.

    Below ``gemv_threshold`` rows the fused path wins, because the weights are
    read once and never written back out in 16-bit. Above it, materialising
    the matrix and handing it to cuBLAS is faster: the dequantization cost is
    paid once for the whole batch and cuBLAS's GEMM is far better tuned than
    anything hand-rolled here.
    """
    if not t.qweight.is_cuda:
        # Host tier: read the packed weights in place rather than copying them
        # to the GPU or expanding them to 16 bits first.
        return nvfp4_matmul_cpu(x, t)

    ext = get_extension()
    orig_shape = x.shape
    xf = x.reshape(-1, x.shape[-1])
    n = xf.shape[0]

    if ext is not None and n <= gemv_threshold:
        if t.padded_in != xf.shape[-1]:
            xf = torch.nn.functional.pad(xf, (0, t.padded_in - xf.shape[-1]))
        y = ext.nvfp4_gemv(
            t.qweight.contiguous(),
            t.block_scale.view(torch.uint8).contiguous(),
            float(t.global_scale.item()), xf.contiguous(), t.padded_in)
        return y.to(x.dtype).reshape(*orig_shape[:-1], t.shape[0])

    # Prefill. Try the FP4 tensor cores first: at this batch size the
    # activation quantization error is amortised and the datapath is roughly
    # four times BF16. Falls straight through when unavailable.
    if n > gemv_threshold:
        tc = nvfp4_mm_tensorcore(x, t)
        if tc is not None:
            return tc

    w = nvfp4_dequant(t, x.dtype if x.dtype != torch.float32 else torch.bfloat16)
    return torch.nn.functional.linear(x, w.to(x.dtype))


# --------------------------------------------------------------------------
# INT4
# --------------------------------------------------------------------------


def int4_dequant(t: INT4Tensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    ext = get_extension()
    if ext is None or not t.qweight.is_cuda:
        return dequantize_int4(t, dtype)
    out = ext.int4_dequant(
        t.qweight.contiguous(), t.scales.contiguous(), t.zeros.contiguous(),
        t.padded_in, t.group_size, dtype)
    return out[:, : t.shape[-1]] if t.padded_in != t.shape[-1] else out


def int4_matmul(x: torch.Tensor, t: INT4Tensor,
                gemv_threshold: int = 8) -> torch.Tensor:
    if not t.qweight.is_cuda:
        return int4_matmul_cpu(x, t)

    ext = get_extension()
    orig_shape = x.shape
    xf = x.reshape(-1, x.shape[-1])
    n = xf.shape[0]

    if ext is not None and n <= gemv_threshold:
        if t.padded_in != xf.shape[-1]:
            xf = torch.nn.functional.pad(xf, (0, t.padded_in - xf.shape[-1]))
        y = ext.int4_gemv(
            t.qweight.contiguous(), t.scales.contiguous(), t.zeros.contiguous(),
            xf.contiguous(), t.padded_in, t.group_size)
        return y.to(x.dtype).reshape(*orig_shape[:-1], t.shape[0])

    w = int4_dequant(t, x.dtype if x.dtype != torch.float32 else torch.float16)
    return torch.nn.functional.linear(x, w.to(x.dtype))
