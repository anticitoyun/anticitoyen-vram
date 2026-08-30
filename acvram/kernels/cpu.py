"""CPU dequantize-and-multiply, for the host memory tier.

Built on demand from ``acvram_cpu.cpp`` into a plain shared object and loaded
through ctypes. No torch extension, so no Python headers and no ninja are
needed on the machine that runs this -- only a C++ compiler, and if even that
is missing everything falls back to the PyTorch reference path.

Why this exists at all: a layer whose weights sit in host RAM can either be
copied to the GPU over PCIe or computed where it already is. Computing in
place only wins if the CPU reads the packed 4-bit weights directly. Going
through ``dequantize() @ x`` materialises a 32-bit copy of the whole matrix
first -- eight times the traffic -- which hands the advantage straight back.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import warnings
from typing import Any, Optional

import torch

from ..quant.int4 import INT4Tensor, dequantize_int4
from ..quant.nvfp4 import NVFP4Tensor, dequantize_nvfp4

__all__ = ["get_cpu_lib", "cpu_kernels_available", "cpu_build_info",
           "int4_matmul_cpu", "nvfp4_matmul_cpu"]

_LIB: Optional[ctypes.CDLL] = None
_TRIED = False
_ERROR = ""
_HAS_AVX2 = False

_FLAGS = ["-O3", "-fPIC", "-shared", "-fopenmp", "-funroll-loops"]


def _cache_dir() -> str:
    d = os.path.expanduser("~/.cache/acvram/cpu")
    os.makedirs(d, exist_ok=True)
    return d


def _build() -> Optional[str]:
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "acvram_cpu.cpp")
    if not os.path.isfile(src):
        raise FileNotFoundError(src)
    with open(src, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()[:16]
    out = os.path.join(_cache_dir(), f"libacvram_cpu-{digest}.so")
    if os.path.isfile(out):
        return out

    compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        raise RuntimeError("no C++ compiler found (install g++)")
    cmd = [compiler, *_FLAGS, "-o", out, src]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{proc.stderr[-2000:]}")
    return out


def get_cpu_lib() -> Optional[ctypes.CDLL]:
    global _LIB, _TRIED, _ERROR, _HAS_AVX2
    if _TRIED:
        return _LIB
    _TRIED = True
    if os.environ.get("ACVRAM_DISABLE_CPU_KERNELS"):
        _ERROR = "disabled by ACVRAM_DISABLE_CPU_KERNELS"
        return None
    try:
        path = _build()
        lib = ctypes.CDLL(path)
        lib.acvram_cpu_has_avx2.restype = ctypes.c_int
        lib.acvram_int4_gemv.restype = None
        lib.acvram_int4_gemv.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64]
        lib.acvram_nvfp4_gemv.restype = None
        lib.acvram_nvfp4_gemv.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        _HAS_AVX2 = bool(lib.acvram_cpu_has_avx2())
        _LIB = lib
    except Exception as exc:                          # noqa: BLE001
        _ERROR = f"{type(exc).__name__}: {exc}"
        warnings.warn(f"acvram: CPU kernels unavailable ({_ERROR}); "
                      f"host-tier layers will use the reference path")
        _LIB = None
    return _LIB


def cpu_kernels_available() -> bool:
    return get_cpu_lib() is not None


def cpu_build_info() -> dict:
    lib = get_cpu_lib()
    return {"available": lib is not None, "error": _ERROR, "avx2": _HAS_AVX2,
            "threads": torch.get_num_threads()}


def _prep(x: torch.Tensor, padded_in: int) -> tuple[torch.Tensor, tuple, int]:
    orig = x.shape
    xf = x.reshape(-1, x.shape[-1]).to(torch.float32)
    if padded_in != xf.shape[-1]:
        xf = torch.nn.functional.pad(xf, (0, padded_in - xf.shape[-1]))
    return xf.contiguous(), orig, xf.shape[0]


def int4_matmul_cpu(x: torch.Tensor, t: INT4Tensor) -> torch.Tensor:
    lib = get_cpu_lib()
    if lib is None or t.qweight.is_cuda:
        return torch.nn.functional.linear(x, dequantize_int4(t, x.dtype))
    xf, orig, n = _prep(x, t.padded_in)
    m = t.qweight.shape[0]
    out = torch.empty(n, m, dtype=torch.float32)
    q = t.qweight.contiguous()
    sc = t.scales.contiguous()
    zr = t.zeros.contiguous()
    lib.acvram_int4_gemv(
        ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(sc.data_ptr()),
        ctypes.c_void_p(zr.data_ptr()), ctypes.c_void_p(xf.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        m, t.padded_in, n, t.group_size)
    return out.to(x.dtype).reshape(*orig[:-1], m)


def nvfp4_matmul_cpu(x: torch.Tensor, t: NVFP4Tensor) -> torch.Tensor:
    lib = get_cpu_lib()
    if lib is None or t.qweight.is_cuda:
        return torch.nn.functional.linear(x, dequantize_nvfp4(t, x.dtype))
    xf, orig, n = _prep(x, t.padded_in)
    m = t.qweight.shape[0]
    out = torch.empty(n, m, dtype=torch.float32)
    q = t.qweight.contiguous()
    bs = t.block_scale.view(torch.uint8).contiguous()
    lib.acvram_nvfp4_gemv(
        ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(bs.data_ptr()),
        ctypes.c_float(float(t.global_scale)), ctypes.c_void_p(xf.data_ptr()),
        ctypes.c_void_p(out.data_ptr()), m, t.padded_in, n)
    return out.to(x.dtype).reshape(*orig[:-1], m)
