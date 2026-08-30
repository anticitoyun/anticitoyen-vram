"""Chargement des noyaux fusionnés, avec un repli qui fonctionne en leur absence.

L'extension CUDA est compilée au premier usage par
``torch.utils.cpp_extension`` et mise en cache sous ``~/.cache/acvram/kernels``.
Si nvcc manque, si l'architecture n'est pas gérée ou si la compilation échoue,
chaque point d'entrée retombe sur l'implémentation PyTorch de référence. Ce
repli est lent — il matérialise une copie 16 bits des poids — mais il est
numériquement identique : le moteur tourne donc correctement sur une machine
sans compilateur, et la suite de tests peut vérifier les noyaux face à lui.
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

# Blackwell exige CUDA 12.8 ou plus récent ; rien de plus ancien ne sait émettre du sm_120.
_MIN_CUDA_FOR_SM120 = (12, 8)


def _arch_flags() -> list[str]:
    """Émet du code pour exactement les architectures présentes, plus un repli PTX."""
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
    """Compile une fois, puis rend le module d'extension, ou None."""
    global _EXT, _TRIED, _ERROR
    if _TRIED:
        return _EXT
    _TRIED = True

    if os.environ.get("ACVRAM_DISABLE_KERNELS"):
        _ERROR = "desactive par ACVRAM_DISABLE_KERNELS"
        return None
    if not torch.cuda.is_available():
        _ERROR = "aucun peripherique CUDA"
        return None

    caps = {torch.cuda.get_device_capability(i)
            for i in range(torch.cuda.device_count())}
    if any(c >= (12, 0) for c in caps) and _cuda_version() < _MIN_CUDA_FOR_SM120:
        _ERROR = (f"un peripherique Blackwell est present mais torch a ete "
                  f"compile pour CUDA {torch.version.cuda} ; sm_120 exige 12.8 "
                  f"ou plus recent. Installez une version cu128 ou cu130.")
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
    except Exception as exc:                      # noqa: BLE001 — signaler, pas planter
        _ERROR = f"{type(exc).__name__}: {exc}"
        _EXT = None
        warnings.warn(f"acvram : repli sur les noyaux de reference ({_ERROR})")
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
    """``x @ W.T`` avec W stocké en NVFP4.

    En dessous de ``gemv_threshold`` lignes, le chemin fusionné l'emporte : les
    poids sont lus une fois et jamais réécrits en 16 bits. Au-dessus,
    matérialiser la matrice et la confier à cuBLAS est plus rapide, car le coût
    de déquantification est payé une seule fois pour tout le lot et le produit
    matriciel de cuBLAS est bien mieux réglé que tout ce qu'on écrirait ici.
    """
    if not t.qweight.is_cuda:
        # Étage hôte : on lit les poids empaquetés sur place, plutôt que de les
        # copier vers le GPU ou de les étendre d'abord en 16 bits.
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

    # Prefill. On essaie d'abord les tensor cores FP4 : à cette taille de lot,
    # l'erreur de quantification de l'activation est amortie et le chemin de
    # données vaut environ quatre fois le BF16. On passe outre s'il est
    # indisponible.
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
