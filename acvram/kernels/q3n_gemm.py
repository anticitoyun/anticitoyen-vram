"""Contrat public du GEMV Q3N — miroir de ``fp4_gemm``, verrouillé par
``tests/test_q3n.py``.

Volontairement SANS plancher de capacité : Q3N décode des bits et accumule en
fp32, il n'a besoin d'aucun cœur tensoriel — l'éteindre sous sm_100 aurait
privé la 3080 Ti, précisément la carte où llama.cpp nous bat encore.
"""
from __future__ import annotations

from typing import Optional

import torch

from . import get_extension
from ..quant.q3n import Q3NTensor


def q3n_gemv_available(device: Optional[torch.device] = None) -> bool:
    ext = get_extension()
    return ext is not None and hasattr(ext, "q3n_gemv")


def q3n_gemv_fused(x: torch.Tensor, t: Q3NTensor) -> Optional[torch.Tensor]:
    """``x @ W.T`` par le noyau fusionné, ou None si indisponible ou forme
    inadaptée — jamais d'exception de forme : le registre essaie le chemin
    suivant, une levée ferait tomber la requête entière."""
    if not q3n_gemv_available(x.device):
        return None
    if x.dim() < 1 or x.shape[-1] != t.shape[1] or t.block % 8:
        return None
    if not x.is_cuda or not t.qweight.is_cuda:
        return None
    ext = get_extension()
    return ext.q3n_gemv(t.qweight, t.block_scale.view(torch.uint8),
                        t.global_scale_float(), t.table_gpu(x.device), x,
                        t.shape[1], t.block)
