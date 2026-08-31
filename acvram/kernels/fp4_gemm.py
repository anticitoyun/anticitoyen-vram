"""Produit matriciel FP4 sur tensor cores Blackwell, quand le torch installé en
expose un.

Le GEMV fusionné d'``acvram_kernels.cu`` résout le problème du décodage : il lit
des poids sur 4 bits et ne les étend jamais. Il ne résout pas celui du
*prefill*, car le prefill est limité par le calcul, et le nombre intéressant n'y
est pas le peu d'octets lus mais le nombre d'opérations que les tensor cores
peuvent retirer. Sur Blackwell, le chemin de données FP4 vaut environ quatre
fois le BF16, et le chemin de prefill actuel jette cela en déquantifiant vers le
BF16 pour appeler cuBLAS.

Y accéder demande un produit matriciel FP4 à échelle par bloc. En écrire un de
zéro signifie CUTLASS ; en emprunter un signifie prendre ce que le PyTorch
installé expose, ce qui évolue vite et diffère d'une version à l'autre. Ce
module *sonde* donc au lieu de supposer : il tente l'opération une fois sur une
petite matrice, retient si elle a marché, et rapporte la raison sinon.
``acvram doctor`` affiche cette raison, si bien que la réponse à « est-ce que
j'obtiens du FP4 sur tensor cores ? » est toujours un fait et non un espoir.

Le repli n'est pas un mode de panne : c'est le même chemin
déquantification-puis-cuBLAS qu'avant, à numérique identique.
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
    """Les échelles de bloc dans la disposition « tuilée » qu'exige cuBLAS.

    Le produit FP4 par blocs de 16 ne lit pas les échelles ligne par ligne : il
    attend des tuiles de 128 lignes sur 4 blocs, elles-mêmes rangées en
    sous-tuiles de 32×4×4 octets. Les lignes sont donc remplies au multiple de
    128 et les blocs au multiple de 4 — c'est pour cela qu'un lot de 64
    requêtes était refusé : 64 lignes d'échelles là où la tuile en veut 128.
    """
    m, ng = bs.shape
    bs = bs.view(torch.float8_e4m3fn)
    mb = (m + 127) // 128
    nb = (ng + 3) // 4
    if (mb * 128 != m) or (nb * 4 != ng):
        bs = torch.nn.functional.pad(bs.view(torch.uint8),
                                     (0, nb * 4 - ng, 0, mb * 128 - m)
                                     ).view(torch.float8_e4m3fn)
    t = bs.view(mb, 128, nb, 4).permute(0, 2, 1, 3)       # [mb, nb, 128, 4]
    t = t.reshape(-1, 4, 32, 4).transpose(1, 2)           # [mb*nb, 32, 4, 4]
    return t.reshape(-1).contiguous()


def _probe() -> None:
    global _PROBED, _OK, _REASON, _IMPL
    if _PROBED:
        return
    _PROBED = True

    if os.environ.get("ACVRAM_DISABLE_FP4_GEMM"):
        _REASON = "desactive par ACVRAM_DISABLE_FP4_GEMM"
        return
    if not torch.cuda.is_available():
        _REASON = "aucun peripherique CUDA"
        return
    caps = {torch.cuda.get_device_capability(i)
            for i in range(torch.cuda.device_count())}
    if not any(c >= (10, 0) for c in caps):
        _REASON = (f"aucun peripherique Blackwell (trouve "
                   f"{', '.join(f'sm_{a}{b}' for a, b in sorted(caps))}) ; "
                   f"les tensor cores FP4 exigent sm_100 ou plus recent")
        return
    if not hasattr(torch, "float4_e2m1fn_x2"):
        _REASON = (f"torch {torch.__version__} n'a pas le type float4_e2m1fn_x2 ; "
                   f"il faut 2.8 ou plus recent")
        return
    if not hasattr(torch, "_scaled_mm"):
        _REASON = "torch._scaled_mm est absent"
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
        _REASON = (f"torch._scaled_mm a rejete les operandes FP4 "
                   f"({type(exc).__name__} : {str(exc)[:180]})")
        _OK = False


def fp4_mm_available() -> bool:
    _probe()
    return _OK


def fp4_mm_info() -> dict:
    _probe()
    return {"available": _OK, "impl": _IMPL, "reason": _REASON,
            "torch": torch.__version__}


def nvfp4_mm_tensorcore(x: torch.Tensor, t: NVFP4Tensor) -> Optional[torch.Tensor]:
    """``x @ W.T`` sur les tensor cores FP4, ou None si ce chemin est indisponible.

    L'activation est quantifiée en NVFP4 à la volée, avec des échelles par bloc
    de 16 : c'est ce qui fait de l'opération un produit matriciel FP4 *sur
    tensor cores* et non une opération sur les poids seuls. Cela n'a de sens que
    pour le prefill : à 4 bits, la quantification de l'activation est une source
    d'erreur réelle, amortie sur un grand lot mais pas sur un unique jeton
    décodé. C'est à l'appelant de trancher.
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
        _REASON = "la sonde FP4 a reussi mais un appel reel a echoue"
        return None
