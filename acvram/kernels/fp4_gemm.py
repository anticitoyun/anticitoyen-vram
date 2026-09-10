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

import warnings

import torch

from ..quant.nvfp4 import BLOCK, NVFP4Tensor

__all__ = ["fp4_mm_available", "fp4_mm_info", "nvfp4_mm_tensorcore"]

_PROBED = False
_OK = False
_REASON = "not probed"
_IMPL = ""

# Extinction par CAPACITE de carte, et non globale.
#
# L'echec d'un appel reel eteignait le chemin FP4 pour tout le processus. Sur
# une machine a cartes inegales — une Blackwell et une Ampere, par exemple —
# un echec survenu sur celle qui n'a pas les tensor cores FP4 privait aussi
# l'autre, qui les a. Le repli restait correct, mais il coutait le gain sur la
# carte capable.
#
# La cle est la CAPACITE (major, minor), pas l'index du peripherique : un
# changement de CUDA_VISIBLE_DEVICES renumerote les cartes, et un etat range
# par index se retrouverait attribue a la mauvaise.
#
# Ce qui reste GLOBAL, a dessein : les echecs qui ne dependent pas de la carte
# — type float4_e2m1fn_x2 absent, _scaled_mm absent, desactivation par
# l'environnement. Les rendre par carte multiplierait le cout de sonde sans
# rien changer au verdict.
_ETEINT_PAR_CAPACITE: dict[tuple[int, int], str] = {}


def _capacite(dev) -> tuple[int, int]:
    """Capacite de calcul de la carte visee, ou (0, 0) hors CUDA."""
    try:
        return tuple(torch.cuda.get_device_capability(dev))
    except Exception:                                 # noqa: BLE001
        return (0, 0)


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


def fp4_mm_available(device=None) -> bool:
    """Le chemin FP4 est-il utilisable, pour CETTE carte.

    Sans argument, la question porte sur le processus : la sonde a-t-elle
    reussi quelque part. Avec un peripherique, elle porte aussi sur les
    extinctions locales survenues sur des cartes de meme capacite.
    """
    _probe()
    if not _OK:
        return False
    if device is None:
        return True
    return _capacite(device) not in _ETEINT_PAR_CAPACITE


def fp4_mm_info() -> dict:
    _probe()
    return {"available": _OK, "impl": _IMPL, "reason": _REASON,
            "torch": torch.__version__,
            "eteint_par_capacite": {f"sm_{a}{b}": r
                                    for (a, b), r in _ETEINT_PAR_CAPACITE.items()}}


def nvfp4_mm_tensorcore(x: torch.Tensor, t: NVFP4Tensor) -> Optional[torch.Tensor]:
    """``x @ W.T`` sur les tensor cores FP4, ou None si ce chemin est indisponible.

    L'activation est quantifiée en NVFP4 à la volée, avec des échelles par bloc
    de 16 : c'est ce qui fait de l'opération un produit matriciel FP4 *sur
    tensor cores* et non une opération sur les poids seuls. Cela n'a de sens que
    pour le prefill : à 4 bits, la quantification de l'activation est une source
    d'erreur réelle, amortie sur un grand lot mais pas sur un unique jeton
    décodé. C'est à l'appelant de trancher.
    """
    if getattr(t, "global_scale_rows", None) is not None:
        return None     # pile à une échelle par segment : seul le GEMV et la
                        # déquantification la lisent (GLM-4.7, 6/09/2026)
    # La carte est passee : sans elle, une extinction survenue sur une autre
    # capacite serait ignoree ici, et la correction ne servirait a rien.
    if not fp4_mm_available(x.device):
        return None
    # ``torch._scaled_mm`` veut une dimension contractée multiple de 16 octets,
    # soit 32 poids FP4. Une couche qui ne s'y plie pas — 688 colonnes, donc
    # 344 octets — n'est pas un défaut du chemin : c'est une forme qu'il ne sait
    # pas prendre. La refuser ici, et non par l'exception plus bas, évite qu'une
    # seule couche atypique n'éteigne les tensor cores pour tout le modèle.
    if t.padded_in % 32 or t.qweight.shape[-1] % 16:
        return None
    # Garde par carte, AVANT toute tentative.
    #
    # Les tensor cores FP4 exigent sm_100. Sans ce test, une carte plus
    # ancienne atteignait l'appel reel, echouait, et eteignait le chemin — pour
    # elle et, avant la correction du 8 septembre 2026, pour toutes les autres.
    # Mieux vaut ne pas essayer que d'essayer, echouer, et devoir reparer les
    # degats de l'echec.
    if _capacite(x.device) < (10, 0):
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
    except Exception as exc:                          # noqa: BLE001
        # Une capture de graphe en cours n'est PAS une panne du chemin.
        #
        # Mesuré le 10/09/2026 : à douze séquences concurrentes, le décodage
        # franchit le seuil de lot, ``torch._scaled_mm`` entre dans la région
        # de capture, cuBLASLt y interroge son heuristique et réserve son
        # espace de travail — opération interdite pendant une capture. Le
        # même appel réussit à huit séquences, et réussit à douze hors
        # capture : c'est le CONTEXTE qui échoue, pas le chemin.
        #
        # Éteindre ici transformait cet accident en état permanent du
        # processus : un serveur ayant vu une seule fois plus de huit
        # séquences perdait les tensor cores FP4 pour toutes ses requêtes
        # suivantes, sans autre trace qu'un avertissement. Et rendre ``None``
        # laisserait l'appelant bâtir un graphe sur une capture déjà
        # invalidée. On relaie donc l'erreur telle quelle.
        if torch.cuda.is_current_stream_capturing():
            raise
        # Extinction globale : elle ne doit plus concerner qu'une panne du
        # chemin lui-même, les formes étant écartées plus haut. Elle change le
        # résultat de toutes les couches suivantes, donc elle s'annonce — le
        # repli silencieux avait fait diverger la première requête d'un
        # processus de toutes les suivantes.
        cap = _capacite(x.device)
        _ETEINT_PAR_CAPACITE[cap] = (
            f"la sonde FP4 a reussi mais un appel reel a echoue : {exc}")
        warnings.warn(f"acvram : chemin FP4 tensor cores eteint pour sm_{cap[0]}{cap[1]} "
                      f"apres un echec reel, repli sur les noyaux fusionnes ({exc}). "
                      f"Les cartes d'une autre capacite le gardent.")
        return None


# --------------------------------------------------------------------------
# Prefill W4A8 : poids NVFP4 élargis en FP8 par colonne, activation FP8 par
# ligne, produit sur les tensor cores FP8 (rowwise scaled_mm).
#
# Le chemin W4A4 quantifie l'activation en FP4 : ~9,5 % d'erreur relative par
# couche, la plus grosse source d'imprécision du prefill. En FP8 l'activation
# garde ~2 % ; le poids repasse par une matérialisation FP8 par couche et par
# lot — c'est le prix, amorti sur le lot comme l'était la déquantification du
# repli bf16, mais le GEMM court ensuite deux fois plus vite qu'en bf16.
# --------------------------------------------------------------------------

_F8_MAX = 448.0


def nvfp4_mm_w4a8(x: torch.Tensor, t: NVFP4Tensor) -> Optional[torch.Tensor]:
    """``x @ W.T`` en FP8×FP8 rowwise, ou None si indisponible."""
    if getattr(t, "global_scale_rows", None) is not None:
        return None     # pile à une échelle par segment : seul le GEMV et la
                        # déquantification la lisent (GLM-4.7, 6/09/2026)
    if not torch.cuda.is_available():
        return None
    if torch.cuda.get_device_capability(x.device) < (8, 9):
        return None                       # e4m3 sur tensor cores : Ada+
    from . import nvfp4_dequant

    orig = x.shape
    xf = x.reshape(-1, x.shape[-1]).to(torch.float32)
    w = nvfp4_dequant(t, torch.float32)   # [M, K] — matérialisation par lot
    if xf.shape[-1] != w.shape[-1]:
        xf = torch.nn.functional.pad(xf, (0, w.shape[-1] - xf.shape[-1]))
    try:
        sx = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / _F8_MAX
        sw = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / _F8_MAX
        x8 = (xf / sx).clamp(-_F8_MAX, _F8_MAX).to(torch.float8_e4m3fn)
        w8 = (w / sw).clamp(-_F8_MAX, _F8_MAX).to(torch.float8_e4m3fn)
        y = torch._scaled_mm(x8, w8.t(), sx, sw.t(), out_dtype=torch.bfloat16)
        return y.to(x.dtype).reshape(*orig[:-1], t.shape[0])
    except Exception:                     # noqa: BLE001 — repli silencieux
        return None
