"""Fake-quant des ACTIVATIONS, bloc 16 — sonde de recherche, pas un chemin
d'exécution livré.

Bead anticitoyen-vram-brd, étape 1 (Laurine, 13/09/2026) : la RTX 5090 a une
MMA FP4 native (`mma.sync...kind::mxf4nvf4...e2m1.e2m1...ue4m3`) qui exige
les DEUX opérandes en E2M1 avec échelle bloc UE4M3 (W4A4). Notre moteur
quantifie aujourd'hui les poids en NVFP4 mais garde les activations en
BF16 (W4A16, déquantification logicielle) — plafond mesuré à +24 % par
Laurine. Avant d'écrire un noyau W4A4, il faut savoir ce que ça coûte en
qualité : ce module fait le calcul en PyTorch pur (aucun GPU requis), en
« fake quant » — arrondi puis reconstruction en pleine précision, le calcul
matriciel qui suit reste exact. C'est la méthode standard pour ESTIMER une
perte de quantification sans écrire le noyau qui l'exploite.

Trois fonctions, pour les trois régimes du protocole :
  W4A16  bf16_identite         (témoin : aucune perte d'activation)
  W4A4   fake_quantize_nvfp4_activation   (E2M1 bloc 16, échelle UE4M3)
  W4A8   fake_quantize_e4m3_activation    (E4M3 bloc 16, repli mxf8f6f4)

Le bloc est le MÊME (16) dans les trois cas : seule la représentation de
l'élément change, pas la granularité de l'échelle — sinon un écart de PPL
mesurerait deux choses à la fois.
"""
from __future__ import annotations

import torch

from .nvfp4 import BLOCK, dequantize_nvfp4, quantize_nvfp4

__all__ = [
    "fake_quantize_nvfp4_activation",
    "fake_quantize_e4m3_activation",
    "fake_quantize_a8",
    "fake_quantize_w8_row",
]


def fake_quantize_w8_row(w: torch.Tensor, bloc_lignes: int = 4096) -> torch.Tensor:
    """Porte W8r (Sage, 19/09) : un poids [..., M, K] arrondi en int8
    SYMÉTRIQUE PAR LIGNE de sortie (échelle amax_ligne/127, arrondi à demi
    éloigné de zéro, ± 127) puis reconstruit — ce qu'un expert stocké en int8
    par ligne rendrait, sans le stocker. Torch pur, dtype de w conservé.

    EN PLACE et PAR BLOCS de lignes (verdict-porte-w8r-19-09 : la version
    d'un seul tenant allouait des temporaires fp32 de la taille de la pile
    bf16 [128, 768, 2048] = 768 Mio × 3 → OOM) : le pic temporaire est celui
    d'un bloc de ``bloc_lignes`` lignes (4 096 × 2 048 × 4 o = 32 Mio), la
    sortie est w lui-même. Le résultat est identique à la version d'un tenant
    (test), mais ne demande aucune allocation de la taille du poids."""
    plat = w.reshape(-1, w.shape[-1])
    for a in range(0, plat.shape[0], bloc_lignes):
        b = plat[a:a + bloc_lignes]
        bf = b.to(torch.float32)
        s = torch.clamp_min(bf.abs().amax(dim=-1, keepdim=True), 1e-8) / 127.0
        bf.div_(s)
        r = torch.where(bf >= 0, torch.floor(bf + 0.5), -torch.floor(-bf + 0.5)).clamp_(-127.0, 127.0)
        b.copy_((r * s).to(w.dtype))
        del bf, r, s
    return w


def fake_quantize_a8(x: torch.Tensor, fmt: str = "int8") -> torch.Tensor:
    """Porte W4A8 / FP8-MLA (19/09) : ``int8`` = par jeton, amax/127, arrondi
    à demi éloigné de zéro (`quantifier_a8_torch`, au bit du noyau
    `_quant_a8_kernel` des chemins a8/cublas) ; ``e4m3`` = E4M3 par bloc de 16
    (`fake_quantize_e4m3_activation`, le format d'activation de la MMA
    mxf8f6f4). Sortie dans le dtype et la forme de x. Torch pur."""
    if fmt == "e4m3":
        return fake_quantize_e4m3_activation(x)
    if fmt != "int8":
        raise ValueError(f"fake_quantize_a8 : format {fmt!r}, attendu int8 | e4m3")
    from ..kernels.gemm_w8a8 import quantifier_a8_torch
    a, s = quantifier_a8_torch(x)
    return (a.to(torch.float32) * s[:, None]).view(x.shape).to(x.dtype)


def fake_quantize_nvfp4_activation(x: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """Aller-retour E2M1/bloc 16, échelle UE4M3 par bloc + FP32 par appel.

    Réutilise `quantize_nvfp4`/`dequantize_nvfp4` telles quelles : ce sont
    les MÊMES fonctions que celles qui quantifient les poids, donc le même
    format, sans code dupliqué qui pourrait diverger du noyau CUDA que ces
    fonctions documentent devoir reproduire au bit près.

    DYNAMIQUE, pas calibrée à l'avance : chaque appel choisit sa propre
    échelle globale à partir du lot d'activations reçu. C'est le seul choix
    cohérent avec un GEMM W4A4 en production — les activations ne se
    calibrent pas hors ligne, contrairement aux poids.
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    x2 = x.reshape(-1, orig_shape[-1])
    t = quantize_nvfp4(x2, block=block)
    deq = dequantize_nvfp4(t, torch.float32)
    return deq.reshape(orig_shape).to(orig_dtype)


def fake_quantize_e4m3_activation(x: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """Aller-retour E4M3 (FP8), échelle absmax par bloc de MÊME TAILLE que
    NVFP4 — repli W4A8 (mxf8f6f4 dans la nomenclature NVIDIA : A en FP8,
    poids restent FP4).

    Bloc choisi égal à `fake_quantize_nvfp4_activation` (16), pas la
    convention MXFP8 usuelle (bloc 32, échelle UE8M0) : le protocole compare
    des FORMATS D'ÉLÉMENT à granularité d'échelle ÉGALE, sinon l'écart de
    PPL entre A4 et A8 mesurerait aussi un écart de granularité et la
    comparaison ne répondrait plus à la question posée.
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    n = orig_shape[-1]
    x2 = x.reshape(-1, n).to(torch.float32)
    rem = n % block
    pad = 0 if rem == 0 else block - rem
    if pad:
        x2 = torch.nn.functional.pad(x2, (0, pad))
    rows, k = x2.shape
    xb = x2.view(rows, k // block, block)
    e4m3_max = torch.finfo(torch.float8_e4m3fn).max
    amax = xb.abs().amax(dim=-1, keepdim=True)
    scale = (amax / e4m3_max).clamp(min=torch.finfo(torch.float32).tiny)
    q = (xb / scale).to(torch.float8_e4m3fn)
    deq = q.to(torch.float32) * scale
    deq = deq.reshape(rows, k)
    if pad:
        deq = deq[:, :n]
    return deq.reshape(orig_shape).to(orig_dtype)
