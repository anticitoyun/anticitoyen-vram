"""SmoothQuant — déplace la difficulté de quantification des activations
vers les poids par une échelle par CANAL D'ENTRÉE, avant fake-quant A4.

Suite du bead anticitoyen-vram-brd étape 1 (verdict A4 : PPL 5,7550, +2,58 %,
seuil dépassé — revue/verdict-a4-fakequant-llama2-7b.md). Hypothèse de
chef, 13/09/2026 : une calibration STATIQUE (comme les poids) pourrait
combler l'écart entre notre mesure et la littérature (QuaRot/SpinQuant/
DuQuant), qui suppose un lissage préalable des activations, pas un
fake-quant nu.

    s_j = max|X_j|^a / max|W_j|^(1-a)     (par canal d'entree j)
    X'_j = X_j / s_j                       W'_{:,j} = W_{:,j} . s_j

Identité algébrique EXACTE avant toute quantification :
    X' @ W'^T = sum_j (X_j/s_j)(W_ij s_j) = sum_j X_j W_ij = X @ W^T
Le lissage ne change donc RIEN au calcul en pleine précision — il ne fait
que redistribuer la plage dynamique entre les deux opérandes AVANT que
chacun soit quantifié séparément, avec sa propre grille et sa propre perte.

alpha = 0 pousse toute la difficulté vers le poids (s petit, X inchangé),
alpha = 1 la pousse entièrement vers l'activation (s grand, W absorbe le
canal difficile). Les deux extrêmes n'ont pas de raison d'être meilleurs
que l'entre-deux — c'est pourquoi on balaie plusieurs valeurs plutôt que
d'en choisir une.
"""
from __future__ import annotations

import torch

from .nvfp4 import dequantize_nvfp4, quantize_nvfp4

__all__ = ["echelle_smoothquant", "replier_smoothquant"]


def echelle_smoothquant(w_absmax_par_entree: torch.Tensor,
                        x_absmax_par_entree: torch.Tensor,
                        alpha: float, eps: float = 1e-5) -> torch.Tensor:
    """s_j = max|X_j|^alpha / max|W_j|^(1-alpha), par canal d'entrée j.

    `eps` protège les deux bases de puissances négatives sur zéro : un canal
    jamais activé (max|X_j| = 0, une porte MoE non routée dans le lot de
    calibration par exemple) ou un poids nul recevrait sinon un exposant
    négatif de zéro, `inf` ou `nan`, en silence.
    """
    if w_absmax_par_entree.shape != x_absmax_par_entree.shape:
        raise ValueError(
            f"formes incompatibles : poids {tuple(w_absmax_par_entree.shape)} "
            f"vs activation {tuple(x_absmax_par_entree.shape)}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha doit etre dans [0, 1], recu {alpha}")
    x = x_absmax_par_entree.to(torch.float64).clamp(min=eps)
    w = w_absmax_par_entree.to(torch.float64).clamp(min=eps)
    s = x.pow(alpha) / w.pow(1.0 - alpha)
    return s.to(torch.float32)


def replier_smoothquant(qweight, x_absmax_par_entree: torch.Tensor,
                        alpha: float, block: int = 16):
    """Replie l'échelle SmoothQuant dans un poids NVFP4 déjà quantifié.

    Déquantifie, calcule `max|W_j|` sur le poids EN PLEINE PRÉCISION (pas
    sur la version déjà quantifiée puis dequantifiée — l'écart serait le
    bruit de quantification des poids, pas leur vraie dynamique), applique
    l'échelle par colonne, requantifie en NVFP4 avec le MÊME bloc que
    l'original (16 : cohérent avec `fakequant_activation`, qui compare des
    formats à granularité d'échelle égale).

    Rend (nouveau_qweight, s) : `s` doit ensuite diviser l'activation
    AVANT le fake-quant E2M1, dans le hook qui appelle cette fonction.
    """
    w = dequantize_nvfp4(qweight, torch.float32)
    w_absmax = w.abs().amax(dim=0)
    s = echelle_smoothquant(w_absmax, x_absmax_par_entree, alpha)
    w_smooth = w * s.unsqueeze(0)
    return quantize_nvfp4(w_smooth, block=block), s
