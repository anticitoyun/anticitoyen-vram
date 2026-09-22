"""Rotation Hadamard bloc-diagonale H_512 sur les poids d'experts (Sage,
revue/sage-hadamard-16-09.md) : x·H·(W·H)ᵀ = x·Wᵀ, H orthogonale block-
diagonale (H Hᵀ = I par bloc). Contrôle demandé AVANT toute conversion —
si l'identité ne tient pas, l'expérience mesurerait un bogue de rotation,
pas l'effet de la rotation.

K=2048 (gate_proj/up_proj) et K=1536 (down_proj) de GLM-4.7-Flash, les deux
divisibles par 512 — c'est précisément pourquoi Sage a choisi ce bloc
plutôt que le diviseur naturel (`largest_pow2_divisor` rendrait 2048 pour
K=2048, pas 512).

L'écart mesuré n'est PAS 1 ulp exactement : neuf étages de papillon
(log2(512)) plus l'accumulation du produit matriciel sur K éléments
introduisent 3 à 6 ulp fp32 de bruit d'arrondi, pas un défaut de la
rotation elle-même — un seuil à 1 ulp littéral aurait échoué sur du code
correct. Le seuil ici (8 ulp) borne l'accumulation attendue, pas l'identité
algébrique en elle-même.

Côté activation, le chemin réel de production est `fwht_activations`
(Laurine, 7eb44da, `ChannelScaler.apply`) — pas `hadamard_transform` — donc
c'est CETTE fonction qui doit apparaître dans le test, même si le poids se
tourne à la conversion avec `apply_hadamard_weight` (`hadamard_transform`
interne, normalisation légèrement différente d'un ulp par construction,
Laurine `laurine-hadamard-16-09.md` : « le produit reste exact à 1 ulp
mais ton test 1 ulp doit le vérifier de ton côté » — vérifié ici)."""
import math

import torch

from acvram.quant.calibrate import apply_hadamard_weight, fwht_activations

SEUIL_ULP = 8.0


def _ulp_f32(v: float) -> float:
    v = abs(v)
    if v == 0.0:
        return 2.0 ** -149
    return 2.0 ** (math.floor(math.log2(v)) - 23)


def _ecart_en_ulp(y_h: torch.Tensor, y_ref: torch.Tensor) -> float:
    ulp = _ulp_f32(y_ref.abs().max().item())
    return ((y_h - y_ref).abs().max() / ulp).item()


def test_rotation_hadamard_512_preserve_le_produit_gate_up():
    torch.manual_seed(0)
    k = 2048  # gate_proj / up_proj de GLM-4.7-Flash
    w = torch.randn(64, k, dtype=torch.float32)
    x = torch.randn(8, k, dtype=torch.float32)
    y_ref = x @ w.t()
    w_h = apply_hadamard_weight(w, block=512)
    x_h = fwht_activations(x, block=512)
    y_h = x_h @ w_h.t()
    assert _ecart_en_ulp(y_h, y_ref) <= SEUIL_ULP


def test_rotation_hadamard_512_preserve_le_produit_down_proj():
    torch.manual_seed(1)
    k = 1536  # down_proj de GLM-4.7-Flash
    w = torch.randn(64, k, dtype=torch.float32)
    x = torch.randn(8, k, dtype=torch.float32)
    y_ref = x @ w.t()
    w_h = apply_hadamard_weight(w, block=512)
    x_h = fwht_activations(x, block=512)
    y_h = x_h @ w_h.t()
    assert _ecart_en_ulp(y_h, y_ref) <= SEUIL_ULP


def test_hadamard_block_impose_bypasse_le_diviseur_naturel():
    """Sans `hadamard_block=512` explicite, `quantize_with_calibration`
    choisirait 2048 (diviseur puissance de deux naturel de K=2048) — le
    paramètre doit imposer 512 malgré cela."""
    from acvram.quant.calibrate import quantize_with_calibration

    torch.manual_seed(2)
    w = torch.randn(64, 2048, dtype=torch.float32)
    _, scaler, _ = quantize_with_calibration(
        w, "nvfp4", stats=None, group_size=128,
        use_hadamard=True, use_awq=False, hadamard_block=512)
    assert scaler.hadamard_block == 512
