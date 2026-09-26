"""poste7-awq-relu2-garde-repli-17-09 geste (d), CORRIGÉ par
poste7-awq-plancher-median-faute-18-09 : plancher borné sur l'ÉTENDUE
(max(mean_abs)/4096), pas sur une statistique de POSITION comme la
médiane, dans la recherche de canaux salients.

Cause réelle du 145/5935 de Nemotron calibA (pas le nombre
d'échantillons, déjà exclu) : l'entrée de `down_proj` chez nemotron_h est
ReLU²(up(x)) -- creuse par construction, les canaux jamais activés sur le
corpus s'écrasent au plancher contre ~1 ailleurs, étendue 1,7e6× mesurée.

Le premier correctif (geste d, commit 982ad80, `1e-2 × médiane`) était
LUI-MÊME fautif : sur un tenseur où PLUS DE LA MOITIÉ des canaux sont
écrasés (le régime normal de ReLU², pas un cas limite), la médiane
elle-même tombe à ~0, et `1e-2 × médiane` devient un plancher PLUS BAS
que l'ancien 1e-6 -- ÉLARGISSANT l'étendue au lieu de la borner. Mesuré
sur 10 tenseurs Nemotron : 6/10 aggravés jusqu'à ×319 (étendue 511 ->
162589). Un plancher sur le MAXIMUM du tenseur (`max(mean_abs)/4096`)
tient quelle que soit la fraction de canaux nuls.
"""
import torch

from acvram.quant.calibrate import (ActStats, _magnitude_avec_plancher_relatif,
                                    quantize_with_calibration)


def _stats_relu2_creuse(k=256, graine=0, frac_nulle=0.4):
    """Motif ReLU² réaliste : `frac_nulle` des canaux jamais activés sur le
    corpus (magnitude quasi nulle), le reste avec une magnitude normale --
    un motif DISTRIBUÉ sur toute la largeur du tenseur, pas un seul canal
    isolé comme le witness du seuil d'échantillons."""
    torch.manual_seed(graine)
    m = torch.rand(k) * 0.9 + 0.1        # canaux actifs : [0.1, 1.0]
    jamais_actifs = torch.rand(k) < frac_nulle
    m[jamais_actifs] = 1e-6              # ce que ReLU² rend pour ces canaux
    return ActStats(m, m, 128)


def _plancher_median_fautif(mean_abs: torch.Tensor) -> torch.Tensor:
    """Le correctif du 982ad80, FIGÉ ici comme bras cassant -- REGLES §5 :
    la faute d'hier doit rester un témoin qui rougit, pas une note dans un
    commentaire. Si la médiane tombe à ~0 (majorité de canaux nuls), le
    plancher devient quasi nul lui aussi -- l'exact mécanisme du bogue."""
    plancher = 1e-2 * mean_abs.median().clamp(min=1e-12)
    return mean_abs.clamp(min=plancher)


def test_plancher_borne_letendue_meme_a_90_pour_cent_de_canaux_nuls():
    """90 % des canaux jamais activés -- le régime le plus dur pour un
    plancher par médiane (qui y tombe lui-même à ~0), et un cas parfaitement
    normal pour ReLU² sur un expert peu actif. Un changement qui doit
    casser : remplacer l'appel par `_plancher_median_fautif` (le bogue
    d'hier) fait sortir l'étendue très au-delà de 4096."""
    torch.manual_seed(1)
    w = torch.randn(256, 256) * 0.02
    stats = _stats_relu2_creuse(k=256, graine=0, frac_nulle=0.9)
    _, scaler, _ = quantize_with_calibration(w, "nvfp4", stats, use_awq=True)
    assert scaler.scale is not None, "la recherche doit trouver une echelle non triviale"
    etendue = (scaler.scale.max() / scaler.scale.min().clamp(min=1e-12)).item()
    assert etendue <= 4096, f"etendue de l'echelle AWQ = {etendue:.1f} (attendu <= 4096)"


def test_bras_casse_le_plancher_median_dhier_depasse_4096_a_90_pour_cent_de_canaux_nuls(monkeypatch):
    """Le bogue de `poste7-awq-relu2-garde-repli-17-09` (982ad80), figé en
    test : le plancher par médiane laisse l'étendue exploser dès qu'une
    majorité de canaux sont nuls -- exactement le régime que le correctif
    du 18/09 répare."""
    import acvram.quant.calibrate as calmod
    monkeypatch.setattr(calmod, "_magnitude_avec_plancher_relatif", _plancher_median_fautif)

    torch.manual_seed(1)
    w = torch.randn(256, 256) * 0.02
    stats = _stats_relu2_creuse(k=256, graine=0, frac_nulle=0.9)
    _, scaler, _ = calmod.quantize_with_calibration(w, "nvfp4", stats, use_awq=True)
    etendue = (scaler.scale.max() / scaler.scale.min().clamp(min=1e-12)).item()
    assert etendue > 4096, (
        f"le plancher par mediane devrait laisser l'etendue exploser "
        f"(mesure : {etendue:.1f}) -- s'il ne le fait plus, ce test ne "
        f"prouve plus rien")


def test_hors_du_motif_creux_le_plancher_ne_change_rien():
    """Coder/GLM (SiLU à porte, pas de motif ReLU² creux) : le maximum
    d'un tenseur SANS canaux écrasés reste proche des canaux bas, le
    plancher (max/4096) est alors sous l'ancien plancher absolu (1e-6) et
    ne borne jamais rien -- comportement inchangé, à 1e-3 près (mesure
    poste7 sur Coder/GLM)."""
    torch.manual_seed(2)
    m = torch.rand(256) * 0.9 + 0.1      # aucun canal ecrase, motif ordinaire
    ancien = m.clamp(min=1e-6)
    nouveau = _magnitude_avec_plancher_relatif(m)
    assert torch.allclose(ancien, nouveau, atol=1e-3)
