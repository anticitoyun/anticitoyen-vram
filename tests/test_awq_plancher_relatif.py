"""poste7-awq-relu2-garde-repli-17-09, geste (d) : plancher RELATIF (1 % de
la médiane du tenseur) au lieu d'une constante absolue (1e-6) dans la
recherche de canaux salients. Cause réelle du 145/346 de Nemotron calibA
(pas le nombre d'échantillons, déjà exclu) : l'entrée de `down_proj` chez
nemotron_h est ReLU²(up(x)) -- creuse par construction, les canaux jamais
activés sur le corpus s'écrasent au plancher absolu contre ~1 ailleurs,
étendue 1,7e6× mesurée. Un plancher à 1 % de la médiane borne cette
étendue à ~100× quelle que soit l'échelle du tenseur.
"""
import torch

from acvram.quant.calibrate import (ActStats, _magnitude_avec_plancher_relatif,
                                    quantize_with_calibration)


def _stats_relu2_creuse(k=256, graine=0):
    """Motif ReLU² réaliste : ~40 % des canaux jamais activés sur le
    corpus (magnitude quasi nulle), le reste avec une magnitude normale --
    pas un seul canal isolé comme le witness du seuil d'échantillons,
    un motif DISTRIBUÉ sur toute la largeur du tenseur."""
    torch.manual_seed(graine)
    m = torch.rand(k) * 0.9 + 0.1        # canaux actifs : [0.1, 1.0]
    jamais_actifs = torch.rand(k) < 0.4
    m[jamais_actifs] = 1e-6              # ce que ReLU² rend pour ces canaux
    return ActStats(m, m, 128)


def test_plancher_relatif_borne_letendue_de_lechelle_awq():
    """Un changement qui doit casser : `git stash`/revenir à
    `clamp(min=1e-6)` (constante absolue) fait sortir l'étendue au-delà de
    64, exactement le motif du 17/09."""
    torch.manual_seed(1)
    w = torch.randn(256, 256) * 0.02
    stats = _stats_relu2_creuse(k=256, graine=0)
    _, scaler, _ = quantize_with_calibration(w, "nvfp4", stats, use_awq=True)
    assert scaler.scale is not None, "la recherche doit trouver une echelle non triviale"
    etendue = (scaler.scale.max() / scaler.scale.min().clamp(min=1e-12)).item()
    assert etendue <= 64, f"etendue de l'echelle AWQ = {etendue:.1f} (attendu <= 64)"


def test_hors_du_motif_creux_le_plancher_relatif_ne_change_rien():
    """Coder/GLM (SiLU à porte, pas de motif ReLU² creux) : la médiane
    d'un tenseur SANS canaux écrasés reste proche des canaux bas, le
    plancher relatif (1 % de la médiane) est alors sous l'ancien plancher
    absolu (1e-6) et ne borne jamais rien -- comportement inchangé, à
    1e-3 près (mesure poste7 sur Coder/GLM)."""
    torch.manual_seed(2)
    m = torch.rand(256) * 0.9 + 0.1      # aucun canal ecrase, motif ordinaire
    ancien = m.clamp(min=1e-6)
    nouveau = _magnitude_avec_plancher_relatif(m)
    assert torch.allclose(ancien, nouveau, atol=1e-3)
