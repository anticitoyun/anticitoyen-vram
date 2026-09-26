"""`search_channel_scales_commun` — item A7 de l'audit poste7 (14/09).

`gate_proj` et `up_proj` lisent la même entrée : chercher un alpha AWQ
séparément pour chacun choisit deux exposants différents la plupart du
temps, et `_scaler_commun` (layers.py) refuse alors de porter le scaler
sur la pile empilée. La recherche commune force UN alpha, identique par
construction — testé ici SANS carte (pure arithmétique CPU)."""
import torch

from acvram.quant.calibrate import (ActStats, search_channel_scales,
                                    search_channel_scales_commun)


def _stats_avec_canaux_saillants(k: int, graine: int) -> ActStats:
    g = torch.Generator().manual_seed(graine)
    mean_abs = torch.rand(k, generator=g) * 0.1
    # quelques canaux saillants — c'est ce qui donne un alpha != 0 a la
    # recherche AWQ ; sans eux, l'echelle identite est deja optimale.
    mean_abs[::7] *= 50.0
    return ActStats(mean_abs, None, 64)


def test_rend_un_scaler_unique_pour_deux_tenseurs():
    k = 32
    stats = _stats_avec_canaux_saillants(k, graine=1)
    g1 = torch.Generator().manual_seed(2)
    g2 = torch.Generator().manual_seed(3)
    w_gate = torch.randn(16, k, generator=g1)
    w_up = torch.randn(16, k, generator=g2)

    scaler, erreurs = search_channel_scales_commun(
        [w_gate, w_up], stats, fmt="int8", group_size=None)

    assert len(erreurs) == 2
    assert all(e >= 0 for e in erreurs)


def test_le_scaler_commun_egale_ou_depasse_les_optima_individuels():
    """Le prix de la fusion : la somme des erreurs a l'alpha COMMUN ne peut
    pas etre MEILLEURE que la somme des erreurs a l'alpha optimal de
    CHAQUE tenseur pris separement — sinon la recherche individuelle
    aurait un bogue (elle explore la meme grille)."""
    k = 32
    stats = _stats_avec_canaux_saillants(k, graine=10)
    g1 = torch.Generator().manual_seed(11)
    g2 = torch.Generator().manual_seed(12)
    w_gate = torch.randn(16, k, generator=g1)
    w_up = torch.randn(16, k, generator=g2)

    _, err_gate_seul = search_channel_scales(w_gate, stats, fmt="int8")
    _, err_up_seul = search_channel_scales(w_up, stats, fmt="int8")
    somme_individuelle = err_gate_seul + err_up_seul

    _, erreurs_communes = search_channel_scales_commun(
        [w_gate, w_up], stats, fmt="int8")
    somme_commune = sum(erreurs_communes)

    assert somme_commune >= somme_individuelle - 1e-9, (
        f"la recherche commune ({somme_commune}) bat la somme des optima "
        f"individuels ({somme_individuelle}) — impossible, meme grille")


def test_journal_conserve_la_grille_commune():
    k = 16
    stats = _stats_avec_canaux_saillants(k, graine=20)
    w_gate = torch.randn(8, k, generator=torch.Generator().manual_seed(21))
    w_up = torch.randn(8, k, generator=torch.Generator().manual_seed(22))
    journal: dict = {}

    search_channel_scales_commun([w_gate, w_up], stats, fmt="int8",
                                 n_grid=20, journal=journal)

    assert len(journal["erreurs_grille_commune"]) == 21
    assert 0.0 <= journal["alpha_commun_retenu"] <= 1.0


def test_refuse_des_tenseurs_de_largeurs_differentes():
    stats = _stats_avec_canaux_saillants(16, graine=30)
    w_gate = torch.randn(8, 16)
    w_up_mauvaise_largeur = torch.randn(8, 24)
    import pytest
    with pytest.raises(ValueError, match="canaux d'entree"):
        search_channel_scales_commun([w_gate, w_up_mauvaise_largeur], stats,
                                     fmt="int8")


def test_refuse_moins_de_deux_tenseurs():
    import pytest
    with pytest.raises(ValueError, match="au moins deux"):
        search_channel_scales_commun([torch.randn(8, 16)], None, fmt="int8")


def test_trois_tenseurs_a_la_fois():
    """Pas limite a une paire — n'importe quel groupe partageant l'entree."""
    k = 24
    stats = _stats_avec_canaux_saillants(k, graine=40)
    ws = [torch.randn(8, k, generator=torch.Generator().manual_seed(41 + i))
         for i in range(3)]
    scaler, erreurs = search_channel_scales_commun(ws, stats, fmt="int8")
    assert len(erreurs) == 3
