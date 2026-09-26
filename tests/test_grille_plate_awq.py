"""Pièce 23 : une grille AWQ dont la magnitude par canal est CONSTANTE donne
la même échelle pour les 21 valeurs d alpha — elle se calcule une fois.

C est le cas de tout expert que le corpus n a pas routé (`stats is None` →
`act = 1`) : sur le 30B-VL du 22/09, 5 235 experts sur 6 144 sans statistique,
soit la grande majorité des tenseurs quantifiés, parcouraient 21 quantifications
complètes pour un résultat connu d avance.

Le raccourci doit être AU BIT — même échelle, même erreur rendue, même journal
— et il doit cesser dès que la magnitude n est plus constante. Ce test casse
dans les deux sens.
"""
from __future__ import annotations

import time

import torch

from acvram.quant.calibrate import ActStats, search_channel_scales


def _w(n=128, k=256, graine=3):
    g = torch.Generator().manual_seed(graine)
    return torch.randn(n, k, generator=g) * 0.05


def test_stats_absentes_rendent_l_identite_et_la_meme_erreur():
    from acvram.quant.calibrate import _quant_dequant
    w = _w()
    j_court = {}
    sc, err = search_channel_scales(w, None, "nvfp4", group_size=16, n_grid=20, journal=j_court)
    # La référence, calculée à la main : alpha = 0 donne s = 1, donc
    # l évaluation que la boucle complète aurait faite au premier tour.
    x = torch.diag(torch.ones(w.shape[1]))
    y_ref = x @ w.float().t()
    wq = _quant_dequant(w.float(), "nvfp4", 16)
    attendu = ((x @ wq.t() - y_ref).norm() / y_ref.norm().clamp(min=1e-12)).item()
    assert sc.scale is None
    assert err == attendu                                  # au bit : la même évaluation
    assert j_court["grille_plate"] is True
    assert len(j_court["erreurs_grille"]) == 21            # journal inchangé
    assert len(set(j_court["erreurs_grille"])) == 1
    assert j_court["alpha_retenu"] == 0.0


def test_magnitude_constante_non_unitaire_aussi():
    """`act` constant mais ≠ 1 : l échelle normalisée vaut encore 1 partout."""
    w = _w()
    st = ActStats(mean_abs=torch.full((256,), 0.7), max_abs=None, n_samples=4096)
    sc, _ = search_channel_scales(w, st, "nvfp4", group_size=16, n_grid=20, journal=(j := {}))
    assert j["grille_plate"] is True and sc.scale is None


def test_magnitude_variee_parcourt_la_grille():
    """Le raccourci ne doit PAS s appliquer quand il y a quelque chose à
    optimiser — sinon il choisirait l identité à la place du bon alpha."""
    m = torch.full((256,), 0.02)
    m[::8] *= 8.0
    st = ActStats(mean_abs=m, max_abs=None, n_samples=4096)
    sc, _ = search_channel_scales(_w(), st, "nvfp4", group_size=16, n_grid=20, journal=(j := {}))
    assert j["grille_plate"] is False
    assert len(set(j["erreurs_grille"])) > 1               # la grille varie réellement
    assert sc.scale is not None                            # et un alpha non trivial est retenu


def test_le_raccourci_est_plus_rapide():
    """Le gain est la raison d être du raccourci : au moins 5× sur une grille
    de 21 points (la borne est lâche exprès, la machine varie)."""
    w = _w(256, 512)
    t0 = time.perf_counter()
    search_channel_scales(w, None, "nvfp4", group_size=16, n_grid=20)
    court = time.perf_counter() - t0
    m = torch.full((512,), 0.02)
    m[::8] *= 8.0
    t0 = time.perf_counter()
    search_channel_scales(w, ActStats(mean_abs=m, max_abs=None, n_samples=4096), "nvfp4",
                          group_size=16, n_grid=20)
    long = time.perf_counter() - t0
    assert court * 5 < long, f"raccourci {court * 1e3:.1f} ms contre grille complète {long * 1e3:.1f} ms"
