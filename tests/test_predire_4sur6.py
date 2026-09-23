"""Pièce 26 : le prédicteur Four Over Six dit-il ce qu il prétend ?

Ce qui doit casser si l instrument ment : G < 0 (impossible — 4sur6 prend le
meilleur des deux blocs par bloc), ou un verdict « d accord » rendu sans
qu aucune erreur maximale n augmente.
"""
from __future__ import annotations

import importlib.util
import os

import pytest
import torch

CHEMIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outils", "gpu", "mesure", "predire-4sur6.py")


def _module():
    spec = importlib.util.spec_from_file_location("predire_4sur6", CHEMIN)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def p4():
    return _module()


def test_g_positif_par_construction(p4):
    """Sur trois lois très différentes, G ≥ 0 — c est la règle de sélection,
    pas une propriété des poids : un G positif ne prédit donc rien."""
    torch.manual_seed(0)
    lois = {"gaussien": torch.randn(64, 256),
            "queue_lourde": torch.randn(64, 256) * torch.exp(torch.randn(64, 256) * 1.5),
            "uniforme": (torch.rand(64, 256) - 0.5) * 2}
    for nom, w in lois.items():
        r = p4.par_bloc(w)
        g = (r["mse6"] - r["mse4"]) / r["mse6"]
        assert g >= -1e-9, f"{nom} : G = {g} < 0, la sélection n est pas celle du convertisseur"
        assert 0.0 <= r["part4"] <= 1.0


def test_classe_gagnante_est_le_bloc_plat(p4):
    """La classe de blocs gagnants, MESURÉE le 22/09 — l inverse de l intuition
    « amax/4 sert les queues lourdes » : un bloc PLAT (seize valeurs du même
    ordre) gagne à amax/4, un bloc PIQUÉ ne gagne jamais. Les niveaux E2M1 hauts
    sont espacés (2, 3, 4, 6) : à amax/6 une valeur à 0,8·amax tombe entre 4 et
    6 (erreur ≤ 0,167·amax), à amax/4 entre 3 et 4 (≤ 0,125) ; sur un bloc piqué
    au contraire amax/6 donne le pas fin (amax/12 contre amax/8) là où sont les
    quinze petites valeurs, et le pic reste exact (6·s = amax).
    Ce test casse si l on réintroduit la lecture inverse."""
    torch.manual_seed(1)
    plat = torch.randn(64, 256) * 0.1 + 1.0                       # 16 valeurs voisines
    pique = torch.randn(64, 256) * 0.02
    pique[:, ::16] = 8.0                                          # une valeur par bloc écrase le reste
    r_plat, r_pique = p4.par_bloc(plat), p4.par_bloc(pique)
    assert r_plat["part4"] > 0.5 > r_pique["part4"], (r_plat["part4"], r_pique["part4"])
    assert r_pique["part4"] == 0.0
    assert r_plat["platitude_gagnants"] > 0.5


def test_verdict_accord_exige_une_degradation(p4):
    """Un G > 0 sans aucune dégradation L∞ ni saturation = prédicteur RÉFUTÉ
    (il annoncerait 4sur6 gagnant, contre le scellé E : KL 1,39 > 0,87)."""
    assert "RÉFUTÉ" in p4.verdict({"G_global": 0.03, "linf_pire": 0.0, "clamp4": 0.0})["verdict"]
    assert "D ACCORD" in p4.verdict({"G_global": 0.03, "linf_pire": 0.31, "clamp4": 0.0})["verdict"]
    assert "INVALIDE" in p4.verdict({"G_global": -0.01, "linf_pire": 0.3, "clamp4": 0.1})["verdict"]
