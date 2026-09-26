"""Pièce 48 : le classement des issues est dans le programme, pas dans l'œil du
lecteur (revue/poste1-piece48-ncu-etroites-22-09.md). Ces tests fixent les
quatre issues et, surtout, le cas « indécis » — un diagnostic qui ne peut pas
rendre « je ne sais pas » choisirait toujours l'issue la plus commode."""
import importlib.util
import os
import sys

_spec = importlib.util.spec_from_file_location(
    "ncu_resume_p48",
    os.path.join(os.path.dirname(__file__), "..", "outils", "gpu", "mesure", "ncu-resume-p48.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
issue = _m.issue


def test_bande_saturee_ferme_la_piece():
    assert issue({"emission": 85.0, "dram": 88.0}).startswith("I4")


def test_emission_saturee_et_bande_libre_est_le_depaquetage():
    assert issue({"emission": 74.0, "dram": 29.0, "long_sb": 10.0}).startswith("I1")


def test_emission_basse_et_attente_dram_est_la_latence():
    """Le cas que je prédis : 0,45 To/s sans lancement en cause."""
    assert issue({"emission": 32.0, "dram": 27.0, "long_sb": 51.0, "warps": 45.0}).startswith("I2")


def test_occupation_quand_ni_emission_ni_attente():
    assert issue({"emission": 30.0, "dram": 25.0, "long_sb": 12.0, "warps": 18.0}).startswith("I3")


def test_indecis_quand_aucune_condition_ne_porte():
    """Émission moyenne, pas d'attente, occupation correcte : aucune des
    quatre. Le résumé doit le DIRE au lieu de trancher."""
    r = issue({"emission": 58.0, "dram": 30.0, "long_sb": 15.0, "warps": 70.0})
    assert r.startswith("indécis") and "58" in r


def test_indecis_si_les_metriques_manquent():
    assert issue({"warps": 40.0}).startswith("indécis")


def test_lordre_des_regles_bande_avant_emission():
    """Une bande saturée ET une émission saturée : c'est I4 qui gagne, sinon un
    noyau au plafond de bande passerait pour un problème de dépaquetage."""
    assert issue({"emission": 95.0, "dram": 92.0}).startswith("I4")
