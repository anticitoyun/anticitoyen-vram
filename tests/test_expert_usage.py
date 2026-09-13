"""Persistance du profil d'usage et AUTOPIN (bead anticitoyen-vram-pds, poste7
§4, revue/poste7-cache-experts-13-09.md) : `sauvegarder`/`charger` (le
`.acvram_usage` persistant) et `choisir_residents` (le geste AUTOPIN — les
experts les plus demandés d'une couche, sous garde de confiance). À sec,
aucune carte : ce sont des fonctions pures sur des dicts et un fichier JSON.
`concentration_top_fraction` est déjà couverte par test_usage_routage.py.
"""
from __future__ import annotations

import torch

from acvram.memory.expert_usage import (charger, choisir_residents,
                                       decider_residents, sauvegarder)


# --------------------------------------------------------------------------
# sauvegarder / charger — le profil persistant
# --------------------------------------------------------------------------

def test_aller_retour_preserve_les_comptes(tmp_path):
    # La couche 1 est présente mais TOTALEMENT à zéro (jamais sollicitée, ou
    # dense sans routage) : elle ne doit pas survivre à l'aller-retour — un
    # profil qui écrirait des zéros grossirait sans jamais rien dire de plus
    # qu'une absence.
    usage = {0: torch.tensor([5, 0, 3]), 1: torch.tensor([0, 0]),
            2: torch.tensor([0, 7])}
    chemin = str(tmp_path / "profil.json")
    sauvegarder(usage, chemin)
    relu = charger(chemin)
    assert relu == {0: {0: 5, 2: 3}, 2: {1: 7}}
    assert 1 not in relu


def test_charger_fichier_absent_rend_none(tmp_path):
    assert charger(str(tmp_path / "n-existe-pas.json")) is None


def test_charger_fichier_corrompu_rend_none(tmp_path):
    chemin = tmp_path / "casse.json"
    chemin.write_text("{ceci n'est pas du JSON")
    assert charger(str(chemin)) is None


def test_sauvegarder_est_atomique(tmp_path):
    """Le fichier temporaire ne doit pas survivre à un écrit réussi — sinon
    un lecteur qui liste le dossier verrait un `.tmp` qu'il ne sait pas lire."""
    chemin = tmp_path / "profil.json"
    sauvegarder({0: torch.tensor([1])}, str(chemin))
    assert chemin.exists()
    assert not (tmp_path / "profil.json.tmp").exists()


# --------------------------------------------------------------------------
# choisir_residents — AUTOPIN, avec sa garde de confiance
# --------------------------------------------------------------------------

def test_sous_le_seuil_de_confiance_rend_none():
    # 4999 sélections au total : juste sous le seuil (5000, comme colibrì).
    compte = {0: 4999}
    assert choisir_residents(compte, capacite=1) is None


def test_au_seuil_de_confiance_decide():
    compte = {0: 5000}
    assert choisir_residents(compte, capacite=1) == [0]


def test_pique_les_plus_demandes_dans_l_ordre():
    compte = {0: 100, 1: 9000, 2: 500, 3: 50}   # total 9650 >= seuil
    assert choisir_residents(compte, capacite=2) == [1, 2]


def test_egalite_departagee_par_l_identifiant():
    # Deux experts à égalité de compte : le résultat ne doit pas dépendre de
    # l'ordre d'insertion du dict (non garanti comme critère de tri stable
    # pour l'appelant) — l'identifiant le plus bas gagne, déterministe.
    compte = {5: 3000, 2: 3000, 9: 6000}        # total 12000 >= seuil
    assert choisir_residents(compte, capacite=2) == [9, 2]


def test_capacite_superieure_au_nombre_d_experts_rend_tout():
    compte = {0: 3000, 1: 3000}                 # total 6000 >= seuil
    assert sorted(choisir_residents(compte, capacite=10)) == [0, 1]


def test_dict_vide_est_sous_le_seuil():
    assert choisir_residents({}, capacite=4) is None


# --------------------------------------------------------------------------
# decider_residents — AUTOPIN au chargement, avec repli explicite
# --------------------------------------------------------------------------

def test_sans_profil_rend_le_defaut_par_indice():
    residents, source = decider_residents(None, n_experts=8, capacite=3)
    assert residents == {0, 1, 2}
    assert source == "defaut"


def test_profil_insuffisant_rend_aussi_le_defaut():
    residents, source = decider_residents({0: 100}, n_experts=8, capacite=3)
    assert residents == {0, 1, 2}          # 100 < seuil : pas assez d'historique
    assert source == "defaut"


def test_profil_suffisant_rend_l_autopin():
    compte = {0: 100, 1: 9000, 2: 500, 3: 50}       # total 9650 >= seuil
    residents, source = decider_residents(compte, n_experts=8, capacite=2)
    assert residents == {1, 2}
    assert source == "autopin"


def test_defaut_ne_depasse_pas_le_nombre_d_experts():
    residents, source = decider_residents(None, n_experts=2, capacite=10)
    assert residents == {0, 1}
    assert source == "defaut"
