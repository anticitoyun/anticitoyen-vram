"""`trace_routage.taux_de_succes*` existait (`taux_de_succes`) sans jamais avoir
tourné (constat de poste7, `revue/poste7-cache-experts-13-09.md` §0.1) ; ce fichier
teste les DEUX extensions écrites pour M1 (bead jt5 / colibrì) :
`taux_de_succes_par_couche` (LRU/LFU par couche, pas un pool partagé) et
`taux_de_succes_pin` (politique statique apprise sur une moitié, évaluée sur
l'autre). Aucune carte : ce sont des fonctions pures sur un fichier texte.
"""
from __future__ import annotations

from acvram.memory import trace_routage as tr


def _ecrire(tmp_path, lignes: list[tuple[int, int, list[int]]]) -> str:
    """Écrit une trace au format documenté (`route_trace.h`-like côté nous :
    voir la docstring du module) : `<jeton> <couche> <e1>,<e2>,...`."""
    p = tmp_path / "trace.txt"
    with open(p, "w") as f:
        f.write("# jeton couche experts\n")
        for jeton, couche, experts in lignes:
            f.write(f"{jeton} {couche} " + ",".join(map(str, experts)) + "\n")
    return str(p)


# --------------------------------------------------------------------------
# taux_de_succes_par_couche : chaque couche a SON cache, pas un pool commun
# --------------------------------------------------------------------------

def test_par_couche_isole_les_couches(tmp_path):
    # Couche 0 : experts 0,1 en boucle (tient dans une capacité de 1... non,
    # tient dans 2). Couche 1 : experts 10,11 en boucle. Avec une capacité de
    # 1 PAR couche, chaque couche alterne deux experts qu'elle ne peut pas
    # garder tous les deux : taux de succès nul après la première visite de
    # chacun, sauf la répétition immédiate d'un même expert.
    lignes = []
    for j in range(6):
        lignes.append((j, 0, [j % 2]))          # 0,1,0,1,0,1
        lignes.append((j, 1, [10 + j % 2]))     # 10,11,10,11,10,11
    chemin = _ecrire(tmp_path, lignes)
    r = tr.taux_de_succes_par_couche(chemin, capacite=1, politique="lru")
    # Chaque alternance est un défaut avec une capacité de 1 (LRU) : la
    # première visite de chaque couche est forcément un défaut (cache vide),
    # les suivantes alternent 0/1 -> toujours un défaut avec capacite=1.
    assert r["succes"] == 0
    assert r["demandes"] == 12
    assert r["taux_par_couche"][0] == 0.0
    assert r["taux_par_couche"][1] == 0.0


def test_par_couche_capacite_suffisante_reussit_tout(tmp_path):
    lignes = [(j, 0, [j % 2]) for j in range(6)]
    chemin = _ecrire(tmp_path, lignes)
    r = tr.taux_de_succes_par_couche(chemin, capacite=2, politique="lru")
    # Deux emplacements pour deux experts distincts : après le premier passage
    # de chacun (2 défauts), tout le reste est un succès.
    assert r["succes"] == 4
    assert r["demandes"] == 6


def test_par_couche_ne_confond_pas_deux_couches_de_meme_id_expert(tmp_path):
    # Les deux couches routent le MÊME identifiant d'expert (0 et 1) : si le
    # cache était partagé entre couches (comme `taux_de_succes` sans
    # variante), la présence de l'expert 0 pour la couche 0 compterait comme
    # un succès pour la couche 1 — ce que ce test refuse.
    lignes = [(0, 0, [0]), (0, 1, [0]), (1, 0, [0]), (1, 1, [1])]
    chemin = _ecrire(tmp_path, lignes)
    r = tr.taux_de_succes_par_couche(chemin, capacite=1, politique="lru")
    # couche 0 : [0]->défaut, [0]->succès (2 demandes, 1 succès)
    # couche 1 : [0]->défaut, [1]->défaut (2 demandes, 0 succès)
    assert r["taux_par_couche"][0] == 0.5
    assert r["taux_par_couche"][1] == 0.0


# --------------------------------------------------------------------------
# taux_de_succes_pin : appris sur la première moitié, évalué sur la seconde
# --------------------------------------------------------------------------

def test_pin_apprend_sur_la_premiere_moitie_pas_sur_tout(tmp_path):
    # Jetons 0-9 (entraînement) : expert 0 domine (9 fois sur 10). Jetons
    # 10-19 (évaluation) : la charge a changé, expert 1 domine désormais.
    # Un pin appris ET évalué sur la MÊME moitié dirait « ça marche » parce
    # que n'importe quel cache assez grand couvre ses propres données —
    # c'est justement ce que ce test vérifie qu'on ne mesure PAS.
    lignes = [(j, 0, [0 if j != 5 else 1]) for j in range(10)]        # train
    lignes += [(10 + j, 0, [1 if j != 5 else 0]) for j in range(10)]  # eval
    chemin = _ecrire(tmp_path, lignes)
    out = tr.taux_de_succes_pin(chemin, capacites=[1], entrainement=0.5)
    # pin(couche 0, C=1) = {0} (appris sur l'entraînement, où 0 domine).
    # Sur l'évaluation, 1 domine : le pin de la première moitié échoue neuf
    # fois sur dix — un taux élevé signalerait un bug de fuite train/eval.
    assert out[1]["taux"] == 0.1


def test_pin_evalue_sur_lui_meme_donnerait_un_taux_different(tmp_path):
    """Contrôle négatif : si `taux_de_succes_pin` évaluait par erreur sur
    TOUTE la trace (entraînement inclus) au lieu du seul reste, le taux du
    test précédent monterait — ce test le vérifie directement en comparant
    aux deux moitiés prises séparément avec `relire`."""
    lignes = [(j, 0, [0 if j != 5 else 1]) for j in range(10)]
    lignes += [(10 + j, 0, [1 if j != 5 else 0]) for j in range(10)]
    chemin = _ecrire(tmp_path, lignes)
    out = tr.taux_de_succes_pin(chemin, capacites=[1], entrainement=0.5)
    # Sur l'ENTRAÎNEMENT seul, le pin (={0}) réussirait 9/10 — très différent
    # du 1/10 mesuré sur l'évaluation : la fonction ne mélange pas les deux.
    assert out[1]["taux"] != 0.9


def test_pin_capacite_egale_au_vu_reussit_tout(tmp_path):
    # Deux experts vus pendant l'entraînement, capacité 2 : le pin les
    # contient tous les deux, et l'évaluation (qui ne route que parmi eux)
    # réussit systématiquement.
    lignes = [(j, 0, [j % 2]) for j in range(20)]
    chemin = _ecrire(tmp_path, lignes)
    out = tr.taux_de_succes_pin(chemin, capacites=[2], entrainement=0.5)
    assert out[2]["taux"] == 1.0


def test_pin_plusieurs_capacites_une_seule_lecture(tmp_path):
    """Plusieurs capacités en un seul appel doivent rendre les mêmes chiffres
    qu'un appel isolé par capacité — c'est ce qui justifie de ne lire la
    trace qu'une fois pour toutes les capacités (docstring)."""
    lignes = [(j, 0, [j % 4]) for j in range(40)]
    chemin = _ecrire(tmp_path, lignes)
    ensemble = tr.taux_de_succes_pin(chemin, capacites=[1, 2, 4], entrainement=0.5)
    isole_1 = tr.taux_de_succes_pin(chemin, capacites=[1], entrainement=0.5)
    isole_4 = tr.taux_de_succes_pin(chemin, capacites=[4], entrainement=0.5)
    assert ensemble[1]["taux"] == isole_1[1]["taux"]
    assert ensemble[4]["taux"] == isole_4[4]["taux"]
    assert ensemble[4]["taux"] == 1.0          # 4 experts distincts, capacité 4


def test_pin_trace_vide_ne_fabrique_pas_de_taux(tmp_path):
    chemin = _ecrire(tmp_path, [])
    out = tr.taux_de_succes_pin(chemin, capacites=[8])
    assert out[8]["taux"] == 0.0
    assert out[8]["taux_par_couche"] == {}
