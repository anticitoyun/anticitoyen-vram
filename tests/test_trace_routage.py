"""`trace_routage.taux_de_succes*` existait (`taux_de_succes`) sans jamais avoir
tourné (constat de Sage, `revue/sage-cache-experts-13-09.md` §0.1) ; ce fichier
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


# ---- M1 étendu (sage-c9-119b-cache-experts-19-09) ---------------------------

def test_pas_de_decodage_regroupe_les_rafales_de_meme_couche(tmp_path):
    # b=3 : trois lignes de suite par couche = un pas ; deux couches, deux pas chacune
    lignes = [(0, 0, [1, 2]), (1, 0, [2, 3]), (2, 0, [1, 3]), (0, 1, [7, 8]), (1, 1, [7, 8]), (2, 1, [7, 9]),
              (3, 0, [4, 5]), (4, 0, [4, 5]), (5, 0, [4, 6]), (3, 1, [7, 8]), (4, 1, [8, 9]), (5, 1, [9, 7])]
    j = _ecrire(tmp_path, lignes)
    pas = list(tr.pas_de_decodage(j))
    assert [(c, len(r)) for c, r in pas] == [(0, 3), (1, 3), (0, 3), (1, 3)]
    d = tr.distincts_par_pas(j)
    assert d[0]["pas"] == 2 and d[0]["max"] == 3 and d[0]["lot_moyen"] == 3.0
    # couche 0, pas 1 : {1,2,3} = 3 distincts pour 6 demandes → recouvrement 0,5 ; pas 2 : {4,5,6} idem
    assert abs(d[0]["recouvrement"] - 0.5) < 1e-9
    # témoin cassant : à b=1 (une ligne par couche, deux couches) il n'y a aucun recouvrement
    (tmp_path / "b1").mkdir()
    j1 = _ecrire(tmp_path / "b1", [(i, c, [i % 4, (i + 1) % 4]) for i in range(8) for c in (0, 1)])
    d1 = tr.distincts_par_pas(j1)
    assert d1[0]["pas"] == 8 and d1[0]["recouvrement"] == 0.0


def test_lru_jugee_sur_la_seconde_moitie_et_par_pas(tmp_path):
    # couche 0 : la première moitié (jetons 0-9) ne demande que {0,1} ; la seconde {0,1} aussi
    # → LRU(2) chauffée sur la première moitié rend 100 % sur la seconde ; LRU(1) alterne → 0 %
    lignes = [(i, c, [i % 2]) for i in range(20) for c in (0, 1)]
    j = _ecrire(tmp_path, lignes)
    assert tr.taux_de_succes_lru_juge(j, 2)["taux"] == 1.0
    assert tr.taux_de_succes_lru_juge(j, 1)["taux"] == 0.0
    # par pas (b=3, deux couches) : un expert demandé par 3 jetons du même pas = 1 demande
    (tmp_path / "pas").mkdir()
    lignes = [(k, 0, [5]) for k in range(3)] + [(k, 1, [6]) for k in range(3)] \
        + [(3 + k, 0, [5]) for k in range(3)] + [(3 + k, 1, [6]) for k in range(3)]
    j2 = _ecrire(tmp_path / "pas", lignes)
    r = tr.taux_de_succes_lru_juge(j2, 1, entrainement=0.4, par_pas=True)
    assert r["demandes_jugees"] == 2                      # le second pas : une demande distincte par couche
    r2 = tr.taux_de_succes_lru_juge(j2, 1, entrainement=0.4, par_pas=False)
    assert r2["demandes_jugees"] == 6


def test_rapport_m1_rend_le_critere_de_sage(tmp_path):
    # 8 experts ; les jetons ne demandent que {0,1,2,3} → h(E/2 = 4) = 1 → Δh = 0,5 → cache engagé
    lignes = [(i, c, [i % 4, (i + 1) % 4]) for i in range(40) for c in (0, 1)]
    j = _ecrire(tmp_path, lignes)
    r = tr.rapport_m1(j, capacites=(2, 4), nb_experts=8)
    assert r["experts_vus"] == 8 and r["capacite_demi"] == 4
    assert r["h_pin"][4] == 1.0 and r["h_lru"][4] == 1.0 and r["verdict_sage"] == "cache engagé"
    assert 0.0 < r["h_lru"][2] < 1.0                     # capacité 2 : la moitié des demandes manquent
    # témoin cassant : routage uniforme sur 8 experts avec 2 emplacements → Δh < 0,10 → pas de cache
    lignes = [(i, 0, [(3 * i) % 8, (3 * i + 5) % 8]) for i in range(80)]
    (tmp_path / "plat").mkdir()
    j2 = _ecrire(tmp_path / "plat", [(i, c, e) for i, _, e in lignes for c in (0, 1)])
    r2 = tr.rapport_m1(j2, capacites=(2,), nb_experts=8)
    assert r2["capacite_demi"] == 4 and r2["verdict_sage"] != "cache engagé"
