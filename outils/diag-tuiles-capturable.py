#!/usr/bin/env python3
"""Bead runner, prérequis (ii) du levier MoE MMA décodage (Sage) : vérifie
que `MoEBlock._tuiles`, à grille fixe (14/09 soir), est bien capturable
dans un graphe CUDA -- LE point bloquant nommé par Sage (l'ancien
`int(ntiles.sum())` synchronisait l'hôte).

Capture un graphe autour d'un appel à `_tuiles` avec un `cnt` variable
(écrit en place entre deux rejeux, comme le ferait un vrai pas de
décodage) et vérifie que le rejeu produit bien un résultat DIFFÉRENT du
premier quand `cnt` change -- pas seulement que la capture ne lève pas :
un graphe qui rejoue toujours la même sortie capturée une fois, sans
rien lire de `cnt` à l'exécution, passerait un test moins strict.

    outils/carte.sh .venv/bin/python outils/diag-tuiles-capturable.py
"""
import os
import sys

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
sys.path.insert(0, _REPO)

import torch

from acvram.engine.model import MoEBlock

E = 8
BT = 16
T_MAX = 20  # ceil(48/16) + 8 = 11, marge large pour le test


def main():
    dev = torch.device("cuda:0")
    cnt = torch.zeros(E, dtype=torch.int64, device=dev)

    # -- capture -----------------------------------------------------------
    cnt.copy_(torch.tensor([6, 0, 0, 0, 0, 0, 0, 0], device=dev))
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):  # chauffe hors capture (allocateur)
            MoEBlock._tuiles(cnt, BT, T_MAX)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    try:
        with torch.cuda.graph(g):
            e_out, t0_out, n_out = MoEBlock._tuiles(cnt, BT, T_MAX)
    except Exception as exc:
        print(f"ÉCHEC DE CAPTURE : {type(exc).__name__}: {exc}")
        sys.exit(1)
    print("capture réussie, aucune exception.")

    # -- rejeu 1 : cnt inchangé, doit correspondre à la capture ------------
    g.replay()
    torch.cuda.synchronize()
    e1, t0_1, n1 = e_out.clone(), t0_out.clone(), n_out.clone()
    print(f"rejeu 1 (cnt=[6,0,...]) : e[:3]={e1[:3].tolist()} n[:3]={n1[:3].tolist()}")

    # -- rejeu 2 : cnt change EN PLACE (meme adresse), doit changer la sortie
    cnt.copy_(torch.tensor([0, 3, 3, 0, 0, 0, 0, 0], device=dev))
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    e2, t0_2, n2 = e_out.clone(), t0_out.clone(), n_out.clone()
    print(f"rejeu 2 (cnt=[0,3,3,...]) : e[:3]={e2[:3].tolist()} n[:3]={n2[:3].tolist()}")

    if torch.equal(e1, e2) and torch.equal(n1, n2):
        print("ÉCHEC : le rejeu 2 rend la MÊME sortie que le rejeu 1 -- "
              "le graphe ne lit pas `cnt` à l'exécution, capture illusoire.")
        sys.exit(1)

    # -- comparaison directe (hors graphe) pour valider les VALEURS --------
    e_attendu, t0_attendu, n_attendu = MoEBlock._tuiles(
        torch.tensor([0, 3, 3, 0, 0, 0, 0, 0], device=dev), BT, T_MAX)
    if not (torch.equal(e2, e_attendu) and torch.equal(t0_2, t0_attendu)
            and torch.equal(n2, n_attendu)):
        print("ÉCHEC : le rejeu 2 diverge de l'appel direct (hors graphe) -- "
              "valeurs incorrectes, pas seulement une histoire de capture.")
        sys.exit(1)

    print("\nVERDICT : capturable, rejeu sensible à `cnt`, valeurs correctes.")


if __name__ == "__main__":
    main()
