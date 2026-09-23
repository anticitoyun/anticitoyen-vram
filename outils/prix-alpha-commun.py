#!/usr/bin/env python3
"""Prix en qualite d'un exposant AWQ COMMUN a chaque groupe empilable.

Lit `erreurs_grille` au manifeste (ecrit par `acvram convert --grille-erreurs`)
et repond a la seule question qui decide : **combien coute, en erreur de
sortie, d'imposer un meme alpha aux projections d'un groupe, pour qu'elles
fusionnent ?**

Ce que la fusion rapporte est deja mesure : +0,19 % sur Llama-2-7b-int8 avec
5 groupes empiles sur 64, et l'extrapolation a couverture complete vaut
+2,43 %, ancree par le +2,60 % mesure en bf16 pur. Le cout memoire est nul
depuis que les quatre empileurs rendent des vues. Il ne manque que ce prix.

Aucun GPU, aucune reconversion : le manifeste porte les 21 erreurs.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# k/v n'est PAS candidat : leurs exposants ne s'accordent jamais (0 couche sur
# 31 sur Llama-2-7b-int8), et l'ecart large y est probablement une propriete
# des tenseurs et non un defaut de la recherche.
GROUPES = (("gate_proj", "up_proj"), ("q_proj", "k_proj", "v_proj"))


def prix_du_groupe(grilles: list[list[float]]) -> dict:
    """Erreur au meilleur alpha COMMUN contre la somme des meilleurs individuels."""
    n = len(grilles[0])
    if any(len(g) != n for g in grilles):
        return {}
    indiv = [min(g) for g in grilles]
    i_indiv = [g.index(m) for g, m in zip(grilles, indiv)]
    somme = [sum(g[i] for g in grilles) for i in range(n)]
    i_commun = somme.index(min(somme))
    au_commun = [g[i_commun] for g in grilles]
    return {
        "alpha_individuels": [round(i / (n - 1), 3) for i in i_indiv],
        "alpha_commun": round(i_commun / (n - 1), 3),
        "erreur_individuelle": [round(e, 6) for e in indiv],
        "erreur_au_commun": [round(e, 6) for e in au_commun],
        # Le surcout RELATIF par tenseur : c'est lui qui decide, une erreur
        # absolue ne se compare pas entre tenseurs de tailles differentes.
        "surcout_relatif": [round(a / max(b, 1e-12) - 1, 5)
                            for a, b in zip(au_commun, indiv)],
        "deja_accordes": i_indiv.count(i_indiv[0]) == len(i_indiv),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifeste", help="chemin d'un acvram_manifest.json")
    ap.add_argument("--seuil", type=float, default=0.02,
                    help="surcout relatif maximal accepte (defaut 2 %%)")
    a = ap.parse_args()
    d = json.loads(Path(a.manifeste).read_text())
    t = d.get("tensors", {})
    avec = [k for k, v in t.items() if "erreurs_grille" in v]
    if not avec:
        print("ECHEC / CAUSE: aucun tenseur ne porte erreurs_grille. "
              "Reconvertir avec --grille-erreurs. / "
              "SUITE: ce dossier ne peut pas repondre.")
        return 2
    print(f"{Path(a.manifeste).parent.name} : {len(avec)} tenseurs sur "
          f"{len(t)} portent la grille\n")
    par_couche: dict[str, dict] = defaultdict(dict)
    for k, v in t.items():
        m = re.match(r"(.*layers\.\d+)\.(?:self_attn|mlp)\.(\w+)\.weight$", k)
        if m and "erreurs_grille" in v:
            par_couche[m.group(1)][m.group(2)] = v["erreurs_grille"]
    bilan: dict[str, list] = defaultdict(list)
    for couche, projs in par_couche.items():
        for g in GROUPES:
            if all(p in projs for p in g):
                r = prix_du_groupe([projs[p] for p in g])
                if r:
                    bilan["/".join(g)].append((couche, r))
    if not bilan:
        print("Aucun groupe empilable complet dans ce dossier.")
        return 0
    for nom, lignes in bilan.items():
        deja = sum(1 for _, r in lignes if r["deja_accordes"])
        pires = sorted(max(r["surcout_relatif"]) for _, r in lignes)
        sous_seuil = sum(1 for p in pires if p <= a.seuil)
        med = pires[len(pires) // 2]
        print(f"  {nom}")
        print(f"    {len(lignes)} groupes, {deja} deja accordes "
              f"(ils fusionnent DEJA)")
        print(f"    surcout relatif du pire tenseur du groupe : "
              f"median {100 * med:+.3f} %, max {100 * pires[-1]:+.3f} %")
        print(f"    groupes sous le seuil de {100 * a.seuil:.1f} % : "
              f"{sous_seuil} sur {len(lignes)} "
              f"({100 * sous_seuil / len(lignes):.0f} %)")
        for couche, r in lignes[:3]:
            print(f"      {couche.rsplit('.', 1)[-1]:>3s} : alpha "
                  f"{r['alpha_individuels']} -> {r['alpha_commun']}, "
                  f"surcout {[f'{100*x:+.2f}%' for x in r['surcout_relatif']]}")
    total = sum(len(v) for v in bilan.values())
    gagnables = sum(1 for v in bilan.values() for _, r in v
                    if not r["deja_accordes"] and max(r["surcout_relatif"]) <= a.seuil)
    print(f"\n  RECUPERABLES sous le seuil : {gagnables} groupes sur {total}")
    print(f"  Ce que ca vaut en debit : le gain est proportionnel a la "
          f"couverture\n  (verifie a deux ancrages, 6,5 % d'ecart). "
          f"Passer de la couverture actuelle\n  a {100*(gagnables+sum(1 for v in bilan.values() for _, r in v if r['deja_accordes']))/total:.0f} % "
          f"vaut donc environ "
          f"{2.43 * (gagnables + sum(1 for v in bilan.values() for _, r in v if r['deja_accordes'])) / total:.2f} % "
          f"sur ce modele.")
    print("\n  ET CE QUE CE CALCUL NE DIT PAS : le surcout est une erreur de "
          "SORTIE par\n  tenseur, pas une perplexite. Un surcout de 2 % sur "
          "l'erreur ne se traduit pas\n  lineairement en perplexite — il faut "
          "une reconversion et un `acvram eval`\n  contre l'etalon 5,4141 "
          "pour trancher. Ce chiffre-ci sert a savoir SI cela\n  vaut la "
          "reconversion, pas a la remplacer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
