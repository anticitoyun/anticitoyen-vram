"""Premier (pas, couche) où deux exécutions divergent (sage-p1-situ-verdict-
18-09 : (b) capturé contre (b) eager, ACVRAM_DUMP_MOE) : lit pas-NNNNN.pt de
deux dossiers, compare la sortie MoE de chaque couche pour la forme voulue
(défaut : lot de 1, décodage) et imprime le premier écart relatif > seuil, puis
le profil par couche de ce pas et le nombre de pas divergents.

    python outils/comparer-dump-moe-18-09.py <dossier_A> <dossier_B> [--forme "(1, 2048)"] [--seuil 1e-5]
"""
import argparse
import glob
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--forme", default="")
    ap.add_argument("--seuil", type=float, default=1e-5)
    o = ap.parse_args()
    fa = sorted(glob.glob(os.path.join(o.a, "pas-*.pt"))); fb = sorted(glob.glob(os.path.join(o.b, "pas-*.pt")))
    n = min(len(fa), len(fb))
    print(f"{len(fa)} pas dans A, {len(fb)} dans B, comparés : {n}")
    premier, divergents = None, 0
    for i in range(n):
        A, B = torch.load(fa[i]), torch.load(fb[i])
        pire = 0.0
        profil = []
        for c in sorted(A):
            formes = [f for f in A[c] if f in B.get(c, {}) and (not o.forme or f == o.forme)]
            if not formes:
                continue
            f = o.forme or [f for f in formes if f.startswith("(1,")][0] if any(f.startswith("(1,") for f in formes) else formes[0]
            a, b = A[c][f].float(), B[c][f].float()
            rel = float((a - b).abs().max() / max(float(a.abs().max()), 1e-30))
            profil.append((c, rel)); pire = max(pire, rel)
            if rel > o.seuil and premier is None:
                premier = (i, c, rel, f)
        if pire > o.seuil:
            divergents += 1
        if premier is not None and premier[0] == i:
            print(f"PREMIER ÉCART : pas {i}, couche {premier[1]}, forme {premier[3]}, écart relatif max {premier[2]:.2e}")
            print("profil du pas (couche : écart) : " + " ".join(f"{c}:{r:.1e}" for c, r in profil))
    if premier is None:
        print(f"aucun écart > {o.seuil:g} sur {n} pas")
    else:
        print(f"pas divergents (écart > {o.seuil:g} sur au moins une couche) : {divergents}/{n}")


if __name__ == "__main__":
    main()
