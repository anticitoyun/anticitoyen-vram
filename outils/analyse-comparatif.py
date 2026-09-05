#!/usr/bin/env python3
"""Lit le TSV du comparatif des quatre moteurs et en tire ce qui compte.

    analyse-comparatif.py docs/comparatif-20260905.tsv [--md]

Par modèle : débit de chaque moteur, gagnant, écart d'acvram au gagnant,
famille (dense, MoE, hybride à état). Puis le bilan : modèles gagnés par
moteur, médianes, énergie, temps de chargement, et la liste des modèles où
acvram perd — triée par écart, c'est la liste de travail.
"""
import csv, json, os, sys
from collections import defaultdict

KIMI = os.path.expanduser("~/.kimi-code")
MOTEURS = ("acvram", "llamacpp", "vllm", "tabby")


def famille(alias_acvram: str) -> str:
    """dense | moe | hybride | hybride-moe, d'après le manifeste acvram."""
    for l in open(os.path.join(KIMI, "acvram-chemins.tsv")):
        c = l.rstrip("\n").split("\t")
        if c[0] == alias_acvram:
            try:
                t = json.load(open(os.path.join(c[1], "acvram_manifest.json")))["tensors"]
            except Exception:
                return "?"
            moe = any(".mlp.experts." in k for k in t)
            hyb = any("linear_attn" in k or ".mamba" in k or "kv_a_proj" in k for k in t)
            return ("hybride-" if hyb else "") + ("moe" if moe else "dense") if (hyb or moe) else "dense"
    return "?"


def main():
    chemin = sys.argv[1]
    md = "--md" in sys.argv
    rows = list(csv.DictReader(open(chemin), delimiter="\t"))
    par = defaultdict(dict)
    for r in rows:
        par[r["modele"]][r["moteur"]] = r
    lignes = []
    for nom, ms in sorted(par.items()):
        if len(ms) < 2:
            continue
        deb = {m: float(r["t_s"]) for m, r in ms.items() if r["etat"] == "ok"}
        if not deb:
            continue
        gagnant = max(deb, key=deb.get)
        acv = deb.get("acvram")
        ecart = (acv / deb[gagnant] - 1) * 100 if acv else None
        fam = famille(ms["acvram"]["alias"]) if "acvram" in ms else "?"
        lignes.append({"modele": nom, "fam": fam, "deb": deb, "gagnant": gagnant, "ecart": ecart,
                       "charge": {m: float(r["chargement_s"]) for m, r in ms.items() if r["etat"] == "ok"},
                       "jkj": {m: float(r["j_kJ"]) for m, r in ms.items() if r["etat"] == "ok"},
                       "erreurs": [m for m, r in ms.items() if r["etat"] != "ok"]})

    def med(v):
        v = sorted(v); return v[len(v) // 2] if v else 0.0

    print(f"{len(lignes)} modèles servis par au moins deux moteurs\n")
    gagnes = defaultdict(int)
    for l in lignes:
        gagnes[l["gagnant"]] += 1
    print("| moteur | modèles gagnés | médiane t/s | médiane j/kJ | chargement médian |")
    print("|---|---|---|---|---|")
    for m in MOTEURS:
        debs = [l["deb"][m] for l in lignes if m in l["deb"]]
        if not debs:
            continue
        print(f"| {m} | {gagnes[m]} | {med(debs):.0f} | "
              f"{med([l['jkj'][m] for l in lignes if m in l['jkj']]):.0f} | "
              f"{med([l['charge'][m] for l in lignes if m in l['charge']]):.0f} s |")
    print()
    print("| modèle | famille | " + " | ".join(MOTEURS) + " | gagnant | acvram vs gagnant |")
    print("|---|---|" + "---|" * len(MOTEURS) + "---|---|")
    for l in sorted(lignes, key=lambda l: (l["ecart"] if l["ecart"] is not None else 1e9)):
        cells = [f"{l['deb'][m]:.0f}" if m in l["deb"] else ("éch." if m in l["erreurs"] else "—") for m in MOTEURS]
        e = f"{l['ecart']:+.0f} %" if l["ecart"] is not None else "—"
        print(f"| {l['modele'][:44]} | {l['fam']} | " + " | ".join(cells) + f" | {l['gagnant']} | {e} |")
    print()
    perd = [l for l in lignes if l["ecart"] is not None and l["ecart"] < -5]
    print(f"acvram perd de plus de 5 % sur {len(perd)} modèles :")
    fam = defaultdict(list)
    for l in perd:
        fam[l["fam"]].append(l["ecart"])
    for f, e in sorted(fam.items(), key=lambda kv: med(kv[1])):
        print(f"  {f:14s} {len(e):2d} modèles, écart médian {med(e):+.0f} %")


if __name__ == "__main__":
    main()
