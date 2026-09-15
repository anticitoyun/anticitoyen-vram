#!/usr/bin/env python3
"""Quel agregat par tenseur suit la PPL A BUDGET FIXE ?

La courbe du quota (agregat-qui-predit-la-perplexite.py) fait varier le BUDGET :
tout y decroit ensemble, une monotonie commune gonfle toute correlation. Ici
les quatre points ont le MEME budget (6,00 Gio) et ne different QUE par l'ordre
du sac a dos — donc par l'ensemble promu. Le confondant de budget disparait ;
ce qui reste est le pouvoir predictif de l'ordre.

QUATRE points seulement (les quatre ordres essayes). Pearson sur n=4 est
anemique et un seul point le domine : on rapporte donc l'ACCORD DE RANG (l'ordre
que l'agregat impose vs l'ordre de la PPL), pas un coefficient qu'on
sur-interpreterait. Quatre est le minimum utile ; trois serait sous le minimum.
"""
import json
from pathlib import Path
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.racine_modeles import MODELES  # noqa: E402
from outils._chemins import sorties  # noqa: E402

BASE = Path(MODELES)
ECH = sorties() / "normes-poids-source.json"
# (dossier, PPL mesuree, budget 6,00 Gio, deux exemplaires pour base_croissant)
POINTS = [("Llama-2-7b-ordre-inverse", 5.4482),        # B
          ("Llama-2-7b-ordre-normal", 5.4918),         # A (defaut)
          ("Llama-2-7b-ordre-erreur", 5.4927),
          ("Llama-2-7b-ordre-base_croissant", 5.4981)]


def rang(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0] * len(xs)
    for pos, i in enumerate(o):
        r[i] = pos
    return r


def kendall(a, b):
    n, c, d = len(a), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            c += s > 0
            d += s < 0
    return (c - d) / (c + d) if (c + d) else float("nan")


S = json.loads(ECH.read_text())
rows = []
for nom, ppl in POINTS:
    m = json.loads((BASE / nom / "acvram_manifest.json").read_text())
    t = m["tensors"]
    s_rel = s_abs = s_quad = 0.0
    n = 0
    for k, v in t.items():
        snr = v.get("out_snr_db")
        if snr is None or k not in S:
            continue
        rel = 10.0 ** (-snr / 20.0)
        ech = S[k]
        s_rel += rel
        s_abs += rel * ech
        s_quad += (rel * ech) ** 2
        n += 1
    promus = m["budget"]["promus"]
    rows.append({"nom": nom.replace("Llama-2-7b-ordre-", ""), "ppl": ppl,
                 "n": n, "rel": s_rel, "abs": s_abs, "quad": s_quad ** 0.5,
                 "promus": promus})

print(f"{'ordre':16s}{'PPL':>9s}{'promus':>8s}{'somme_rel':>12s}"
      f"{'somme_abs':>12s}{'quadrature':>12s}")
for r in rows:
    print(f"{r['nom']:16s}{r['ppl']:9.4f}{r['promus']:8d}{r['rel']:12.2f}"
          f"{r['abs']:12.4f}{r['quad']:12.6f}")

ppl = [r["ppl"] for r in rows]
rp = rang(ppl)
print(f"\nordre PPL (0=meilleur) : "
      + ", ".join(f"{r['nom']}={rp[i]}" for i, r in enumerate(rows)))
print(f"\n{'agregat':14s}{'tau de Kendall vs PPL':>26s}   ordre impose")
for cle, signe in (("rel", +1), ("abs", +1), ("quad", +1),
                   ("promus", -1)):
    vals = [signe * r[cle] for r in rows]
    tau = kendall(rang(vals), rp)
    ordre = [rows[i]["nom"] for i in sorted(range(len(rows)),
                                            key=lambda i: vals[i])]
    print(f"{cle:14s}{tau:26.3f}   {' < '.join(ordre)}")

print("\nLecture : tau=+1 -> l'agregat classe les quatre ordres EXACTEMENT")
print("comme la PPL ; tau=-1 -> a l'envers ; 0 -> aucun lien. n=4, minimum")
print("utile — un tau parfait sur quatre points est une PISTE, pas une preuve.")
print("Le temoin 'promus' (nombre de tenseurs promus) doit ECHOUER : B a le")
print("MOINS de promus (149) et la MEILLEURE PPL.")
