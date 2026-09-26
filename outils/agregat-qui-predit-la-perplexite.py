#!/usr/bin/env python3
"""Quel agregat par tenseur predit la perplexite MESUREE ? Six points, sans carte.

Un sac a dos glouton suppose que l'objectif est ADDITIF : il ajoute des gains
marginaux. Or aucune de nos cles ne l'est.

    decibels          ne s'additionnent pas du tout
    erreur relative    un rapport ne s'additionne pas entre tenseurs
    erreur absolue     s'additionne en QUADRATURE, pas lineairement
    somme g^2 dw^2     additive, et dans l'unite de la perte (formule de chef)

Donc l'ordre n'est pas le seul defaut : meme avec la bonne cle par tenseur, un
glouton additif sur un objectif non additif se trompe. Ce script le teste
contre la SEULE verite dont nous disposons — six perplexites mesurees.

Pour chaque dossier, on calcule plusieurs agregats des erreurs par tenseur et
on regarde lequel suit la perplexite mesuree. L'agregat qui la suit le mieux
est celui qu'un sac a dos devrait minimiser.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.racine_modeles import MODELES  # noqa: E402
from outils._chemins import sorties  # noqa: E402

BASE = Path(MODELES)
# (dossier, perplexite MESUREE, corpus wiki-gptq, etalon exterieur 5,4141)
POINTS = [("Llama-2-7b-nvfp4", 5.6102), ("Llama-2-7b-quota-4g50", 5.5643),
          ("Llama-2-7b-quota-5g00", 5.5394), ("Llama-2-7b-quota-5g50", 5.5125),
          ("Llama-2-7b-quota-6g00", 5.4918), ("Llama-2-7b-int8", 5.4144)]
ECHELLES = sorties() / "normes-poids-source.json"


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main() -> int:
    if not ECHELLES.exists():
        print("ECHEC / CAUSE: normes des poids source absentes / "
              "SUITE: lancer outils/echelle-de-sortie-approchee.py")
        return 2
    S = json.loads(ECHELLES.read_text())
    lignes = []
    for nom, ppl in POINTS:
        p = BASE / nom / "acvram_manifest.json"
        if not p.exists():
            print(f"ECHEC / CAUSE: {nom} absent")
            return 2
        t = json.loads(p.read_text())["tensors"]
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
        if n == 0:
            print(f"  {nom:26s} aucun tenseur avec out_snr_db — dossier "
                  f"produit avant l'ajout du champ")
            continue
        # TEMOINS VOLONTAIREMENT BETES. Troisieme reserve de chef, et la
        # plus genante : les six points sont ORDONNES PAR BUDGET, donc
        # perplexite et somme d'erreurs decroissent toutes deux avec lui.
        # N'importe quelle grandeur monotone en le budget correlera au-dessus
        # de +0,9 — la correlation mesurerait alors une MONOTONIE COMMUNE et
        # non un pouvoir predictif. Ces temoins le disent : s'ils correlent
        # aussi bien que les agregats, les agregats n'ont rien demontre.
        n_int8 = sum(1 for v in t.values() if v.get("format") == "int8")
        octets_dossier = sum(
            (BASE / nom).glob("*.safetensors") and
            f.stat().st_size for f in (BASE / nom).glob("*.safetensors"))
        lignes.append({"nom": nom, "ppl": ppl, "n": n, "rel": s_rel,
                       "abs": s_abs, "quad": s_quad ** 0.5,
                       "bete_promus": -n_int8, "bete_octets": -octets_dossier,
                       "bete_rang": 0})
    if len(lignes) < 4:
        print(f"ECHEC / CAUSE: seulement {len(lignes)} dossiers exploitables "
              f"sur {len(POINTS)} — pas assez pour correler / "
              f"SUITE: les dossiers sans out_snr_db par tenseur ne peuvent "
              f"pas repondre")
        return 3
    print(f"{'dossier':26s} {'n':>4s} {'PPL':>8s} "
          f"{'somme rel':>12s} {'somme abs':>12s} {'quadrature':>12s}")
    for l in lignes:
        print(f"{l['nom'][:26]:26s} {l['n']:4d} {l['ppl']:8.4f} "
              f"{l['rel']:12.4f} {l['abs']:12.2f} {l['quad']:12.2f}")
    print()
    ppl = [l["ppl"] for l in lignes]
    # le rang pur : 1, 2, 3... c'est la MONOTONIE SEULE, sans aucun contenu
    for i, l in enumerate(sorted(lignes, key=lambda l: l["bete_promus"])):
        l["bete_rang"] = -(i + 1)
    for cle, etiq in (("rel", "somme des erreurs RELATIVES"),
                      ("abs", "somme des erreurs ABSOLUES (approchees)"),
                      ("quad", "QUADRATURE des erreurs absolues")):
        r = pearson([l[cle] for l in lignes], ppl)
        print(f"  agregat  {etiq:44s} {r:+.4f}")
    print()
    for cle, etiq in (("bete_promus", "TEMOIN BETE : nombre de tenseurs promus"),
                      ("bete_octets", "TEMOIN BETE : octets du dossier"),
                      ("bete_rang", "TEMOIN BETE : le RANG seul (1,2,3...)")):
        r = pearson([l[cle] for l in lignes], ppl)
        print(f"  temoin   {etiq:44s} {r:+.4f}")
    meilleur_bete = max(abs(pearson([l[c] for l in lignes], ppl))
                        for c in ("bete_promus", "bete_octets", "bete_rang"))
    meilleur_agr = max(abs(pearson([l[c] for l in lignes], ppl))
                       for c in ("rel", "abs", "quad"))
    print(f"\n  meilleur agregat {meilleur_agr:+.4f} contre meilleur temoin "
          f"bete {meilleur_bete:+.4f}")
    # SEUIL EXIGEANT, et il faut qu'il le soit : sur six points ordonnes par
    # budget, une marge de quelques centiemes ne distingue rien. Mon premier
    # seuil etait a 0,01 et il m'aurait laisse conclure « pas de la pure
    # monotonie » avec une marge de 0,016 — c'est-a-dire me donner raison de
    # justesse contre un temoin qui n'a AUCUN contenu. Un seuil qui laisse
    # passer sa propre these de peu n'est pas un seuil.
    if meilleur_bete >= meilleur_agr - 0.05:
        print(f"  VERDICT : le meilleur temoin BETE atteint {meilleur_bete:+.4f} "
              f"pour une marge de\n  seulement {meilleur_agr - meilleur_bete:+.4f}. "
              f"Sur six points ORDONNES PAR BUDGET, cela ne distingue\n  rien : "
              f"les octets du dossier n'ont AUCUN contenu predictif et arrivent "
              f"presque\n  au meme niveau. LA CORRELATION MESURE LA MONOTONIE "
              f"COMMUNE.\n\n  LE +0,9931 EST RETIRE COMME DEMONSTRATION. Ce qui "
              f"survit, et rien de plus : l'ordre\n  rel < abs < quad va dans le "
              f"sens de l'argument de propagation, sans le\n  soutenir. Le "
              f"trancher demande des points NON ordonnes — plusieurs dossiers "
              f"au\n  MEME budget avec des cles differentes, ce que les bras "
              f"A/B produisent.")
    else:
        print(f"  VERDICT : les agregats depassent le meilleur temoin bete de "
              f"{meilleur_agr - meilleur_bete:+.4f}.\n  L'ecart est faible sur "
              f"six points ordonnes ; il ne suffit pas a couronner un agregat, "
              f"mais\n  il n'est pas nul, donc les agregats ne sont pas de la "
              f"pure monotonie.")
    print(f"\n  ({len(lignes)} points ; une correlation sur {len(lignes)} "
          f"points ne distingue pas des ecarts fins,\n   et les trois agregats "
          f"derivent des memes out_snr_db — ils ne sont pas independants.)")
    print("""
  CE QUE CE TEST PEUT ET NE PEUT PAS DIRE
  Il peut ecarter un agregat qui NE SUIT PAS la perplexite : c'est une
  refutation, et elle vaut. Il ne peut pas couronner le meilleur des trois
  sur six points, ni prouver qu'un quatrieme agregat ne ferait pas mieux.
  Et il ne teste PAS l'additivite elle-meme : pour cela il faudrait comparer
  la perplexite d'un ensemble de promotions a la SOMME des perplexites de
  chaque promotion prise seule — une mesure par tenseur, donc un chantier.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
