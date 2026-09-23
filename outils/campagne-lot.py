#!/usr/bin/env python3
"""Le lot conditionne PA_WARPS — huit points pour le savoir.

LA GRILLE DU NOYAU EST (BQ, HQ, C). Tout le dossier de l'attention a ete
mesure a BQ = 1, un regime que le service ne verra presque jamais : douze
sequences concurrentes donnent x2,84 en debit (mesure).

MECANISME, verifiable sans mesure : 48 warps par SM se repartissent en 12
blocs de 4 warps mais seulement 3 blocs de 16. Les blocs RESIDENTS passent de
2040 a 510. A BQ = 1, 80 blocs tiennent partout et les warps sont gratuits ;
a BQ = 12, 960 blocs tiennent en UNE vague a 4 warps et en demandent 1,9 a 16.

    BQ   blocs (chunk 512, ctx 1024)   vagues a 4w   a 16w
     1                            80          0,0     0,2
     4                           320          0,2     0,6
     8                           640          0,3     1,3
    12                           960          0,5     1,9

PREDICTION ECRITE AVANT LA MESURE : a BQ = 1, PA_WARPS=16 gagne ~73 % sur
l'attention (mesure : 5,198 -> 1,409 ms). L'ecart se REDUIT quand BQ monte et
PEUT S'INVERSER a BQ = 12.

CLAUSE DE REFUTATION : si 16 gagne encore autant a BQ = 12, le modele de vagues
est faux et il faudra chercher ailleurs pourquoi les warps paient. Le modele a
deja ete juste sur la forme et faux sur son domaine deux fois aujourd'hui.

CE QUI SE DECIDE ICI : PA_WARPS = 16 ne va pas sur main avant ce verdict, et
PA_WARPS = 32 est suspendu — a BQ = 12 il donnerait 3,8 vagues contre 0,5.
"""
import argparse
import json
import os
import subprocess
import sys

BANC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "part-attention-trois-bras.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", required=True)
    ap.add_argument("--bq", default="1,4,8,12")
    ap.add_argument("--warps", default="4,16")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--sortie", default="/tmp/campagne-lot.json")
    a = ap.parse_args()

    releves, empreintes = [], {}
    for w in [x.strip() for x in a.warps.split(",")]:
        for bq in [int(x) for x in a.bq.split(",")]:
            env = dict(os.environ, ACVRAM_PA_WARPS=w,
                       ACVRAM_PA_ARM_SORTIE_FAUSSE="1")
            out = f"/tmp/lot-w{w}-bq{bq}.json"
            print(f"== PA_WARPS={w} BQ={bq} ==", flush=True)
            r = subprocess.run(
                [sys.executable, BANC, "--modele", a.modele, "--sortie", out,
                 "--ctx", str(a.ctx), "--generes", str(min(200, a.ctx // 2)),
                 "--bq", str(bq), "--ordre", "A,B,B,A"],
                env=env, timeout=7200)
            if r.returncode == 2:
                print(f"MANCHE SANS OBJET a PA_WARPS={w} BQ={bq} : plan degrade "
                      "(exil ou graphes inactifs). Arret — un point manquant "
                      "vaut mieux qu'un point faux.", file=sys.stderr)
                return 2
            if r.returncode != 0:
                print(f"ECHEC a PA_WARPS={w} BQ={bq} (code {r.returncode})",
                      file=sys.stderr)
                return 1
            d = json.load(open(out))
            r0 = d["releves"][0]
            releves.append({"warps": int(w), "bq": bq, "p_ms": d["p_ms"],
                            "pas_ms": r0["median_ms"],
                            "empreinte_so": r0["empreinte_so"],
                            "seqs_finies": r0.get("seqs_finies")})
            empreintes.setdefault(w, set()).add(r0["empreinte_so"])
            print(f"   -> p {d['p_ms']:.3f} ms, pas {r0['median_ms']:.3f} ms, "
                  f"{r0.get('seqs_finies')} sequences finies, "
                  f"empreinte {r0['empreinte_so']}", flush=True)

    par_valeur = {w: next(iter(e)) for w, e in empreintes.items()}
    if len(set(par_valeur.values())) != len(par_valeur):
        print(f"ARRET : deux valeurs de PA_WARPS ont donne le meme binaire "
              f"({par_valeur}) — le hachage ne voit pas les flags et un temps "
              "plat se lirait « pas d'effet ». Rien n'est publie.",
              file=sys.stderr)
        return 3

    print(f"\n{'BQ':>4} {'p a 4w':>9} {'p a 16w':>9} {'16 contre 4':>12}"
          f" | {'pas 4w':>9} {'pas 16w':>9} {'debit 16/4':>11}")
    for bq in [int(x) for x in a.bq.split(",")]:
        d4 = next((r for r in releves if r["bq"] == bq and r["warps"] == 4), None)
        d16 = next((r for r in releves if r["bq"] == bq and r["warps"] == 16), None)
        if not (d4 and d16):
            continue
        print(f"{bq:>4} {d4['p_ms']:>9.3f} {d16['p_ms']:>9.3f} "
              f"{100 * (d16['p_ms'] / d4['p_ms'] - 1):>11.1f} % | "
              f"{d4['pas_ms']:>9.3f} {d16['pas_ms']:>9.3f} "
              f"{100 * (d4['pas_ms'] / d16['pas_ms'] - 1):>10.1f} %")
    json.dump(releves, open(a.sortie, "w"), indent=1)
    print(f"\nTERMINE — releve dans {a.sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
