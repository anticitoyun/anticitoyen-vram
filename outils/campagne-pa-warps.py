#!/usr/bin/env python3
"""Effet de PA_WARPS sur le temps de l'attention paginee, a C CONSTANT.

PA_WARPS ajoute des warps DANS un bloc ; PA_CHUNK ajoute des blocs. Les deux
augmentent le parallelisme, mais par des chemins differents et a des couts
differents : PA_WARPS ne touche ni le reduce, ni le nombre de lancements, ni la
memoire de travail. On les mesure donc separement — PA_CHUNK reste fixe ici.

CE QUE LA CAPACITE DIT, lue au binaire (REG 40, SHARED 3648, 170 SM) :

    PA_WARPS   threads   blocs/SM   warps/SM   a 80 blocs : warps actifs
           4       128         12         48        320   ( 3,9 % de 8160)
           8       256          6         48        640   ( 7,8 %)
          16       512          3         48       1280   (15,7 %)
          32      1024          1         32       2560   (31,4 %)

warps/SM vaut 48 dans les trois premiers cas : A SM SATURE, PA_WARPS EST
NEUTRE. Il ne paie que parce que nous sommes a 80 blocs sur 2040 et que chaque
SM actif ne recoit qu'un bloc, qu'il remplit alors mieux. Le jour ou PA_CHUNK
remplira la carte, PA_WARPS cessera de payer.

LE POINT DE MESURE PRINCIPAL EST ctx 128, pas 3007 : a ctx 128 avec chunk 512,
C = 1, donc 40 blocs et 130 SM VIDES — aucune valeur de PA_CHUNK n'y peut rien
puisque le contexte lui-meme borne C. C'est le regime du decodage a contexte
frais, le debut de toute conversation.

PREDICTION ECRITE AVANT LA MESURE : a ctx 1024, -20 a -45 % sur les 5,198 ms
mesurees a PA_WARPS=4, et rien de plus a 32 (voire une regression, l'occupation
par SM y retombant de 48 a 32 warps). A ctx 128, le maximum du gain relatif de
la campagne, mais un SENS et non une valeur — le temps de l'attention n'y a
jamais ete mesure.

CE QUI ME REFUTERAIT : un temps plat de 4 a 32. Cela voudrait dire que 4 warps
cachent deja toute la latence, donc que le noyau est limite par autre chose que
le parallelisme — ce qui contredirait les 6,1x et 8,6x etablis par convergence
et rouvrirait la question au lieu de la fermer.
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
    ap.add_argument("--warps", default="4,8,16,32")
    ap.add_argument("--ctx", default="128,1024")
    ap.add_argument("--sortie", default="/tmp/campagne-pa-warps.json")
    a = ap.parse_args()
    warps = [int(w) for w in a.warps.split(",")]
    ctxs = [int(c) for c in a.ctx.split(",")]

    releves, empreintes = [], {}
    for w in warps:
        for ctx in ctxs:
            env = dict(os.environ, ACVRAM_PA_WARPS=str(w),
                       ACVRAM_PA_ARM_SORTIE_FAUSSE="1")
            out = f"/tmp/pa-warps-{w}-ctx{ctx}.json"
            print(f"== PA_WARPS={w} ctx={ctx} ==", flush=True)
            r = subprocess.run(
                [sys.executable, BANC, "--modele", a.modele, "--sortie", out,
                 "--ctx", str(ctx), "--generes", str(min(200, ctx // 2)),
                 "--ordre", "A,B,B,A"],
                env=env, timeout=3600)
            if r.returncode == 2:
                print(f"MANCHE SANS OBJET a PA_WARPS={w} ctx={ctx} : plan "
                      "degrade (exil ou graphes inactifs). La campagne "
                      "s'arrete — un point manquant vaut mieux qu'un point "
                      "faux.", file=sys.stderr)
                return 2
            if r.returncode != 0:
                print(f"ECHEC a PA_WARPS={w} ctx={ctx} (code {r.returncode})",
                      file=sys.stderr)
                return 1
            d = json.load(open(out))
            emp = d["releves"][0]["empreinte_so"]
            r0 = d["releves"][0]
            releves.append({"warps": w, "ctx": ctx, "p_ms": d["p_ms"],
                            "pas_ms": r0["median_ms"], "empreinte_so": emp,
                            "graphes": r0.get("graphes"),
                            "captures": r0.get("captures"),
                            "plan_est_decode_tok_s":
                                r0.get("plan_est_decode_tok_s")})
            print(f"   -> p {d['p_ms']:.3f} ms, pas {r0['median_ms']:.3f} ms, "
                  f"empreinte {emp}, graphes {r0.get('graphes')} "
                  f"({r0.get('captures')} captures), plan annonce "
                  f"{r0.get('plan_est_decode_tok_s')}", flush=True)
            empreintes.setdefault(w, set()).add(emp)

    # LE CONTROLE QUI EMPECHE LE RESULTAT LE PLUS INSIDIEUX. Un parametre passe
    # par -D ne change pas le fichier source : si le hachage du binaire ne tient
    # pas compte des flags, quatre reglages rendent LE MEME binaire, quatre fois
    # le meme chiffre, et un temps plat se lit « le reglage n'a pas d'effet » —
    # refutant a tort une prediction juste. Quatre reglages doivent donner
    # quatre empreintes.
    par_valeur = {w: next(iter(e)) for w, e in empreintes.items()}
    if len(set(par_valeur.values())) != len(par_valeur):
        print("ARRET : deux valeurs de PA_WARPS ont donne LE MEME binaire "
              f"({par_valeur}) — le hachage ne voit pas les flags, et un temps "
              "plat se lirait « pas d'effet ». Rien n'est publie.",
              file=sys.stderr)
        return 2
    print(f"\ncontrole des binaires : {len(par_valeur)} valeurs, "
          f"{len(set(par_valeur.values()))} empreintes distinctes")

    print(f"\n{'PA_WARPS':>9} {'ctx':>5} {'pas (ms)':>10} {'p (ms)':>9} "
          f"{'p / pas':>8}  {'vs PA_WARPS=4':>14}")
    for ctx in ctxs:
        base = next((r["p_ms"] for r in releves
                     if r["ctx"] == ctx and r["warps"] == warps[0]), None)
        for r in [x for x in releves if x["ctx"] == ctx]:
            delta = (f"{100 * (r['p_ms'] / base - 1):+7.1f} %"
                     if base else "     —")
            print(f"{r['warps']:>9} {r['ctx']:>5} {r['pas_ms']:>10.3f} "
                  f"{r['p_ms']:>9.3f} {100 * r['p_ms'] / r['pas_ms']:>7.2f} % "
                  f"{delta:>14}")
    json.dump(releves, open(a.sortie, "w"), indent=1)
    print(f"\nTERMINE — releve dans {a.sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
