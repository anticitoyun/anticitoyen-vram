#!/usr/bin/env python3
"""La perplexite change-t-elle quand les logits passent en fp32 ?

CE QUI EST A TRANCHER. L'arrondi bf16 des logits produit deux erreurs de
natures differentes, et elles ne se lisent pas au meme endroit :

    bruit   erreur centree sur log p, ~0,35 nat par jeton, qui DECROIT en
            1/racine(N) : 2,2 % a N = 256, 0,09 % a N = 148 920
    biais   logsumexp est CONVEXE, donc par Jensen l'arrondi le surestime
            systematiquement de 1/2 sigma^2 (1 - somme p^2). Ce terme NE
            DECROIT PAS avec N. Verifie par simulation sur les logits reels de
            GLM-4.7 : 0,009875 nat mesure contre 0,010410 predit, accord a 5 %,
            soit +1,05 % de perplexite SUR-estimee, a toute longueur.

D'ou le protocole a deux longueurs, et c'est lui qui separe les deux termes :

    N = 256      le bruit domine (2,2 %), le biais y est noye
    N complet    le bruit s'efface (0,09 %), SEUL LE BIAIS SUBSISTE (~1 %)

    ecart au complet nul      -> le biais de Jensen est de trop, il se retire
    ecart au complet ~1 %     -> il est etabli, et il faut alors reprendre nos
                                 comparaisons de PPL entre modeles centres et
                                 modeles decales : le biais depend du PAS, donc
                                 de la magnitude des logits, donc du MODELE —
                                 1,05 % sur GLM contre 0,001 % sur un modele
                                 centre, un rapport de 1024. Il fausse la
                                 comparaison, pas la mesure.

CONDITION DE RECEVABILITE, sans laquelle la mesure ne prouve rien : le bras
fp32 doit rendre PLUS DE VALEURS DISTINCTES que le bras bf16. Si le compte ne
bouge pas, le correctif n'a pas pris, et deux perplexites identiques ne
diraient rien du tout. L'instrument doit avoir montre qu'il sait rendre
« different » avant qu'on lise son verdict — c'est le denombrement, 44 contre
~2 600 sur GLM, 2 473 contre 149 356 sur Qwen3-4B.

Le sens attendu est inscrit : la PPL du bras bf16 doit etre SUPERIEURE, jamais
inferieure. Un ecart de signe oppose refuterait le mecanisme entier.
"""
import argparse
import json
import os
import subprocess
import sys


def mesurer(modele: str, n: int, bf16: bool) -> dict:
    import torch
    from acvram.evaluate import perplexity
    from acvram.engine import model as M

    niveaux = {"min": None, "max": None, "n": 0}
    brut = M.ACVRamModel._logits_finaux

    def espion(self, logits):
        u = torch.unique(logits.detach().reshape(-1)).numel()
        niveaux["n"] += 1
        niveaux["min"] = u if niveaux["min"] is None else min(niveaux["min"], u)
        niveaux["max"] = u if niveaux["max"] is None else max(niveaux["max"], u)
        return brut(self, logits)

    M.ACVRamModel._logits_finaux = espion
    r = perplexity(modele, max_tokens=n, min_context=256)
    d = r.to_dict() if hasattr(r, "to_dict") else {"perplexity": r.perplexity}
    return {"bf16": bf16, "n_demande": n, "ppl": d.get("perplexity"),
            "jetons": d.get("tokens") or d.get("n_tokens"),
            "niveaux_min": niveaux["min"], "niveaux_max": niveaux["max"],
            "appels": niveaux["n"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", required=True)
    ap.add_argument("--n", default="256,8192")
    ap.add_argument("--un-point", nargs=2, metavar=("N", "BF16"))
    ap.add_argument("--sortie", default="/tmp/ppl-logits-fp32.json")
    a = ap.parse_args()

    if a.un_point:
        n, bf = int(a.un_point[0]), a.un_point[1] == "1"
        print(json.dumps(mesurer(a.modele, n, bf)))
        return 0

    releves = []
    for n in [int(x) for x in a.n.split(",")]:
        for bf in ("1", "0"):
            env = dict(os.environ)
            if bf == "1":
                env["ACVRAM_LOGITS_BF16"] = "1"
            else:
                env.pop("ACVRAM_LOGITS_BF16", None)
            r = subprocess.run([sys.executable, __file__, "--modele", a.modele,
                                "--un-point", str(n), bf], env=env,
                               capture_output=True, text=True, timeout=7200)
            if r.returncode != 0:
                print(r.stderr[-1500:], file=sys.stderr)
                return 1
            d = json.loads(r.stdout.strip().splitlines()[-1])
            releves.append(d)
            print(f"  N={n:>6} {'bf16' if bf == '1' else 'fp32'} : "
                  f"PPL {d['ppl']:.4f}  sur {d['jetons']} jetons, "
                  f"niveaux {d['niveaux_min']}-{d['niveaux_max']} "
                  f"({d['appels']} appels)", flush=True)

    print()
    verdict = []
    for n in [int(x) for x in a.n.split(",")]:
        b = next((d for d in releves if d["n_demande"] == n and d["bf16"]), None)
        f = next((d for d in releves if d["n_demande"] == n and not d["bf16"]), None)
        if not (b and f):
            continue
        # RECEVABILITE d'abord : l'instrument sait-il rendre « different » ?
        if not (f["niveaux_max"] > b["niveaux_max"]):
            print(f"N={n} IRRECEVABLE : le bras fp32 ne rend pas plus de valeurs "
                  f"distinctes ({f['niveaux_max']} contre {b['niveaux_max']}) — "
                  "le correctif n'a pas pris, et deux PPL egales ne diraient rien.")
            verdict.append(None)
            continue
        ec = 100 * (b["ppl"] / f["ppl"] - 1)
        print(f"N={n:>6} : PPL bf16 {b['ppl']:.4f} contre fp32 {f['ppl']:.4f} "
              f"-> {ec:+.3f} %   (niveaux {b['niveaux_max']} -> {f['niveaux_max']})")
        verdict.append(ec)
    if verdict and verdict[-1] is not None:
        ec = verdict[-1]
        if abs(ec) < 0.2:
            print("\nVERDICT : ecart nul au complet — le biais de Jensen est de "
                  "trop et se retire. Seul le bruit agissait, et il se moyenne.")
        elif ec > 0:
            print(f"\nVERDICT : {ec:+.3f} % au complet, dans le sens predit — le "
                  "biais convexe est ETABLI. Nos comparaisons de PPL entre "
                  "modeles de magnitudes differentes sont a reprendre.")
        else:
            print(f"\nVERDICT : {ec:+.3f} %, SIGNE OPPOSE au mecanisme. Le "
                  "raisonnement entier est a refaire : l'arrondi ne peut pas "
                  "SOUS-estimer logsumexp.")
    json.dump(releves, open(a.sortie, "w"), indent=1)
    print(f"TERMINE — releve dans {a.sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
