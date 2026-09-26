#!/usr/bin/env python3
"""Etape A du chantier de la speculation : le taux d'acceptation, et rien d'autre.

AUCUNE MESURE DE TEMPS. Le taux commande les trois motifs qui menent a la
speculation — occupation, energie, amortissement de la tete — et il les
commande seul : chiffres avec le taux deja publie (1,105 a 1,163 jeton par pas
sur nemo-12b-thinking-exl3, ETABLI.md:454), les trois donnent des gains petits.

    amortissement de la tete   a 1,163 : 14 % d'une tete qui pese ~1,4 % du pas
                               sur un converti nvfp4  ->  ~0,2 % du pas
    occupation                 q_len = 2 double les blocs, mais a BQ = 12 le lot
                               remplit deja la carte
    energie                    +2,1 % mesure, le seul des trois deja chiffre

CRITERE D'ARRET ECRIT D'AVANCE : si le taux reste SOUS 1,3 sur nos convertis,
les etapes B et C sont sans objet et le chantier s'arrete ici. Une etape qui
peut clore le dossier a elle seule, sans toucher au debit ni a l'energie.

CE QUE LA LECTURE DU PARC IMPOSE AU PROTOCOLE, et qui evite une faute :
14 convertis sur 114 portent une tete MTP, et Qwen2.5-Coder-14B — notre modele
de reference pour tout le dossier de l'attention — N'EN PORTE PAS. Comparer
donc `ngram` sur Qwen2.5-Coder-14B a `mtp` sur un autre modele melangerait deux
modeles : c'est la faute du jour. Les deux modes se mesurent sur LE MEME
modele, necessairement l'un des 14.

Convertis porteurs reperes : Qwen3.6-27B-Fable-Fusion-711-Heretic-MTP,
Huihui-Qwen3.8-27B-abliterated, Ornith-1.5-35B-A3B-Heretic,
Qwen3.6-27B-Fable-Fusion-Uncensored-Heretic, qwen36-27b-heretic (et 9 autres).

TROIS FAMILLES D'INVITE, parce que le taux depend du domaine et que `ngram`
propose depuis le texte deja produit : repetitif (son terrain), prose (son
terrain le plus pauvre), et code — le regime reel d'un modele Coder, jamais
mesure sur cet axe.
"""
import argparse
import json
import os
import sys

SEUIL_ARRET = 1.3
GENERES = 128

FAMILLES = {
    "repetitif": "Repete exactement la liste suivante dix fois, sans rien ajouter : "
                 "alpha, beta, gamma, delta, epsilon, zeta, eta, theta.",
    "prose": "Ecris un paragraphe de prose continue sur la maniere dont la "
             "lumiere change au-dessus d'un estuaire entre l'aube et midi.",
    "code": "Ecris une fonction Python qui prend une liste de dictionnaires et "
            "la regroupe par une cle donnee, avec les annotations de type et "
            "une docstring.",
}


def mesurer(modele: str, mode: str, famille: str) -> dict:
    import torch
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    from transformers import AutoTokenizer

    charge = load_model(modele, dtype=torch.bfloat16, device_override="cuda:0")
    exiles = [c.index for c in charge.plan.layers
              if "cpu" in (c.attn_storage, c.mlp_storage) or c.mlp_exec == "cpu"]
    if exiles:
        print(f"MANCHE SANS OBJET : {len(exiles)} couche(s) exilee(s)",
              file=sys.stderr)
        sys.exit(2)
    tok = AutoTokenizer.from_pretrained(modele, trust_remote_code=True)
    # Le moteur attend un OBJET speculateur, pas un nom de mode. On reprend la
    # fabrique du CLI (cli.py:363) plutot que d'en ecrire une seconde : deux
    # fabriques divergent, et la divergence se lirait comme un ecart de taux.
    a_mtp = getattr(charge.model, "mtp", None) is not None
    speculateur = None
    if mode == "ngram":
        from acvram.engine.speculative import NGramProposer
        speculateur = NGramProposer()
    elif mode == "mtp":
        if not a_mtp:
            print("MANCHE SANS OBJET : ce converti ne porte pas de tete MTP. "
                  "Le CLI REPLIE alors sur ngram (cli.py:364, mode auto) : "
                  "mesurer ici rendrait le taux du n-gramme sous l'etiquette "
                  "mtp, et un repli silencieux se lit comme un resultat.",
                  file=sys.stderr)
            sys.exit(2)
        from acvram.engine.speculative import MTPProposer
        speculateur = MTPProposer(charge.model, max_model_len=1024)
    moteur = Engine(charge, tok, max_batch_size=1, max_model_len=1024,
                    speculator=speculateur)
    ids = tok(FAMILLES[famille], return_tensors=None)["input_ids"]
    seq = moteur.add_request(ids, SamplingParams(temperature=0.0,
                                                 max_tokens=GENERES))
    while not seq.finished and len(seq.output_ids) < GENERES:
        moteur.step()
    st = moteur.stats.to_dict()
    return {"mode": mode, "famille": famille, "a_mtp": a_mtp,
            "jetons": len(seq.output_ids),
            "proposes": st.get("proposed_tokens"),
            "acceptes": st.get("accepted_tokens"),
            "par_pas": st.get("tokens_per_step"),
            "pas": st.get("steps")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", required=True,
                    help="DOIT porter une tete MTP pour que la comparaison "
                         "ngram/mtp ait un objet")
    ap.add_argument("--modes", default="none,ngram,mtp")
    ap.add_argument("--un-point", nargs=2, metavar=("MODE", "FAMILLE"))
    ap.add_argument("--sortie", default="/tmp/etape-a-taux.json")
    a = ap.parse_args()

    if a.un_point:
        print(json.dumps(mesurer(a.modele, *a.un_point)))
        return 0

    import subprocess
    releves = []
    for mode in [m.strip() for m in a.modes.split(",")]:
        for famille in FAMILLES:
            r = subprocess.run(
                [sys.executable, __file__, "--modele", a.modele,
                 "--un-point", mode, famille], capture_output=True, text=True,
                timeout=3600)
            if r.returncode == 2:
                print(f"  {mode:>6} / {famille:<10} SANS OBJET : "
                      f"{r.stderr.strip().splitlines()[-1][:90]}")
                continue
            if r.returncode != 0:
                print(r.stderr[-1200:], file=sys.stderr)
                return 1
            d = json.loads(r.stdout.strip().splitlines()[-1])
            releves.append(d)
            print(f"  {d['mode']:>6} / {d['famille']:<10} "
                  f"{d['jetons']:>4} jetons en {d['pas']} pas, "
                  f"proposes {d['proposes']}, acceptes {d['acceptes']}, "
                  f"par pas {d['par_pas']}", flush=True)

    print()
    verdict, pires = "aucun mode speculatif mesurable", []
    for mode in {d["mode"] for d in releves} - {"none"}:
        pts = [d for d in releves if d["mode"] == mode]
        m = max(d["par_pas"] or 0 for d in pts)
        pires.append((mode, m))
        print(f"{mode:>6} : taux maximal {m:.3f} sur {len(pts)} familles")
    if pires:
        meilleur, m = max(pires, key=lambda x: x[1])
        if m < SEUIL_ARRET:
            verdict = (f"ARRET : le meilleur taux est {m:.3f} < {SEUIL_ARRET} "
                       f"({meilleur}). Les etapes B et C sont sans objet — les "
                       "trois motifs etaient reels, le gain ne l'est pas.")
        else:
            verdict = (f"POURSUITE : {meilleur} atteint {m:.3f} >= "
                       f"{SEUIL_ARRET}. B (compte de blocs) puis C (energie et "
                       "debit a BQ realiste, graphes declares).")
    print(f"\nVERDICT : {verdict}")
    json.dump({"releves": releves, "verdict": verdict},
              open(a.sortie, "w"), indent=1)
    print(f"TERMINE — releve dans {a.sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
