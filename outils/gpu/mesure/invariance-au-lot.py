#!/usr/bin/env python3
"""Deux requetes identiques rendent-elles la meme sortie selon leurs voisines de lot ?

FAIT ETABLI PAR LE CODE, pas par une IA : `model.py:87` dimensionne la table de
blocs sur le MAXIMUM du lot —

    n = bucket_blocks(max(t.shape[0] for t in self.block_tables))

— et C = ceil(N x 16 / PA_CHUNK) en decoule. Une requete courte placee dans un
lot contenant une longue recoit donc un C plus grand, l'ordre de reduction
change, et sa sortie peut changer au dernier bit.

CE QUI MANQUE EST L'AMPLITUDE. Une variation au dernier bit qui ne deplace
jamais un argmax n'a pas le meme statut qu'une variation qui en deplace un sur
cent. Ce banc mesure les deux, a deux niveaux.

NIVEAU 1 — LE NOYAU SEUL. Meme q, meme cache, meme seq_len, mais deux tables de
tailles differentes (deux godets). Isole exactement le mecanisme et donne
l'amplitude numerique. Reponse binaire : les sorties sont identiques au bit, ou
elles ne le sont pas.

NIVEAU 2 — LE MOTEUR. La meme invite servie seule, puis accompagnee d'une
requete longue qui pousse le godet vers le haut. Donne l'IMPACT : les jetons
produits changent-ils ?

Les godets etant des puissances de deux depuis 8 blocs de 16 positions, la
non-invariance est PAR GODET et non continue :

    godet     positions    C a PA_CHUNK=512
        8           128           1
       32           512           1
       64          1024           2      <- premier godet ou C > 1
      256          4096           8

A PA_CHUNK = 512, les trois premiers godets donnent tous C = 1 : aucun
decoupage, donc aucune sensibilite attendue. Le premier ecart possible est
entre le godet 32 (C=1) et le godet 64 (C=2).

DOMAINE DE VALIDITE DE CE TEST — ecrit avec lui, parce que mes deux modeles
refutes aujourd'hui etaient justes sur la forme et faux sur leur domaine. Ce
banc ne conclut RIEN si l'une de ces trois conditions manque :

  (a) le moteur est repetable a configuration fixe. serie-determinisme.sh le
      verifie pour des passages IDENTIQUES, mais pas dans ce montage-ci : d'ou
      le bras A2 ci-dessous. Sans lui, un ecart observe pourrait venir du bruit
      d'execution et non du lot, et l'instrument ne saurait pas distinguer sa
      propre instabilite de l'effet cherche — le defaut du temoin a cache nul
      de ce matin, ou A et B rendaient 0 tous les deux.
  (b) le godet CHANGE reellement entre les deux lots. Si les deux lots tombent
      dans le meme godet, N est identique, C est identique, et « rien ne
      bouge » ne prouve rien : le test n'aurait alors pu conclure que dans un
      sens. Le banc AFFICHE les deux N et refuse de conclure s'ils sont egaux.
  (c) la requete courte n'est pas servie depuis le cache de prefixe de la
      longue. Les deux invites sont tirees de graines differentes, donc sans
      prefixe commun.

QUATRE CASES, et elles decident (le reglage se choisit par ACVRAM_PA_CHUNK
quand le noyau le lit, sinon par recompilation) :

    rien ne bouge dans les deux cas   -> propriete theorique sans effet
                                         mesurable : on documente et on passe
    ca bouge avec la tranche adaptative seulement
                                      -> elle a un cout qu'on n'avait pas vu
    ca bouge dans les deux            -> fait ancien du moteur que personne
                                         n'avait nomme, et le gain de la
                                         tranche adaptative n'en est pas
                                         responsable
"""
import argparse
import json
import os
import sys

COURT, LONG = 200, 3000


def niveau1(modele: str) -> dict:
    """Le noyau seul, deux tailles de table pour une meme sequence."""
    import torch
    from acvram import kernels
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams

    charge = load_model(modele, dtype=torch.bfloat16, device_override="cuda:0")
    e = Engine(charge, None, max_batch_size=2, max_model_len=COURT + 64,
               enable_cuda_graphs=False)
    e.add_request(list(range(1, COURT)), SamplingParams(temperature=0.0,
                                                        max_tokens=2))
    e.step()
    cache = charge.model.caches[0]
    dec = [s for s in e.running if not s.finished]
    for s in dec:
        e._grow(s)
    batch = e._build_batch(dec, prefill=False)
    tables, lens = batch.fixed_decode_views(torch.device("cuda:0"))
    hq, hd = charge.spec.num_attention_heads, charge.spec.head_dim
    n_rep = hq // charge.spec.num_key_value_heads
    torch.manual_seed(0)
    q = torch.randn(len(dec), hq, hd, device="cuda:0")

    res = {"n_natif": int(tables.shape[1]), "positions": int(lens.max()),
           "comparaisons": []}
    base = kernels.paged_attention(q, cache, tables, lens, n_rep, hd ** -0.5)
    if base is None:
        raise RuntimeError("noyau pagine indisponible")
    # MEME sequence, MEME longueur, table completee a un godet SUPERIEUR : c'est
    # exactement ce qu'un voisin de lot plus long provoque.
    for facteur in (2, 4, 8):
        n2 = tables.shape[1] * facteur
        t2 = torch.zeros(tables.shape[0], n2, dtype=tables.dtype,
                         device=tables.device)
        t2[:, : tables.shape[1]] = tables
        autre = kernels.paged_attention(q, cache, t2, lens, n_rep, hd ** -0.5)
        d = (autre - base).abs()
        res["comparaisons"].append({
            "n": n2, "facteur": facteur,
            "identique_au_bit": bool(torch.equal(autre, base)),
            "ecart_max": d.max().item(), "ecart_moyen": d.mean().item(),
        })
    return res


def niveau2(modele: str, avec_long: bool) -> dict:
    """Le moteur : la meme invite, seule ou accompagnee d'une longue."""
    import torch
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    import random

    charge = load_model(modele, dtype=torch.bfloat16, device_override="cuda:0")
    n_max = (LONG if avec_long else COURT) + 64
    e = Engine(charge, None, max_batch_size=2, max_model_len=n_max)
    e._eos = set()
    inv = [random.Random(7).randrange(10, 150000) for _ in range(COURT)]
    s_court = e.add_request(inv, SamplingParams(temperature=0.0, max_tokens=32))
    if avec_long:
        inv_long = [random.Random(99).randrange(10, 150000) for _ in range(LONG)]
        e.add_request(inv_long, SamplingParams(temperature=0.0, max_tokens=32))
    while not s_court.finished and len(s_court.output_ids) < 32:
        e.step()
    # LE N EFFECTIF, la grandeur qui decide : c'est lui que le lot deplace.
    dec = [s for s in e.running if not s.finished]
    n_blocs = None
    if dec:
        b = e._build_batch(dec, prefill=False)
        n_blocs = int(b.fixed_decode_views(torch.device("cuda:0"))[0].shape[1])
    return {"avec_long": avec_long, "jetons": list(s_court.output_ids),
            "n_jetons": len(s_court.output_ids), "n_blocs": n_blocs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", required=True)
    ap.add_argument("--niveau", type=int, choices=(1, 2))
    ap.add_argument("--avec-long", action="store_true")
    ap.add_argument("--sortie", default="/tmp/invariance-au-lot.json")
    a = ap.parse_args()
    if a.niveau == 1:
        print(json.dumps(niveau1(a.modele)))
        return 0
    if a.niveau == 2:
        print(json.dumps(niveau2(a.modele, a.avec_long)))
        return 0

    import subprocess
    def lance(*args):
        r = subprocess.run([sys.executable, __file__, "--modele", a.modele,
                            *args], capture_output=True, text=True,
                           timeout=3600)
        if r.returncode != 0:
            print(r.stderr[-1500:], file=sys.stderr)
            sys.exit(1)
        return json.loads(r.stdout.strip().splitlines()[-1])

    print("NIVEAU 1 — le noyau seul, table completee a un godet superieur")
    n1 = lance("--niveau", "1")
    print(f"  table native {n1['n_natif']} blocs, {n1['positions']} positions")
    for c in n1["comparaisons"]:
        print(f"    n={c['n']:>5} (x{c['facteur']}) : "
              f"{'IDENTIQUE au bit' if c['identique_au_bit'] else 'DIFFERENT'}"
              f"   ecart max {c['ecart_max']:.3e}  moyen {c['ecart_moyen']:.3e}")

    print("\nNIVEAU 2 — le moteur, meme invite seule puis accompagnee")
    seul = lance("--niveau", "2")
    seul2 = lance("--niveau", "2")          # A2 : temoin de repetabilite
    if seul["jetons"] != seul2["jetons"]:
        print("ARRET : la meme invite servie DEUX FOIS SEULE ne rend pas la "
              "meme sortie. Le montage n'est pas repetable, donc un ecart "
              "avec le lot ne serait pas attribuable au lot. Rien n'est "
              "conclu.", file=sys.stderr)
        return 3
    print("  temoin de repetabilite : deux passages seuls identiques")
    accompagne = lance("--niveau", "2", "--avec-long")
    if accompagne.get("n_blocs") == seul.get("n_blocs"):
        print(f"ARRET : les deux lots donnent le MEME godet "
              f"(N = {seul.get('n_blocs')} blocs). Le test n'a pas franchi la "
              "frontiere qu'il devait franchir : « rien ne bouge » ne "
              "prouverait rien. Augmentez LONG.", file=sys.stderr)
        return 4
    print(f"  godets franchis : N = {seul.get('n_blocs')} blocs seule, "
          f"{accompagne.get('n_blocs')} accompagnee")
    memes = seul["jetons"] == accompagne["jetons"]
    prem = next((i for i, (x, y) in enumerate(
        zip(seul["jetons"], accompagne["jetons"])) if x != y), None)
    print(f"  {seul['n_jetons']} jetons seule, {accompagne['n_jetons']} "
          f"accompagnee : {'IDENTIQUES' if memes else 'DIFFERENTS'}")
    if not memes:
        print(f"  premier ecart au jeton {prem}")
        print(f"    seule       {seul['jetons'][:12]}")
        print(f"    accompagnee {accompagne['jetons'][:12]}")

    verdict = ("aucun effet mesurable, ET le test POUVAIT conclure "
               "(repetabilite verifiee, godets franchis) : propriete "
               "theorique, a documenter"
               if all(c["identique_au_bit"] for c in n1["comparaisons"])
               and memes else
               "effet mesure : la sortie depend des voisines de lot")
    print(f"\nVERDICT : {verdict}")
    json.dump({"niveau1": n1, "seul": seul, "seul2": seul2,
               "accompagne": accompagne,
               "jetons_identiques": memes, "verdict": verdict},
              open(a.sortie, "w"), indent=1)
    print(f"TERMINE — releve dans {a.sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
