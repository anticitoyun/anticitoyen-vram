#!/usr/bin/env python3
"""Part du pas prise par `paged_attn_partial`, mesuree SANS profileur.

Tranche « 26,09 Gio / 28,14 ms = 995 Go/s effectifs » (ETABLI.md:2476) en
mesurant p, la part du pas occupee par des noyaux qui ne lisent pas de poids.

    A   l'attention reelle
    B   rien n'est lu : le plancher du montage (lancement + ecritures)
    C   le KV est lu selon le MEME parcours, sans le softmax ni les produits

    C - B   lecture du KV seule      PREDICTION ECRITE D'AVANCE : 0,097 ms
    A - C   le calcul de l'attention
    A - B   p, dont le biais L2 est mesure au lieu d'etre suppose

SEUIL INSCRIT AVANT LA MESURE — la bande passante exigee des GEMM vaut
26,06 Gio / (28,14 ms x (1 - p)) et franchit la borne de la carte (1050 Go/s)
des que p > 5,24 % du temps GPU, soit 1,475 ms par pas, soit 5,13 % du mural.
Trois issues, aucune ne se lit « le test n'a rien donne » :

    p ~ 0,35 %          l'attention est memoire-bornee : la conclusion TIENT
    0,35 % < p < 5,24 % elle tient encore, mais l'attention est deja hors
                        bande passante — le dossier gagne une cible
    p > 5,24 %          elle tombe par contradiction, et l'attention est a
                        15x son plancher d'octets

Les bras B et C rendent une sortie FAUSSE. Le moteur est construit sans
tokenizer : il n'y a aucun texte a publier, par construction et non par garde.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

# Conditions du tableau conteste (ETABLI.md:1527, :2278).
CTX, GENERES = 1024, 200
SEUIL_PCT_GPU = 5.24
T_GPU_MS, T_MURAL_MS = 28.140, 28.748
PREDICTION_C_MOINS_B_MS = 0.097
ORDRE = ["A", "B", "C", "C", "B", "A"]      # ABBA : la derive thermique biaise
                                            # l'ordre, alterner ne corrige rien


def mesurer(bras: str, modele: str) -> dict:
    os.environ["ACVRAM_PA_ARM"] = bras
    if bras != "A":
        os.environ["ACVRAM_PA_ARM_SORTIE_FAUSSE"] = "1"
    import torch
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams

    charge = load_model(modele, dtype=torch.bfloat16)
    moteur = Engine(charge, None, max_batch_size=1, max_model_len=CTX + 32)
    # Sans ignore_eos, une fin de sequence au premier jeton rendrait un pas
    # mesure sur un seul passage. On publie le compte REELLEMENT execute.
    moteur._eos = set()
    invite = [random.Random(1234).randrange(10, 150000)
              for _ in range(CTX - GENERES)]      # ni jeton constant (cache de
                                                  # prefixe) ni hash() (sale)
    seq = moteur.add_request(invite, SamplingParams(temperature=0.0,
                                                    max_tokens=GENERES))
    pas = []
    while not seq.finished and len(pas) < GENERES:
        t = time.perf_counter()
        moteur.step()
        pas.append((time.perf_counter() - t) * 1000.0)
    ordonnes = sorted(pas[1:]) or sorted(pas)     # le premier passage est froid
    n = len(ordonnes)
    return {"bras": bras, "pas_executes": len(pas), "n_retenus": n,
            "median_ms": ordonnes[n // 2],
            "p10_ms": ordonnes[max(0, int(0.10 * n))],
            "p90_ms": ordonnes[min(n - 1, int(0.90 * n))],
            "premier_ms": pas[0]}


def pilote(modele: str, sortie: str) -> int:
    # Le changement qui doit casser : un bras inconnu leve, donc le .so charge
    # contient bien ce code. Sans ce controle, une mesure sur l'ancien binaire
    # rendrait trois bras identiques — et « aucun effet » se lirait comme un
    # resultat.
    env = dict(os.environ, ACVRAM_PA_ARM="9", ACVRAM_PA_ARM_SORTIE_FAUSSE="1")
    t = subprocess.run([sys.executable, __file__, "--bras", "9",
                        "--modele", modele], env=env, capture_output=True,
                       text=True, timeout=600)
    if t.returncode == 0:
        print("ARRET : ACVRAM_PA_ARM=9 n'a pas leve — le binaire charge est "
              "l'ancien. Rien n'est mesure.", file=sys.stderr)
        return 2
    print("controle du binaire : le bras inconnu leve, le .so est a jour")

    releves = []
    for i, bras in enumerate(ORDRE, 1):
        env = dict(os.environ, ACVRAM_PA_ARM=bras,
                   ACVRAM_PA_ARM_SORTIE_FAUSSE="1")
        t0 = time.time()
        r = subprocess.run([sys.executable, __file__, "--bras", bras,
                            "--modele", modele, "--json"], env=env,
                           capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            print(f"ECHEC bras {bras} :\n{r.stderr[-2000:]}", file=sys.stderr)
            return 1
        d = json.loads(r.stdout.strip().splitlines()[-1])
        releves.append(d)
        print(f"  {i}/{len(ORDRE)} bras {bras} : {d['median_ms']:7.3f} ms "
              f"[{d['p10_ms']:.3f} – {d['p90_ms']:.3f}] sur {d['n_retenus']} "
              f"pas, {time.time() - t0:.0f} s")

    med = {b: sorted(d["median_ms"] for d in releves if d["bras"] == b)
           for b in "ABC"}
    a, bb, c = (m[len(m) // 2] for m in (med["A"], med["B"], med["C"]))
    p_ms, kv_ms, calc_ms = a - bb, c - bb, a - c
    lignes = [
        "",
        f"A (reel)   {a:7.3f} ms     B (plancher) {bb:7.3f} ms"
        f"     C (KV lu) {c:7.3f} ms",
        f"  ecart entre les deux occurrences de chaque bras : "
        f"A {abs(med['A'][0] - med['A'][-1]):.3f}  "
        f"B {abs(med['B'][0] - med['B'][-1]):.3f}  "
        f"C {abs(med['C'][0] - med['C'][-1]):.3f} ms",
        "",
        f"C - B  lecture du KV      {kv_ms:7.3f} ms   "
        f"predit {PREDICTION_C_MOINS_B_MS:.3f} ms   "
        f"ecart x{kv_ms / PREDICTION_C_MOINS_B_MS:.2f}"
        if kv_ms > 0 else f"C - B  {kv_ms:7.3f} ms  NEGATIF : montage invalide",
        f"A - C  calcul de l'attention {calc_ms:7.3f} ms",
        f"A - B  p                  {p_ms:7.3f} ms = "
        f"{100 * p_ms / T_GPU_MS:.3f} % du GPU (28,140 ms) = "
        f"{100 * p_ms / T_MURAL_MS:.3f} % du mural (28,748 ms)",
        "",
        f"seuil inscrit d'avance : {SEUIL_PCT_GPU} % du GPU = "
        f"{SEUIL_PCT_GPU * T_GPU_MS / 100:.3f} ms",
    ]
    pct = 100 * p_ms / T_GPU_MS
    if pct > SEUIL_PCT_GPU:
        lignes.append("ISSUE 3 : la conclusion d'ETABLI.md:2476 TOMBE par "
                      "contradiction — les GEMM exigeraient plus que 1050 Go/s.")
    elif pct > 0.35:
        lignes.append("ISSUE 2 : la conclusion tient arithmetiquement, MAIS "
                      "l'attention est deja hors bande passante — une cible.")
    else:
        lignes.append("ISSUE 1 : l'attention est memoire-bornee, la conclusion "
                      "TIENT et la ligne sort de « suspecte ».")
    lignes.append(f"controle du montage : A median {a:.3f} ms contre "
                  f"{T_MURAL_MS} ms attendus au banc "
                  f"(ecart {100 * (a - T_MURAL_MS) / T_MURAL_MS:+.1f} %) — "
                  "au-dela de quelques pourcents, c'est le montage qu'il faut "
                  "corriger, pas la part.")
    texte = "\n".join(lignes)
    print(texte)
    with open(sortie, "w") as f:
        json.dump({"releves": releves, "p_ms": p_ms, "kv_ms": kv_ms,
                   "calc_ms": calc_ms, "rapport": texte}, f, indent=1)
    print(f"\nTERMINE — releve dans {sortie}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modele", required=True)
    ap.add_argument("--bras", choices=["A", "B", "C", "9"])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--sortie", default="/tmp/part-attention-trois-bras.json")
    a = ap.parse_args()
    if a.bras:
        d = mesurer(a.bras, a.modele)
        print(json.dumps(d) if a.json else d)
        sys.exit(0)
    sys.exit(pilote(a.modele, a.sortie))
