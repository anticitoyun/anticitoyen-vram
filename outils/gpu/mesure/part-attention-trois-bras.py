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
CTX, GENERES, BQ = 1024, 200, 1          # surcharges par --ctx / --generes / --bq
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
    # LE PLAN EST-IL CELUI DU TABLEAU ? La campagne d'ETABLI.md declare « 0
    # couche exilee ». Si une autre session tient de la VRAM au moment du
    # chargement, le plan exile — et l'exil d'une seule couche coute 70 % du
    # debit. Six chargements seraient perdus sur un `p` qui ne porte pas sur
    # le meme moteur. On refuse AVANT de mesurer, pas apres.
    exiles = [c.index for c in charge.plan.layers
              if "cpu" in (c.attn_storage, c.mlp_storage) or c.mlp_exec == "cpu"]
    if exiles:
        print(f"MANCHE SANS OBJET : {len(exiles)} couche(s) exilee(s) sur "
              f"{len(charge.plan.layers)} (indices {exiles[:8]}"
              f"{'...' if len(exiles) > 8 else ''}). Le tableau d'ETABLI.md "
              "porte sur 0 couche exilee, et l'exil d'une seule couche coute "
              "70 % du debit : ce chiffre ne mesure pas ce qu'on croit et ne "
              "doit pas pouvoir etre publie par megarde.", file=sys.stderr)
        sys.exit(2)
    moteur = Engine(charge, None, max_batch_size=max(1, BQ),
                    max_model_len=CTX + 32)
    # LES GRAPHES SONT-ILS ACTIFS ? Un pas sans graphe paie ses lancements un
    # par un : il ne mesure pas le meme moteur. Meme garde que l'exil.
    graphes = moteur.graphs is not None and getattr(moteur.graphs, "enabled",
                                                    False)
    if not graphes:
        print("MANCHE SANS OBJET : les graphes CUDA sont inactifs. Un pas sans "
              "graphe paie ses lancements un par un et ne mesure pas le moteur "
              "du tableau.", file=sys.stderr)
        sys.exit(2)
    # Sans ignore_eos, une fin de sequence au premier jeton rendrait un pas
    # mesure sur un seul passage. On publie le compte REELLEMENT execute.
    moteur._eos = set()
    # ni jeton constant (cache de prefixe) ni hash() (sale d'un processus a
    # l'autre) : graine fixe, invite differente par sequence.
    # LE LOT. La grille du noyau est (BQ, HQ, C) : BQ=1 est un cas
    # PARTICULIER, et le service reel tourne a douze sequences (x2,84 en
    # debit mesure). Une invite differente par sequence, sinon le cache de
    # prefixe les fusionnerait et BQ ne serait qu'une apparence.
    seqs = []
    for i in range(BQ):
        inv = [random.Random(1234 + i).randrange(10, 150000)
               for _ in range(CTX - GENERES)]
        seqs.append(moteur.add_request(
            inv, SamplingParams(temperature=0.0, max_tokens=GENERES)))
    seq = seqs[0]
    pas = []
    while any(not s.finished for s in seqs) and len(pas) < GENERES:
        t = time.perf_counter()
        moteur.step()
        pas.append((time.perf_counter() - t) * 1000.0)
    # TEMOIN QUE LE BRAS A REELLEMENT PRIS, propre a CETTE mesure. Les bras B
    # et C amputent l'attention : les jetons produits DOIVENT differer de ceux
    # du bras A. Une empreinte identique signifie que l'interrupteur n'a pas
    # agi — et « aucun effet » se lirait comme un resultat. Le controle
    # ACVRAM_PA_ARM=9 prouve que le code est dans le binaire ; celui-ci prouve
    # qu'il a agi pendant ce passage-ci.
    import hashlib as _h
    empreinte_jetons = _h.sha256(
        ",".join(map(str, seq.output_ids)).encode()).hexdigest()[:12]
    ordonnes = sorted(pas[1:]) or sorted(pas)     # le premier passage est froid
    n = len(ordonnes)
    import hashlib
    import re as _re
    from acvram.kernels import get_extension
    src = os.path.join(os.path.dirname(
        os.path.abspath(sys.modules["acvram.kernels"].__file__)),
        "acvram_kernels.cu")
    _src = open(src).read()
    # UNE TRANCHE ADAPTATIVE NE DOIT PAS PASSER POUR CELLE DU TABLEAU. Si le
    # source lit la tranche dans l'environnement, la valeur doit etre POSEE :
    # sinon un bras A « par defaut » mesurerait le decoupage adaptatif en
    # croyant mesurer le moteur d'ETABLI.md. Poser ACVRAM_PA_CHUNK sur un
    # binaire qui l'ignore serait la faute symetrique — une fausse assurance —
    # donc on verifie le SOURCE, pas la variable.
    if "ACVRAM_PA_CHUNK" in _src and not os.environ.get("ACVRAM_PA_CHUNK"):
        raise RuntimeError(
            "ce noyau lit la tranche dans l'environnement et ACVRAM_PA_CHUNK "
            "n'est pas posee : le bras A porterait sur le decoupage adaptatif "
            "et non sur celui du tableau. Posez ACVRAM_PA_CHUNK=512.")
    chunk = os.environ.get("ACVRAM_PA_CHUNK") or _re.search(
        r"PA_CHUNK\s*=\s*(\d+)", _src).group(1)
    so = getattr(get_extension(), "__file__", "") or ""
    emp = (hashlib.sha256(open(so, "rb").read()).hexdigest()[:12]
           if so and os.path.exists(so) else "inconnu")
    # L'ANNONCE DU PLAN, relevee a cote de chaque bras. Laurine a mesure
    # qu'elle sous-provisionne de 1,39 Gio sur un modele quadratique, et Laure
    # qu'un plan bascule sur 123 Mio : l'annonce se trompe de onze fois la
    # marge qui decide. Si un chargement exile ou desactive les graphes, ce
    # releve dira que ce n'est pas la mesure qui est en cause.
    _annonce = getattr(charge.plan, "est_decode_tok_s", None)
    return {"bras": bras, "pa_chunk": int(chunk), "so": so, "empreinte_so": emp,
            "graphes": graphes, "captures": getattr(moteur.graphs, "captures", -1),
            "plan_est_decode_tok_s": _annonce,
            "pa_warps": os.environ.get("ACVRAM_PA_WARPS", "defaut"),
            "couches": len(charge.plan.layers), "exilees": 0,
            "jetons": empreinte_jetons, "n_jetons": len(seq.output_ids),
            "bq": BQ, "seqs_finies": sum(1 for s in seqs if s.finished),
            "pas_executes": len(pas), "n_retenus": n,
            "median_ms": ordonnes[n // 2],
            "p10_ms": ordonnes[max(0, int(0.10 * n))],
            "p90_ms": ordonnes[min(n - 1, int(0.90 * n))],
            "premier_ms": pas[0]}


def rendre_le_cache(modele: str) -> int:
    """Rend au systeme le page cache des poids qu'on vient de lire.

    Lire 27,5 Gio remplit le page cache ; MemAvailable reste haut mais MemFree
    s'effondre, et le superviseur tue sur MemFree. Six bras enchaines, c'est
    six fois la meme lecture. On ne contourne pas le garde-fou : on retire la
    pression qui le declenche.
    """
    rendus = 0
    for racine, _, fichiers in os.walk(os.path.realpath(modele)):
        for f in fichiers:
            if not f.endswith((".safetensors", ".bin", ".gguf")):
                continue
            chemin = os.path.join(racine, f)
            try:
                fd = os.open(chemin, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    rendus += 1
                finally:
                    os.close(fd)
            except OSError:
                pass
    return rendus


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
                            "--modele", modele, "--json",
                            "--ctx", str(CTX), "--generes", str(GENERES),
                            "--bq", str(BQ)],
                           env=env,
                           capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            print(f"ECHEC bras {bras} :\n{r.stderr[-2000:]}", file=sys.stderr)
            return 1
        d = json.loads(r.stdout.strip().splitlines()[-1])
        releves.append(d)
        n_rendus = rendre_le_cache(modele)      # avant le bras suivant
        print(f"  {i}/{len(ORDRE)} bras {bras} : {d['median_ms']:7.3f} ms "
              f"[{d['p10_ms']:.3f} – {d['p90_ms']:.3f}] sur {d['n_retenus']} "
              f"pas, {time.time() - t0:.0f} s, "
              f"{n_rendus} fichiers rendus au cache")

    bras_vus = sorted({d["bras"] for d in releves})
    emp = {b: [d["jetons"] for d in releves if d["bras"] == b] for b in bras_vus}
    if len({e[0] for e in emp.values()}) != len(bras_vus):
        print("ARRET : les trois bras ne produisent pas trois sorties "
              f"distinctes ({emp}) — l'interrupteur n'a pas agi sur au moins "
              "un bras, et « aucun effet » se lirait comme un resultat. "
              "Rien n'est calcule.", file=sys.stderr)
        return 3
    for b, e in emp.items():
        if len(set(e)) != 1:
            print(f"ARRET : le bras {b} n'a pas produit deux fois la meme "
                  f"sortie ({e}) — le moteur n'est pas deterministe a "
                  "temperature 0, la comparaison des medianes ne tient pas.",
                  file=sys.stderr)
            return 4
    print("temoin : trois sorties distinctes, chaque bras reproductible")
    med = {b: sorted(d["median_ms"] for d in releves if d["bras"] == b)
           for b in bras_vus}
    a, bb = (med["A"][len(med["A"]) // 2], med["B"][len(med["B"]) // 2])
    if "C" in med:
        c = med["C"][len(med["C"]) // 2]
    else:                       # campagne qui ne cherche que p : C non mesure
        c = float("nan")
    p_ms, kv_ms, calc_ms = a - bb, c - bb, a - c
    lignes = [
        "",
        f"A (reel)   {a:7.3f} ms     B (plancher) {bb:7.3f} ms"
        f"     C (KV lu) {c:7.3f} ms",
        "  ecart entre les occurrences de chaque bras : " + "  ".join(
            f"{b} {abs(med[b][0] - med[b][-1]):.3f}" for b in bras_vus) + " ms",
        "",
        (f"C - B  lecture du KV      {kv_ms:7.3f} ms   "
        f"predit {PREDICTION_C_MOINS_B_MS:.3f} ms   "
        f"ecart x{kv_ms / PREDICTION_C_MOINS_B_MS:.2f}"
         if kv_ms > 0 else f"C - B  {kv_ms:7.3f} ms  NEGATIF : montage invalide")
        if "C" in med else "C - B  non mesure (bras C absent de --ordre)",
        f"A - C  calcul de l'attention {calc_ms:7.3f} ms"
        if "C" in med else "",
        # LE DENOMINATEUR EST CELUI QU'ON MESURE. Diviser par les 28,140 ms
        # du tableau etait un denominateur EMPRUNTE : mon pas valait 33,286.
        f"A - B  p                  {p_ms:7.3f} ms = "
        f"{100 * p_ms / a:.3f} % de MON pas ({a:.3f} ms)",
        f"       pour comparaison, et sous l'hypothese NON VERIFIEE que le "
        f"noyau coute pareil dans le moteur du tableau : "
        f"{100 * p_ms / T_GPU_MS:.3f} % de ses 28,140 ms",
        "",
        f"seuil inscrit d'avance : {SEUIL_PCT_GPU} % du GPU = "
        f"{SEUIL_PCT_GPU * T_GPU_MS / 100:.3f} ms",
    ]
    pct = 100 * p_ms / a          # sur le pas mesure, pas sur celui du tableau
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
    ap.add_argument("--ctx", type=int, default=CTX)
    ap.add_argument("--bq", type=int, default=1,
                    help="sequences concurrentes ; la grille est (BQ, HQ, C) "
                         "et BQ=1 est un cas particulier")
    ap.add_argument("--generes", type=int, default=GENERES)
    ap.add_argument("--ordre", default="A,B,C,C,B,A",
                    help="bras a mesurer, dans l'ordre ; A,B,B,A suffit pour "
                         "une campagne qui ne cherche que p")
    a = ap.parse_args()
    CTX, GENERES, BQ = a.ctx, a.generes, a.bq
    ORDRE = [b.strip() for b in a.ordre.split(",")]
    if CTX <= GENERES:
        ap.error(f"--ctx {CTX} doit depasser --generes {GENERES} : l'invite "
                 "serait vide ou negative")
    globals()["CTX"], globals()["GENERES"], globals()["BQ"] = CTX, GENERES, BQ
    globals()["ORDRE"] = ORDRE
    if a.bras:
        d = mesurer(a.bras, a.modele)
        print(json.dumps(d) if a.json else d)
        sys.exit(0)
    sys.exit(pilote(a.modele, a.sortie))
