#!/usr/bin/env python3
"""Courbe du quota : ce que chaque octet supplementaire achete en qualite.

Six budgets entre le plancher tout-nvfp4 et le plafond tout-int8 de
Llama-2-7B. DEUX des six sont des TEMOINS a valeur prevue : le budget du
plancher doit reproduire le dossier tout-nvfp4 (4,7296 bits/poids, PPL
5,6102) et celui du plafond le dossier tout-int8 (8,3452, PPL 5,4142). Si un
temoin ne reproduit pas sa valeur, la campagne est invalide AVANT d'avoir
publie un point — c'est le seul ordre acceptable.

Regles du projet appliquees ici :
  - un test ne doit ABSOLUMENT JAMAIS bloquer le PC : chaque point tourne dans
    un service systemd-run --user borne en memoire, detache, et le disque est
    rendu au cache page entre deux points ;
  - chaque test previent quand il est termine : sentinelle + ligne d'etat ;
  - un point ne se publie que si son propre manifeste confirme le budget
    reellement depense (bloc "budget", ajoute le 10/09) — sans quoi un dossier
    peut porter un budget dans son nom et une autre grandeur dans ses octets ;
  - rien ne part sur la carte sans l'accord des sessions : --pour-de-vrai est
    obligatoire, et sans lui le script imprime le plan et s'arrete.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

P_POIDS = 6_738_417_664          # poids comptes dans le manifeste des etalons
GIO = 1024 ** 3

BASE = Path("/media/anticitoyenlm/2TO_2023_980PRO1/Modeles/models_acvram")
SOURCE = BASE / "Llama-2-7b-hf"
CORPUS = Path("~/Bureau/Claude/acvram-memoire/corpus/wiki-gptq.txt")
RELEVES = Path("~/Bureau/Claude/acvram-memoire/corpus")

# Budget en Gio -> (nom du dossier, valeur prevue si temoin)
POINTS = [
    (3.75, "quota-3g75", {"bpw": 4.7296, "ppl": 5.6102, "role": "temoin plancher"}),
    (4.50, "quota-4g50", None),
    (5.00, "quota-5g00", None),
    (5.50, "quota-5g50", None),
    (6.00, "quota-6g00", None),
    (6.55, "quota-6g55", {"bpw": 8.3452, "ppl": 5.4142, "role": "temoin plafond"}),
]

# Etalon exterieur, mesure par transformers sans rien importer d'acvram.
PPL_REFERENCE = 5.4141

EVAL = ["--corpus", str(CORPUS), "--window", "2048", "--stride", "2048",
        "--max-tokens", "344064", "--min-context", "0", "--json"]


def gio(x: float) -> str:
    return f"{x:.3f} Gio"


def bits(octets: int) -> float:
    return octets * 8 / P_POIDS


def rendre_le_cache(chemin: Path) -> int:
    """posix_fadvise(DONTNEED) sur les fragments : le superviseur a tue deux
    bancs avec 83 Go disponibles parce que le cache page, non le MemFree,
    remplissait le cgroup. Rend les octets effectivement relaches."""
    total = 0
    for f in sorted(chemin.glob("*.safetensors")):
        fd = os.open(f, os.O_RDONLY)
        try:
            taille = os.fstat(fd).st_size
            os.posix_fadvise(fd, 0, taille, os.POSIX_FADV_DONTNEED)
            total += taille
        finally:
            os.close(fd)
    return total


def service(nom: str, argv: list[str], journal: Path, memoire_max: str,
            minutes: int) -> int:
    """Lance argv dans un service utilisateur borne, attend sa fin, rend son code.

    Borne en memoire ET en temps : un point qui derape est tue par le cgroup,
    jamais par le superviseur de la machine."""
    sentinelle = journal.with_suffix(".fini")
    sentinelle.unlink(missing_ok=True)
    cmd = ["systemd-run", "--user", "--unit", nom, "--collect",
           f"--property=MemoryHigh={memoire_max}",
           f"--property=MemoryMax={memoire_max}",
           f"--property=RuntimeMaxSec={minutes * 60}",
           "--property=CPUWeight=60",
           "--setenv=ACVRAM_CUDA_HOME=/usr/local/cuda-13.2",
           f"--setenv=ACVRAM_KERNEL_CACHE={os.environ.get('ACVRAM_KERNEL_CACHE', '')}",
           "/bin/bash", "-c",
           # le code de sortie part dans la sentinelle : le service peut mourir
           # sans que la commande ait parle
           f"{' '.join(map(repr_sh, argv))} > {journal!s} 2>&1; "
           f"echo $? > {sentinelle!s}"]
    subprocess.run(cmd, check=True)
    t0 = time.time()
    dernier = 0.0
    while not sentinelle.exists():
        if time.time() - t0 > minutes * 60 + 120:
            subprocess.run(["systemctl", "--user", "stop", nom], check=False)
            return 124
        if time.time() - dernier > 300:      # un point toutes les 5 min
            dernier = time.time()
            print(f"    [{int(time.time() - t0)} s] {nom} tourne toujours",
                  flush=True)
        time.sleep(5)
    return int(sentinelle.read_text().strip() or "1")


def repr_sh(x: str) -> str:
    return "'" + x.replace("'", "'\\''") + "'"


def lire_budget(dossier: Path) -> dict | None:
    m = json.loads((dossier / "acvram_manifest.json").read_text())
    return m.get("budget")


def octets_safetensors(dossier: Path) -> int:
    return sum(f.stat().st_size for f in dossier.glob("*.safetensors"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pour-de-vrai", action="store_true",
                    help="sans ce drapeau, imprime le plan et s'arrete")
    ap.add_argument("--points", help="sous-ensemble, ex. 4.50,5.00")
    ap.add_argument("--sortie", default=str(RELEVES / "quota"))
    ap.add_argument("--memoire-max", default="40G")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    voulus = None
    if a.points:
        voulus = {float(x) for x in a.points.split(",")}
    points = [p for p in POINTS if voulus is None or p[0] in voulus]

    print("CAMPAGNE QUOTA — Llama-2-7B, "
          f"{len(points)} point(s) sur {len(POINTS)}")
    print(f"  source   {SOURCE}")
    print(f"  corpus   {CORPUS.name} ({CORPUS.stat().st_size} o)")
    print(f"  etalon   reference exterieure PPL {PPL_REFERENCE}")
    print(f"  plancher {gio(3_983_834_740 / GIO)} (tout-nvfp4, "
          f"{bits(3_983_834_740):.4f} b/p)")
    print(f"  plafond  {gio(7_029_266_352 / GIO)} (tout-int8, "
          f"{bits(7_029_266_352):.4f} b/p)")
    for b, nom, temoin in points:
        marque = f"   <- {temoin['role']}, prevu {temoin['bpw']} b/p "\
                 f"et PPL {temoin['ppl']}" if temoin else ""
        print(f"  {b:5.2f} Gio  {nom}{marque}")
    if not a.pour_de_vrai:
        print("\nPLAN SEULEMENT. Rien n'a tourne. "
              "Relancer avec --pour-de-vrai une fois l'accord des sessions obtenu.")
        return 0

    if not SOURCE.exists():
        print(f"ECHEC / CAUSE: source absente {SOURCE}")
        return 2
    sortie = Path(a.sortie)
    sortie.mkdir(parents=True, exist_ok=True)

    # --- Manche zero : la RESOLUTION de l'instrument, avant tout point ---
    # Remarque de claude-0a le 10/09 : l'accord du temoin negatif (tout-int8
    # 5,4142 contre etalon 5,4141, soit 0,002 %) n'a de valeur que si la
    # dispersion entre deux executions identiques est PLUS PETITE que 0,002 %.
    # Sinon le temoin passe par construction et ne prouve rien — c'est ainsi
    # qu'un instrument aveugle a rendu quatre fois 8,825 au millieme le meme
    # jour. Une passe rejouee a l'identique donne ce chiffre pour le prix d'une
    # evaluation, et rien ne se publie avant de l'avoir.
    dispersion = None
    ref_dossier = BASE / "Llama-2-7b-int8"
    if ref_dossier.exists():
        passes = []
        for k in (1, 2):
            j = sortie / f"dispersion-{k}.log"
            code = service(f"acvram-disp-{k}",
                           [a.python, "-m", "acvram", "eval", str(ref_dossier)] + EVAL,
                           j, a.memoire_max, minutes=45)
            if code:
                print(f"ECHEC / CAUSE: manche de dispersion {k}, code {code}")
                return 2
            brut = j.read_text()
            d = brut.find("{")
            r = json.loads(brut[d:]) if d >= 0 else {}
            if isinstance(r, dict) and "models" in r:
                r = r["models"][0]
            passes.append(r.get("perplexity"))
            print(f"  passe {k} : PPL {passes[-1]}", flush=True)
            rendre_le_cache(ref_dossier)
        if None in passes:
            print("ECHEC / CAUSE: une passe de dispersion n'a pas rendu de PPL")
            return 2
        dispersion = abs(passes[0] - passes[1])
        rel = dispersion / passes[0] * 100
        print(f"  RESOLUTION : dispersion {dispersion:.6f} PPL soit {rel:.5f} % "
              f"— l'ecart du temoin negatif vaut {abs(5.4142 - PPL_REFERENCE):.4f} "
              f"({100 * abs(5.4142 - PPL_REFERENCE) / PPL_REFERENCE:.5f} %)")
        if dispersion >= abs(5.4142 - PPL_REFERENCE):
            print("  ATTENTION : la dispersion couvre l'ecart du temoin negatif. "
                  "Son accord a 0,002 % ne prouve donc rien, et le plancher de "
                  "reproduction des temoins est porte a 3x la dispersion.")
    else:
        print("  dispersion NON MESUREE : "
              f"{ref_dossier.name} absent. Les verdicts de temoin seront "
              "rendus avec un seuil pose a priori, ce qui est plus faible.")

    resultats = []
    for b, nom, temoin in points:
        dossier = BASE / f"Llama-2-7b-{nom}"
        print(f"\n=== {nom} — budget {gio(b)}", flush=True)

        if not dossier.exists():
            code = service(
                f"acvram-conv-{nom}",
                [a.python, "-m", "acvram", "convert", str(SOURCE),
                 "--out", str(dossier), "--bits-budget", str(b)],
                sortie / f"conv-{nom}.log", a.memoire_max, minutes=90)
            if code:
                print(f"  ECHEC / CAUSE: conversion code {code}, "
                      f"voir {sortie / f'conv-{nom}.log'}")
                resultats.append({"budget_gib": b, "etat": "conversion echouee",
                                  "code": code})
                continue
        else:
            print("  dossier deja present, conversion sautee")

        bud = lire_budget(dossier)
        oct_reels = octets_safetensors(dossier)
        bpw_reel = bits(oct_reels)
        print(f"  octets {oct_reels:,} = {bpw_reel:.4f} bits/poids".replace(",", " "))
        if bud is None:
            print("  REFUS DE PUBLIER CE POINT : le manifeste ne porte pas de "
                  "bloc 'budget'. Le dossier a ete produit par une version "
                  "d'avant le 10/09 et on ne peut pas savoir ce que le budget "
                  "a reellement achete.")
            resultats.append({"budget_gib": b, "etat": "manifeste sans bloc budget",
                              "bpw": bpw_reel})
            continue
        print(f"  budget : plancher {bud['plancher_gib']} plafond "
              f"{bud['plafond_gib']} depense {bud['depense_gib']} "
              f"restant {bud['restant_mio']} Mio — "
              f"{bud['promus']}/{bud['candidats']} promus")
        if bud.get("sous_le_plancher"):
            print("  POINT SANS OBJET : budget sous le plancher, zero promotion. "
                  "Ce dossier est une conversion de base, pas un point de courbe.")
            resultats.append({"budget_gib": b, "etat": "sous le plancher",
                              "bpw": bpw_reel, "budget": bud})
            continue

        rendre = rendre_le_cache(dossier)
        print(f"  cache page rendu : {rendre / 2**20:.0f} Mio", flush=True)

        jppl = sortie / f"ppl-{nom}.json"
        code = service(
            f"acvram-eval-{nom}",
            [a.python, "-m", "acvram", "eval", str(dossier)] + EVAL,
            sortie / f"eval-{nom}.log", a.memoire_max, minutes=45)
        if code:
            print(f"  ECHEC / CAUSE: eval code {code}, "
                  f"voir {sortie / f'eval-{nom}.log'}")
            resultats.append({"budget_gib": b, "etat": "eval echouee",
                              "code": code, "bpw": bpw_reel, "budget": bud})
            continue
        brut = (sortie / f"eval-{nom}.log").read_text()
        deb = brut.find("{")
        releve = json.loads(brut[deb:]) if deb >= 0 else {}
        if isinstance(releve, dict) and "models" in releve:
            releve = releve["models"][0]
        jppl.write_text(json.dumps(releve, indent=2, ensure_ascii=False))
        ppl = releve.get("perplexity")
        print(f"  PPL {ppl}  (reference exterieure {PPL_REFERENCE}, "
              f"ecart {100 * (ppl / PPL_REFERENCE - 1):+.3f} %)"
              if ppl else "  PPL absente du releve")

        verdict = None
        if temoin and ppl:
            db = abs(bpw_reel - temoin["bpw"])
            dp = abs(ppl - temoin["ppl"])
            # Le seuil de reproduction suit la RESOLUTION mesuree quand on l'a :
            # exiger mieux que ce que l'instrument distingue rend un verdict
            # ininterpretable dans les deux sens.
            seuil_ppl = max(0.005, 3 * dispersion) if dispersion else 0.005
            ok = db < 0.02 and dp < seuil_ppl
            verdict = {"role": temoin["role"], "reproduit": ok,
                       "ecart_bpw": round(db, 4), "ecart_ppl": round(dp, 4),
                       "seuil_ppl": round(seuil_ppl, 6),
                       "dispersion_mesuree": dispersion}
            print(f"  TEMOIN {temoin['role']} : "
                  f"{'reproduit' if ok else 'NE REPRODUIT PAS'} "
                  f"(ecart {db:.4f} b/p, {dp:.4f} PPL)")
            if not ok:
                print("  ARRET : un temoin qui ne reproduit pas invalide la "
                      "campagne. Aucun point ne sera publie avant diagnostic.")

        resultats.append({"budget_gib": b, "dossier": str(dossier),
                          "bpw": round(bpw_reel, 4), "octets": oct_reels,
                          "perplexity": ppl, "budget": bud, "temoin": verdict,
                          "etat": "ok"})
        (sortie / "campagne-quota.json").write_text(
            json.dumps({"points": resultats, "reference": PPL_REFERENCE,
                        "dispersion_ppl": dispersion,
                        "corpus": "wiki-gptq.txt (344 402 jetons, 168 segments) "
                                  "— le MEME que l'etalon exterieur, verifie "
                                  "au jeton : suites identiques sur les 344 064 "
                                  "premiers, tokenizer rapide et lent confondus"},
                       indent=2, ensure_ascii=False))
        if verdict and not verdict["reproduit"]:
            return 3

    print("\n--- COURBE DU QUOTA ---")
    print(f"{'budget':>8}  {'bits/poids':>10}  {'PPL':>8}  {'vs ref':>8}  promus")
    for r in resultats:
        if r.get("etat") != "ok":
            print(f"{r['budget_gib']:8.2f}  {r.get('bpw', 0):10.4f}  "
                  f"{'—':>8}  {'—':>8}  {r['etat']}")
            continue
        p = r["perplexity"]
        print(f"{r['budget_gib']:8.2f}  {r['bpw']:10.4f}  {p:8.4f}  "
              f"{100 * (p / PPL_REFERENCE - 1):+7.3f} %  "
              f"{r['budget']['promus']}/{r['budget']['candidats']}")
    print(f"\nFAIT / TESTE: {sum(1 for r in resultats if r.get('etat') == 'ok')} "
          f"point(s) publiables sur {len(points)} / "
          f"RESTE: releves dans {sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
