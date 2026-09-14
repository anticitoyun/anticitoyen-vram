#!/usr/bin/env python3
"""Protocole A/B : KLD couche-par-couche contre SNR-bloc, budget d'octets EGAL.

Suite de revue/protocole-ab-kld-vs-snr.md (poste2, 13/09/2026), qui pose la
prediction et le seuil AVANT toute mesure — a lire avant de lancer.

Deux bras, meme budget demande (`--bits-budget`, PAS le budget reellement
depense : le sac a dos de chaque bras choisit independamment jusqu'a ce
budget) :

  bras A  ordre=snr   ACVRAM_ORDRE_SAC absent (defaut du convertisseur)
  bras B  ordre=kld   ACVRAM_ORDRE_SAC=kld  --mesurer-kld

JUMELLES (regle du 10/09, claude-f2) : chaque bras est EVALUE deux fois. La
conversion elle-meme est deterministe (memes poids, meme code, aucun alea) ;
les deux exemplaires verifient la stabilite du banc PPL — qui charge le
modele sur la carte et peut varier d'une passe a l'autre — pas la
conversion.

Verdict, selon le seuil pose dans le protocole (0,01 PPL = bruit de
mesure documente par courbe-du-quota-deux-points.md) :

  |PPL(A) - PPL(B)| < 0,01           -> INDECIDABLE, ni egalite ni gagnant
  |PPL(A) - PPL(B)| >= 0,01          -> le bras au PPL le plus bas est nomme

Regles du projet appliquees ici (memes que outils/campagne-quota.py) :
  - un test ne doit ABSOLUMENT JAMAIS bloquer le PC : chaque manche tourne
    dans un service systemd-run --user borne en memoire, detache, et le
    disque est rendu au cache page entre deux manches ;
  - rien ne part sur la carte sans --pour-de-vrai : sans lui, le plan
    s'imprime et le script s'arrete.

Ecarts DELIBERES par rapport a campagne-quota.py :
  - `ACVRAM_ORDRE_SAC` est transmis au service via `--setenv`, jamais lu
    dans l'environnement ambiant du pilote. campagne-quota.py lit la
    variable cote pilote pour NOMMER le dossier, mais ne la transmet nulle
    part au service systemd-run qui fait la conversion — elle n'atteint le
    processus enfant que si le gestionnaire systemd importe deja
    l'environnement du pilote, ce qui n'est pas garanti. Ici la variable
    est explicite, dans les deux sens : jamais devinee.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import sys as _s, pathlib as _p  # noqa: E401
_s.path.insert(0, str(_p.Path(__file__).resolve().parent.parent))
from outils.racine_modeles import MODELES  # noqa: E402
from outils._chemins import sorties  # noqa: E402

GIO = 1024 ** 3
CARTE = Path(__file__).resolve().parent / "carte.sh"
BASE = Path(MODELES)
SOURCE = BASE / "Llama-2-7b-hf"

# MEME CORPUS QUE campagne-quota.py. Deux campagnes qui ne lisent pas le
# meme fichier ne sont pas sur la meme courbe, et ce defaut ne se voit qu'a
# la fin (regle posee par f2 le 10/09). Duplique ici plutot qu'importe :
# campagne-quota.py est un script (le tiret dans son nom interdit `import`),
# pas un module.
CORPUS = Path("/mnt/4TO_SATACMR_2022/Modeles/corpus/wiki-gptq.txt")
CORPUS_SHA = "e52922746ad09bac73b0dba32b2987c0d7924da14337dcd43c1d9113a9f6d0ae"
PPL_REFERENCE = 5.4141
RELEVES = sorties()

# Memes bornes que outils/campagne-quota.py (plancher/plafond mesures,
# tout-nvfp4 / tout-int8 de Llama-2-7B). Le budget par defaut de CE
# protocole est le meme point "serre" que tests/test_kld_couche.py::
# test_ordre_sac_kld_publie_son_nom_et_change_les_promotions : a 40 % de
# l'intervalle plancher-plafond, assez etroit pour qu'un ordre different
# change reellement l'ensemble des promus (a budget large, tous les
# candidats passent et aucun tri ne se distingue d'un autre).
PLANCHER_OCTETS = 3_983_834_740
PLAFOND_OCTETS = 7_029_266_352
PLANCHER_GIO = PLANCHER_OCTETS / GIO
PLAFOND_GIO = PLAFOND_OCTETS / GIO
BUDGET_SERRE_DEFAUT = round(PLANCHER_GIO + (PLAFOND_GIO - PLANCHER_GIO) * 0.4, 4)

EVAL = ["--corpus", str(CORPUS), "--window", "2048", "--stride", "2048",
        "--max-tokens", "344064", "--min-context", "0", "--json"]

# Seuil de decision pose dans revue/protocole-ab-kld-vs-snr.md, AVANT
# mesure. Ne se change pas apres coup pour faire gagner un bras.
SEUIL_DECISION_PPL = 0.01

BRAS = [
    {"nom": "snr", "ordre_sac": None, "mesurer_kld": False,
     "ordre_glouton_attendu": "snr_par_octet_decroissant"},
    {"nom": "kld", "ordre_sac": "kld", "mesurer_kld": True,
     "ordre_glouton_attendu": "kld_couche_par_octet_decroissant"},
]


def gio(x: float) -> str:
    return f"{x:.4f} Gio"


def bits(octets: int, p_poids: int) -> float:
    return octets * 8 / p_poids


def mon_unite() -> str:
    """L'unite systemd qui contient CE processus, ou une chaine vide.

    Duplique de campagne-quota.py : sans elle, le garde des orphelins
    compterait le pilote lui-meme parmi les services vivants et refuserait
    de demarrer."""
    try:
        for ligne in open("/proc/self/cgroup", encoding="utf-8"):
            morceaux = [m for m in ligne.strip().split("/")
                        if m.endswith(".service")]
            if morceaux:
                return morceaux[-1]
    except OSError:
        pass
    return ""


def rendre_le_cache(chemin: Path) -> int:
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


def repr_sh(x: str) -> str:
    return "'" + x.replace("'", "'\\''") + "'"


def service(nom: str, argv: list[str], journal: Path, memoire_max: str,
            minutes: int, ordre_sac: str | None = None) -> int:
    """Lance argv dans un service utilisateur borne, attend sa fin, rend son
    code. Duplique de campagne-quota.py, avec `ordre_sac` transmis
    explicitement (voir l'ecart deliberee dans le docstring du module)."""
    sentinelle = journal.with_suffix(".fini")
    sentinelle.unlink(missing_ok=True)
    racine = Path(__file__).resolve().parent.parent
    cmd = ["systemd-run", "--user", "--unit", nom, "--collect",
           f"--property=WorkingDirectory={racine}",
           f"--setenv=PYTHONPATH={racine}",
           f"--property=MemoryHigh={memoire_max}",
           f"--property=MemoryMax={memoire_max}",
           f"--property=RuntimeMaxSec={minutes * 60}",
           "--property=CPUWeight=60",
           "--setenv=ACVRAM_CUDA_HOME=/usr/local/cuda-13.2",
           f"--setenv=ACVRAM_KERNEL_CACHE={os.environ.get('ACVRAM_KERNEL_CACHE', '')}"]
    if ordre_sac:
        cmd.append(f"--setenv=ACVRAM_ORDRE_SAC={ordre_sac}")
    cmd += ["/bin/bash", "-c",
           f"( while kill -0 {os.getpid()} 2>/dev/null; do sleep 5; done; "
           f"  echo 'PILOTE MORT : ce bras s arrete, il tenait la carte pour "
           f"rien' >> {journal!s}; kill -TERM 0 ) & "
           f"{repr_sh(str(CARTE))} {' '.join(map(repr_sh, argv))} "
           f"> {journal!s} 2>&1; echo $? > {sentinelle!s}"]
    subprocess.run(cmd, check=True)
    t0 = time.time()
    dernier = 0.0
    while not sentinelle.exists():
        if time.time() - t0 > minutes * 60 + 120:
            subprocess.run(["systemctl", "--user", "stop", nom], check=False)
            return 124
        if time.time() - dernier > 300:
            dernier = time.time()
            print(f"    [{int(time.time() - t0)} s] {nom} tourne toujours",
                  flush=True)
        time.sleep(5)
    return int(sentinelle.read_text().strip() or "1")


def releve_du_journal(journal: Path) -> dict | None:
    """Duplique de campagne-quota.py : le journal d'`acvram eval` contient
    d'abord le plan (un objet JSON), puis la progression, puis le releve."""
    t = journal.read_text(encoding="utf-8", errors="replace")
    dec = json.JSONDecoder()
    i = 0
    dernier = None
    while True:
        d = t.find("{", i)
        if d < 0:
            break
        try:
            objet, fin = dec.raw_decode(t, d)
            i = fin
        except json.JSONDecodeError:
            i = d + 1
            continue
        for c in (objet.get("models", [objet]) if isinstance(objet, dict) else []):
            if isinstance(c, dict) and "perplexity" in c:
                dernier = c
    return dernier


def verifier_que_c_est_notre_code(r: dict, journal: Path) -> str | None:
    manquants = [k for k in ("corpus_sha256", "nbytes_detail",
                             "bits_par_poids_en_memoire") if not r.get(k)]
    if manquants:
        return (f"le releve de {journal.name} ne porte pas {manquants} : ce "
                f"n'est pas le code de ce worktree qui a tourne. Aucun "
                f"chiffre de cette passe ne vaut.")
    if r["corpus_sha256"] != CORPUS_SHA[:24]:
        return (f"le releve porte le corpus {r['corpus_sha256']} au lieu de "
                f"{CORPUS_SHA[:24]}")
    return None


def lire_budget(dossier: Path) -> dict | None:
    m = json.loads((dossier / "acvram_manifest.json").read_text())
    return m.get("budget")


def octets_safetensors(dossier: Path) -> int:
    return sum(f.stat().st_size for f in dossier.glob("*.safetensors"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pour-de-vrai", action="store_true",
                    help="sans ce drapeau, imprime le plan et s'arrete")
    ap.add_argument("--budget-gib", type=float, default=BUDGET_SERRE_DEFAUT,
                    help=f"budget DEMANDE, EGAL pour les deux bras "
                         f"(defaut {BUDGET_SERRE_DEFAUT:.4f}, point serre a "
                         f"40 %% de l'intervalle plancher-plafond)")
    ap.add_argument("--sortie", default=str(RELEVES / "ab-kld-vs-snr"))
    ap.add_argument("--memoire-max", default="40G")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    sortie = Path(a.sortie)

    print("PROTOCOLE A/B — KLD couche-par-couche contre SNR-bloc")
    print(f"  source        {SOURCE}")
    print(f"  corpus        {CORPUS.name}")
    print(f"  etalon fp16   PPL {PPL_REFERENCE} (reference exterieure)")
    print(f"  plancher      {gio(PLANCHER_GIO)}   plafond {gio(PLAFOND_GIO)}")
    print(f"  budget EGAL   {gio(a.budget_gib)}  (les deux bras)")
    print(f"  seuil verdict {SEUIL_DECISION_PPL} PPL "
          f"(revue/protocole-ab-kld-vs-snr.md)")
    for bras in BRAS:
        print(f"  bras {bras['nom']:<4} ordre_sac={bras['ordre_sac']!r:<8} "
              f"mesurer_kld={bras['mesurer_kld']}  x2 jumelles")
    if not a.pour_de_vrai:
        print("\nPLAN SEULEMENT. Rien n'a tourne. "
              "Relancer avec --pour-de-vrai une fois l'accord des sessions "
              "obtenu et la carte libre.")
        return 0

    restes = subprocess.run(["systemctl", "--user", "list-units", "acvram*",
                             "--no-legend", "--plain"],
                            capture_output=True, text=True).stdout.split()
    vivants = [m for m in restes if m.endswith(".service") and m != mon_unite()]
    if vivants:
        print(f"ECHEC / CAUSE: {len(vivants)} service(s) acvram encore "
              f"vivant(s) — {', '.join(vivants[:4])}. Orphelins probables, "
              f"tenant la carte pour rien. / SUITE: systemctl --user stop "
              f"<unite>, puis relancer.")
        return 2
    if not SOURCE.exists():
        print(f"ECHEC / CAUSE: source absente {SOURCE}")
        return 2
    if not CORPUS.exists():
        print(f"ECHEC / CAUSE: corpus absent {CORPUS}")
        return 2
    sha = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    if sha != CORPUS_SHA:
        print(f"ECHEC / CAUSE: corpus de sha {sha[:16]} au lieu de "
              f"{CORPUS_SHA[:16]}. Un corpus different rend une "
              f"perplexite differente sans que rien ne le signale.")
        return 2
    sortie.mkdir(parents=True, exist_ok=True)

    resultats: dict[str, dict] = {}
    for bras in BRAS:
        nom = bras["nom"]
        ppls: list[float] = []
        for k in (1, 2):
            dossier = BASE / f"Llama-2-7b-ab-{nom}-{k}"
            if not dossier.exists():
                argv = [a.python, "-m", "acvram", "convert", str(SOURCE),
                        "--out", str(dossier), "--bits-budget", str(a.budget_gib)]
                if bras["mesurer_kld"]:
                    argv.append("--mesurer-kld")
                code = service(f"acvram-ab-conv-{nom}-{k}", argv,
                               sortie / f"conv-{nom}-{k}.log", a.memoire_max,
                               minutes=90, ordre_sac=bras["ordre_sac"])
                if code:
                    print(f"ECHEC / CAUSE: conversion {nom} exemplaire {k}, "
                          f"code {code}, voir "
                          f"{sortie / f'conv-{nom}-{k}.log'}")
                    return 2
            bud = lire_budget(dossier)
            if bud is None:
                print(f"ECHEC / CAUSE: {dossier} sans bloc budget au manifeste")
                return 2
            vu = bud.get("ordre_glouton")
            if vu != bras["ordre_glouton_attendu"]:
                print(f"ECHEC / CAUSE: {dossier} porte ordre_glouton={vu!r} "
                      f"au lieu de {bras['ordre_glouton_attendu']!r} attendu "
                      f"pour le bras {nom}.")
                return 2
            print(f"  {nom}/{k}  budget demande {gio(a.budget_gib)}  "
                  f"depense {bud['depense_gib']} Gio  "
                  f"{bud['promus']}/{bud['candidats']} promus")

            rendre_le_cache(dossier)
            jl = sortie / f"eval-{nom}-{k}.log"
            code = service(f"acvram-ab-eval-{nom}-{k}",
                           [a.python, "-m", "acvram", "eval", str(dossier)] + EVAL,
                           jl, a.memoire_max, minutes=45)
            if code:
                print(f"ECHEC / CAUSE: eval {nom} exemplaire {k}, code {code}, "
                      f"voir {jl}")
                return 2
            releve = releve_du_journal(jl) or {}
            souci = verifier_que_c_est_notre_code(releve, jl) if releve else "releve absent"
            if souci:
                print(f"REFUS DE PUBLIER : {souci}")
                return 2
            ppl = releve.get("perplexity")
            if ppl is None:
                print(f"ECHEC / CAUSE: {jl} sans perplexite au releve")
                return 2
            print(f"  {nom}/{k}  PPL {ppl}")
            ppls.append(ppl)
            rendre_le_cache(dossier)

        dispersion = abs(ppls[0] - ppls[1])
        print(f"  bras {nom} : jumelles {ppls[0]} / {ppls[1]}, "
              f"dispersion {dispersion:.6f} PPL")
        resultats[nom] = {"ppl_jumelles": ppls, "ppl_moyenne": sum(ppls) / 2,
                          "dispersion": dispersion}

    ppl_a = resultats["snr"]["ppl_moyenne"]
    ppl_b = resultats["kld"]["ppl_moyenne"]
    ecart = ppl_a - ppl_b
    if any(r["dispersion"] >= SEUIL_DECISION_PPL for r in resultats.values()):
        verdict = ("INDECIDABLE : la dispersion INTRA-bras depasse deja le "
                   "seuil de decision — aucune comparaison ENTRE bras n'est "
                   "interpretable tant que ceci n'est pas resolu.")
    elif abs(ecart) < SEUIL_DECISION_PPL:
        verdict = (f"INDECIDABLE : ecart {ecart:+.4f} PPL, sous le seuil "
                   f"{SEUIL_DECISION_PPL}. Ni egalite ni gagnant — voir "
                   f"l'issue nommee dans revue/protocole-ab-kld-vs-snr.md.")
    else:
        gagnant = "kld" if ecart > 0 else "snr"
        verdict = (f"{gagnant.upper()} gagne : ecart {ecart:+.4f} PPL, "
                   f"au-dela du seuil {SEUIL_DECISION_PPL}.")

    print(f"\nPPL bras A (snr) : {ppl_a:.4f}   PPL bras B (kld) : {ppl_b:.4f}")
    print(f"VERDICT : {verdict}")
    (sortie / "resultats.json").write_text(json.dumps(
        {"budget_gib_demande": a.budget_gib, "seuil_ppl": SEUIL_DECISION_PPL,
         "resultats": resultats, "ecart_ppl": ecart, "verdict": verdict},
        indent=2, ensure_ascii=False))
    print(f"\nFAIT / TESTE: {sortie / 'resultats.json'} / RESTE: rien")
    return 0


if __name__ == "__main__":
    sys.exit(main())
