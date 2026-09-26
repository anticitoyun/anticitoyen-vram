#!/usr/bin/env python3
"""Courbe du quota : ce que chaque octet supplementaire achete en qualite.

Six budgets entre le plancher tout-nvfp4 et le plafond tout-int8 de
Llama-2-7B. DEUX des six sont des TEMOINS a valeur prevue : le budget du
plancher doit reproduire le dossier tout-nvfp4 (4,7296 bits/poids, PPL
5,6102) et celui du plafond le dossier tout-int8 (8,3452, PPL 5,4144). Si un
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

P_POIDS = 6_738_417_664          # poids comptes dans le manifeste des etalons
GIO = 1024 ** 3

CARTE = Path(__file__).resolve().parent / "carte.sh"
RACINE_OUTILS = Path(__file__).resolve().parent
BASE = Path(MODELES)
SOURCE = BASE / "Llama-2-7b-hf"
# Corpus PARTAGE, pas dans un worktree : 0a l'a cherche et ne l'a pas
# trouve, parce qu'il ne vivait que dans mon arbre et n'etait pas suivi par git.
# Deux campagnes qui ne lisent pas le meme fichier ne sont pas sur la meme
# courbe, et ce defaut-la ne se voit qu'a la fin.
CORPUS = Path("/mnt/4TO_SATACMR_2022/Modeles/corpus/wiki-gptq.txt")
CORPUS_SHA = "e52922746ad09bac73b0dba32b2987c0d7924da14337dcd43c1d9113a9f6d0ae"
# Pour memoire, le brut de llama.cpp — PAS celui de l'etalon exterieur :
#   wiki.test.raw  173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08
#   335 688 jetons, 163 segments, PPL de reference 5,5625 (contre 5,4141)
# L'ecart de 2,67 % entre les deux corpus serait attribue aux bits.
RELEVES = sorties()

# LE PLANCHER EST UNE VALEUR CALCULEE, PAS UN CHIFFRE ROND.
#
# Le temoin plancher portait 3,75 Gio quand le plancher reel vaut 3,7102 — donc
# 40,7 Mio DE PLUS que le plancher. Le sac a dos a promu six tenseurs, le
# dossier est sorti a 4,7848 bits/poids au lieu de 4,7297, et le garde a
# declare une non-reproduction. Le garde avait raison ; c'est la CIBLE qui
# etait fausse. On demandait au temoin de reproduire le plancher avec 40 Mio
# de plus que le plancher — un denominateur emprunte, dans le temoin cense
# valider les autres. Releve par f2 le 10/09.
#
# Le script CONNAISSAIT le plancher : il l'imprime. Il ne l'utilisait pas.
PLANCHER_OCTETS = 3_983_834_740        # tout-nvfp4 mesure, safetensors seuls
PLAFOND_OCTETS = 7_029_266_352         # tout-int8 mesure
PLANCHER_GIO = PLANCHER_OCTETS / GIO
PLAFOND_GIO = PLAFOND_OCTETS / GIO

# Budget en Gio -> (nom du dossier, valeur prevue si temoin)
POINTS = [
    (round(PLANCHER_GIO, 4), "quota-plancher",
     {"bpw": 4.7297, "ppl": 5.6102, "promus_attendus": 0,
      "role": "temoin plancher (budget EGAL au plancher : zero promotion "
              "attendue, donc le dossier doit etre celui du tout-nvfp4)"}),
    (4.50, "quota-4g50", None),
    (5.00, "quota-5g00", None),
    (5.50, "quota-5g50", None),
    (6.00, "quota-6g00", None),
    (round(PLAFOND_GIO, 4), "quota-plafond",
     {"bpw": 8.3452, "ppl": 5.4144, "role": "temoin plafond (budget EGAL au "
      "plafond : tout doit etre promu, et le dossier doit valoir le tout-int8 "
      "— c'est ce point qui dira si l'ordre du glouton est mal oriente, ou si "
      "mon plafond de reference differait par son MECANISME)"}),
]

# Etalon exterieur, mesure par transformers sans rien importer d'acvram.
PPL_REFERENCE = 5.4141
# Perplexite du tout-int8 SUR LE BINAIRE COURANT. Elle valait 5,4142 dans les
# releves archives ; le binaire a change le 10/09 et elle vaut 5,4144, mesure
# deux fois a six decimales identiques. Ce n'est pas du bruit — la dispersion
# de l'instrument est nulle — c'est un delta de code, et c'est pourquoi la
# constante doit vivre ici et non recopiee dans trois messages.
PPL_PLAFOND = 5.4144

EVAL = ["--corpus", str(CORPUS), "--window", "2048", "--stride", "2048",
        "--max-tokens", "344064", "--min-context", "0", "--json"]


def mon_unite() -> str:
    """L'unite systemd qui contient CE processus, ou une chaine vide.

    Le garde des orphelins comptait le PILOTE lui-meme parmi les services
    vivants et refusait de demarrer — un garde qui accuse son operateur, et qui
    rendait impossible le detachement du pilote, c'est-a-dire la protection
    meme qu'il devait servir. Il faut donc s'exclure, et le seul moyen fiable
    est de lire son propre cgroup : `systemd-run --unit=X` place le processus
    dans `X.service`, et rien dans l'environnement ne le dit.
    """
    try:
        for ligne in open("/proc/self/cgroup", encoding="utf-8"):
            # LE DERNIER segment, pas le premier. Le chemin vaut
            #   /user.slice/user-1000.slice/user@1000.service/app.slice/X.service
            # et le premier segment en « .service » est `user@1000.service` —
            # jamais l'unite. Ma premiere version renvoyait donc toujours le
            # gestionnaire de session, le garde ne s'excluait jamais, et il
            # refusait le lancement une seconde fois apres avoir ete corrige.
            morceaux = [m for m in ligne.strip().split("/")
                        if m.endswith(".service")]
            if morceaux:
                return morceaux[-1]
    except OSError:
        pass
    return ""


def gio(x: float) -> str:
    """UNE convention d'affichage, quatre decimales.

    Le script affichait « plafond 6,547 Gio » a trois decimales et listait le
    meme budget « 6.5465 » a quatre : le meme nombre vu par deux conventions,
    et quelqu'un qui compare les deux se demande lequel est faux. Aucun des
    deux. Releve par f2 le 10/09, en meme temps que ma propre erreur —
    j'avais ecrit 6,5464 dans un message, en TRONQUANT la ou le script
    ARRONDIT. Le code etait juste ; c'est ma prose qui l'etait pas.
    """
    return f"{x:.4f} Gio"


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
    racine = Path(__file__).resolve().parent.parent
    cmd = ["systemd-run", "--user", "--unit", nom, "--collect",
           f"--property=WorkingDirectory={racine}",
           f"--setenv=PYTHONPATH={racine}",
           f"--property=MemoryHigh={memoire_max}",
           f"--property=MemoryMax={memoire_max}",
           f"--property=RuntimeMaxSec={minutes * 60}",
           "--property=CPUWeight=60",
           "--setenv=ACVRAM_CUDA_HOME=/usr/local/cuda-13.2",
           f"--setenv=ACVRAM_KERNEL_CACHE={os.environ.get('ACVRAM_KERNEL_CACHE', '')}",
           "/bin/bash", "-c",
           # CHIEN DE GARDE SUR LE PILOTE. Releve par f2 le 10/09 : le
           # superviseur du harnais a tue son PILOTE (MemFree 10,1 Go alors que
           # MemAvailable valait 85,6) et les bras detaches ont SURVECU — un
           # acvram-disp-1.service encore vivant, tenant la carte pour une
           # mesure qui ne menait plus nulle part, et que carte-libre.sh voyait
           # comme une occupation legitime. Le detachement protege les bras,
           # pas celui qui les enchaine.
           #
           # `kill -0` sur le PID du pilote suffit, et fonctionne meme si le
           # pilote est tue par SIGKILL — ce qu'un `trap` ou un atexit ne
           # couvrirait pas.
           # carte.sh A L'INTERIEUR du service, jamais autour — precaution de
           # 0a, et son garde refuse maintenant activement l'inverse :
           # enveloppant systemd-run, le premier verrou est relache des que le
           # travail est parti et un second concurrent obtient la carte sur un
           # service encore actif. C'est ainsi que deux mesures ont charge en
           # meme temps le 10/09 a 11h05, l'une morte en OOM et l'autre rendant
           # un chiffre que rien ne signalait comme faux.
           # le code de sortie part dans la sentinelle : le service peut mourir
           # sans que la commande ait parle
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
        if time.time() - dernier > 300:      # un point toutes les 5 min
            dernier = time.time()
            print(f"    [{int(time.time() - t0)} s] {nom} tourne toujours",
                  flush=True)
        time.sleep(5)
    return int(sentinelle.read_text().strip() or "1")


def repr_sh(x: str) -> str:
    return "'" + x.replace("'", "'\\''") + "'"


def releve_du_journal(journal: Path) -> dict | None:
    """Rend l'objet JSON qui porte une perplexite, pas le premier accolade venu.

    Le journal d'`acvram eval` contient d'abord le plan (un objet JSON), puis
    168 lignes de progression, puis le releve. Un `find("{")` suivi de
    `json.loads` a donc echoue sur « Extra data: line 46 » — le plan etait lu,
    le releve jamais atteint.
    """
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
    """Le releve doit porter les champs ajoutes le 10/09. Sinon ce n'est pas ce
    code qui a tourne.

    Premiere manche zero : `systemd-run` demarre hors du worktree, donc
    `python -m acvram` a resolu vers l'acvram INSTALLE et non vers le worktree.
    Le releve rendu ne portait ni `corpus_sha256` ni `nbytes_detail` — les deux
    champs ajoutes le jour meme. C'est le temoin : un champ neuf absent prouve
    que le binaire est l'ancien, et aucun chiffre de cette passe ne vaut.
    """
    manquants = [k for k in ("corpus_sha256", "nbytes_detail",
                             "bits_par_poids_en_memoire") if not r.get(k)]
    if manquants:
        return (f"le releve de {journal.name} ne porte pas {manquants} : ce "
                f"n'est pas le code de ce worktree qui a tourne, mais l'acvram "
                f"installe. Aucun chiffre de cette passe ne vaut.")
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
    ap.add_argument("--points", help="sous-ensemble, ex. 4.50,5.00")
    ap.add_argument("--dispersion-seule", action="store_true",
                    help="manche zero seule : mesure la resolution de "
                         "l'instrument et s'arrete, sans convertir")
    ap.add_argument("--sortie", default=str(RELEVES / "quota"))
    ap.add_argument("--memoire-max", default="40G")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    voulus = set() if a.dispersion_seule else None
    if a.points and not a.dispersion_seule:
        voulus = {float(x) for x in a.points.split(",")}
    # APPARIEMENT TOLERANT, et un REFUS si une valeur demandee ne correspond a
    # rien. Depuis que les temoins derivent des octets mesures, leurs budgets
    # ne sont plus ronds — le plafond vaut 6,5464 et non 6,55. Un
    # `--points 6.55` ne correspondait alors a AUCUN point et la campagne
    # tournait sa dispersion puis ne mesurait rien, en silence. Un lancement
    # qui ne mesure rien doit le DIRE.
    def _demande(budget: float) -> bool:
        return voulus is None or any(abs(budget - v) < 0.02 for v in voulus)

    points = [p for p in POINTS if _demande(p[0])]
    if voulus and not a.dispersion_seule:
        orphelines = [v for v in voulus
                      if not any(abs(p[0] - v) < 0.02 for p in POINTS)]
        if orphelines:
            print(f"ECHEC / CAUSE: {orphelines} ne correspond a aucun point. "
                  f"Budgets disponibles : "
                  f"{[p[0] for p in POINTS]}. / SUITE: relancer avec une de "
                  f"ces valeurs — les temoins derivent des octets mesures, "
                  f"leurs budgets ne sont pas ronds.")
            return 2

    print("CAMPAGNE QUOTA — Llama-2-7B, "
          f"{len(points)} point(s) sur {len(POINTS)}")
    print(f"  source   {SOURCE}")
    print(f"  corpus   {CORPUS.name} ({CORPUS.stat().st_size} o)")
    print(f"  etalon   reference exterieure PPL {PPL_REFERENCE}")
    print(f"  plancher {gio(PLANCHER_GIO)} (tout-nvfp4, "
          f"{bits(PLANCHER_OCTETS):.4f} b/p)")
    print(f"  plafond  {gio(PLAFOND_GIO)} (tout-int8, "
          f"{bits(PLAFOND_OCTETS):.4f} b/p)")
    for b, nom, temoin in points:
        marque = f"   <- {temoin['role']}, prevu {temoin['bpw']} b/p "\
                 f"et PPL {temoin['ppl']}" if temoin else ""
        print(f"  {gio(b)}  {nom}{marque}")
    if not a.pour_de_vrai:
        print("\nPLAN SEULEMENT. Rien n'a tourne. "
              "Relancer avec --pour-de-vrai une fois l'accord des sessions obtenu.")
        return 0

    # ORPHELINS D'UNE MANCHE MORTE : ils tiennent la carte et ecrivent dans des
    # journaux que personne ne lira. Les detecter au demarrage plutot que de
    # les decouvrir par un verrou qui n'arrive jamais.
    restes = subprocess.run(["systemctl", "--user", "list-units", "acvram*",
                             "--no-legend", "--plain"],
                            capture_output=True, text=True).stdout.split()
    vivants = [m for m in restes if m.endswith(".service") and m != mon_unite()]
    if vivants:
        print(f"ECHEC / CAUSE: {len(vivants)} service(s) acvram encore vivant(s) "
              f"— {', '.join(vivants[:4])}. Ce sont probablement des orphelins "
              f"d'une manche dont le pilote est mort : ils tiennent la carte "
              f"pour rien. / SUITE: systemctl --user stop <unite>, puis "
              f"relancer.")
        return 2

    if not SOURCE.exists():
        print(f"ECHEC / CAUSE: source absente {SOURCE}")
        return 2
    # Le sha du corpus se VERIFIE, il ne se suppose pas.
    if not CORPUS.exists():
        print(f"ECHEC / CAUSE: corpus absent {CORPUS}. Il se reconstruit par "
              f"outils/etalon-ppl-transformers.py (\"\\n\\n\".join du split "
              f"test de wikitext-2-raw-v1).")
        return 2
    sha = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    if sha != CORPUS_SHA:
        print(f"ECHEC / CAUSE: corpus {CORPUS.name} de sha {sha[:16]} au lieu "
              f"de {CORPUS_SHA[:16]}. Un corpus different rend une perplexite "
              f"differente sans que rien ne le signale.")
        return 2
    print(f"  corpus verifie : sha {sha[:16]}")
    sortie = Path(a.sortie)
    sortie.mkdir(parents=True, exist_ok=True)

    # --- Manche zero : la RESOLUTION de l'instrument, avant tout point ---
    # Remarque de 0a le 10/09 : l'accord du temoin negatif (tout-int8
    # 5,4144 contre etalon 5,4141, soit 0,0055 %) n'a de valeur que si la
    # dispersion entre deux executions identiques est PLUS PETITE que 0,002 %.
    # Sinon le temoin passe par construction et ne prouve rien — c'est ainsi
    # qu'un instrument aveugle a rendu quatre fois 8,825 au millieme le meme
    # jour. Une passe rejouee a l'identique donne ce chiffre pour le prix d'une
    # evaluation, et rien ne se publie avant de l'avoir.
    # LE CACHE SE REND AVANT LA PREMIERE MESURE, pas seulement entre les
    # points. Le 10/09, 75,7 Go de cache page ont fait tuer un pilote — et ce
    # n'est pas un point qui les avait remplis, c'est la manche de DISPERSION,
    # avant le premier point. Une campagne lancee sur une machine deja chaude
    # meurt donc avant d'avoir rien mesure. Releve par f2, qui n'a pas
    # pu attribuer son cache a sa propre manche : il est partage.
    rendu = 0
    for d in BASE.glob("Llama-2-7b-*"):
        if d.is_dir():
            rendu += rendre_le_cache(d)
    print(f"  cache page rendu AVANT la dispersion : {rendu / 2**20:.0f} Mio",
          flush=True)

    dispersion = None
    # Precision de f2 : la dispersion se prend SUR LE POINT qu'on
    # comparera, pas sur un autre — la reproductibilite n'a aucune raison
    # d'etre la meme a 3,75 et a 6,55 Gio. Quand la campagne ne porte qu'un
    # temoin, on la mesure sur le dossier deja converti qui lui correspond ;
    # sinon sur le plafond, le plus exigeant des deux (son ecart a l'etalon
    # est de 0,002 %, celui du plancher de 3,6 %).
    ref_dossier = BASE / ("Llama-2-7b-nvfp4"
                          if points and points[0][0] < 4.0 and len(points) == 1
                          else "Llama-2-7b-int8")
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
            r = releve_du_journal(j) or {}
            souci = verifier_que_c_est_notre_code(r, j) if r else "releve absent"
            if souci:
                print(f"ECHEC / CAUSE: {souci}")
                return 2
            passes.append(r.get("perplexity"))
            print(f"  passe {k} : PPL {passes[-1]}", flush=True)
            rendre_le_cache(ref_dossier)
        if None in passes:
            print("ECHEC / CAUSE: une passe de dispersion n'a pas rendu de PPL")
            return 2
        dispersion = abs(passes[0] - passes[1])
        rel = dispersion / passes[0] * 100
        print(f"  RESOLUTION : dispersion {dispersion:.6f} PPL soit {rel:.5f} % "
              f"— l'ecart du temoin negatif vaut {abs(PPL_PLAFOND - PPL_REFERENCE):.4f} "
              f"({100 * abs(PPL_PLAFOND - PPL_REFERENCE) / PPL_REFERENCE:.5f} %)")
        if dispersion >= abs(PPL_PLAFOND - PPL_REFERENCE):
            print("  ATTENTION : la dispersion couvre l'ecart du temoin negatif. "
                  "Son accord a 0,002 % ne prouve donc rien, et le plancher de "
                  "reproduction des temoins est porte a 3x la dispersion.")
    else:
        print("  dispersion NON MESUREE : "
              f"{ref_dossier.name} absent. Les verdicts de temoin seront "
              "rendus avec un seuil pose a priori, ce qui est plus faible.")

    if a.dispersion_seule:
        print(f"\nFAIT / TESTE: dispersion {dispersion} PPL / "
              f"RESTE: rien, manche zero seule")
        (sortie / "dispersion.json").write_text(json.dumps(
            {"dispersion_ppl": dispersion, "dossier": str(ref_dossier),
             "corpus": str(CORPUS), "corpus_sha": CORPUS_SHA}, indent=2))
        return 0

    # LE NOM DU DOSSIER PORTE LE MODE D'ORDRE, et c'est structurel.
    #
    # Le 10/09 au soir, une manche entiere a ete perdue : le dossier
    # `Llama-2-7b-quota-6g00` existait deja, converti en mode `snr`, et la
    # branche « dossier deja present, conversion sautee » l'a reutilise pour
    # une campagne lancee avec ACVRAM_ORDRE_SAC=erreur. La variable n'a jamais
    # servi — il n'y a pas eu de conversion a ordonner — et la campagne a
    # remesure le bras precedent en silence. Seul le compte de promus (198,
    # valeur annoncee d'avance comme signal d'alarme) l'a revele.
    #
    # Le defaut n'etait pas dans le cache : il etait dans le NOM. Trois modes
    # produisent trois dossiers differents ; leur donner un seul nom les fait
    # se confondre, et aucune vigilance ne repare cela durablement. Le mode
    # par defaut garde le nom historique pour ne pas orpheliner les dossiers
    # deja produits ; tout autre mode porte son suffixe.
    _ordre = os.environ.get("ACVRAM_ORDRE_SAC", "snr").strip().lower()
    if os.environ.get("ACVRAM_ORDRE_SAC_INVERSE"):
        _ordre = "inverse"
    _suffixe = "" if _ordre == "snr" else f"-{_ordre}"
    if _suffixe:
        print(f"[campagne] ordre du sac a dos : {_ordre} — les dossiers "
              f"porteront le suffixe « {_suffixe} »", flush=True)

    resultats = []
    for b, nom, temoin in points:
        dossier = BASE / f"Llama-2-7b-{nom}{_suffixe}"
        print(f"\n=== {nom} — budget {gio(b)} — ordre {_ordre}", flush=True)

        if not dossier.exists():
            code = service(
                f"acvram-conv-{nom}",
                # --grille-erreurs : les 21 erreurs de la grille AWQ par
                # tenseur, au manifeste. Decision du circuit le 10/09 — les
                # deux chantiers demandent des conversions, donc le prix d'un
                # exposant COMMUN a un groupe empilable tombe en prime, sur
                # sept modeles au lieu d'un, sans une conversion de plus. Et
                # cela repond mieux que la question initiale : ce prix depend
                # du modele et de sa calibration, un seul point ne dirait pas
                # s'il est stable.
                [a.python, "-m", "acvram", "convert", str(SOURCE),
                 "--out", str(dossier), "--bits-budget", str(b),
                 "--grille-erreurs"],
                sortie / f"conv-{nom}.log", a.memoire_max, minutes=90)
            if code:
                print(f"  ECHEC / CAUSE: conversion code {code}, "
                      f"voir {sortie / f'conv-{nom}.log'}")
                resultats.append({"budget_gib": b, "etat": "conversion echouee",
                                  "code": code})
                continue
        else:
            # Un dossier reutilise doit prouver qu'il a ete produit par le
            # mode demande. Le manifeste porte `ordre_glouton` depuis le
            # 10/09 ; un dossier plus ancien ne l'a pas, et son absence ne
            # prouve rien — elle est donc signalee comme telle et non lue
            # comme un accord.
            _bud = lire_budget(dossier) or {}
            _vu = _bud.get("ordre_glouton")
            _attendu = {"base_croissant": "snr_de_base_croissant_sans_cout",
                        "snr": "snr_par_octet_decroissant",
                        "erreur": "erreur_evitee_par_octet_decroissante",
                        "absolu": "erreur_absolue_evitee_par_octet_decroissante",
                        "inverse": "snr_par_octet_croissant"}.get(_ordre)
            if _vu is None:
                print(f"  dossier deja present, conversion sautee — mais son "
                      f"manifeste ne porte PAS d'ordre_glouton : rien ne "
                      f"prouve qu'il vient du mode « {_ordre} ». Supprimez-le "
                      f"ou renommez-le si vous mesurez un mode precis.")
            elif _attendu and _vu != _attendu:
                print(f"  ECHEC / CAUSE: dossier deja present mais produit "
                      f"avec ordre_glouton={_vu!r}, alors que la campagne "
                      f"demande {_attendu!r} ({_ordre}).")
                print(f"  SUITE: supprimer {dossier} ou lancer avec un autre "
                      f"ACVRAM_ORDRE_SAC.")
                resultats.append({"budget_gib": b,
                                  "etat": "dossier d'un autre ordre",
                                  "ordre_vu": _vu, "ordre_demande": _ordre})
                continue
            else:
                print(f"  dossier deja present, conversion sautee "
                      f"(ordre_glouton={_vu})")

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

        # Le prix de l'alpha commun, lu au manifeste du point qu'on vient de
        # convertir. Aucun GPU : c'est de l'arithmetique sur des erreurs deja
        # calculees.
        prix = subprocess.run(
            [a.python, str(RACINE_OUTILS / "prix-alpha-commun.py"),
             str(dossier / "acvram_manifest.json")],
            capture_output=True, text=True)
        (sortie / f"prix-alpha-{nom}.txt").write_text(prix.stdout + prix.stderr)
        recup = [l for l in prix.stdout.splitlines() if "RECUPERABLES" in l]
        print(f"  prix alpha commun : {recup[0].strip() if recup else 'non calculable'}",
              flush=True)

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
        jl = sortie / f"eval-{nom}.log"
        releve = releve_du_journal(jl) or {}
        souci = verifier_que_c_est_notre_code(releve, jl) if releve else "releve absent"
        if souci:
            print(f"  REFUS DE PUBLIER CE POINT : {souci}")
            resultats.append({"budget_gib": b, "etat": "releve non attribuable",
                              "cause": souci, "bpw": bpw_reel})
            continue
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
            # Un temoin peut exiger un nombre de promotions : le plancher n'en
            # attend AUCUNE, puisque son budget vaut exactement le plancher.
            # Sans cette verification, un budget legerement au-dessus promeut
            # quelques tenseurs et le temoin echoue en accusant la mesure au
            # lieu de sa propre cible — c'est ce qui s'est produit le 10/09
            # avec 3,75 Gio pour un plancher a 3,7102.
            attendus = temoin.get("promus_attendus")
            if attendus is not None and bud["promus"] != attendus:
                print(f"  MANCHE SANS OBJET : {bud['promus']} promotions pour "
                      f"{attendus} attendues. Le budget de ce temoin n'est pas "
                      f"celui de sa cible — corriger la CIBLE, pas la mesure.")
                ok = False
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
