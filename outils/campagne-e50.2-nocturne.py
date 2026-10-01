#!/usr/bin/env python3
"""Pièce e50.2 (poste2, ordre chef, 01/10) : campagne NOCTURNE tok/s (banc-outils) et
refus (banc-refus) sur les 266 alias du parc, écriture par `parc.ecrire_note` SEULE.

Doit tourner sous /usr/bin/python3 (gi/PyGObject n'existe pas dans le venv du projet) :
    /usr/bin/python3 outils/campagne-e50.2-nocturne.py --simule
    /usr/bin/python3 outils/campagne-e50.2-nocturne.py --executer --je-sais-que-la-carte-est-libre

`--simule` (défaut) : lit le parc, calcule la cible (tps non mesuré/étoilé, refus inconnu),
imprime l'ordre et une durée prédite, aucune carte.
`--executer` : pour chaque alias de la cible, dans l'ordre — charge via le lanceur du
moteur (acvram-serveur/llamacpp-serveur/vllm-serveur/llamacpp-appoint, qui gèrent
eux-mêmes carte.sh et bloquent jusqu'à « prêt »), lance banc-outils et/ou banc-refus
selon ce qui manque, écrit par `ecrire_note`, arrête le serveur (SIGTERM sur le port),
passe au suivant. JAMAIS de `set -e` : un alias en échec/TIMEOUT garde son ANCIENNE
valeur, la raison va au bilan, jamais un « 0 » écrit à la place (ordre chef 01/10).

Fenêtre : 20:00-07:00 ; aucun alias ne DÉMARRE après 06:30 ; carte GPU0 (5090) seule —
jamais ACVRAM_CARTE surchargé ici, donc jamais la 3080 Ti (ordre chef 01/10).

Hors périmètre automatique : tabby et yals « chargent leur modèle eux-mêmes » (pas de
lanceur scriptable, `fenetre.py:1149-1151`) — comptés dans la cible, jamais mesurés par
cette campagne, nommés au bilan avec leur raison.
"""
import argparse
import datetime
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "parc" / "lib"))
from menu_modeles import parc as _parc  # noqa: E402
from menu_modeles.config import MOTEURS, ORDRE_MOTEUR  # noqa: E402
from menu_modeles.moteur import pid_du_port  # noqa: E402

BIN = _parc.PARC.bin if hasattr(_parc, "PARC") else Path.home() / ".local" / "bin"
HORS_PERIMETRE = {"tabby", "yals"}  # chargent leur modèle eux-mêmes, non scriptables ici

HEURE_DEBUT = datetime.time(20, 0)
HEURE_DERNIER_DEPART = datetime.time(6, 30)
HEURE_FIN = datetime.time(7, 0)

PLAFOND_DEFAUT_S = 15 * 60  # 15 min/alias tant que le calibrage n'a pas parlé (ordre chef)


def _besoin_tps(m):
    t = (m.tps or "").strip()
    return (not t) or t in ("?", "non mesuré") or "*" in t


def _besoin_refus(m):
    return (m.refus or "").strip() == "inconnu"


def cible():
    """Alias à mesurer, ordre du parc (= ordre des menus claude-modeles/kimi-modeles,
    tous `lancable` dans ce parc — vérifié : 266/266 le 01/10), groupés par moteur
    (`ORDRE_MOTEUR`) pour ne pas recharger le même moteur en dents de scie."""
    p = _parc.charger_parc()
    c = [m for m in p if _besoin_tps(m) or _besoin_refus(m)]
    c.sort(key=lambda m: (ORDRE_MOTEUR.get(m.provider, 9), p.index(m)))
    return c


def simuler():
    c = cible()
    n_tps = sum(1 for m in c if _besoin_tps(m))
    n_refus = sum(1 for m in c if _besoin_refus(m))
    hors = [m for m in c if m.provider in HORS_PERIMETRE]
    autos = [m for m in c if m.provider not in HORS_PERIMETRE]
    print(f"cible : {len(c)} alias (tps à mesurer : {n_tps} ; refus à mesurer : {n_refus})")
    print(f"hors périmètre automatique (tabby/yals, chargent leur modèle eux-mêmes) : {len(hors)}")
    for m in hors:
        print(f"  hors périmètre : {m.alias} ({m.provider})")
    print(f"à mesurer par cette campagne : {len(autos)}")
    for m in autos:
        besoins = []
        if _besoin_tps(m):
            besoins.append("tps")
        if _besoin_refus(m):
            besoins.append("refus")
        print(f"  {m.alias:55s} [{m.provider:9s}] -> {','.join(besoins)}")
    total_s = len(autos) * PLAFOND_DEFAUT_S
    print(f"durée prédite au plafond par défaut (15 min/alias, à resserrer par calibrage) : "
          f"{total_s/3600:.1f} h pour {len(autos)} alias")
    nuits = -(-total_s // ((HEURE_FIN.hour - HEURE_DEBUT.hour + 24) % 24 * 3600 or 11 * 3600))
    print(f"fenêtre 20:00-07:00 (11 h utiles, départs coupés à 06:30) : "
          f"~{nuits} nuit(s) au pire cas, moins si le calibrage resserre le plafond")
    return 0


def _dans_fenetre():
    h = datetime.datetime.now().time()
    if HEURE_DEBUT <= h or h < HEURE_FIN:
        return True
    return False


def _peut_demarrer():
    h = datetime.datetime.now().time()
    # fenêtre traverse minuit : après 20:00 OU avant 06:30
    if h >= HEURE_DEBUT or h < HEURE_DERNIER_DEPART:
        return True
    return False


def _env():
    e = dict(os.environ)
    e.setdefault("ACVRAM_POSTE", "poste2")
    # jamais de surcharge ACVRAM_CARTE ici : carte.sh reste sur son défaut (0 = 5090),
    # jamais la 3080 Ti qui porte le service de l'utilisatrice (ordre chef 01/10).
    e.pop("ACVRAM_CARTE", None)
    return e


def _charger(m, journal):
    mot = MOTEURS[m.provider]
    if m.provider == "rapide":
        argv = [str(BIN / "llamacpp-appoint")]
    elif m.provider == "acvram":
        argv = [str(BIN / "acvram-serveur"), m.alias]
    elif m.provider == "llamacpp":
        argv = [str(BIN / "llamacpp-serveur"), m.dossier, str(m.ctx_service or m.ctx)]
    else:
        argv = [str(BIN / "vllm-serveur"), m.dossier, str(m.ctx_service or m.ctx * 2)]
    env = _env()
    if m.provider == "llamacpp" and m.gabarit:
        env["GABARIT"] = m.gabarit
    with open(journal, "a") as f:
        f.write(f"=== chargement {m.alias} ({m.provider}) {time.strftime('%H:%M:%S')}\n")
        r = subprocess.run(argv, cwd=RACINE, env=env, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode


def _arreter(m, journal):
    port = MOTEURS[m.provider].port
    pid = pid_du_port(port)
    if pid is None:
        with open(journal, "a") as f:
            f.write(f"=== arrêt {m.alias} : port {port} sans propriétaire identifiable\n")
        return
    os.kill(pid, signal.SIGTERM)
    with open(journal, "a") as f:
        f.write(f"=== arrêt {m.alias} (pid {pid}, SIGTERM)\n")
    time.sleep(2)


def _lancer_banc(outil, port, journal):
    r = subprocess.run([str(BIN / outil), "--port", str(port)], cwd=RACINE,
                        capture_output=True, text=True)
    with open(journal, "a") as f:
        f.write(f"=== {outil} port={port} rc={r.returncode}\n{r.stdout}\n{r.stderr}\n")
    return r.returncode, r.stdout


def _parser_outils(sortie):
    r = re.search(r"→ (\d+)/(\d+) appels corrects · (\d+) tok/s", sortie)
    return r


def _parser_refus(sortie):
    r = re.search(r"\t(\d+)/(\d+) refus\t(\S.*)$", sortie, re.MULTILINE)
    return r


def executer(journal, duree_max_par_moteur):
    c = [m for m in cible() if m.provider not in HORS_PERIMETRE]
    bilan = {"tenu": [], "timeout": [], "refus": [], "echec": [], "sautes": []}
    calibrage = {}
    for m in c:
        # reprise : déjà à jour depuis un passage précédent de CETTE campagne (relu
        # à chaque alias, pas en mémoire — symétrique de prise-tache-275.sh)
        p_actuel = _parc.charger_parc()
        m_actuel = next((x for x in p_actuel if x.alias == m.alias), m)
        if not _besoin_tps(m_actuel) and not _besoin_refus(m_actuel):
            bilan["sautes"].append((m.alias, "déjà à jour, sauté"))
            continue
        if not _peut_demarrer():
            with open(journal, "a") as f:
                f.write(f"=== fenêtre fermée (après 06:30), arrêt propre à {m.alias}\n")
            break

        plafond = duree_max_par_moteur.get(m.provider, PLAFOND_DEFAUT_S)
        debut = time.time()
        ancien_tps, ancien_refus, ancien_qual, ancien_usage = m.tps, m.refus, m.qual, m.usage

        rc = _charger(m, journal)
        if rc != 0:
            bilan["echec"].append((m.alias, f"chargement rc={rc}"))
            continue

        port = MOTEURS[m.provider].port
        nouveau_tps, nouveau_refus = ancien_tps, ancien_refus
        raisons = []
        try:
            if time.time() - debut > plafond:
                raise TimeoutError("plafond dépassé avant le banc")
            if _besoin_tps(m):
                rc2, sortie = _lancer_banc("banc-outils", port, journal)
                r = _parser_outils(sortie) if rc2 == 0 else None
                if r:
                    nouveau_tps = f"{r.group(3)}"
                else:
                    raisons.append(f"banc-outils rc={rc2}, non reconnu")
            if _besoin_refus(m):
                if time.time() - debut > plafond:
                    raise TimeoutError("plafond dépassé avant banc-refus")
                rc3, sortie = _lancer_banc("banc-refus", port, journal)
                r = _parser_refus(sortie) if rc3 == 0 else None
                if r:
                    nouveau_refus = r.group(3).strip()
                else:
                    raisons.append(f"banc-refus rc={rc3}, non reconnu")
        except TimeoutError as e:
            raisons.append(str(e))
            bilan["timeout"].append((m.alias, str(e)))
        finally:
            _arreter(m, journal)

        # ordre chef : un alias en échec/TIMEOUT garde sa valeur ANCIENNE, jamais un
        # « 0 » — on n'écrit que ce qui a vraiment été mesuré cette passe.
        if nouveau_tps != ancien_tps or nouveau_refus != ancien_refus:
            _parc.ecrire_note(m.alias, nouveau_refus, nouveau_tps, ancien_qual, ancien_usage)
            bilan["tenu"].append((m.alias, f"tps={nouveau_tps} refus={nouveau_refus}"))
        elif raisons:
            bilan["refus"].append((m.alias, "; ".join(raisons)))

        dt = time.time() - debut
        if m.provider not in calibrage:
            calibrage[m.provider] = dt
            duree_max_par_moteur[m.provider] = min(PLAFOND_DEFAUT_S, max(60, int(dt * 2)))
            with open(journal, "a") as f:
                f.write(f"=== calibrage {m.provider} : {dt:.0f}s mesurés, "
                        f"plafond resserré à {duree_max_par_moteur[m.provider]}s\n")

    with open(journal, "a") as f:
        f.write(f"=== FIN {time.strftime('%H:%M:%S')}\n")
        for cle, lot in bilan.items():
            f.write(f"BILAN {cle} : {len(lot)}\n")
            for alias, raison in lot:
                f.write(f"  {alias} : {raison}\n")
    print(f"=== FIN {time.strftime('%H:%M:%S')}")
    for cle, lot in bilan.items():
        print(f"BILAN {cle} : {len(lot)}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--simule", action="store_true", default=True)
    g.add_argument("--executer", action="store_true")
    ap.add_argument("--je-sais-que-la-carte-est-libre", action="store_true", dest="confirme")
    ap.add_argument("--journal", default=str(RACINE / "scratchpad" / "poste2-e50.2-01-10" / "campagne.log"))
    a = ap.parse_args()
    if a.executer:
        if not a.confirme:
            print("REFUS : --executer exige --je-sais-que-la-carte-est-libre", file=sys.stderr)
            sys.exit(66)
        if not _dans_fenetre():
            print("REFUS : hors fenêtre 20:00-07:00", file=sys.stderr)
            sys.exit(66)
        Path(a.journal).parent.mkdir(parents=True, exist_ok=True)
        sys.exit(executer(a.journal, {}))
    sys.exit(simuler())
