#!/usr/bin/env python3
"""e50.3 § 7.3 (poste4, 01/10) — orchestre `outils/qualite-e50.sh` sur les alias « non mesuré »
du parc réellement servi (intersection config.toml ∩ notes-modeles.tsv), même modèle que
`outils/campagne-qualite-275.py` (poste2, 30/09) : une seule commande, pas de lancement manuel
par alias.

`--simule` (DÉFAUT, REGLES §3 : aucune carte, aucun processus) : lit le parc réel, construit la
cible (témoin + lots de 10), imprime l'ordre et la durée prédite (méthode § 2/§6,
`acvram-memoire/revue/poste6-e50-3-methode-qualite-01-10.md`), sort 0.

`--executer` : pour chaque lot, joue le témoin (`acvram-qwen3-4b-srcgguf-nvfp4`) EN TÊTE, puis
les 10 alias du lot via `outils/qualite-e50.sh <alias> --executer
--je-sais-que-la-carte-est-libre` ; écrit un journal (`campagne-qualite-e50.tsv`, un rang par
alias, rc et durée). Garde témoin (méthode § 4, E1) : si le S du témoin au lot N s'écarte de
plus de 0,05 (abs) de son premier S mesuré dans CETTE campagne, le lot s'arrête et la campagne
REFUSE de continuer (dérive de l'instrument, pas des modèles) — **ceci est une garde de
cohérence simple, pas le test de McNemar item par item prescrit en § 4 : celui-ci exige les
échantillons appariés (`panel-taches-resume.py`) et reste à la charge de qui mesure (poste2 ou
poste1) avant de publier une étoile, jamais sauté ici.** Pause coopérative : si
`~/.config/acvram/campagne-qualite-e50.pause` existe entre deux alias, la carte est rendue et la
campagne attend sa disparition (même convention que campagne-e50.2-nocturne.py, 01/10).

Usage :
  outils/campagne-qualite-e50.py --simule
  outils/campagne-qualite-e50.py --executer --je-sais-que-la-carte-est-libre
"""
import argparse
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
TEMOIN = "acvram-qwen3-4b-srcgguf-nvfp4"
TAILLE_LOT = 10
ECART_TEMOIN_MAX = 0.05   # garde de cohérence simple (pas McNemar, voir docstring)

# Durées prédites par catégorie (méthode § 2, poste6, scellées AVANT mesure) — minutes.
DUREE_MIN = {"dense_30b": 10, "moe": 3.5, "petit_dense": 3, "grand_dense_exile": 60}
PAUSE_FICHIER = Path(os.path.expanduser("~/.config/acvram/campagne-qualite-e50.pause"))


def _menu():
    conf_path = Path(os.path.expanduser("~/.kimi-code/config.toml"))
    if not conf_path.exists():
        return {}
    return tomllib.load(conf_path.open("rb")).get("models", {})


def _non_mesures():
    """alias → ligne (5 colonnes) pour chaque alias servi dont la colonne qualité vaut
    « non mesuré » dans ~/TSV/notes-modeles.tsv — même lecture que qualite-e50.sh."""
    tsv_path = Path(os.path.expanduser("~/TSV/notes-modeles.tsv"))
    if not tsv_path.exists():
        return []
    menu = _menu()
    cibles = []
    for ligne in tsv_path.read_text().splitlines():
        if not ligne or ligne.startswith("#"):
            continue
        champs = ligne.split("\t")
        if len(champs) < 4:
            continue
        alias, qual = champs[0], champs[3]
        if qual == "non mesuré" and alias in menu:
            cibles.append(alias)
    return sorted(cibles)


def _classe(alias):
    al = alias.lower()
    moe = "a3b" in al or "-moe-" in al or al.endswith("moe") or "flash" in al
    import re
    m = re.search(r"(\d+(?:\.\d+)?)b", al)
    taille = float(m.group(1)) if m else None
    if taille is None:
        return "petit_dense"   # défaut conservateur (minore rarement le temps prédit)
    if moe:
        return "moe"
    if taille >= 60:
        return "grand_dense_exile"
    if taille >= 20:
        return "dense_30b"
    return "petit_dense"


def preparer():
    cibles = _non_mesures()
    lots = [cibles[i:i + TAILLE_LOT] for i in range(0, len(cibles), TAILLE_LOT)]
    duree_min = sum(DUREE_MIN[_classe(a)] for a in cibles)
    duree_min += len(lots) * DUREE_MIN["petit_dense"]   # témoin en tête de chaque lot (~3 min)
    return cibles, lots, duree_min


def simuler():
    cibles, lots, duree_min = preparer()
    print(f"=== campagne-qualite-e50 --simule ({len(cibles)} alias non mesurés, {len(lots)} lot(s) de ≤{TAILLE_LOT})")
    compte = {}
    for a in cibles:
        compte[_classe(a)] = compte.get(_classe(a), 0) + 1
    for cl, n in sorted(compte.items()):
        print(f"  {cl:20s} : {n:3d} alias × {DUREE_MIN[cl]:.1f} min")
    print(f"témoin {TEMOIN} rejoué en tête de chaque lot (+{len(lots)} × {DUREE_MIN['petit_dense']:.1f} min)")
    print(f"durée prédite totale : {duree_min / 60:.1f} h de carte (méthode § 2/§6, prédiction E2)")
    for i, lot in enumerate(lots, 1):
        print(f"  lot {i:2d}/{len(lots)} : {TEMOIN} (témoin) puis {', '.join(lot)}")
    print("PRÊTE")
    return 0


def _attendre_pause():
    while PAUSE_FICHIER.exists():
        print(f"=== pause ({PAUSE_FICHIER}) — carte rendue, en attente")
        time.sleep(30)


def _jouer_un(alias):
    env = {**os.environ, "ACVRAM_POSTE": os.environ.get("ACVRAM_POSTE", "poste2")}
    debut = time.monotonic()
    r = subprocess.run(
        ["outils/qualite-e50.sh", alias, "--executer", "--je-sais-que-la-carte-est-libre"],
        cwd=RACINE, env=env, capture_output=True, text=True,
    )
    duree = time.monotonic() - debut
    s = None
    for ligne in r.stdout.splitlines():
        if ligne.startswith("S="):
            try:
                s = float(ligne.split()[0].split("=", 1)[1])
            except ValueError:
                pass
    return r.returncode, s, duree, r.stdout, r.stderr


def executer():
    cibles, lots, _ = preparer()
    if not cibles:
        print("=== rien à mesurer (aucun alias non mesuré au parc)")
        return 0
    journal = RACINE / "outils" / "campagne-qualite-e50.tsv"
    s_temoin_initial = None
    for num_lot, lot in enumerate(lots, 1):
        _attendre_pause()
        print(f"=== lot {num_lot}/{len(lots)} — témoin {TEMOIN}")
        rc, s, duree, out, err = _jouer_un(TEMOIN)
        with journal.open("a") as f:
            f.write(f"{TEMOIN}\tlot{num_lot}-temoin\t{rc}\t{s}\t{duree:.0f}\n")
        if rc != 0:
            print(f"ÉCHEC témoin (rc={rc}) — campagne arrêtée au lot {num_lot}\n{err}", file=sys.stderr)
            return 1
        if s_temoin_initial is None:
            s_temoin_initial = s
            print(f"témoin de référence S={s}")
        elif s is None or abs(s - s_temoin_initial) > ECART_TEMOIN_MAX:
            print(f"TÉMOIN FAUX : S={s} s'écarte de plus de {ECART_TEMOIN_MAX} du premier S={s_temoin_initial} "
                  f"— dérive de l'instrument suspectée, campagne arrêtée au lot {num_lot} "
                  "(garde de cohérence simple ; McNemar item par item reste à faire avant de publier, voir § 4)",
                  file=sys.stderr)
            return 1
        for alias in lot:
            _attendre_pause()
            print(f"--- {alias}")
            rc, s, duree, out, err = _jouer_un(alias)
            with journal.open("a") as f:
                f.write(f"{alias}\tlot{num_lot}\t{rc}\t{s}\t{duree:.0f}\n")
            if rc != 0:
                print(f"ÉCHEC {alias} (rc={rc}) — ANCIENNE valeur conservée, raison au journal, "
                      "campagne continue (ordre chef e50.2, même règle)", file=sys.stderr)
    print(f"=== campagne terminée, journal : {journal}")
    return 0


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--simule", action="store_true", default=True)
    g.add_argument("--executer", action="store_true")
    p.add_argument("--je-sais-que-la-carte-est-libre", action="store_true", dest="confirme")
    a = p.parse_args()
    if a.executer:
        if not a.confirme:
            print("REFUS : --executer exige --je-sais-que-la-carte-est-libre (REGLES §3)", file=sys.stderr)
            return 65
        return executer()
    return simuler()


if __name__ == "__main__":
    raise SystemExit(main())
