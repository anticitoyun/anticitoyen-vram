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
alias, rc et durée). Garde témoin (méthode § 4, E1, chef 01/10) : **McNemar ITEM PAR ITEM**
(`outils/qualite-e50-mcnemar.py`), pas un simple écart de S — la 1re passe du témoin pose la
référence (`~/.cache/acvram/qualite-e50-temoin-reference.json`, persiste entre deux lancements de
cette campagne) ; chaque passe suivante compare ses échantillons (`SAMPLES <tâche>: <chemin>`,
imprimés par `qualite-e50.sh`) à cette référence par `qualite_e50_mcnemar.temoin_tenu` : TENU
seulement si p > 0,05 ET taux de discordance ≤ 10 % (E1 : « réponses identiques ≥ 90 % ») — les
deux conditions sont nécessaires, une dérive SYMÉTRIQUE (autant d'items justes→faux que
faux→justes) donne un p proche de 1 malgré 100 % de discordance ; voir
`tests/test_qualite_e50_mcnemar.py` pour ce cas construit. Rejeu FAUX → le lot s'arrête, la
campagne REFUSE de continuer (dérive de l'instrument, pas des modèles, § 4). Pause coopérative :
si `~/.config/acvram/campagne-qualite-e50.pause` existe entre deux alias, la carte est rendue et
la campagne attend sa disparition (même convention que campagne-e50.2-nocturne.py, 01/10).

Usage :
  outils/campagne-qualite-e50.py --simule
  outils/campagne-qualite-e50.py --executer --je-sais-que-la-carte-est-libre
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
TEMOIN = "acvram-qwen3-4b-srcgguf-nvfp4"
TAILLE_LOT = 10
TACHES_E50 = ("mmlu_e50_hsm", "mmlu_e50_law", "mmlu_e50_ccs", "gsm8k_e50", "humaneval_e50")
REFERENCE_TEMOIN = Path(os.path.expanduser("~/.cache/acvram/qualite-e50-temoin-reference.json"))


def _mcnemar_lib():
    spec = importlib.util.spec_from_file_location("qe50_mcnemar", RACINE / "outils" / "qualite-e50-mcnemar.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

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
    echantillons = {}
    for ligne in r.stdout.splitlines():
        if ligne.startswith("S="):
            try:
                s = float(ligne.split()[0].split("=", 1)[1])
            except ValueError:
                pass
        elif ligne.startswith("SAMPLES "):
            # "SAMPLES <tâche>: <chemin|?>"
            reste = ligne[len("SAMPLES "):]
            tache, _, chemin = reste.partition(":")
            chemin = chemin.strip()
            if chemin and chemin != "?":
                echantillons[tache.strip()] = chemin
    return r.returncode, s, duree, echantillons, r.stdout, r.stderr


def _corrects_echantillons(echantillons):
    """{tâche: chemin} -> {tâche: {doc_id: correct}}, via qualite-e50-mcnemar.py (bac à sable
    pour HumanEval, lecture directe des métriques lm-eval pour MMLU/GSM8K)."""
    mcn = _mcnemar_lib()
    out = {}
    for tache, chemin in echantillons.items():
        if not os.path.exists(chemin):
            continue
        if tache == "humaneval_e50":
            out[tache] = mcn.corrects_humaneval(chemin)
        else:
            out[tache] = mcn.corrects_mmlu_gsm8k(chemin)
    return out


def executer():
    cibles, lots, _ = preparer()
    if not cibles:
        print("=== rien à mesurer (aucun alias non mesuré au parc)")
        return 0
    mcn = _mcnemar_lib()
    journal = RACINE / "outils" / "campagne-qualite-e50.tsv"
    reference = None
    if REFERENCE_TEMOIN.exists():
        try:
            reference = json.loads(REFERENCE_TEMOIN.read_text())
            print(f"référence témoin reprise de {REFERENCE_TEMOIN} (campagne précédente)")
        except (json.JSONDecodeError, OSError):
            reference = None

    for num_lot, lot in enumerate(lots, 1):
        _attendre_pause()
        print(f"=== lot {num_lot}/{len(lots)} — témoin {TEMOIN}")
        rc, s, duree, echantillons, out, err = _jouer_un(TEMOIN)
        with journal.open("a") as f:
            f.write(f"{TEMOIN}\tlot{num_lot}-temoin\t{rc}\t{s}\t{duree:.0f}\n")
        if rc != 0:
            print(f"ÉCHEC témoin (rc={rc}) — campagne arrêtée au lot {num_lot}\n{err}", file=sys.stderr)
            return 1
        corrects = _corrects_echantillons(echantillons)
        if reference is None:
            reference = corrects
            REFERENCE_TEMOIN.parent.mkdir(parents=True, exist_ok=True)
            REFERENCE_TEMOIN.write_text(json.dumps(corrects))
            print(f"témoin de référence posé (S={s}), {sum(len(d) for d in corrects.values())} items, "
                  f"{REFERENCE_TEMOIN}")
        else:
            tenu, disc_b, disc_c, p, taux = mcn.temoin_tenu(reference, corrects)
            print(f"témoin lot {num_lot} : S={s}, McNemar b={disc_b} c={disc_c} p={p:.4f} "
                  f"discordance={taux:.1%} -> {'TENU' if tenu else 'FAUX'}")
            if not tenu:
                print(f"TÉMOIN FAUX (McNemar, méthode §4/E1) au lot {num_lot} — dérive de "
                      "l'instrument suspectée, campagne arrêtée (pas des modèles)", file=sys.stderr)
                return 1
        for alias in lot:
            _attendre_pause()
            print(f"--- {alias}")
            rc, s, duree, echantillons, out, err = _jouer_un(alias)
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
