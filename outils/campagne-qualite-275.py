#!/usr/bin/env python3
"""Pièce campagne-275 (poste2, ordre chef, 30/09) : orchestre `outils/qualite.sh`
sur les 3 modèles coordonnés avec poste1 (piece 274/275), une seule commande au lieu
de trois lancements manuels.

`--simule` (défaut) : aucune carte, aucun processus — imprime le plan (alias présent
au menu, référence trouvée, doublon de dossier, durée prédite) et sort 0.
`--executer` : génère d'abord la référence manquante (Qwen3-4B-srcgguf-nvfp4, décision
chef 30/09 — les 3 modèles, pas 2) via `generer-reference-v2.sh`, puis lance
`outils/qualite.sh <alias> <bras>` pour chaque candidat, dans l'ordre, sous
ATTENDU=<HEAD court> ; s'arrête au premier FAUX (REGLES : un FAUX est un résultat, on
ne l'enterre pas sous une suite qui continue quand même). Requiert ATTENDU déjà posé
dans l'environnement (même contrôle que qualite.sh) ET la confirmation explicite
`--je-sais-que-la-carte-est-libre`, jamais déduite ici. `ACVRAM_POSTE=poste2` posé pour
chaque sous-appel (carte.sh, ordre chef 30/09) sauf s'il est déjà dans l'environnement.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
TSV_ACVRAM = Path.home() / "TSV" / "acvram-chemins.tsv"
CONFIG_KIMI = Path.home() / ".kimi-code" / "config.toml"
CACHE_275 = Path.home() / ".cache" / "acvram" / "qualite-275"

# Coordonnés avec poste1 (piece 274, scellé 275, 26/09) : un MoE, un mixte, un dense.
# Nom du dossier de référence (~/.cache/acvram/qualite-275/<dossier>/) diffère de
# l'alias GUI (casse d'origine du modèle) — les deux sont donnés ici, pas déduits.
CANDIDATS = [
    {"alias": "acvram-qwen3-coder-30b-a3b-qkvo-i8c-nvfp4", "ref_dossier": "Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c", "role": "MoE"},
    {"alias": "acvram-qwen3.8-27b-unsloth-mixte-i8c-nvfp4", "ref_dossier": "Qwen3.8-27B-unsloth-mixte-i8c", "role": "mixte"},
    {"alias": "acvram-qwen3-4b-srcgguf-nvfp4", "ref_dossier": "Qwen3-4B-srcgguf-nvfp4", "role": "dense"},
]
BRAS = "defaut"

# Durées observées le 26/09 sur la génération de référence mixte-27B (seul modèle
# chronométré poste par poste, timestamps mtime des sorties, aucun autre poste
# n'a ces marqueurs) — PPL et gsm8k/hsm/ccs sous le plafond ACVRAM_DUREE_MAX=1800s
# de carte.sh ; professional_law a expiré une fois à 1800s puis a fini sous 3600s
# (poste2-piece275-verdict-reference-mixte-26-09.md). Ce n'est PAS une mesure de ce
# run (bras=defaut contre une référence déjà écrite peut différer), c'est le seul
# repère chiffré disponible à sec — d'où une fourchette, pas un chiffre unique.
DUREE_PPL_S = (600, 1800)              # jamais chronométré au démarrage, borné par le plafond
DUREE_GSM8K_S = (600, 1800)
DUREE_HSM_S = (1500, 1800)
DUREE_CCS_S = (1200, 1800)
DUREE_LAW_S = (1500, 3600)             # seule tâche à avoir dépassé 1800s une fois
DUREE_MODELE_S = tuple(
    sum(v) for v in zip(DUREE_PPL_S, DUREE_GSM8K_S, DUREE_HSM_S, DUREE_CCS_S, DUREE_LAW_S)
)


def _menu_actuel() -> dict:
    """Alias → max_context_size depuis le config.toml réellement installé (source
    commune claude-modeles/kimi-modeles, PARC.config_kimi) — pas une copie locale."""
    import tomllib

    if not CONFIG_KIMI.exists():
        return {}
    conf = tomllib.load(CONFIG_KIMI.open("rb"))
    return {a: m.get("max_context_size", 0) for a, m in conf.get("models", {}).items()}


def _dossiers_tsv() -> dict:
    d = {}
    if not TSV_ACVRAM.exists():
        return d
    for ligne in TSV_ACVRAM.read_text().splitlines():
        c = ligne.split("\t")
        if len(c) >= 2:
            d[c[0]] = c[1]
    return d


def preparer():
    menu = _menu_actuel()
    dossiers = _dossiers_tsv()
    vus_dossiers = {}
    lignes = []
    for c in CANDIDATS:
        alias, ref_dossier = c["alias"], c["ref_dossier"]
        present = alias in menu
        ctx = menu.get(alias, 0)
        ref_ok = (CACHE_275 / ref_dossier / "ppl.json").exists() and (CACHE_275 / ref_dossier / "panel.json").exists()
        dossier = dossiers.get(alias, "?")
        doublon = vus_dossiers.get(dossier)
        vus_dossiers[dossier] = alias
        # hors champ = exclu des deux GUI faute de fenêtre : aucun des deux clients
        # n'a de plancher dur commun (claude-modele refuse sous 15096, kimi-modele
        # ne refuse jamais — poste3-gui-verdict-29-09.md) ; qualite.sh appelle
        # `acvram serve`/`acvram eval` directement, sans passer par un client, donc
        # ce seuil ne s'applique pas à la campagne — vérifié, pas supposé.
        hors_champ = present and ctx and ctx < 15096
        lignes.append({
            "alias": alias, "role": c["role"], "present": present, "ctx": ctx,
            "reference": ref_ok, "dossier": dossier,
            "doublon_de": doublon, "hors_champ": hors_champ,
        })
    return lignes


def simuler():
    lignes = preparer()
    print(f"=== campagne-qualite-275 --simule (HEAD {_head()}, bras={BRAS})")
    ok = True
    for l in lignes:
        etat = []
        if not l["present"]:
            etat.append("ABSENT DU MENU")
            ok = False
        if not l["reference"]:
            etat.append("SANS RÉFÉRENCE")
            ok = False
        if l["doublon_de"]:
            etat.append(f"DOUBLON de {l['doublon_de']}")
            ok = False
        if l["hors_champ"]:
            etat.append("HORS CHAMP")
            ok = False
        etat_txt = ", ".join(etat) if etat else "prêt"
        print(f"  {l['alias']:55s} [{l['role']:5s}] ctx={l['ctx']:>6} dossier={l['dossier']} -> {etat_txt}")
    lo, hi = DUREE_MODELE_S
    n = len(CANDIDATS)
    print(f"durée prédite : {lo*n/3600:.1f}-{hi*n/3600:.1f} h pour {n} modèles séquentiels "
          f"(1 GPU, carte.sh exclusive ; par modèle {lo/3600:.1f}-{hi/3600:.1f} h, "
          "voir DUREE_*_S pour la source)")
    print("aucune référence tierce ne se recouvre (3 dossiers distincts)" if not any(l["doublon_de"] for l in lignes) else "DOUBLONS DÉTECTÉS")
    print("PRÊTE" if ok else "BLOQUÉE — voir les lignes ci-dessus")
    return 0 if ok else 1


def _head() -> str:
    return subprocess.run(["git", "-C", str(RACINE), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, check=True).stdout.strip()


def _env():
    e = dict(os.environ)
    e.setdefault("ACVRAM_POSTE", "poste2")
    return e


def executer():
    if "ATTENDU" not in os.environ:
        print("REFUS : ATTENDU=<HEAD court> requis (même contrôle que qualite.sh)", file=sys.stderr)
        return 66
    lignes = preparer()
    for l in lignes:
        if not l["present"] or l["doublon_de"] or l["hors_champ"]:
            print(f"REFUS : {l['alias']} n'est pas prêt (rejoue --simule)", file=sys.stderr)
            return 66

    for l in lignes:
        if l["reference"]:
            continue
        ref_dossier = next(c["ref_dossier"] for c in CANDIDATS if c["alias"] == l["alias"])
        print(f"=== référence manquante {ref_dossier} — génération (5 prises)")
        r = subprocess.run(
            ["scratchpad/poste2-p275-26-09/generer-reference-v2.sh", ref_dossier],
            cwd=RACINE, env=_env(),
        )
        if r.returncode != 0:
            print(f"REFUS : génération de référence échouée pour {ref_dossier} (rc={r.returncode})", file=sys.stderr)
            return r.returncode

    for c in CANDIDATS:
        print(f"=== {c['alias']} (bras={BRAS})")
        r = subprocess.run(["outils/qualite.sh", c["alias"], BRAS], cwd=RACINE, env=_env())
        if r.returncode != 0:
            print(f"FAUX sur {c['alias']} (rc={r.returncode}) — arrêt, pas de suite en aveugle", file=sys.stderr)
            return r.returncode
    print("TENU sur les 3 modèles")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--simule", action="store_true", default=True)
    g.add_argument("--executer", action="store_true")
    p.add_argument("--je-sais-que-la-carte-est-libre", action="store_true", dest="confirme",
                   help="requis avec --executer, jamais déduit du programme")
    a = p.parse_args()
    if a.executer:
        if not a.confirme:
            print("REFUS : --executer exige --je-sais-que-la-carte-est-libre (REGLES §3, "
                  "carte.sh est la seule vérité mais ce script ne l'interroge pas lui-même)",
                  file=sys.stderr)
            sys.exit(66)
        sys.exit(executer())
    sys.exit(simuler())
