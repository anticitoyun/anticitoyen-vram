#!/usr/bin/env python3
"""Dédoublonne les alias acvram qui pointent le MÊME dossier converti.

t5e (poste1, 27/09, verdict `revue/poste1-t5e-verdict-27-09.md` §11) : le
renommage d'une conversion (`renommer-convertis.py`) AJOUTE un alias
supplémentaire portant le nom corrigé et GARDE l'ancien — voulu (les scripts
qui référencent déjà l'ancien nom continuent de marcher), mais les deux
restent alors dans `acvram-chemins.tsv` ET dans `config.toml`, donc dans le
menu claude-modeles/kimi-modeles : deux lignes pour un seul modèle
(« …-srci1q4km-… » et « …-srcq4-k-m-… », même sha256, huihui et kat-coder).

Ce script ne fusionne QUE des lignes dont le DOSSIER (2e colonne du TSV) est
identique au caractère près, jamais par ressemblance de nom : deux alias
dont les dossiers diffèrent restent deux alias, même s'ils se ressemblent
(« srcq4km » et « srcq5k » ne sont jamais confondus, dossiers différents).
Dans chaque groupe, l'alias gardé est celui que produirait AUJOURD'HUI la
même règle que `renommer-convertis.py` (`acvram-<basename slugifié>`) ; à
défaut de correspondance, le dernier par ordre alphabétique (le nom le plus
récent/complet, en pratique). Les autres sont retirés du TSV et, s'ils ont
une entrée `[models.<alias>]`, de `config.toml` — sans réécriture TOML
(tomlkit absent des dépendances) : suppression du bloc au caractère près,
rien d'autre ne bouge.

Écrit un journal de retour AVANT d'agir. --appliquer pour exécuter.
"""
import re
import sys
from pathlib import Path

ALIAS_TSV_PAR_DEFAUT = Path.home() / ".kimi-code" / "acvram-chemins.tsv"
CONFIG_PAR_DEFAUT = Path.home() / ".kimi-code" / "config.toml"


def canonique(dossier):
    """Même règle que la « na » de renommer-convertis.py : c'est elle qui
    écrit les alias neufs, donc c'est elle qui dit lequel est le nom actuel."""
    base = Path(dossier.rstrip("/")).name
    return "acvram-" + re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")


def lire_tsv(chemin):
    """Rend les lignes brutes (texte, commentaires et vides compris) et un
    dict alias -> [dossier, ...reste]."""
    lignes = chemin.read_text().splitlines() if chemin.exists() else []
    entrees = {}
    for ligne in lignes:
        if not ligne.strip() or ligne.startswith("#"):
            continue
        c = ligne.split("\t")
        if len(c) >= 2:
            entrees[c[0]] = c[1:]
    return lignes, entrees


def plan_doublons(entrees):
    """Groupe par dossier EXACT (colonne 2) ; rend {alias_garde: [alias_retires...]}."""
    par_dossier = {}
    for alias, reste in entrees.items():
        par_dossier.setdefault(reste[0], []).append(alias)
    plan = {}
    for dossier, alias_liste in par_dossier.items():
        if len(alias_liste) < 2:
            continue
        canon = canonique(dossier)
        garde = canon if canon in alias_liste else sorted(alias_liste)[-1]
        retires = [a for a in alias_liste if a != garde]
        plan[garde] = retires
    return plan


def retirer_du_toml(texte, alias_a_retirer):
    """Retire chaque table [models.<alias>] (jusqu'au prochain [...] en tête
    de ligne ou la fin), sans toucher au reste. Ni tomllib (lecture seule)
    ni tomlkit (absent) : une réécriture TOML complète reformaterait des
    commentaires écrits à la main que ce script n'a pas à connaître."""
    entetes_a_retirer = {f"[models.{a}]" for a in alias_a_retirer}
    sortie = []
    saute = False
    for ligne in texte.splitlines(keepends=True):
        entete = ligne.strip()
        if entete.startswith("[") and entete.endswith("]"):
            saute = entete in entetes_a_retirer
        if not saute:
            sortie.append(ligne)
    return "".join(sortie)


def executer(tsv, config, appliquer, journal):
    lignes, entrees = lire_tsv(tsv)
    plan = plan_doublons(entrees)
    if not plan:
        return 0, plan
    with journal.open("w") as f:
        f.write("garde\tretire\tdossier\n")
        for garde, retires in plan.items():
            for r in retires:
                f.write(f"{garde}\t{r}\t{entrees[r][0]}\n")
    if not appliquer:
        return sum(len(v) for v in plan.values()), plan
    a_retirer = {r for retires in plan.values() for r in retires}
    lignes_gardees = [l for l in lignes if l.split("\t", 1)[0] not in a_retirer]
    tsv.write_text("\n".join(lignes_gardees) + ("\n" if lignes_gardees else ""))
    if config.exists():
        config.write_text(retirer_du_toml(config.read_text(), a_retirer))
    return len(a_retirer), plan


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    appliquer = "--appliquer" in argv
    total, plan = executer(ALIAS_TSV_PAR_DEFAUT, CONFIG_PAR_DEFAUT, appliquer,
                           Path(__file__).with_name("dedoublonnages.tsv"))
    if not plan:
        print("aucun doublon (même dossier, deux alias)")
        return 0
    print(f"{total} alias en doublon (sur {len(plan)} dossiers) ; "
          f"journal : {Path(__file__).with_name('dedoublonnages.tsv')}")
    if not appliquer:
        print("(essai à blanc — relancer avec --appliquer)")
    else:
        print(f"{total} alias retirés (TSV"
              f"{' + config.toml' if CONFIG_PAR_DEFAUT.exists() else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
