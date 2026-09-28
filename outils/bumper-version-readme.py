#!/usr/bin/env python3
"""Passe le numéro de version dans README.md § État ET ses 31 traductions, d'un
seul coup — demandé par chef (28/09, pièce 295) : `outils/sortir-version.sh`
refuse désormais si README.md ne porte pas la version qu'on sort (code 74), donc
ce geste est à refaire à CHAQUE sortie, jamais un seul fichier.

Ne touche QU'AU numéro de version, dans la section État (entre <a id="etat"></a>
et l'ancre suivante) — jamais un autre chiffre ancré (dates de mesure, débits,
contexte tenu…), jamais le mot « Version »/« Versión »/« Versione »… qui reste
dans la langue de chaque traduction. Le numéro ANCIEN doit apparaître EXACTEMENT
UNE FOIS dans cette section ; sinon REFUS nommé (fichier, nombre d'occurrences)
plutôt qu'une réécriture au hasard.

    outils/bumper-version-readme.py 0.7.10 0.7.11              # essai à blanc
    outils/bumper-version-readme.py 0.7.10 0.7.11 --appliquer  # écrit

Écrit un journal de retour AVANT d'agir (même convention que renommer-convertis.py
et dedoublonner-alias.py)."""
import re
import sys
from pathlib import Path

DEPOT = Path(__file__).resolve().parent.parent
DOCS = DEPOT / "docs"
README = DEPOT / "README.md"

LANGUES = ("ar", "bn", "ca", "cs", "da", "de", "el", "en", "eo", "es", "fa", "fi", "he", "hi",
           "hu", "id", "it", "ja", "ko", "nb", "nl", "pl", "pt", "ro", "ru", "sv", "th", "tr",
           "uk", "vi", "zh")

_ANCRE_ETAT = re.compile(r'<a id="etat"></a>')
_ANCRE_SUIVANTE = re.compile(r'<a id="[^"]+"></a>')


def _bloc_etat(texte: str) -> tuple[int, int] | None:
    """(début, fin) du contenu entre l'ancre État et la suivante — None si absente
    (traduction pas encore migrée au modèle à ancres, hors périmètre)."""
    m = _ANCRE_ETAT.search(texte)
    if not m:
        return None
    suite = _ANCRE_SUIVANTE.search(texte, m.end())
    fin = suite.start() if suite else len(texte)
    return m.end(), fin


def fichiers(depot: Path = DEPOT) -> list[Path]:
    docs = depot / "docs"
    return [depot / "README.md"] + [docs / f"README.{c}.md" for c in LANGUES if (docs / f"README.{c}.md").exists()]


def plan_bascule(ancienne: str, nouvelle: str, chemins: list[Path] | None = None
                  ) -> tuple[dict[Path, str], list[tuple[Path, str]]]:
    """Rend {fichier: nouveau_texte} pour chaque fichier OK, et une liste de
    (fichier, raison) pour chaque refus (0 ou ≥2 occurrences dans le bloc État,
    ou bloc État absent). `chemins` par défaut : README.md + les 31 traductions
    du dépôt réel ; surchargeable pour les tests (fichiers jetables)."""
    plan, refus = {}, []
    for f in (chemins if chemins is not None else fichiers()):
        texte = f.read_text(encoding="utf-8")
        bloc = _bloc_etat(texte)
        if bloc is None:
            refus.append((f, "aucune section État (ancre etat absente) — pas encore migrée, hors périmètre"))
            continue
        debut, fin = bloc
        n = texte.count(ancienne, debut, fin)
        if n == 0:
            refus.append((f, f"« {ancienne} » absente de la section État"))
            continue
        if n > 1:
            refus.append((f, f"« {ancienne} » apparaît {n} fois dans la section État — ambigu"))
            continue
        nouveau_texte = texte[:debut] + texte[debut:fin].replace(ancienne, nouvelle, 1) + texte[fin:]
        plan[f] = nouveau_texte
    return plan, refus


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    positionnels = [a for a in argv if not a.startswith("--")]
    if len(positionnels) != 2:
        print("usage : bumper-version-readme.py <ancienne> <nouvelle> [--appliquer]", file=sys.stderr)
        return 64
    ancienne, nouvelle = positionnels
    appliquer = "--appliquer" in argv

    plan, refus = plan_bascule(ancienne, nouvelle)
    journal = Path(__file__).with_name("bumper-version-readme.journal.tsv")
    with journal.open("w") as j:
        j.write("fichier\tstatut\n")
        for f in plan:
            j.write(f"{f.relative_to(DEPOT)}\tOK\n")
        for f, raison in refus:
            j.write(f"{f.relative_to(DEPOT)}\tREFUS : {raison}\n")

    for f, raison in refus:
        print(f"REFUS : {f.relative_to(DEPOT)} — {raison}", file=sys.stderr)
    print(f"{len(plan)} fichier(s) {'à passer' if not appliquer else 'passés'} de {ancienne} à {nouvelle} "
          f"({len(refus)} refus) ; journal : {journal}")
    if not appliquer:
        print("(essai à blanc — relancer avec --appliquer)")
        return 1 if refus else 0

    for f, nouveau_texte in plan.items():
        f.write_text(nouveau_texte, encoding="utf-8")
    return 1 if refus else 0


if __name__ == "__main__":
    raise SystemExit(main())
