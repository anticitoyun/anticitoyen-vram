"""Pièce 263 (chef, 26/09) : aucune garde n'exigeait qu'une note de `acvram-memoire/revue/`
soit indexée dans `INDEX.md` — les notes 231 et 244 en sont sorties sans que rien ne le voie
(contre-lecture poste2). `INDEX.md` référence chaque note par son NOM DE FICHIER exact entre
parenthèses ; ce test le vérifie plutôt que de le supposer.

CLIQUET, pas un ménage complet (746 notes pré-24/09 n'ont jamais été indexées — les indexer
toutes est un chantier à part, hors de cette pièce) : le nombre de notes non indexées ne doit
JAMAIS AUGMENTER. Toute note ajoutée à partir d'aujourd'hui (26/09, 724 manquantes après les
22 lignes 213-262/25-26-09 de cette pièce) doit être indexée dans le même commit, sous peine
de faire remonter le compteur et casser ce test — descendre le seuil (en indexant du passif)
est encouragé, jamais l'inverse."""
import pathlib
import re

RACINE = pathlib.Path(__file__).resolve().parent.parent
REVUE = RACINE / "acvram-memoire" / "revue"
INDEX = REVUE / "INDEX.md"

# L'instantané public (outils/publier-github.sh) publie une partie des notes mais garde INDEX.md
# privé : sans index il n'y a rien à garder, et le module entier est sauté.
if not INDEX.exists():
    import pytest
    pytest.skip("acvram-memoire/revue/INDEX.md absent de cet arbre (instantané public)", allow_module_level=True)

# Cliquet : nombre de notes non indexées le 26/09, après la pièce 263 (213-262/25-26-09
# ajoutées). Baisser ce chiffre est bienvenu (l'indexation d'un passif) ; jamais le monter.
PLAFOND_NON_INDEXEES = 0


def _notes_du_dossier(revue_dir: pathlib.Path = REVUE) -> set[str]:
    return {p.name for p in revue_dir.glob("*.md") if p.name != "INDEX.md"}


def _notes_non_indexees(revue_dir: pathlib.Path, index_path: pathlib.Path) -> list[str]:
    """La fonction réellement exercée par les trois tests ci-dessous — pas réimplémentée deux
    fois (une version en dur dans le témoin aurait pu diverger de celle qui protège pour de
    vrai, et le témoin n'aurait alors rien prouvé sur la garde réelle)."""
    texte = index_path.read_text(encoding="utf-8")
    return sorted(n for n in _notes_du_dossier(revue_dir) if n not in texte)


def test_index_existe():
    assert INDEX.exists(), "acvram-memoire/revue/INDEX.md est introuvable"


def test_notes_non_indexees_ne_depasse_pas_le_plafond():
    manquantes = _notes_non_indexees(REVUE, INDEX)
    assert len(manquantes) <= PLAFOND_NON_INDEXEES, (
        f"{len(manquantes)} notes non indexees > plafond {PLAFOND_NON_INDEXEES} — "
        f"une note NEUVE n'a pas ete ajoutee a INDEX.md dans le meme commit "
        f"(nouvelles : {[m for m in manquantes if m not in _MANQUANTES_CONNUES_26_09][:10]})")


def test_une_note_neuve_non_indexee_est_bien_detectee(tmp_path):
    """Témoin, avec la MÊME fonction que la garde réelle (`_notes_non_indexees`) : une note
    présente sur disque mais absente d'INDEX.md doit être vue, jamais passer inaperçue."""
    revue = tmp_path / "revue"
    revue.mkdir()
    (revue / "connue-26-09.md").write_text("x")
    (revue / "orpheline-26-09.md").write_text("x")
    index = revue / "INDEX.md"
    index.write_text("- [une note connue](connue-26-09.md) — RAS\n", encoding="utf-8")
    assert _notes_non_indexees(revue, index) == ["orpheline-26-09.md"]


# Notes déjà connues comme manquantes au 26/09 (pré-24/09, hors périmètre de la 263) — sert
# uniquement à préciser le message d'erreur si le plafond est dépassé, jamais à les exempter :
# un jour où elles seraient indexées, cette liste devient obsolète et peut être vidée.
_MANQUANTES_CONNUES_26_09: set[str] = set(_notes_non_indexees(REVUE, INDEX)) if INDEX.exists() else set()
