"""Pièce 257 : garde des liens avant publication — README.md et les 31 `docs/README.*.md`.

Casse si un lien relatif ou un `src` d'image pointe vers un fichier absent du dépôt (résolu depuis
le dossier du fichier qui le contient, `../` compris), ou si une ancre `#x` ne correspond à aucun
`<a id="x">` dans le fichier ciblé (le même fichier pour une ancre seule, `chemin#x`). Les liens
externes (`http://`, `https://`, `mailto:`) sont exclus : aucun réseau dans ce test.
"""
import re
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[1]
FICHIERS = [RACINE / "README.md"] + sorted((RACINE / "docs").glob("README.*.md"))

_LIEN = re.compile(r"\]\(([^)]+)\)")
_SRC = re.compile(r'src="([^"]+)"')
_ANCRE_DEF = re.compile(r'<a id="([^"]+)"')

_EXTERNE = ("http://", "https://", "mailto:")


def _ancres(texte: str) -> set[str]:
    return set(_ANCRE_DEF.findall(texte))


def _cibles(texte: str) -> list[str]:
    return _LIEN.findall(texte) + _SRC.findall(texte)


def _cas() -> list[tuple[Path, str]]:
    cas = []
    for f in FICHIERS:
        texte = f.read_text(encoding="utf-8")
        for cible in _cibles(texte):
            cas.append((f, cible))
    return cas


@pytest.mark.parametrize("fichier,cible", _cas(), ids=lambda v: str(v)[-60:] if isinstance(v, (str, Path)) else v)
def test_lien_resout(fichier, cible):
    if cible.startswith(_EXTERNE):
        pytest.skip("lien externe, hors périmètre (aucun réseau dans ce test)")

    chemin, _, ancre = cible.partition("#")

    if chemin == "":
        # ancre seule : doit exister dans CE fichier
        assert ancre in _ancres(fichier.read_text(encoding="utf-8")), (
            f"{fichier.relative_to(RACINE)} : ancre #{ancre} absente (aucun <a id=\"{ancre}\">)")
        return

    cible_resolue = (fichier.parent / chemin).resolve()
    assert cible_resolue.exists(), (
        f"{fichier.relative_to(RACINE)} : lien vers {chemin!r} — fichier absent ({cible_resolue})")

    if ancre:
        texte_cible = cible_resolue.read_text(encoding="utf-8")
        assert ancre in _ancres(texte_cible), (
            f"{fichier.relative_to(RACINE)} : lien vers {chemin}#{ancre} — ancre #{ancre} absente de {chemin}")


def test_au_moins_31_fichiers_verifies():
    assert len(FICHIERS) == 32, f"attendu README.md + 31 traductions, trouvé {len(FICHIERS)}"


def test_le_test_sait_dire_faux(tmp_path, monkeypatch):
    faux = tmp_path / "faux.md"
    faux.write_text("[cassé](chemin/qui/n/existe/pas.md)\n", encoding="utf-8")
    cible = _LIEN.findall(faux.read_text(encoding="utf-8"))[0]
    resolue = (faux.parent / cible).resolve()
    assert not resolue.exists(), "le fichier factice ne doit pas exister — sinon le test ne prouve rien"
