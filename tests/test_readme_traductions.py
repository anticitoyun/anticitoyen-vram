"""Le README public existe dans les 31 langues de la GUI (docs/README.<code>.md) et le README
français propose un lien vers chaque traduction AVANT la ligne « Soutenir » (utilisateur 21/09).

Contrôles qui peuvent rendre faux : fichier absent, barre de langues absente ou incomplète ou placée
après « Soutenir », lien de soutien absent d'une traduction, structure divergente (titres, blocs de
code, tableaux) — un bloc de code traduit ou raccourci est un défaut.
"""
import re
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
README = RACINE / "README.md"
DOCS = RACINE / "docs"
SOUTIEN = "buymeacoffee.com/anticitoyen"

# code → nom natif (la barre de langues affiche le nom natif, jamais le nom français)
LANGUES = {
    "ar": "العربية", "bn": "বাংলা", "ca": "Català", "cs": "Čeština", "da": "Dansk", "de": "Deutsch",
    "el": "Ελληνικά", "en": "English", "eo": "Esperanto", "es": "Español", "fa": "فارسی", "fi": "Suomi",
    "he": "עברית", "hi": "हिन्दी", "hu": "Magyar", "id": "Bahasa Indonesia", "it": "Italiano", "ja": "日本語",
    "ko": "한국어", "nb": "Norsk bokmål", "nl": "Nederlands", "pl": "Polski", "pt": "Português", "ro": "Română",
    "ru": "Русский", "sv": "Svenska", "th": "ไทย", "tr": "Türkçe", "uk": "Українська", "vi": "Tiếng Việt", "zh": "中文",
}


def _blocs_de_code(texte: str) -> list[str]:
    return re.findall(r"```.*?```", texte, re.S)


LOGO = '<p align="center"><img src="logo-acvram.png" alt="acvram" width="420"></p>'


def _titres(texte: str) -> int:
    return sum(1 for l in texte.splitlines() if re.match(r"^#{1,6} ", l))


def _lignes_de_tableau(texte: str) -> int:
    return sum(1 for l in texte.splitlines() if l.startswith("|"))


def test_les_31_langues_de_la_gui_sont_celles_du_readme():
    codes = sorted(p.stem for p in (RACINE / "packaging" / "langues").glob("*.json") if not p.stem.startswith("_"))
    assert codes == sorted(LANGUES), "la liste des langues du README doit être celle de packaging/langues/"


def test_la_barre_de_langues_precede_soutenir_dans_le_readme():
    lignes = README.read_text(encoding="utf-8").splitlines()
    i_barre = next((i for i, l in enumerate(lignes) if l.startswith("🌐")), None)
    i_soutien = next(i for i, l in enumerate(lignes) if SOUTIEN in l)
    assert i_barre is not None, "barre de langues « 🌐 … » absente du README"
    assert i_barre < i_soutien, "la barre de langues doit précéder la ligne « Soutenir »"
    barre = lignes[i_barre]
    for code, nom in LANGUES.items():
        assert f"[{nom}](docs/README.{code}.md)" in barre, f"lien manquant ou mal nommé : {code} ({nom})"


def test_chaque_traduction_existe_et_garde_la_structure_du_readme():
    src = README.read_text(encoding="utf-8")
    blocs, titres, tab = _blocs_de_code(src), _titres(src), _lignes_de_tableau(src)
    for code in LANGUES:
        p = DOCS / f"README.{code}.md"
        assert p.exists(), f"docs/README.{code}.md absent"
        t = p.read_text(encoding="utf-8")
        assert SOUTIEN in t, f"{code} : lien de soutien absent"
        # 21/09 utilisateur : le logo officiel ouvre chaque README, centré, AVANT le titre.
        lignes = t.splitlines()
        assert lignes[0] == LOGO, f"{code} : logo officiel absent ou différent en tête"
        assert lignes[2].startswith("#"), f"{code} : pas de titre sous le logo"
        assert any(l.startswith("🌐") and "README.md" in l for l in t.splitlines()[:8]), f"{code} : barre de langues (retour vers README.md) absente en tête"
        assert _blocs_de_code(t) == blocs, f"{code} : blocs de code modifiés ou manquants (ils ne se traduisent pas)"
        assert _titres(t) == titres, f"{code} : nombre de titres {_titres(t)} ≠ {titres}"
        assert _lignes_de_tableau(t) == tab, f"{code} : lignes de tableau {_lignes_de_tableau(t)} ≠ {tab}"
        assert len(t) > 0.5 * len(src), f"{code} : traduction trop courte ({len(t)} o contre {len(src)})"
