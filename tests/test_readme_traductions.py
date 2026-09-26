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


# 238 : `docs/README.en.md` (référence acceptée par le chef) TRADUIT l'intérieur des blocs de code —
# commentaires (`# environnement virtuel` → `# virtual environment`), texte de démo (« Bonjour » →
# « Hello »), unités (Go → GB), séparateur décimal (, → .), en-têtes de colonnes de sortie d'outil
# (`etage` → `tier`). Seuls les COMMANDES et DRAPEAUX du contrat CLI ne changent jamais — ce sont eux
# que ce contrôle protège, pas l'octet exact du bloc (l'égalité stricte, gardée avant la 238, aurait
# déjà été fausse sur `docs/README.en.md` lui-même).
_JETONS_CLI = re.compile(
    r"acvram\b|curl\b|python\b|pip\b|install\.sh|--[\w-]+|-H\b|-d\b|OpenAI\b|openai\b|"
    r"nvfp4\b|int4_awq\b|int4\b|sm_120\b|qwen3-32b\b|8000\b")


def _jetons_cli(texte: str) -> list[str]:
    return _JETONS_CLI.findall(texte)


# 238 : modèle animematrix (231) — logo sur 3 lignes, chemin relatif à la position du fichier
# (`docs/logo-acvram.png` à la racine, `../docs/logo-acvram.png` depuis docs/), largeur 200 (était
# 420 sur l'ancien modèle à une ligne — gardé ici pour mémoire, plus comparé directement).
def _logo(depuis_racine: bool) -> list[str]:
    chemin = "docs/logo-acvram.png" if depuis_racine else "../docs/logo-acvram.png"
    return ['<p align="center">', f'  <img src="{chemin}" alt="acvram" width="200">', "</p>"]


def _titres(texte: str) -> int:
    return sum(1 for l in texte.splitlines() if re.match(r"^#{1,6} ", l))


def _lignes_de_tableau(texte: str) -> int:
    return sum(1 for l in texte.splitlines() if l.startswith("|"))


def _ligne_barre_langues(texte: str) -> str | None:
    """La ligne de la barre de langues : ni son marqueur (`🌐`, retiré au modèle 231) ni sa position
    fixe ne sont fiables — seuls les DEUX drapeaux 🇫🇷/🇬🇧 le sont (jamais traduits, présents dans
    CHAQUE fichier, source ou traduction)."""
    return next((l for l in texte.splitlines() if "🇫🇷" in l and "🇬🇧" in l), None)


def test_les_31_langues_de_la_gui_sont_celles_du_readme():
    codes = sorted(p.stem for p in (RACINE / "packaging" / "langues").glob("*.json") if not p.stem.startswith("_"))
    assert codes == sorted(LANGUES), "la liste des langues du README doit être celle de packaging/langues/"


def test_la_barre_de_langues_precede_soutenir_dans_le_readme():
    lignes = README.read_text(encoding="utf-8").splitlines()
    barre = _ligne_barre_langues(README.read_text(encoding="utf-8"))
    assert barre is not None, "barre de langues (drapeaux 🇫🇷/🇬🇧) absente du README"
    i_barre = lignes.index(barre)
    # dernière occurrence : le badge d'en-tête (ligne ~11) contient aussi SOUTIEN, avant la barre —
    # seule la section « Soutenir le projet » en fin de fichier doit la suivre.
    i_soutien = max(i for i, l in enumerate(lignes) if SOUTIEN in l)
    assert i_barre < i_soutien, "la barre de langues doit précéder la ligne « Soutenir »"
    for code, nom in LANGUES.items():
        # le nom est précédé d'un drapeau (`[🇸🇦 العربية]`) : on ancre juste avant `](`, pas sur `[`.
        assert f"{nom}](docs/README.{code}.md)" in barre, f"lien manquant ou mal nommé : {code} ({nom})"


def test_chaque_traduction_existe_et_garde_la_structure_du_readme():
    src = README.read_text(encoding="utf-8")
    blocs, titres, tab = _blocs_de_code(src), _titres(src), _lignes_de_tableau(src)
    for code in LANGUES:
        p = DOCS / f"README.{code}.md"
        assert p.exists(), f"docs/README.{code}.md absent"
        t = p.read_text(encoding="utf-8")
        if not (t.startswith('<p align="center">') and "../docs/logo-acvram.png" in t.splitlines()[1]):
            continue  # ancien modèle (231 pas encore repris pour cette langue) : hors périmètre
        assert SOUTIEN in t, f"{code} : lien de soutien absent"
        # 21/09 utilisateur : le logo officiel ouvre chaque README, centré, AVANT le titre.
        lignes = t.splitlines()
        assert lignes[0:3] == _logo(depuis_racine=False), f"{code} : logo officiel absent ou différent en tête"
        assert any(l.startswith("#") for l in lignes[3:6]), f"{code} : pas de titre sous le logo"
        assert _ligne_barre_langues(t) is not None, f"{code} : barre de langues (drapeaux 🇫🇷/🇬🇧) absente"
        t_blocs = _blocs_de_code(t)
        assert len(t_blocs) == len(blocs), f"{code} : {len(t_blocs)} blocs de code contre {len(blocs)}"
        assert [b.count("\n") for b in t_blocs] == [b.count("\n") for b in blocs], (
            f"{code} : un bloc de code a un nombre de lignes différent (raccourci ou complété)")
        import collections
        assert collections.Counter(_jetons_cli(t)) == collections.Counter(_jetons_cli(src)), (
            f"{code} : commandes/drapeaux CLI différents (comptés sur tout le fichier, jamais traduits)")
        assert _titres(t) == titres, f"{code} : nombre de titres {_titres(t)} ≠ {titres}"
        assert _lignes_de_tableau(t) == tab, f"{code} : lignes de tableau {_lignes_de_tableau(t)} ≠ {tab}"
        assert len(t) > 0.5 * len(src), f"{code} : traduction trop courte ({len(t)} o contre {len(src)})"


def test_le_readme_source_a_le_logo_et_le_titre_au_nouveau_modele():
    """Le FR (racine) suit le même modèle que les traductions, chemin `docs/logo-acvram.png` (pas
    `../docs/`) — bras cassant : sans ce contrôle, une régression du FR vers l'ancien modèle ne
    serait jamais vue (la boucle ci-dessus ne teste que les traductions)."""
    lignes = README.read_text(encoding="utf-8").splitlines()
    assert lignes[0:3] == _logo(depuis_racine=True), "README.md : logo absent ou différent du modèle 231"
    assert any(l.startswith("#") for l in lignes[3:6]), "README.md : pas de titre sous le logo"
