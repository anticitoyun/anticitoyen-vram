"""Le README public existe dans les 31 langues de la GUI (docs/README.<code>.md) et le README
français propose un lien vers chaque traduction AVANT la ligne « Soutenir » (utilisateur 21/09).

Contrôles qui peuvent rendre faux : fichier absent, barre de langues absente ou incomplète ou placée
après « Soutenir », lien de soutien absent d'une traduction, structure divergente (titres, blocs de
code, tableaux) — un bloc de code traduit ou raccourci est un défaut.
"""
import collections
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


LOGO = '<p align="center"><img src="logo-acvram.png" alt="acvram" width="420"></p>'   # modèle d'avant la 231 (une ligne)

# 238 (poste2) : `docs/README.en.md` (référence acceptée par le chef) TRADUIT l'intérieur des blocs de code —
# commentaires, texte de démo (« Bonjour » → « Hello »), unités (Go → GB), séparateur décimal, en-têtes de colonnes.
# Seuls les COMMANDES et DRAPEAUX du contrat CLI ne changent jamais — comptés sur tout le fichier, en plus du
# contrôle par bloc de `_commandes_et_chiffres` (242).
# 249 : frontière ASCII (?![A-Za-z0-9_]) et non \b — en coréen, thaï ou chinois une particule se colle au nom
# (« acvram은 ») et \b, qui tient le hangul pour une lettre de mot, ne comptait plus la mention (ko : 35 contre 45).
_F = r"(?![A-Za-z0-9_])"
_JETONS_CLI = re.compile(
    r"acvram" + _F + r"|curl" + _F + r"|python" + _F + r"|pip" + _F + r"|install\.sh|--[\w-]+|-H" + _F + r"|-d" + _F +
    r"|OpenAI" + _F + r"|openai" + _F + r"|nvfp4" + _F + r"|int4_awq" + _F + r"|int4" + _F + r"|sm_120" + _F +
    r"|qwen3-32b" + _F + r"|8000" + _F)


def _jetons_cli(texte: str) -> list[str]:
    return _JETONS_CLI.findall(texte)


# 238 : modèle animematrix (231) — logo sur 3 lignes, chemin relatif à la position du fichier
# (`docs/logo-acvram.png` à la racine, `../docs/logo-acvram.png` depuis docs/), largeur 200.
def _logo(depuis_racine: bool) -> list[str]:
    chemin = "docs/logo-acvram.png" if depuis_racine else "../docs/logo-acvram.png"
    return ['<p align="center">', f'  <img src="{chemin}" alt="acvram" width="200">', "</p>"]


def _ligne_barre_langues(texte: str) -> str | None:
    """La ligne de la barre de langues : ni son marqueur (`🌐`, retiré au modèle 231) ni sa position
    fixe ne sont fiables — seuls les DEUX drapeaux 🇫🇷/🇬🇧 le sont (jamais traduits, présents dans
    CHAQUE fichier, source ou traduction)."""
    return next((l for l in texte.splitlines() if "🇫🇷" in l and "🇬🇧" in l), None)


def _titres(texte: str) -> int:
    return sum(1 for l in texte.splitlines() if re.match(r"^#{1,6} ", l))


def _lignes_de_tableau(texte: str) -> int:
    return sum(1 for l in texte.splitlines() if l.startswith("|"))


def test_les_31_langues_de_la_gui_sont_celles_du_readme():
    codes = sorted(p.stem for p in (RACINE / "packaging" / "langues").glob("*.json") if not p.stem.startswith("_"))
    assert codes == sorted(LANGUES), "la liste des langues du README doit être celle de packaging/langues/"


def test_la_barre_de_langues_precede_soutenir_dans_le_readme():
    """Style 231 (animematrix) : la barre est la ligne de drapeaux « **🇫🇷 Français** · [🇬🇧 English](docs/README.en.md) … »,
    avant la section « Soutenir le projet » (le badge Buy Me a Coffee des badges d'en-tête, lui, est au-dessus par dessein :
    on prend la DERNIÈRE occurrence du lien de soutien, 238)."""
    texte = README.read_text(encoding="utf-8")
    lignes = texte.splitlines()
    barre = _ligne_barre_langues(texte)
    assert barre is not None, "barre de drapeaux 🇫🇷/🇬🇧 (231) absente du README"
    i_barre = lignes.index(barre)
    i_soutien = max(i for i, l in enumerate(lignes) if SOUTIEN in l)
    assert i_barre < i_soutien, "la barre de langues doit précéder la section « Soutenir le projet »"
    assert "**🇫🇷 Français**" in barre, "le français, langue courante du README, doit être en gras et sans lien"
    for code, nom in LANGUES.items():
        # le nom est précédé d'un drapeau (`[🇸🇦 العربية]`) : ancré sur le drapeau ET sur `](`
        assert re.search(r"\[\S+ " + re.escape(nom) + r"\]\(docs/README\." + code + r"\.md\)", barre), f"lien manquant ou mal nommé : {code} ({nom})"


RTL = {"ar", "fa", "he"}                       # 242 : sens droite → gauche, règle d'animematrix (docs/readme/README.ar.md)
DEBUT_231 = '<p align="center">'                # 231 : logo en bloc <p>, badges, barre de drapeaux (la langue courante en gras)


def _commandes_et_chiffres(bloc: str) -> tuple:
    """Ce qui ne se traduit pas dans un bloc de code : les commandes (lignes qui commencent par $, ./, curl, acvram, from,
    client., -H, -d) sans leur commentaire, et tous les nombres. Les commentaires et les libellés de sortie peuvent être
    traduits (modèle 231) ; une commande changée, un chiffre changé ou une ligne en moins est un défaut."""
    lignes = bloc.splitlines()
    cmds = []
    for l in lignes:
        if re.match(r"^\s*(\$ |\./|curl |acvram |from |client\.|-H |-d |import )", l):
            sans_commentaire = l.split(" #", 1)[0]
            mots = sans_commentaire.split()
            # programme (et sous-commande acvram) + options : fixes ; les chemins d'exemple (~/modeles → ~/models) et les
            # commentaires se traduisent (modèle 231, README.en.md)
            cmds.append([m for i, m in enumerate(mots) if i == 0 or m.startswith("-") or (mots[0] in ("$", "acvram") and i == 1)])
    # nombres à séparateurs locaux (30,3 / 30.3 / 39 843 / 39,843) : comparés sur leurs chiffres seuls
    chiffres = [re.sub(r"\D", "", n) for n in re.findall(r"\d+(?:[.,\s\u202f\u00a0]\d+)*", bloc)]
    return (len(lignes), cmds, chiffres)


def _verifier_231(code: str, t: str, src: str) -> None:
    lignes = t.splitlines()
    assert lignes[0:3] == _logo(depuis_racine=False), f"{code} : logo officiel (3 lignes, ../docs/logo-acvram.png, 238) absent ou différent en tête"
    assert lignes[4].startswith("# "), f"{code} : pas de titre sous le logo"
    assert collections.Counter(_jetons_cli(t)) == collections.Counter(_jetons_cli(src)), (
        f"{code} : commandes/drapeaux CLI différents (comptés sur tout le fichier, jamais traduits)")
    assert "img.shields.io" in t[:2000], f"{code} : badges absents"
    barre = [l for l in lignes[:25] if "🇫🇷" in l and "README.md" in l]
    assert barre, f"{code} : barre de drapeaux (retour vers README.md) absente"
    assert f"**{'🇬🇧' if code == 'en' else ''}" in barre[0] or re.search(r"\*\*[^*]*" + re.escape(LANGUES[code]) + r"\*\*", barre[0]), \
        f"{code} : la langue courante doit être en gras et sans lien dans la barre"
    assert f"README.{code}.md" not in barre[0].split("**")[1] if barre[0].count("**") >= 2 else True
    i_barre = next(i for i, l in enumerate(lignes) if l == barre[0])
    if code in RTL:
        # animematrix (docs/readme/README.ar.md) : la barre reste dans son <div align="center"> LTR, fermé d'abord ;
        # le corps s'ouvre ensuite, et seulement ensuite, par <div dir="rtl">
        apres = [l for l in lignes[i_barre + 1:i_barre + 6] if l.strip()]
        assert len(apres) >= 2 and apres[0] == "</div>" and apres[1] == '<div dir="rtl">', \
            f"{code} : après la barre de langues (LTR, fermée par </div>), le corps doit s'ouvrir par <div dir=\"rtl\">"
        fin = [l for l in lignes if l.strip()][-1]
        assert fin == "</div>", f"{code} : <div dir=\"rtl\"> non fermé à la dernière ligne"
        assert t.count('<div dir="rtl">') == 1, f"{code} : une seule balise RTL, autour du corps entier"
    else:
        assert 'dir="rtl"' not in t, f"{code} : balise RTL sur une langue écrite de gauche à droite"


def test_chaque_traduction_existe_et_garde_la_structure_du_readme():
    src = README.read_text(encoding="utf-8")
    blocs, titres, tab = _blocs_de_code(src), _titres(src), _lignes_de_tableau(src)
    for code in LANGUES:
        p = DOCS / f"README.{code}.md"
        assert p.exists(), f"docs/README.{code}.md absent"
        t = p.read_text(encoding="utf-8")
        assert SOUTIEN in t, f"{code} : lien de soutien absent"
        lignes = t.splitlines()
        if lignes[0] == DEBUT_231:
            _verifier_231(code, t, src)
            assert _titres(t) == titres, f"{code} : nombre de titres {_titres(t)} ≠ {titres}"
            assert _lignes_de_tableau(t) == tab, f"{code} : lignes de tableau {_lignes_de_tableau(t)} ≠ {tab}"
            bt = _blocs_de_code(t)
            assert len(bt) == len(blocs), f"{code} : {len(bt)} blocs de code ≠ {len(blocs)}"
            for k, (a, b) in enumerate(zip(blocs, bt)):
                assert _commandes_et_chiffres(a) == _commandes_et_chiffres(b), f"{code} : bloc de code {k} — commande, chiffre ou ligne changés"
        else:
            # style d'avant la 231 (lots 2 non encore refaits) : contrat d'origine, inchangé
            assert lignes[0] == LOGO, f"{code} : logo officiel absent ou différent en tête"
            assert lignes[2].startswith("#"), f"{code} : pas de titre sous le logo"
            assert any(l.startswith("🌐") and "README.md" in l for l in lignes[:8]), f"{code} : barre de langues (retour vers README.md) absente en tête"
            assert 'dir="rtl"' not in t or code in RTL
        assert len(t) > 0.5 * len(src), f"{code} : traduction trop courte ({len(t)} o contre {len(src)})"


def test_les_langues_rtl_sont_celles_d_animematrix():
    assert RTL == {"ar", "fa", "he"}


def test_le_readme_source_a_le_logo_et_le_titre_au_nouveau_modele():
    """238 : le FR (racine) suit le même modèle que les traductions, chemin `docs/logo-acvram.png` (pas
    `../docs/`) — bras cassant : sans ce contrôle, une régression du FR vers l'ancien modèle ne
    serait jamais vue (la boucle ci-dessus ne teste que les traductions)."""
    lignes = README.read_text(encoding="utf-8").splitlines()
    assert lignes[0:3] == _logo(depuis_racine=True), "README.md : logo absent ou différent du modèle 231"
    assert any(l.startswith("#") for l in lignes[3:6]), "README.md : pas de titre sous le logo"
