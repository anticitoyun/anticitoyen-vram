"""Pièce 238 (poste2, à sec, ordre chef) : extension de `test_readme_traductions.py` (structure)
avec une garde qui compare directement chaque traduction au README FR sur trois invariants qu'une
traduction ne doit JAMAIS changer :

1. **les ancres** `<a id="...">` — jamais traduites, sinon la barre de langues et le sommaire des
   autres traductions visent une page morte ;
2. **les chiffres** — chaque nombre (français `12 345,67` à séparateur d'espace/virgule ou anglais
   `12,345.67`) doit apparaître, dans le MÊME ORDRE, avec la MÊME VALEUR NUMÉRIQUE, dans chaque
   traduction. Un chiffre recopié à l'identique passe ; un chiffre retraduit, arrondi ou pris dans
   le mauvais ordre de grandeur (virgule décimale prise pour un séparateur de milliers) casse ;
3. **les liens vers ce dépôt** — chemins relatifs (`../LICENSE`, `ARCHITECTURE.md`…) et ancres
   internes (`#resultats`…) doivent pointer sur la MÊME cible que dans le README FR (à la
   translation de dossier `docs/` → racine près, déjà vérifiée par le gabarit `docs/README.en.md`).

Ce fichier ne refait PAS les contrôles de structure de `test_readme_traductions.py` (blocs de code,
titres, tableaux) : il se concentre sur le contenu numérique et les cibles de liens, que ces
contrôles ne lisent pas."""
import re
from collections import Counter
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
README = RACINE / "README.md"
DOCS = RACINE / "docs"

LANGUES = {
    "ar", "bn", "ca", "cs", "da", "de", "el", "en", "eo", "es", "fa", "fi", "he", "hi", "hu", "id",
    "it", "ja", "ko", "nb", "nl", "pl", "pt", "ro", "ru", "sv", "th", "tr", "uk", "vi", "zh",
}

# Un chiffre, puis toute suite de (séparateur optionnel suivi IMMÉDIATEMENT d'un chiffre) — le
# séparateur (espace, virgule, point) ne prolonge le nombre que s'il est suivi d'un chiffre, jamais
# d'une espace ou d'une virgule : "43.447, 16,383" (EN, deux nombres séparés par ", ") ne fusionne
# pas, alors qu'un simple `\s` dans la classe de caractères les fusionnait (bogue trouvé en écrivant
# ce test). Exclu si collé à une lettre (INT4, NVFP4, FP8, E4M3, BF16, sm_120, k8v4…) : ce sont des
# noms de format/version, pas des chiffres mesurés — leur ORDRE dans la phrase change légitimement
# d'une langue à l'autre (« groupes de 128 de l'INT4 » / « INT4's groups of 128 »), ce qui n'est PAS
# ce que ce contrôle protège.
_LETTRE = r"A-Za-zÀ-ÖØ-öø-ÿ_"
# Exclu seulement si une lettre PRÉCÈDE (INT4, NVFP4, FP8, E4M3, BF16, sm_120, k8v4 — la lettre vient
# avant le chiffre). Une lettre qui SUIT n'exclut pas : plusieurs langues soudent un suffixe à un
# nombre (allemand « 128er-Gruppen », « 16er-Blöcke ») sans que ce soit un nom de format.
_NOMBRE = re.compile(rf"(?<![{_LETTRE}])\d(?:[ .,]?\d)*")


def _ancres(texte: str) -> list[str]:
    return re.findall(r'<a id="([^"]+)"></a>', texte)


def _normalise_nombre(brut: str) -> str:
    """Ne garde que les chiffres, en écartant TOUT séparateur (milliers ou décimal), quelle que
    soit la convention (français : espace milliers, virgule décimale ; anglais : virgule milliers,
    point décimal). `1 995,1` et `1,995.1` deviennent tous deux `19951` ; `40 000` et `40,000`
    deviennent tous deux `40000`. Ambigu entre deux CONVENTIONS d'écriture du MÊME nombre — jamais
    entre deux nombres différents : un chiffre changé change la séquence, une virgule/point/espace
    déplacé par la traduction ne change rien ici (ce n'est pas ce que ce contrôle protège)."""
    return re.sub(r"[^\d]", "", brut)


# Jour 01-31, MOIS TOUJOURS SUR 2 CHIFFRES (01-12, jamais "4" seul) -- sans le mois à 2 chiffres,
# "13.4" (13,4 %, pas une date) matche comme jour=13/mois=4 et disparaît à tort ; sans la borne
# 01-12, "1.44" (facteur) matche comme jour=1/mois=44 (bogues trouvés en écrivant ce test : EN
# perdait ses propres chiffres, alors que docs/README.en.md n'est même pas modifié par la 238).
_JOUR = r"0?[1-9]|[12]\d|3[01]"
_MOIS = r"0[1-9]|1[0-2]"
_MOIS1 = r"0?[1-9]|1[0-2]"          # 249 : mois sur 1 chiffre (finnois « 22.9.2026 », tchèque « 22. 9. 2026 »)
_DATE = re.compile(
    rf"\b(?:{_JOUR})[/.] ?(?:{_MOIS1})\.?(?: ?[/.]? ?\d{{4}})?\b"      # 22/09, 22.09, 22.9., 22. 9. 2026, 22/09/2026
    rf"|\b\d{{4}}[-.](?:{_MOIS})[-.](?:{_JOUR})\b"                       # 2026-09-22, 2026.09.22 (hongrois)
    rf"|\b(?:{_MOIS})\.(?:[0-2]\d|3[01])\b")                            # 09.22 (hongrois : « 09.22-i »)


# Le FR n'écrit ses dates qu'avec « / » ou en ISO : c'est cette forme stricte qui s'applique à la SOURCE
# (`garder` None) — la forme large ci-dessus, appliquée au FR, prendrait « 4.7 » (GLM-4.7-Flash) pour une date.
_DATE_FR = re.compile(rf"\b(?:{_JOUR})/(?:{_MOIS})(?:/\d{{4}})?\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b")


def _sans_dates(texte: str, garder: frozenset | None = None) -> str:
    """Les dates (22/09, 22/09/2026, 2026-09-22, et depuis la 249 les formes locales 22.9.2026, 22. 9. 2026,
    2026.09.22, 09.22) ne sont pas un « chiffre mesuré » au sens de ce contrôle : `docs/README.en.md`
    (référence, déjà acceptée par le chef) écrit certaines en ISO avec l'année et d'autres notes FR
    l'omettent (« Erratum du 22/09 » / une date complète ailleurs) — une convention de rédaction, pas
    une valeur qui a changé. Retirées avant l'extraction plutôt que réordonnées : le nombre de champs
    (jour/mois seuls ou avec année) diffère déjà entre les deux, pas seulement l'ordre.
    ``garder`` (249) : les nombres du FR, normalisés — un candidat-date dont les chiffres sont un nombre
    du FR est un NOMBRE écrit au point décimal (« 16.02 » Gio, « 9.1 » %), pas une date : aucune règle
    de forme ne sépare « 16.02 » de « 22.09 », le FR le fait (il n'écrit jamais 22.09, ni 16,02 en date)."""
    if garder is None:
        return _DATE_FR.sub("", texte)

    def rempl(m):
        return m.group(0) if re.sub(r"\D", "", m.group(0)) in garder else ""
    return _DATE.sub(rempl, texte)


# 249 : en japonais, chinois et coréen les grands nombres se comptent en 億 / 亿 / 억 (10⁸) — « 56 milliards »
# s'écrit « 560 億 ». Ramené au milliard (÷ 10) pour comparer au FR. Un nombre d'億 qui n'est PAS un multiple
# de 10 (« 56 億 » = 5,6 milliards) n'est pas une conversion d'unité mais un chiffre changé : il est rendu
# INVISIBLE au compte (lettre collée devant, que `_NOMBRE` exclut) pour que le « 56 » du FR manque — et non
# laissé tel quel, où « 56 億 » compterait comme le « 56 » du FR (témoin de test_l_elargissement_249_sait_dire_faux).
_OKU = re.compile(r"(\d+)\s*[億亿억]")


def _sans_oku(texte: str) -> str:
    return _OKU.sub(lambda m: str(int(m.group(1)) // 10) if int(m.group(1)) % 10 == 0 else "x" + m.group(0), texte)


def _sans_cibles(texte: str) -> str:
    """Les CIBLES de lien/image (`](url)`, `src="..."`, `href="..."`) portent des paramètres qui ne
    sont pas de la prose — l'URL du bouton « Buy Me a Coffee » encode son PROPRE texte (traduit :
    `text=Offrir%20un%20café` vs `text=Buy%20me%20a%20coffee`, `%20` répété un nombre de fois
    différent selon la longueur du texte) et des couleurs hexadécimales (`FFDD00`, `000000`) — aucun
    des deux n'est un chiffre mesuré. `test_les_cibles_de_liens_internes_sont_les_memes` vérifie déjà
    les cibles elles-mêmes ; ici on les retire pour ne garder que le texte visible."""
    texte = re.sub(r'\]\([^)]*\)', ']', texte)
    texte = re.sub(r'(?:src|href)="[^"]*"', '', texte)
    return texte


def _nombres(texte: str, garder: frozenset | None = None) -> list[str]:
    # hors blocs de code (une sortie de commande n'est pas de la prose ; déjà vérifiée AU BIT par
    # test_readme_traductions.py:_blocs_de_code — la revérifier ici ferait doublon, pas un défaut).
    sans_code = re.sub(r"```.*?```", "", _sans_cibles(_sans_dates(_sans_oku(texte), garder)), flags=re.S)
    return [_normalise_nombre(m.group(0)) for m in _NOMBRE.finditer(sans_code)]


def _cibles_liens_relatifs(texte: str) -> list[str]:
    """Cible de chaque lien/image vers CE dépôt (relatif ou ancre `#...`) — jamais une URL externe
    (github.com/…, buymeacoffee.com/…), qui peut légitimement différer (aucune ici ne le fait, mais
    ce n'est pas ce que ce contrôle protège)."""
    cibles = re.findall(r'(?:\]\(|src="|href=")([^)"]+)[)"]', texte)
    return sorted(c for c in cibles if not c.startswith(("http://", "https://")))


def _migree_au_nouveau_modele(texte: str) -> bool:
    """231 a réécrit README.md/docs/README.en.md sur le modèle animematrix (ancres `<a id=...>`) ;
    les 29 autres traductions se font au fil de l'eau (238 et suivantes) — une traduction qui n'a
    PAS encore ces ancres est sur l'ANCIEN modèle, hors périmètre de ce contrôle jusqu'à sa reprise
    (pas un défaut à signaler ici, juste pas encore fait)."""
    return '<a id="idees"></a>' in texte


def test_les_ancres_ne_sont_jamais_traduites():
    src_ancres = _ancres(README.read_text(encoding="utf-8"))
    assert len(src_ancres) >= 10, "gabarit du test obsolète : moins de 10 ancres dans README.md"
    for code in LANGUES:
        p = DOCS / f"README.{code}.md"
        if not p.exists():
            continue
        t = p.read_text(encoding="utf-8")
        if not _migree_au_nouveau_modele(t):
            continue
        assert _ancres(t) == src_ancres, (
            f"{code} : ancres différentes de README.md (traduites, réordonnées ou manquantes)")


def test_les_chiffres_sont_recopies_a_l_identique():
    """Multiset, pas séquence : une traduction réordonne parfois deux valeurs adjacentes dans la
    MÊME phrase (allemand « bei b=12 um 9,1 % » contre FR « 9,1 % à b=12 » — b=12 avant le
    pourcentage, l'inverse du FR) sans que ce soit un chiffre CHANGÉ. Ce que ce contrôle protège :
    qu'aucune valeur n'apparaisse, disparaisse ou change — pas l'ordre exact des clauses."""
    src = README.read_text(encoding="utf-8")
    src_nombres = _nombres(src)
    assert len(src_nombres) >= 30, "gabarit du test obsolète : moins de 30 chiffres dans README.md"
    src_compte = Counter(src_nombres)
    garder = frozenset(src_nombres)          # 249 : « 16.02 » n'est une date que si le FR n'a pas 16,02
    for code in LANGUES - {"en"}:
        # `docs/README.en.md` écrit ses décimales au point (« 16.02 ») ET certaines de ses dates
        # n'existent qu'avec année (ISO, jamais de DD.MM nu) — mais le FR écrit AUSSI des dates nues
        # au point-décimal-compatible dans d'autres traductions (allemand « 22.09 ») : aucune règle
        # syntaxique ne distingue de façon fiable un « 16.02 » mesuré d'un « 22.09 » daté sans plus
        # de contexte que la forme. EN n'est pas modifié par la 238 (déjà accepté, 231) : exclu de CE
        # contrôle plutôt que de complexifier `_DATE` pour un fichier hors périmètre.
        p = DOCS / f"README.{code}.md"
        if not p.exists():
            continue
        t = p.read_text(encoding="utf-8")
        if not _migree_au_nouveau_modele(t):
            continue
        t_compte = Counter(_nombres(t, garder))
        manquent = list((src_compte - t_compte).elements())
        en_trop = list((t_compte - src_compte).elements())
        assert not manquent and not en_trop, {
            "code": code, "chiffres manquants": sorted(manquent)[:6], "chiffres en trop": sorted(en_trop)[:6]}


_CIBLE_BARRE_LANGUES = re.compile(r"^README(?:\.\w+)?\.md$")


def _depuis_docs(c: str) -> str:
    """`docs/README.en.md` déplace les images (docs/logo→../docs/logo, captures/22-09→captures/22-09
    sans docs/) et les liens (`REPRISE.md`→`../REPRISE.md`) d'un niveau : translation attendue, pas
    une divergence — neutralisé avant comparaison."""
    return c.removeprefix("../").removeprefix("docs/")


def test_les_cibles_de_liens_internes_sont_les_memes():
    src_cibles = set(_cibles_liens_relatifs(README.read_text(encoding="utf-8")))
    for code in LANGUES:
        p = DOCS / f"README.{code}.md"
        if not p.exists():
            continue
        t_texte = p.read_text(encoding="utf-8")
        if not _migree_au_nouveau_modele(t_texte):
            continue
        t_cibles = set(_cibles_liens_relatifs(t_texte))
        # normaliser le déplacement docs/ D'ABORD, puis écarter la barre de langues : elle omet SA
        # PROPRE langue (gras, sans lien) et lie toutes les autres — chaque fichier a donc, par
        # construction, un ensemble de cibles README.*.md différent du FR ; pas une divergence à
        # détecter par CE contrôle.
        depuis_src = {_depuis_docs(c) for c in src_cibles
                      if not c.startswith("#") and not _CIBLE_BARRE_LANGUES.match(_depuis_docs(c))}
        depuis_t = {_depuis_docs(c) for c in t_cibles
                    if not c.startswith("#") and not _CIBLE_BARRE_LANGUES.match(_depuis_docs(c))}
        manquent = sorted(depuis_src - depuis_t)
        en_trop = sorted(depuis_t - depuis_src)
        assert not manquent and not en_trop, {
            "code": code, "cibles manquantes": manquent[:6], "cibles en trop": en_trop[:6]}
        # ancres internes (#resultats…) : identiques aux ancres du FR, jamais traduites
        ancres_src = {c for c in src_cibles if c.startswith("#")}
        ancres_t = {c for c in t_cibles if c.startswith("#")}
        assert ancres_t == ancres_src, {"code": code, "ancres de lien différentes": sorted(ancres_t ^ ancres_src)}


def test_le_test_sait_dire_faux(tmp_path):
    """Un chiffre modifié dans une copie doit casser `test_les_chiffres_sont_recopies_a_l_identique`
    — preuve que le contrôle ci-dessus n'est pas vide."""
    src = README.read_text(encoding="utf-8")
    altere = src.replace("1 995,1", "1 995,2", 1)
    assert Counter(_nombres(altere)) != Counter(_nombres(src)), (
        "l'altération n'a pas changé le multiset de chiffres — gabarit du test à revoir")


# ---- 249 : les dates aussi, et les témoins de l'élargissement ----------------------------------------------

def _dates_jour_mois(texte: str, garder: frozenset | None) -> Counter:
    """Multiset des (jour, mois) des dates du texte — l'année est omise parce que le FR l'omet lui-même
    dans ses notes (« Erratum du 22/09 ») là où d'autres langues l'écrivent (« 2026-09-22 »). Source
    (`garder` None) : forme stricte du FR ; traduction : forme large, hors nombres du FR (« 16.02 »)."""
    t = re.sub(r"```.*?```", "", _sans_cibles(texte), flags=re.S)
    out = []
    for m in (_DATE_FR if garder is None else _DATE).finditer(t):
        if garder is not None and re.sub(r"\D", "", m.group(0)) in garder:
            continue
        champs = re.findall(r"\d+", m.group(0))
        if len(champs[0]) == 4:                                   # 2026-09-22, 2026.09.22
            jour, mois = int(champs[2]), int(champs[1])
        elif len(champs) == 2 and len(champs[0]) == 2 and int(champs[0]) <= 12 < int(champs[1]):
            jour, mois = int(champs[1]), int(champs[0])           # 09.22 (hongrois, mois d'abord)
        else:
            jour, mois = int(champs[0]), int(champs[1])
        out.append((jour, mois))
    return Counter(out)


def test_les_dates_sont_les_memes():
    """Une date changée par la traduction est une erreur comme un chiffre changé — retirer les dates
    des chiffres (ci-dessus) ne doit pas les soustraire à tout contrôle. Comparées en (jour, mois),
    toutes langues, le FR comme référence."""
    src = README.read_text(encoding="utf-8")
    garder = frozenset(_nombres(src))
    src_dates = _dates_jour_mois(src, None)
    assert len(src_dates) >= 3, "gabarit du test obsolète : moins de 3 dates distinctes dans README.md"
    for code in LANGUES:
        p = DOCS / f"README.{code}.md"
        if not p.exists():
            continue
        t = p.read_text(encoding="utf-8")
        if not _migree_au_nouveau_modele(t):
            continue
        assert _dates_jour_mois(t, garder) == src_dates, {
            "code": code, "dates (jour, mois)": dict(_dates_jour_mois(t, garder)), "FR": dict(src_dates)}


def test_l_elargissement_249_sait_dire_faux():
    """REGLES n° 5 : l'élargissement de la 249 (dates locales, nombres du FR gardés, 億 ÷ 10) doit rester
    un contrôle — chaque témoin ci-dessous DOIT casser, sinon c'est un relâchement."""
    src = README.read_text(encoding="utf-8")
    garder = frozenset(_nombres(src))
    src_compte = Counter(_nombres(src))
    ja = (DOCS / "README.ja.md").read_text(encoding="utf-8")
    zh = (DOCS / "README.zh.md").read_text(encoding="utf-8")
    fi = (DOCS / "README.fi.md").read_text(encoding="utf-8")
    hu = (DOCS / "README.hu.md").read_text(encoding="utf-8")
    for nom, t in (("ja", ja), ("zh", zh), ("fi", fi), ("hu", hu)):
        assert Counter(_nombres(t, garder)) == src_compte, f"{nom} : le témoin part d'un fichier déjà rouge"
    # un débit altéré (312,3 → 321,3), en japonais et en chinois
    for nom, t in (("ja", ja), ("zh", zh)):
        assert t.count("312.3") >= 1, f"{nom} : gabarit du témoin obsolète (312.3 absent)"
        assert Counter(_nombres(t.replace("312.3", "321.3", 1), garder)) != src_compte, f"{nom} : débit altéré non vu"
    # 560 億 → 56 億 (ce n'est plus 56 milliards mais 5,6)
    assert ja.count("560 億") >= 1 and zh.count("560 亿") >= 1, "gabarit du témoin obsolète (560 億/亿 absent)"
    assert Counter(_nombres(ja.replace("560 億", "56 億", 1), garder)) != src_compte, "ja : 560 億 → 56 億 non vu"
    assert Counter(_nombres(zh.replace("560 亿", "56 亿", 1), garder)) != src_compte, "zh : 560 亿 → 56 亿 non vu"
    # une date réelle changée, dans une forme locale (finnois 22.9.2026, hongrois 2026.09.22)
    src_dates = _dates_jour_mois(src, None)
    assert fi.count("22.9.2026") >= 1 and hu.count("2026.09.22") >= 1, "gabarit du témoin obsolète (date absente)"
    assert _dates_jour_mois(fi.replace("22.9.2026", "21.9.2026", 1), garder) != src_dates, "fi : date changée non vue"
    assert _dates_jour_mois(hu.replace("2026.09.22", "2026.09.21", 1), garder) != src_dates, "hu : date changée non vue"
    # un « 16.02 » (nombre du FR au point décimal) altéré reste vu, même s'il ressemble à une date
    assert ja.count("16.02") >= 1
    assert Counter(_nombres(ja.replace("16.02", "16.03", 1), garder)) != src_compte, "ja : 16.02 → 16.03 non vu"
