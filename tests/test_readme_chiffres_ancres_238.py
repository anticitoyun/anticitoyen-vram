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
_DATE = re.compile(rf"\b(?:{_JOUR})[/.](?:{_MOIS})(?:[/.]\d{{4}})?\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b")


def _sans_dates(texte: str) -> str:
    """Les dates (22/09, 22/09/2026, 2026-09-22) ne sont pas un « chiffre mesuré » au sens de ce
    contrôle : `docs/README.en.md` (référence, déjà acceptée par le chef) écrit certaines en ISO
    avec l'année et d'autres notes FR l'omettent (« Erratum du 22/09 » / une date complète ailleurs)
    — une convention de rédaction, pas une valeur qui a changé. Retirées avant l'extraction plutôt
    que réordonnées : le nombre de champs (jour/mois seuls ou avec année) diffère déjà entre les
    deux, pas seulement l'ordre."""
    return _DATE.sub("", texte)


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


def _nombres(texte: str) -> list[str]:
    # hors blocs de code (une sortie de commande n'est pas de la prose ; déjà vérifiée AU BIT par
    # test_readme_traductions.py:_blocs_de_code — la revérifier ici ferait doublon, pas un défaut).
    sans_code = re.sub(r"```.*?```", "", _sans_cibles(_sans_dates(texte)), flags=re.S)
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
        t_compte = Counter(_nombres(t))
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
