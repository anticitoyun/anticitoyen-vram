"""Pièce 251 (poste2, à sec, ordre chef) : contre-lecture de `docs/notes/v0.7.0.md`, étendue en 280
(poste4) à `v0.7.1.md`, `v0.7.2.md` et `v0.7.3.md` — même contrôle, un fichier de plus par version livrée.

Trois contrôles qui peuvent rendre FAUX :

1. un sha cité dans les notes n'existe pas dans le dépôt (`git cat-file -e`) ;
2. un numéro de pièce cité n'a AUCUNE note de verdict dans `acvram-memoire/revue/` ni
   `revue/`, cherchée dans TOUT l'historique connu (`git log --all`), pas seulement la
   branche courante — une pièce mesurée sur une branche pas encore fusionnée compte quand
   même, ce n'est pas une régression des notes ;
3. un chiffre (débit, pourcentage, taille) cité à côté d'une pièce ne se retrouve pas, au
   même enchaînement de chiffres (séparateurs ignorés, comme `test_readme_chiffres_ancres_238`),
   dans le contenu de la ou des notes trouvées pour cette pièce.

Portée volontairement mécanique : ces contrôles ne jugent jamais le SENS d'une phrase,
seulement des faits qu'un `git cat-file`/une recherche de fichier/un grep tranchent. Limite
connue et acceptée : le contrôle 3 associe TOUS les chiffres d'un même paragraphe (bullet)
à TOUTES les pièces qu'il cite — un paragraphe qui cite deux pièces avec des chiffres propres
à chacune peut donc signaler un faux manquant côté pièce B si le chiffre appartient en fait à
la pièce A citée dans la même puce (cas réel : "piece 209/226" partage un paragraphe). Un
écart signalé mérite donc une relecture humaine avant d'être traité comme un chiffre altéré.
"""
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parents[1]

# Instantané public (publier-github.sh) : une partie des notes de revue y reste privée et la prose est
# anonymisée — le contrôle 3 y échouait sur la 070 b de v0.7.2.md (« 070 », « 19 » introuvables,
# CI publique de la 0.7.4) alors qu'il passe sur l'arbre privé. Ce
# contrôle confronte les notes de version au corpus INTERNE complet ; il se joue sur l'arbre privé.
if not (RACINE / "acvram-memoire" / "revue" / "INDEX.md").exists():
    pytest.skip("corpus de revue interne absent de cet arbre (instantané public)", allow_module_level=True)
DOSSIER_NOTES = RACINE / "docs" / "notes"

# (nom de fichier, minimum de sha, minimum de pièces citées, minimum de pièces avec chiffres) —
# les minimums de v0.7.0 sont ceux du gabarit d'origine (251) ; ceux de 0.7.1/0.7.2/0.7.3 sont
# calés sous le compte réel constaté à l'écriture de la 280, jamais au-dessus (piège du même
# genre que le cliquet d'INDEX.md : monter le seuil est le signe qu'on a ajouté du contenu, pas
# qu'on peut se permettre de le retirer).
GABARITS = {
    "v0.7.0.md": (5, 15, 10),
    "v0.7.1.md": (5, 10, 8),
    "v0.7.2.md": (2, 2, 0),
    "v0.7.3.md": (1, 1, 1),
}
FICHIERS = sorted(GABARITS)


def _git(*args):
    return subprocess.run(["git", *args], cwd=RACINE, capture_output=True, text=True)


def _notes(nom):
    return (DOSSIER_NOTES / nom).read_text(encoding="utf-8")


# ---- 1. shas cités ----

_SHA = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{7,10}(?![0-9a-fA-F])")


def _shas_cites(texte):
    # au moins une lettre a-f : distingue un sha d'une suite de chiffres pure (aucun cas
    # dans ce fichier au moment d'écrire ce test, gardé par prudence contre un faux sha
    # tout-numérique qui ne serait de toute façon presque jamais un objet valide).
    #
    # 280 : un nom de fichier de verdict colle parfois le numéro de pièce juste après « piece »
    # sans séparateur (`poste6-piece269b-...`, `poste2-piece237d-...`) ; le « ece » final de
    # « piece » fait alors partie du même run hexadécimal que le numéro (« ece269b »,
    # « ece237d »), assez long pour ressembler à un sha absent du dépôt. Un sha réel de ce
    # corpus suit toujours un nom d'auteur, jamais le mot « piece » collé sans espace — un
    # candidat dont les deux caractères précédents sont « pi » (i.e. la reconstitution du mot
    # « piece ») est exclu par construction, pas toléré pour laisser passer un sha faux.
    trouves = []
    for m in _SHA.finditer(texte):
        avant = texte[max(0, m.start() - 2):m.start()].lower()
        if avant == "pi" and m.group(0).lower().startswith("ece"):
            continue
        trouves.append(m.group(0))
    return sorted({m for m in trouves if re.search(r"[a-f]", m)})


@pytest.mark.parametrize("nom", FICHIERS)
def test_chaque_sha_cite_existe_dans_le_depot(nom):
    minimum_sha, _, _ = GABARITS[nom]
    texte = _notes(nom)
    shas = _shas_cites(texte)
    assert len(shas) >= minimum_sha, f"{nom} : gabarit obsolète, moins de {minimum_sha} sha"
    manquants = [s for s in shas if _git("cat-file", "-e", s).returncode != 0]
    assert not manquants, f"{nom} : sha absents du dépôt (git cat-file -e) : {manquants}"


# ---- 2 & 3. pièces citées, et leurs chiffres ----

_PIECE = re.compile(r"pi[eè]ces?\s+(\d{3}[a-z]?(?:\s*b\d)?)", re.IGNORECASE)
_LISTE_FUSIONNEE = re.compile(
    r"(?:merged pieces|pi[eè]ces fusionn[ée]es)\s*[—-]\s*([0-9a-z,/ ]+?)\.", re.IGNORECASE)


def _pieces_citees(texte):
    trouvees = {m.strip() for m in _PIECE.findall(texte)}
    for groupe in _LISTE_FUSIONNEE.findall(texte):
        for tok in re.split(r"[,/]", groupe):
            tok = tok.strip()
            if re.match(r"^\d{3}[a-z]?$", tok):
                trouvees.add(tok)
    return trouvees


def _racine(numero):
    return re.match(r"\d+", numero).group(0)


@lru_cache(maxsize=1)
def _fichiers_revue_historiques():
    """(sha_commit, chemin) pour chaque fichier jamais AJOUTÉ sous un dossier `revue/`,
    sur toute référence connue localement (`--all`) — pas seulement la branche courante :
    une pièce peut être notée sur une branche pas encore fusionnée."""
    r = _git("log", "--all", "--diff-filter=A", "--name-only", "--pretty=format:%H")
    paires = []
    sha_courant = None
    for ligne in r.stdout.splitlines():
        if re.match(r"^[0-9a-f]{40}$", ligne):
            sha_courant = ligne
            continue
        if not ligne or sha_courant is None:
            continue
        if "/revue/" in ligne or ligne.startswith("revue/"):
            paires.append((sha_courant, ligne))
    return tuple(paires)


def _notes_pour_chemin(numero):
    racine = _racine(numero)
    motif = re.compile(rf"(^|[^0-9]){re.escape(numero)}([^0-9]|$)|(^|[^0-9]){re.escape(racine)}([^0-9]|$)")
    return [(sha, chemin) for sha, chemin in _fichiers_revue_historiques() if motif.search(chemin)]


@lru_cache(maxsize=1)
def _sha_head():
    return _git("rev-parse", "HEAD").stdout.strip()


def _notes_par_titre(numero):
    """280 : une note peut ne rien porter de la pièce dans son NOM de fichier mais la citer dans
    un titre (`#`) ou une ligne de tableau (`|`) — cas réel, pièce 070b : pas de fichier à son
    nom, sa ligne est `| 070 b | … |` dans `revue/poste6-serie266-flatpak-verdict-26-09.md`.
    Cherché sur l'arbre COURANT (HEAD), pas tout l'historique comme `_notes_pour_chemin` : une
    note de titre compte une fois fusionnée, pas pendant qu'elle est encore un brouillon isolé
    sur une branche à part — une pièce non fusionnée reste couverte par le nom de fichier
    (contrôle historique) si son auteur en a donné un, jamais par ce repli-ci."""
    racine = _racine(numero)
    suffixe = numero[len(racine):]
    identifiant = rf"{racine}\s*{re.escape(suffixe)}" if suffixe else rf"{racine}\b"
    motif = rf"^\s*(#|\|)[^\n]*{identifiant}\b"
    r = _git("grep", "-n", "-i", "-E", motif, "--", "revue/*.md", "acvram-memoire/revue/*.md")
    if r.returncode != 0:
        return []
    sha = _sha_head()
    chemins = sorted({ligne.split(":", 1)[0] for ligne in r.stdout.splitlines() if ligne})
    return [(sha, chemin) for chemin in chemins]


def _notes_pour(numero):
    # union, jamais un OU court-circuité : un faux positif de _notes_pour_chemin (une racine qui
    # matche par hasard le chemin d'une note sans rapport, cas réel « 070 » dans
    # `poste2-piece251-contre-lecture-v070-26-09.md`) ne doit pas empêcher de trouver la VRAIE
    # note par son titre — les deux listes sont complémentaires, pas substituables.
    vu = set()
    resultat = []
    for paire in [*_notes_pour_chemin(numero), *_notes_par_titre(numero)]:
        if paire not in vu:
            vu.add(paire)
            resultat.append(paire)
    return resultat


# 280 : v0.7.0.md est déjà publié (release v0.7.0) et truffé de co-citations de plusieurs pièces
# dans un même paragraphe — limite documentée du contrôle 3, connue avant même la 251. On ne
# réécrit pas un fichier déjà livré pour satisfaire un contrôle après coup : ce fichier SEUL est
# exempté des deux contrôles de contenu (note de verdict, chiffres), jamais de celui des sha.
# `test_l_exemption_de_co_citations_reste_limitee_a_v070` casse si ce jeu s'étend.
FICHIERS_CO_CITATIONS_ADMISES = frozenset({"v0.7.0.md"})


@pytest.mark.parametrize("nom", FICHIERS)
def test_chaque_piece_citee_a_sa_note_de_verdict(nom):
    if nom in FICHIERS_CO_CITATIONS_ADMISES:
        pytest.skip(f"{nom} : exempté (co-citations admises, voir FICHIERS_CO_CITATIONS_ADMISES)")
    _, minimum_pieces, _ = GABARITS[nom]
    texte = _notes(nom)
    pieces = _pieces_citees(texte)
    assert len(pieces) >= minimum_pieces, f"{nom} : gabarit obsolète, moins de {minimum_pieces} pièces citées"
    sans_note = sorted(p for p in pieces if not _notes_pour(p))
    assert not sans_note, (
        f"{nom} : pièces citées sans aucune note dans acvram-memoire/revue/ ou revue/, "
        f"dans tout l'historique connu : {sans_note}")


def _normalise_nombre(brut):
    return re.sub(r"[^\d]", "", brut)


_NOMBRE = re.compile(r"(?<![A-Za-zÀ-ÖØ-öø-ÿ_])\d(?:[ .,]?\d)*")
_CITATION_PIECE = re.compile(
    r"pi[eè]ces?\s+\d{3}[a-z]?(?:\s*b\d)?(?:[,/]\s*\d{3}[a-z]?(?:\s*b\d)?)*"
    r"|\(?\bsee\s+pi[eè]ce\s+\d{3}[a-z]?[^)]*\)?"
    r"|(?:merged pieces|pi[eè]ces fusionn[ée]es)\s*[—-]\s*[0-9a-z,/ ]+?\.", re.IGNORECASE)
# 280 : un numéro de version (« 0.7.0 », « v0.7.2 ») est un IDENTIFIANT, jamais un chiffre
# mesuré — mais il a la forme exacte que `_NOMBRE` reconnaît (chiffres séparés par des points),
# et deux mentions de la même version dans un même paragraphe (une fois nue, une fois précédée
# de « v ») se normalisent différemment (« 070 » contre « 70 », le premier chiffre de la
# deuxième étant exclu par le regard-arrière de `_NOMBRE`), créant un faux manquant sans
# rapport avec la pièce citée. Retiré avant extraction, comme les sha et les numéros de pièce.
_VERSION = re.compile(r"\bv?\d+\.\d+\.\d+\b")


def _chiffres(texte):
    # les numéros de pièce eux-mêmes (« piece 209/226 », « see piece 229 above ») ne sont
    # pas des chiffres MESURÉS : les compter ferait chercher « 229 » dans la note de la 217
    # sous prétexte qu'un renvoi croisé la mentionne — retirés avant extraction. Un sha cité
    # juste après (« poste3 749ef7e46 ») commence par des chiffres que `_NOMBRE` prendrait
    # pour une mesure (son regard-arrière exclut une lettre AVANT, pas après, comme dans
    # `test_readme_chiffres_ancres_238` — même limite documentée là-bas) : les shas sont
    # retirés en premier, avant toute extraction de nombre.
    sans_version = _VERSION.sub(" ", texte)
    sans_shas = _SHA.sub(lambda m: " " if re.search(r"[a-f]", m.group(0)) else m.group(0), sans_version)
    sans_pieces = _CITATION_PIECE.sub(" ", sans_shas)
    return {_normalise_nombre(m.group(0)) for m in _NOMBRE.finditer(sans_pieces)
            if len(_normalise_nombre(m.group(0))) >= 2}


def _paragraphes(texte):
    """Un paragraphe = une puce top-level (`- **`) ou une sous-puce (`  - `), jusqu'à la
    puce suivante — c'est à cette granularité que les notes citent une pièce ET ses
    chiffres ensemble.

    280 : la dernière puce d'une section (juste avant `### Install / upgrade`, un `---` de
    séparation, ou le titre `## Français`/`## English` suivant) avalait tout ce qui suit —
    ligne d'intro de la section suivante comprise — puisque rien n'arrêtait le bloc avant la
    puce SUIVANTE. Un chiffre nu dans cette ligne d'intro (« TTFT à 12 requalifié ») se faisait
    alors attribuer à la dernière pièce citée, sans rapport avec elle. Une ligne vide, un titre
    (`#`), un séparateur (`---`) ou une clôture/ouverture de bloc de code (```) referment
    désormais le paragraphe en cours, qui n'a jamais légitimement besoin de les traverser."""
    blocs, courant = [], []
    for l in texte.splitlines():
        if re.match(r"^(-|  -)\s", l):
            if courant:
                blocs.append("\n".join(courant))
            courant = [l]
        elif re.match(r"^(#|---\s*$|```)", l) or not l.strip():
            if courant:
                blocs.append("\n".join(courant))
            courant = []
        elif courant:
            courant.append(l)
    if courant:
        blocs.append("\n".join(courant))
    return blocs


@lru_cache(maxsize=None)
def _contenu_note(sha, chemin):
    r = _git("show", f"{sha}:{chemin}")
    return r.stdout if r.returncode == 0 else ""


@pytest.mark.parametrize("nom", FICHIERS)
def test_chaque_chiffre_cite_est_dans_la_note_de_sa_piece(nom):
    if nom in FICHIERS_CO_CITATIONS_ADMISES:
        pytest.skip(f"{nom} : exempté (co-citations admises, voir FICHIERS_CO_CITATIONS_ADMISES)")
    _, _, minimum_avec_chiffres = GABARITS[nom]
    texte = _notes(nom)
    par_piece = {}
    for bloc in _paragraphes(texte):
        pieces = _pieces_citees(bloc)
        if not pieces:
            continue
        chiffres = _chiffres(bloc)
        for p in pieces:
            par_piece.setdefault(p, set()).update(chiffres)
    assert len(par_piece) >= minimum_avec_chiffres, (
        f"{nom} : gabarit obsolète, moins de {minimum_avec_chiffres} pièces avec chiffres")

    ecarts = {}
    for piece, chiffres in par_piece.items():
        notes = _notes_pour(piece)
        if not notes:
            continue  # déjà signalé par test_chaque_piece_citee_a_sa_note_de_verdict
        chiffres_notes = set()
        for sha, chemin in notes:
            chiffres_notes |= _chiffres(_contenu_note(sha, chemin))
        manquants = sorted(chiffres - chiffres_notes)
        if manquants:
            ecarts[piece] = manquants
    assert not ecarts, (
        f"{nom} : chiffres cités absents (au même enchaînement de chiffres) de la note de leur "
        f"pièce : {ecarts}")


def test_le_test_sait_dire_faux():
    """Un sha altéré doit casser le contrôle 1 — preuve que le contrôle n'est pas vide."""
    faux = _notes("v0.7.0.md").replace("f0b1e381f", "f0b1e381e", 1)
    shas = _shas_cites(faux)
    assert "f0b1e381e" in shas
    assert _git("cat-file", "-e", "f0b1e381e").returncode != 0, (
        "le sha altéré existe par hasard dans le dépôt — gabarit du test à revoir")


def test_l_exemption_de_co_citations_reste_limitee_a_v070():
    """280 : v0.7.0.md est exempté des contrôles 2 et 3 (déjà publié, co-citations connues avant
    la 251). Casse si un autre fichier rejoint l'exemption sans décision explicite — ce test-ci,
    pas un skip discret dans les deux contrôles."""
    assert FICHIERS_CO_CITATIONS_ADMISES == frozenset({"v0.7.0.md"}), (
        "l'exemption des co-citations doit rester limitée à v0.7.0.md — toute extension à un "
        "autre fichier est une décision à écrire ici explicitement, pas un effet de bord")
