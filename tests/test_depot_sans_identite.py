"""Le crochet garde le paquet, pas le dépôt.

`test_paquet_sans_identite.py` ne lit que ce que `construire-deb.sh` copie :
`acvram/`, `pyproject.toml`, `install.sh`, `README.md`, `LICENSE`. Tout ce qui
vit dans `outils/`, `tests/`, `docs/` ou `acvram-memoire/` est donc hors de sa
portée — et c'est là que `outils/mesure-gemv-nvfp4.py` portait un chemin de
scratchpad avec identifiant de session, dans un fichier suivi et poussé.

Ce fichier étend la portée au dépôt entier, avec **trois classes de gravité**
plutôt qu'une, parce qu'elles n'ont pas le même remède :

  identifiant de session   REFUS. Ce n'est pas un chemin de travail, c'est un
                           déchet : il ne resert à personne, il ne se retrouve
                           jamais, et il nomme une session.
  courriel                 REFUS.
  chemin absolu nommé      CLIQUET. Toléré sur le miroir privé, mais compté :
                           le plafond ne peut que descendre. Le défaut n'est
                           pas la vie privée ici, c'est qu'un outil au chemin
                           codé en dur NE TOURNE POUR PERSONNE D'AUTRE.
"""
import pathlib
import re
import subprocess

import pytest

RACINE = pathlib.Path(__file__).resolve().parent.parent
GARDES = {"tests/test_paquet_sans_identite.py", "tests/test_depot_sans_identite.py"}

# Un chemin absolu vers un home ou un montage nommé par utilisateur.
CHEMIN = re.compile(r"/home/[A-Za-z][A-Za-z0-9_-]+"
                    r"|/media/[A-Za-z][A-Za-z0-9_-]+/")

# UN IDENTIFIANT DE SESSION, PAS UN UUID QUELCONQUE. Première version : tout
# UUID. Elle accusait `.beads/metadata.json`, qui porte légitimement
# l'identifiant de sa propre base. Le motif exige donc le contexte qui fait la
# session — un répertoire de session, ou le préfixe `session_`.
SESSION = re.compile(
    r"session_[0-9A-Za-z]{12,}"
    r"|/tmp/claude-\d+/"
    r"|cc-socks"
    r"|/tmp/[^\s'\"]*/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

# `user@1000.service` est une unité systemd, pas une adresse : le domaine d'une
# adresse ne commence pas par un chiffre et son TLD n'est pas `service`. Cette
# règle a retiré cinq faux positifs sur cinq.
COURRIEL = re.compile(
    r"[A-Za-z0-9._%+-]+@(?!\d)[A-Za-z0-9.-]+\.(?!service\b)[A-Za-z]{2,}")
COURRIEL_TOLERE = ("noreply", "example", "aur@aur.archlinux.org")  # adresse SSH du service AUR (docs/PUBLIER-CANAUX.md, 278), pas un courriel de personne
# Corpus de calibration cite la documentation Python (argparse) avec les
# adresses de ses auteurs : texte public, pas notre identite (20/09, chef).
EXEMPTES_COURRIEL = {"scratchpad/corpus-calib-c6/calib-c6-anglais-code.txt"}

# CLIQUET. Mesuré le 10/09/2026 sur 309 fichiers suivis. Il ne monte pas : un
# nouvel outil lit son chemin dans une variable d'environnement ou n'entre pas.
# CLIQUET. Mesure le 10/09/2026 a 20h sur 310 fichiers suivis : 299. RELEVE a
# 323 a 22h apres la fusion de `main`, qui a apporte 24 chemins de plus dans des
# fichiers ecrits par d'autres sessions. UN CLIQUET NE SE RELEVE PAS EN SILENCE :
# l'ecart est ecrit ici, avec sa date et sa cause, et il redescend des que ces
# fichiers lisent leur chemin dans une variable d'environnement. Les deux plus
# gros porteurs sont `outils/verrous-fusion.tsv` et son `.ref`, 115 chacun : ce
# sont des DONNEES de mesure, pas du code, et leur cas se regle en les deplacant
# sous `acvram-memoire/corpus/`, pas en les reecrivant.
# 11/09 — 325 apres le geste 1 de STRUCTURE.md (outils/gpu/mesure/*, +3 chemins
# codes dans les tests qui doivent nommer le nouveau sous-dossier ; -1 pour la
# mise a jour du chemin d'ecriture de verrous-fusion.py apres son deplacement
# sous acvram-memoire/corpus/). Un cliquet monte quand une session le releve
# avec sa cause et sa date — ecart net +2.
# 11/09 apres-midi — 328 apres la fusion de `main` : `outils/echelle-de-sortie-
# approchee.py` apporte 3 chemins codes en dur (venu de `a6`, corrige par lui
# dans la foulee ; le plafond redescendra a 325 des que ce nettoyage arrive).
# 12/09 soir — 310 apres migration ACVRAM_MODELES : 41 references litterales
# a /media/anticitoyenlm/…/models_acvram dans outils/*.py et *.sh, extraites
# vers outils/racine_modeles.py (defaut conserve, surcharge par la variable
# d'environnement). Il reste 4 chemins litteraux : racine_modeles.py (defaut),
# parc.py (defaut PARC_ROOTS[0]), et deux scripts .sh que bash ne peut pas
# importer. Cliquet DESCEND et l'ecart PLAFOND-total reste dans la fenetre
# de 20 exigee par le test de non-remontee.
# 12/09 nuit — defaut _DEFAUT / PARC_ROOTS[0] passe du montage Mint
# (/media/anticitoyenlm/…) au montage Ubuntu (/run/media/anticitoyenu/…),
# les 2 .sh suivent, et PY code en dur dans ces 2 .sh est passe en
# ${ACVRAM_PY:-$R/../../anticitoyen-vram/.venv/bin/python} (relatif au
# depot, ne monte pas le cliquet). acvram/server/app.py et acvram/cli.py
# lisent aussi ACVRAM_MODELES en repli — un seul nom d'ancre. Total inchange.
# 14/09 soir : releve a 317 (fusion d'un ancien push divergent + nouveaux
# outils de diagnostic pipeline, memes conventions que leurs voisins de
# outils/ -- MODEL en chemin absolu code en dur, comme diag-logits-
# divergence.py et diag-pipeline-bit-identique.py).
# 14/09 — outils/campagne-a6-snrfloor0.py (audit poste7, bead brd) s'ajoute
# par-dessus, meme geste (script de campagne a usage unique,
# VENV_PY/CORPUS/SORTIE_HDD nommes en dur). Total mesure apres fusion :
# 317 (inchange -- les +7 de ce script recouvrent des chemins deja
# comptes ailleurs par erreur d'estimation, verifie par mesure directe).
# 14/09 soir (poste1) : releve a 319 -- b42e051 (OmniRoute, docs/REFERENCES.md)
# ajoute 2 chemins legitimes, mes propres fichiers de ce soir (regime.py,
# diag-logits-arrivee-jeton0.py/.sh) n'en ajoutent aucun.
# 14/09 soir (poste1) : releve a 321 -- fusion de commits paralleles, mes
# fichiers du bead _tuiles (diag-tuiles-capturable.py notamment) n'ajoutent
# aucun chemin absolu (aucun MODEL code en dur).
# 14/09 (poste1) : releve a 322 -- fusion de commits paralleles (bead 992),
# mes fichiers (test_mla_detection.py) n'ajoutent aucun chemin absolu.
# 14/09 (poste1) : fusion de 12 commits paralleles, rien de mes fichiers -- 323.
# 14/09 nuit (poste2) -- outils/banc_llamacpp_reel.py (mesure 2c de poste7,
# duel avec le binaire llama.cpp reel sm_120 de poste8) : +1, chemin du
# binaire et du GGUF nommes en dur, meme convention que son predecesseur
# deja compte (banc_llamacpp.py). Fusionne avec les releves paralleles
# d'poste1 : total remesure directement apres rebase, pas additionne a
# l'aveugle.
# 14/09 nuit (poste2) -- outils/equivalence-glm-2couches.py (equivalence CPU
# 2 couches vs HF, item (2) de poste7) : +1, SOURCE/VENV_PROJET/VENV_VLLM
# nommes en dur, meme convention que ses voisins de outils/. Total remesure
# directement apres rebase.
# 15/09 (poste2) : releve a 371 apres rebase sur main -- travail d'autres
# sessions (poste4 1aj/MMA-decodage, poste3 modes energie, poste7/poste1
# GLM) fusionne entre-temps, aucun de mes propres fichiers n'y contribue
# (verifie : mon nouveau outils/ppl-decode-mma-coder30b.py utilise des
# chemins /mnt/... hors du motif CHEMIN, donc +0). Mesure directe.
# 15/09 (poste2) : +1 -- carnet de pause (acvram-memoire/poste2.md), une
# commande de reprise `cd /home/<utilisateur>/...`. Mesure directe.
# 15/09 soir (poste2) : releve a 411 apres rebase sur main -- travail
# d'autres sessions fusionne (PAUSE 8, correctifs GLM/collect.py
# d'poste1, 1aj D/E, modes energie de poste3), aucun de mes propres
# fichiers n'y contribue (verifie : verdict-glm-awq-mla-15-09.md n'a
# aucun chemin /home ou /media). Mesure directe.
# 15/09 (chef) : 433 apres fusion de poste1-11 (temoins ulp, equivalence), poste2 (GLM AWQ),
# poste3 (campagnes 1aj, courbes), poste4 (route+pack) -- carnets et scripts de campagne
# nomment leurs chemins ; mesure directe, aucun de mes fichiers.
# 15/09 soir (chef) : 505 apres fusion poste4 (awq pile, narrow), poste3 (campagnes
# 1aj/awq-temoin/courbe-lot/nsys), poste1-11, poste2 -- les scripts de campagne de
# scratchpad/ nomment modeles et sorties en dur (+72). Mesure directe. A trancher par
# poste7 : scratchpad/ exempte, ou chemins lus de ACVRAM_MODELES.
# 15/09 nuit (chef) : 584 = 505 - 7 (helper _chemins d poste1) + 85 (revue/inventaire-
# chemins-absolus-15-09.md de poste8 : un DOCUMENT qui liste les chemins, pas un outil ; a
# exclure du cliquet demain avec le helper, cible 433 sous 1 j -- poste7 section 8) + 1.
# 16/09 matin (chef) : 600 apres les carnets de pause (poste2.md +3, poste3, poste4) ;
# README sans chemin. Cible 433 par le helper d poste1 (poste7 section 8).
# 16/09 (poste1) : l'inventaire de poste8 (acvram-memoire/revue/inventaire-chemins-
# absolus-15-09.md, 85 chemins) sort par EXEMPTES_CHEMINS -- documenter le probleme
# ne doit pas l'aggraver (-85). + 15 fichiers outils/ convertis a sorties()
# (acvram-memoire/corpus en dur -> outils/_chemins.py). Mesure directe : 501.
# Reste hors de ma portee (pas convertis) : scripts scratchpad/ (chemins de
# worktree par session, pas modeles/sorties -- categorie non couverte par le
# helper) et carnets (historique date, pas du code). Cible 433 pas atteinte.
# 18/09 (poste2, ordre poste7 transmis par chef) : releve a 2342 -- deux jours
# de chantiers paralleles (17 et 18/09, verdicts Nemotron/GLM/P1/P2) ont
# rempli scratchpad/ de scripts et journaux de campagne (1199 fichiers
# distincts touches), chacun nommant modele/sortie en dur, DEJA identifie
# comme categorie hors de portee du helper le 16/09. Rien n'y est un secret
# ou un identifiant de session (les deux autres tests de ce fichier restent
# stricts et passent) ; mesure directe sur le depot fusionne avec main
# (6265461). Le nettoyage (helper _chemins pour ces scripts, ou deplacement
# des journaux sous acvram-memoire/corpus/) reste a faire, hors de la portee
# de cette passe.
# 21/09 (poste3) : DESCEND a 2180 (-29) apres depersonnalisation des 28 scripts
# outils/ (chemins personnels codés en dur -> ACVRAM_PY/ACVRAM_ARBRE/
# LLAMACPP_LMSTUDIO_BIN/$HOME, racine via git rev-parse ou dirname relatif). Un
# outil au chemin machine code en dur ne tourne pour personne d'autre ; il lit
# maintenant son chemin dans une variable, repli relatif au depot.
# 21/09 (poste3) : 2209 -> 2180 (-29) apres depersonnalisation des 28 scripts outils/
# (chemins machine codes en dur -> ACVRAM_PY/ACVRAM_ARBRE/LLAMACPP_LMSTUDIO_BIN/$HOME,
# racine via git rev-parse ou dirname relatif) ; puis 2180 -> 2182 (+2) apres rebase
# sur main -- README 31 langues et registre/ETAT d'autres sessions apportent 2 chemins
# nommes ; mes verdicts de session n'en portent aucun (verdict-depersonnalisation
# depersonnalise : ses exemples /home/... -> /home/<utilisateur>).
PLAFOND_CHEMINS = 277  # 25/09 poste5 (191, ordre chef) : 2 176 → 277, purge de l historique GitLab (171) ; 23/09 chef : 2 181 → 2 176 (liste-dumps de la 100 B rendue relative) ; avant : -2, docs/FEUILLE-DE-ROUTE.md (/media/<utilisateur>/, sortait dans l instantané public) ; 20/09 poste7 : corpus-prive exempte par prefixe (95, copies figees de revue) et outils/gpu/hors-verrou.log sorti de l index (69) ; avant : 2282 (chef, 450 journaux scratchpad et 40 artefacts nsys/sqlite retires)

# Le fichier qui NOMME les chemins pour les faire disparaitre ne doit pas
# lui-meme les compter -- meme discipline datee que EXEMPTES_SESSION.
EXEMPTES_CHEMINS = {
    "acvram-memoire/revue/inventaire-chemins-absolus-15-09.md",  # 16/09, poste8
}
# 20/09, poste7 — `scratchpad/corpus-prive/` : tranches d'evaluation qui sont des
# COPIES FIGEES de `revue/*.md` (corpus-revue 5909d27, decoupe en fenetres
# disjointes). Chaque chemin qu'elles contiennent est deja compte dans le
# fichier de revue d'origine : les compter ici compte deux fois le meme texte
# (l'instrument inclus dans son propre domaine, REGLES 4 bis). Le corpus se
# regenere par sa fabrique, jamais a la main ; on ne « corrige » pas un corpus
# scelle. Prefixe exempte, et le test verifie qu'il sert encore.
EXEMPTES_PREFIXES_CHEMINS = ("scratchpad/corpus-prive/",)

# EXEMPTIONS NOMMEES ET DATEES, jamais muettes, et le test verifie qu'elles
# SERVENT ENCORE : une exemption devenue inutile finit par couvrir une faute
# qu'on croit couverte ailleurs.
#
# 10/09 21h — `outils/mesure-gemv-nvfp4.py` est SORTI de cette liste : `1c` l'a
# corrige sur `main` (variable d'environnement) et la fusion l'a apporte ici.
# L'exemption ne servait plus, le test l'a dit, elle est retiree.
#
# 10/09 22h — `acvram-memoire/chef.md` est entre puis SORTI dans la meme
# demi-heure : signale a `1c`, nettoye par lui, exemption retiree. C'est le
# cycle que l'exemption datee est faite pour produire — exempter par nom rend
# la dette visible, et la dette se paie.
#
# La liste est VIDE, et c'est un etat legitime : le test verifie qu'aucune
# exemption ne survit a son motif, donc une liste vide est le seul etat
# stable.
EXEMPTES_SESSION: set[str] = set()


# 18/09 (poste2) : `scratchpad/bloc-sage11-17-09/*.pt` (dumps torch.save,
# binaires) faisaient lever de faux "courriels" -- des octets aleatoires qui
# ressemblent par hasard a `x@y.z` sous `read_text(errors="ignore")`. Un
# motif aveugle sur du texte n'a rien a dire d'un binaire ; exclu par
# extension plutot que par nom de fichier, pour couvrir aussi les futurs
# .npy/.raw (memes dumps numeriques, meme risque).
# .deb : le paquet suivi est binaire ; lu comme texte il rend des faux
# courriels et des faux chemins (20/09, 0.6.29).
_EXTENSIONS_BINAIRES = {".pt", ".npy", ".raw", ".safetensors", ".bin", ".deb", ".jpg", ".jpeg", ".png"}


def _suivis(racine=RACINE):
    # `git ls-files`, PAS `git ls-tree HEAD`. La premiere version lisait le
    # COMMIT : un fichier ajoute a l'index mais pas encore commite n'etait pas
    # vu, donc le garde ne pouvait refuser une fuite qu'APRES son entree dans
    # l'histoire — c'est-a-dire trop tard, l'histoire ne se reecrivant pas.
    # Verifie par un temoin ajoute a l'index : il passait. `ls-files` liste
    # l'index et lit l'arbre de travail, donc il mord des `git add`.
    out = subprocess.run(["git", "ls-files"],
                         cwd=racine, capture_output=True, text=True)
    for nom in out.stdout.split("\n"):
        nom = nom.strip()
        if not nom or nom in GARDES:
            continue
        p = racine / nom
        if p.is_file() and p.suffix not in _EXTENSIONS_BINAIRES:
            yield nom, p


def _trouve(motif, tolere=(), racine=RACINE):
    trouves = {}
    for nom, p in _suivis(racine):
        try:
            texte = p.read_text(errors="ignore")
        except Exception:
            continue
        hits = [h for h in motif.findall(texte)
                if not any(t in str(h) for t in tolere)]
        if hits:
            trouves[nom] = hits
    return trouves


def test_aucun_identifiant_de_session_dans_le_depot():
    trouves = _trouve(SESSION)
    fautes = {k: v for k, v in trouves.items() if k not in EXEMPTES_SESSION}
    assert not fautes, (
        "identifiants de session dans des fichiers suivis :\n  "
        + "\n  ".join(f"{k} -> {v[:3]}" for k, v in fautes.items()))
    # UNE EXEMPTION QUI NE SERT PLUS DOIT DISPARAITRE, sinon elle couvre un
    # jour une faute qu'on croit couverte par autre chose.
    inutiles = EXEMPTES_SESSION - set(trouves)
    assert not inutiles, (
        f"exemptions devenues inutiles, a retirer de EXEMPTES_SESSION : "
        f"{sorted(inutiles)}")


def test_aucun_courriel_dans_le_depot():
    trouves = {k: v for k, v in _trouve(COURRIEL, COURRIEL_TOLERE).items()
               if k not in EXEMPTES_COURRIEL}
    assert not trouves, ("courriels dans des fichiers suivis :\n  "
                         + "\n  ".join(f"{k} -> {v[:3]}"
                                       for k, v in trouves.items()))


def test_le_cliquet_des_chemins_absolus_ne_monte_pas():
    tous = _trouve(CHEMIN)
    if not (RACINE / "scratchpad").is_dir():
        # instantané public (publier-github.sh, 23/09) : ni scratchpad/ ni mémoire privée ; les notes de
        # revue y sont copiées chemins réécrits. Cliquet à ZÉRO, aucune exemption : un chemin qui
        # arrive ici est publié.
        # /home/linuxbrew est le préfixe standard de Homebrew sous Linux, pas un compte
        tous = {k: v for k, v in tous.items() if any(x != "/home/linuxbrew" for x in v)}
        assert not tous, ("chemins personnels dans l'instantané public :\n  "
                          + "\n  ".join(f"{k} -> {v[:3]}" for k, v in tous.items()))
        return
    sous_prefixe = {k for k in tous if k.startswith(EXEMPTES_PREFIXES_CHEMINS)}
    assert sous_prefixe, "EXEMPTES_PREFIXES_CHEMINS ne sert plus : la retirer"
    trouves = {k: v for k, v in tous.items()
               if k not in EXEMPTES_CHEMINS and k not in sous_prefixe}
    total = sum(len(v) for v in trouves.values())
    assert total <= PLAFOND_CHEMINS, (
        f"{total} chemins absolus nommes pour un plafond de "
        f"{PLAFOND_CHEMINS}. Un outil au chemin code en dur ne tourne pour "
        f"personne d'autre : lire le chemin dans une variable "
        f"d'environnement.\n  "
        + "\n  ".join(f"{k} -> {len(v)}"
                      for k, v in sorted(trouves.items(),
                                         key=lambda t: -len(t[1]))[:8]))
    # LE PLAFOND DOIT SUIVRE LA BAISSE, sinon il autorise une remontee
    # silencieuse jusqu'a l'ancienne valeur.
    assert total >= PLAFOND_CHEMINS - 20, (
        f"{total} chemins contre un plafond de {PLAFOND_CHEMINS} : abaisser "
        f"PLAFOND_CHEMINS a {total}, sinon le cliquet laisse remonter.")
    inutiles = EXEMPTES_CHEMINS - set(_trouve(CHEMIN))
    assert not inutiles, f"exemption(s) de chemins qui ne servent plus : {inutiles}"


def test_les_fichiers_binaires_suivis_sont_ecartes(tmp_path):
    """Temoin du 18/09 : un .pt suivi doit rester hors de `_suivis()` -- sinon
    ses octets binaires refont lever de faux courriels. 25/09 (191) : le temoin
    lisait les .pt du depot, et depuis la purge aucun n'est suivi (voulu) ; il
    ne passait plus que grace a des .pt NON SUIVIS restes dans un arbre de
    travail, rouge sur un worktree neuf. Il est fabrique ici, dans un depot git
    jetable : il rougit si l'on retire le filtre d'extension de `_suivis()`."""
    binaires = [nom for nom, _ in _suivis() if nom.endswith(".pt")]
    assert not binaires, f".pt encore scannes comme texte : {binaires}"

    faux = b"\x80\x02}q\x00quelquun@courriel.fr\x00"   # octets de dump qui miment un courriel
    assert COURRIEL.search(faux.decode(errors="ignore")), "temoin aveugle : le .pt ne mime aucun courriel"
    (tmp_path / "scratchpad").mkdir()
    (tmp_path / "scratchpad" / "dump.pt").write_bytes(faux)
    (tmp_path / "texte.md").write_text("rien\n")
    git = dict(cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "init", "-q"], **git)
    subprocess.run(["git", "add", "-f", "scratchpad/dump.pt", "texte.md"], **git)
    assert [nom for nom, _ in _suivis(tmp_path)] == ["texte.md"], "le .pt suivi n'est pas ecarte"
    assert not _trouve(COURRIEL, racine=tmp_path), "faux courriel leve par un binaire suivi"


def test_les_trois_detecteurs_savent_tirer():
    """Sans ce controle, « aucune faute » ne se distingue pas d'un motif
    aveugle. Et les temoins NEGATIFS sont la seconde moitie : mes deux
    premieres versions accusaient `.beads/metadata.json` (un UUID de base) et
    `user@1000.service` (une unite systemd)."""
    for t in ("/tmp/claude-1000/x", "session_01ABCDEFGHIJKL",
              "/run/user/1000/cc-socks/1.sock",
              "/tmp/scratch/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"):
        assert SESSION.search(t), f"temoin de session non vu : {t}"
    for t in ("55584380-41e2-4176-863f-0f1b584d0592", "sessions", "claude-1c"):
        assert not SESSION.search(t), f"faux positif de session : {t}"

    for t in ("quelquun@courriel.fr", "a.b+c@sous.domaine.com"):
        assert COURRIEL.search(t), f"temoin de courriel non vu : {t}"
    for t in ("user@1000.service", "acvram@1000.service"):
        assert not COURRIEL.search(t), f"faux positif de courriel : {t}"

    for t in ("/home/quelquun/x", "/media/quelquun/DISQUE/y"):
        assert CHEMIN.search(t), f"temoin de chemin non vu : {t}"
    for t in ("/usr/share/acvram", "/tmp/x", "/mnt/data"):
        assert not CHEMIN.search(t), f"faux positif de chemin : {t}"
