"""Rien de ce qui part dans le paquet ne doit identifier la machine.

Le 9/09/2026, `cli.py` portait en dur
le parc du 980 PRO (littéral `_DEFAUT` de outils/racine_modeles.py depuis le 20/09) comme repli du
dossier de sortie. `tools/construire-deb.sh` copie `acvram/` tel quel : le
chemin, et le nom d utilisateur dedans, seraient partis chez quiconque
installe le paquet. Ce n est pas un secret cryptographique, c est une fuite
d identite — et elle se corrige une fois puis se garde.
"""
import pathlib
import re

RACINE = pathlib.Path(__file__).resolve().parent.parent
# ce que construire-deb.sh copie dans /usr/share/acvram
COPIE = ["acvram", "pyproject.toml", "install.sh", "README.md", "LICENSE"]

# un chemin absolu vers un home ou un point de montage nomme
# Pas d'IGNORECASE global : il faisait matcher /mnt/data sur la branche des
# points de montage nommes, qui exige des MAJUSCULES ou des chiffres.
SUSPECTS = re.compile(
    r"/home/[A-Za-z][A-Za-z0-9_-]+"          # un home nomme
    r"|/media/[A-Za-z][A-Za-z0-9_-]+/"       # un montage amovible par utilisateur
    r"|/mnt/(?=[A-Z0-9_]*[0-9])[A-Z0-9_]{4,}"  # un disque nomme, ex. 2TO_SSD_2025
)


def _fichiers():
    for nom in COPIE:
        p = RACINE / nom
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in p.rglob("*.py"):
                yield f


def test_aucun_chemin_de_machine_dans_ce_qui_est_distribue():
    fautes = []
    for f in _fichiers():
        for n, ligne in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            m = SUSPECTS.search(ligne)
            if m:
                fautes.append(f"{f.relative_to(RACINE)}:{n} -> {m.group(0)}")
    assert not fautes, "chemins de machine dans le paquet :\n  " + "\n  ".join(fautes)


def test_le_detecteur_sait_tirer():
    """Sans ce controle, un « aucune faute » ne se distingue pas d un motif
    aveugle — trois mesures ont echoue pour cette raison le meme jour."""
    for temoin in ("/home/quelquun/x", "/media/quelquun/DISQUE/y",
                   "/mnt/2TO_SSD_2025_IA/z"):
        assert SUSPECTS.search(temoin), f"le motif ne voit pas {temoin}"
    for innocent in ("/usr/share/acvram", "/tmp/x", "~/.local/share/acvram",
                     "/mnt/data"):
        assert not SUSPECTS.search(innocent), f"faux positif sur {innocent}"


def test_le_paquet_copie_les_docs_par_LISTE_BLANCHE():
    """Une liste noire oublie toujours le document ecrit apres elle.
    La precedente laissait passer PROTOCOLES-EN-ATTENTE.md (trois noms de
    sessions internes), PREDICTION-CAMPAGNE-9SEPT.md (un) et
    FUSIONS-LIBRES-PARC.md (le chemin du parc)."""
    src = (RACINE / "tools" / "construire-deb.sh").read_text()
    assert "cp -r docs" not in src, "docs/ est encore copie en entier"
    assert "DOCS_PUBLIQUES" in src, "la liste blanche a disparu"
    assert "docs/*.md" not in src, "un glob copie encore tous les .md"


def test_aucun_nom_de_session_dans_les_docs_publiees():
    """Les noms de travail des sessions ne concernent pas qui installe."""
    import re
    src = (RACINE / "tools" / "construire-deb.sh").read_text()
    bloc = re.search(r'DOCS_PUBLIQUES="\n(.*?)"', src, re.S)
    assert bloc, "liste blanche illisible"
    motif = re.compile(r"poste2|Oc[eé]ane|poste4|poste8|poste3|anticitoyenlm")
    fautes = []
    for nom in bloc.group(1).split():
        f = RACINE / "docs" / nom
        if not f.is_file():
            continue
        for n, ligne in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            if motif.search(ligne):
                fautes.append(f"{nom}:{n}")
    assert not fautes, "noms internes dans un document publie : " + ", ".join(fautes)
