"""Rien de ce qui part dans le paquet ne doit identifier la machine.

Le 9/09/2026, `cli.py` portait en dur
`/media/anticitoyenlm/2TO_2023_980PRO1/Modeles/models_acvram` comme repli du
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


def test_le_paquet_ne_copie_pas_les_mesures_internes():
    """docs/ embarquait 8 fichiers de mesure — comparatifs, rebancs, releves
    de repetabilite. Le script ne doit copier que des .md, et jamais la
    feuille de route qui porte le chemin de la machine."""
    src = (RACINE / "tools" / "construire-deb.sh").read_text()
    assert "cp -r docs" not in src, "docs/ est encore copie en entier"
    assert "docs/*.md" in src, "la copie selective des .md a disparu"
    assert "FEUILLE-DE-ROUTE.md" in src, "la feuille de route n est plus exclue"
