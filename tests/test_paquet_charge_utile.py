"""La charge utile du `.deb` refuse ce qui n'est pas pour qui installe.

Geste 5 de `acvram-memoire/STRUCTURE.md` : `docs/` public et
`acvram-memoire/` prive ne disent pas ce qui NE DOIT PAS entrer dans le paquet.
C'est une regle de construction, verifiee par un essai — pas par la discipline
de qui range.

Ce test construit le paquet et refuse dans sa charge utile :
  * `acvram-memoire/` (memoire interne, jamais distribuee)
  * `journaux/` (mesures internes)
  * `corpus/` (donnees de campagne)
  * `outils/` (bancs et harnais, sauf `carte.sh` cite explicitement)
  * `tests/`, `revue/`, `.git`, `.beads`, `__pycache__`
Et il verifie que le champ `Depends` du control liste bien des paquets Ubuntu
existants — sinon le paquet s installe mais son lanceur ne demarre pas.
"""
import pathlib
import shutil
import subprocess

import pytest

RACINE = pathlib.Path(__file__).resolve().parent.parent
# Le .deb du dépôt porte la version courante (acvram/__init__.py) : une cible
# en dur (« acvram_0.6.0 ») validait toujours un vieux paquet (Manon, 20/09,
# verdict-paquet-0623-19-09).
import importlib
_VERSION = importlib.import_module("acvram").__version__
DEB = RACINE / f"acvram_{_VERSION}_amd64.deb"

INTERDITS = (
    "acvram-memoire/",
    "journaux/",
    "corpus/",
    "revue/",
    "tests/",
    ".git/",
    ".beads/",
    "__pycache__",
    ".pytest_cache/",
)


def _construit_si_absent():
    if DEB.is_file():
        return
    if not shutil.which("dpkg-deb"):
        pytest.skip("dpkg-deb absent — machine non Debian")
    subprocess.run(["bash", "tools/construire-deb.sh"],
                   cwd=RACINE, check=True, capture_output=True)


def _contenu():
    _construit_si_absent()
    out = subprocess.run(["dpkg-deb", "-c", str(DEB)],
                         capture_output=True, text=True, check=True)
    for ligne in out.stdout.splitlines():
        # dpkg-deb -c rend : perms owner size date time PATH -> lien?
        chemin = ligne.split(None, 5)[-1].split(" -> ")[0]
        yield chemin


def _champ_control(nom):
    _construit_si_absent()
    out = subprocess.run(["dpkg-deb", "-f", str(DEB), nom],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_la_charge_utile_ne_contient_aucun_dossier_interne():
    fautes = []
    for chemin in _contenu():
        for motif in INTERDITS:
            if motif in chemin:
                fautes.append(f"{chemin} contient {motif!r}")
    assert not fautes, "fuites dans le .deb :\n  " + "\n  ".join(fautes)


def test_les_gardes_savent_tirer():
    """Sans ce controle, « aucune faute » ne se distingue pas d un motif
    aveugle. Les temoins sont des chemins qui DOIVENT etre refuses."""
    for temoin in ("./usr/share/acvram/acvram-memoire/notes.md",
                   "./usr/share/acvram/journaux/x.tsv",
                   "./usr/share/acvram/corpus/quota/y.json",
                   "./usr/share/acvram/__pycache__/z.pyc",
                   "./usr/share/acvram/.git/HEAD"):
        assert any(m in temoin for m in INTERDITS), \
            f"aucun motif ne voit {temoin}"
    for innocent in ("./usr/share/acvram/acvram/__init__.py",
                     "./usr/bin/acvram",
                     "./usr/share/doc/acvram/ARCHITECTURE.md",
                     "./usr/share/acvram/carte.sh"):
        assert not any(m in innocent for m in INTERDITS), \
            f"faux positif sur {innocent}"


def test_le_champ_Depends_liste_des_paquets_Ubuntu_existants():
    """Une dependance qui n existe pas fait echouer `apt install`
    silencieusement pour l utilisateur. Le champ est ecrit a la main dans
    construire-deb.sh — ce test attrape une faute de frappe ou un paquet
    renomme entre distributions."""
    if not shutil.which("apt-cache"):
        pytest.skip("apt-cache absent — machine non Debian/Ubuntu")
    depends = _champ_control("Depends")
    manquants = []
    for brique in depends.split(","):
        # « python3 (>= 3.10) » -> « python3 »
        nom = brique.strip().split()[0].split("(")[0].strip()
        r = subprocess.run(["apt-cache", "show", nom],
                           capture_output=True, text=True)
        if r.returncode != 0 or not r.stdout.strip():
            manquants.append(nom)
    assert not manquants, (
        f"Depends liste des paquets absents des sources Ubuntu : {manquants}")


def test_le_control_declare_bien_la_version_et_l_architecture():
    """Un paquet sans Version ou sans Architecture s installe partout, sans
    permettre a apt de le remplacer par une version plus recente."""
    assert _champ_control("Package") == "acvram"
    assert _champ_control("Version"), "Version manquante dans control"
    assert _champ_control("Architecture") == "amd64"


def test_le_lanceur_epingle_transformers_a_la_version_du_moteur():
    """Sage 14 h 10 : transformers est une dépendance du moteur (tour de vision), épinglée ; le lanceur
    du .deb (tools/construire-deb.sh) et acvram/engine/vision.py portent LA MÊME version, et vision.py
    n'importe transformers que quand une tour est demandée (import paresseux : un alias texte n'en
    charge rien — tests/test_mm_moteur.py le prouve à sec)."""
    from acvram.engine import vision
    lanceur = (RACINE / "tools" / "construire-deb.sh").read_text(encoding="utf-8")
    assert f'"transformers=={vision.VERSION_TRANSFORMERS}"' in lanceur, \
        f"le lanceur n'épingle pas transformers=={vision.VERSION_TRANSFORMERS}"
    src = pathlib.Path(vision.__file__).read_text(encoding="utf-8").split("\n")
    tete = [l for l in src if l.startswith(("import ", "from ")) and "transformers" in l]
    assert not tete, f"import de transformers en tête de vision.py : {tete}"
    assert any("import transformers" in l and l.startswith("            ") for l in src), \
        "l'import de transformers doit vivre dans TourVision.depuis_dossier (tour demandée)"


def test_le_deb_contient_les_sources_marlin_port():
    """Les noyaux marlin_port (acvram/kernels/marlin_port/bindings.cpp et
    dépendances) doivent être inclus dans le .deb — sinon un alias acvram-coder-i8c
    servira sans les sources compilées de son runtime."""
    files = list(_contenu())
    marlin_bindings = [f for f in files if "marlin_port" in f and "bindings.cpp" in f]
    assert marlin_bindings, (
        "kernels/marlin_port/bindings.cpp manquant du .deb — "
        "vérifier pyproject.toml:package-data inclut kernels/**/*.cpp")
