"""Un releve manquant n'est pas un releve vide.

`invalidations` faisait `self.pids_fin.get(i, ())` : une carte absente du
releve de fin devenait une carte SANS PROCESSUS. Quand le releve de debut
etait vide lui aussi, les deux tuples etaient egaux et la fenetre passait
pour valide — alors qu'on n'avait simplement pas regarde.

C'est l'absence lue comme un resultat, dans le module meme dont le role est
d'attraper ce qui interdit de conclure.
"""
import pathlib, sys, types

# `outils/gpu/mesure/` n'est pas installe : sans la racine du depot dans
# sys.path, pytest ne trouve pas le namespace package `outils`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _fenetre(pids_debut, pids_fin):
    """Une fenetre minimale, sans NVML ni carte."""
    from outils.gpu.mesure.energie import Energie  # noqa: PLC0415
    f = object.__new__(Energie)
    f.indisponible = None
    f.debut, f.fin = {}, {}
    f.pids_debut, f.pids_fin = pids_debut, pids_fin
    f.bridages = set()
    return f


def test_un_releve_de_fin_manquant_invalide():
    """Le cas exact : deux releves vides, mais l'un n'existe pas."""
    f = _fenetre({0: ()}, {})
    r = f.invalidations
    assert r, "une carte absente du releve de fin passait pour valide"
    assert "aucun releve" in r[0]


def test_deux_releves_vides_mais_PRESENTS_sont_valides():
    """Il ne faut pas crier sur une carte reellement sans processus."""
    assert _fenetre({0: ()}, {0: ()}).invalidations == []


def test_une_carte_apparue_en_cours_est_signalee():
    """La boucle ne parcourait que les cles du DEBUT : une carte apparue
    en route n'etait regardee par personne."""
    r = _fenetre({0: ()}, {0: (), 1: (1234,)}).invalidations
    assert any("apparue en cours" in x for x in r)


def test_un_changement_de_processus_reste_signale():
    r = _fenetre({0: (11,)}, {0: (22,)}).invalidations
    assert r and "processus ont changé" in r[0]
