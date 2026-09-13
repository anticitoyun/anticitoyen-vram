"""Deux gardes ajoutees le 14/09 (poste7/chef), apres le +51 % constate
entre le compteur NVML TotalEnergyConsumption et une mediane de
power.draw.instant sur une fenetre de 2 s (bug distinct, dans
puissance_nvml.py) : une moyenne au-dessus du plafond materiel, ou une
fenetre trop courte, ne disent pas que la carte a depasse sa limite --
elles disent que la fenetre est trop courte pour que le limiteur ait pu
agir, donc trop courte pour qu'on lui fasse confiance.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _fenetre(joules, duree, plafond_w):
    """Une fenetre minimale, NVML simule (pas de carte reelle)."""
    from outils.gpu.mesure import energie as mod  # noqa: PLC0415

    class _FauxNvml:
        cartes = [(0, object())]

        def plafond_w(self, h):
            return plafond_w

    f = object.__new__(mod.Energie)
    f.indisponible = None
    f.debut, f.fin = {0: 0}, {0: int(joules * 1000)}
    f.pids_debut, f.pids_fin = {0: ()}, {0: ()}
    f.bridages = set()
    f.duree = duree
    mod._nvml = _FauxNvml()
    return f


def test_fenetre_courte_est_invalidee():
    f = _fenetre(joules=100.0, duree=5.0, plafond_w=400.0)
    r = f.invalidations
    assert any("trop courte" in x and "5.00" in x for x in r)


def test_fenetre_de_dix_secondes_pile_est_valide_sur_ce_seul_critere():
    # 10.0 s n'est PAS < 10.0 s : la garde ne doit pas se declencher a la
    # limite exacte.
    f = _fenetre(joules=1000.0, duree=10.0, plafond_w=400.0)
    r = f.invalidations
    assert not any("trop courte" in x for x in r)


def test_moyenne_au_dessus_du_plafond_est_invalidee():
    # 100 J / 10 s = 10 W en moyenne -- fixons un plafond tres bas pour
    # forcer le depassement sans dependre d'un vrai calcul de puissance.
    f = _fenetre(joules=100.0, duree=10.0, plafond_w=5.0)
    r = f.invalidations
    assert any("plafond" in x for x in r)


def test_moyenne_sous_le_plafond_est_valide():
    f = _fenetre(joules=100.0, duree=10.0, plafond_w=400.0)
    assert f.invalidations == []


def test_plafond_indisponible_ne_declenche_pas_la_garde():
    """plafond_w() peut rendre 0/-1 (NVML sans info) -- ne pas invalider
    une fenetre juste parce que le plafond est inconnu."""
    f = _fenetre(joules=100.0, duree=10.0, plafond_w=0.0)
    assert not any("plafond" in x for x in f.invalidations)
