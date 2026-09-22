"""Garde de charge CPU, précisée par Sage (18/09, load 70,8, REGLES §2) :
`load1` rejoint la boucle d'échantillons PAR TIC qui existe déjà pour
horloge/température (gratuit, `/proc/loadavg`), jugée sur le MAX de toute
la fenêtre — ni "fin seulement" ni un second instrument. Une charge lancée
à mi-fenêtre, entre deux relevés début/fin, ne doit pas passer entre les
mailles."""
import os

from outils.gpu.mesure import energie as mod


def _fenetre(charges1=(), charge_avant=None, charge_apres=None, nproc=32):
    f = object.__new__(mod.Energie)
    f.indisponible = None
    f.debut = {0: 0}
    f.fin = {0: 100000}
    f.pids_debut = {0: ()}
    f.pids_fin = {0: ()}
    f.bridages = set()
    f.duree = 20.0
    f.charges1 = list(charges1)
    f.charge_avant = charge_avant
    f.charge_apres = charge_apres
    f.nproc = nproc

    class _FauxNvml:
        cartes = [(0, object())]

        def plafond_w(self, h):
            return 400.0

    mod._nvml = _FauxNvml()
    return f


def test_pic_a_mi_fenetre_est_detecte_la_ou_debut_et_fin_ne_le_verraient_pas():
    """Le test qui casse : calme au début, calme à la fin, un pic AU MILIEU
    -- un contrôle « fin seulement » ou « début+fin » ne verrait rien."""
    f = _fenetre(charges1=[1.9, 1.8, 20.0, 1.7, 1.9], nproc=32,
                charge_avant={"load1": 1.9, "nproc": 32, "processus_charges": []},
                charge_apres={"load1": 1.7, "nproc": 32, "processus_charges": [(4242, 1050.0)]})
    r = f.invalidations
    assert any("charge pendant la fenêtre" in x and "20.00" in x for x in r)


def test_load1_bas_toute_la_fenetre_ne_declenche_pas():
    f = _fenetre(charges1=[1.9, 1.8, 2.0, 1.7], nproc=32)
    assert not any("charge pendant la fenêtre" in x for x in f.invalidations)


def test_pile_au_seuil_ne_declenche_pas():
    f = _fenetre(charges1=[16.0], nproc=32)   # nproc/2 = 16.0, pas > 16.0
    assert not any("charge pendant la fenêtre" in x for x in f.invalidations)


def test_acvram_charge_ok_accepte(monkeypatch):
    monkeypatch.setenv("ACVRAM_CHARGE_OK", "1")
    f = _fenetre(charges1=[1.9, 20.0, 1.7], nproc=32)
    assert not any("charge pendant la fenêtre" in x for x in f.invalidations)


def test_load1_max_pur():
    f = _fenetre(charges1=[1.0, 5.0, 2.0], nproc=32)
    assert f._load1_max() == 5.0


def test_load1_max_retombe_sur_avant_apres_sans_boucle():
    f = _fenetre(charges1=[], nproc=32,
                 charge_avant={"load1": 3.0, "nproc": 32, "processus_charges": []},
                 charge_apres={"load1": 7.0, "nproc": 32, "processus_charges": []})
    assert f._load1_max() == 7.0
