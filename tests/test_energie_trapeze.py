"""Contrôle croisé demandé après les six verdicts croisés (qr.md, 22/09) :
`joules` (compteur NVML monotone) reste la seule source publiée -- mais un
lecteur qui sonde plus vite que la période interne de NVML relit plusieurs
fois le même palier de puissance (« même horodatage NVML »).
`_dedupliquer_puissances` réduit chaque palier à ses deux bords (premier et
dernier échantillon de même valeur) avant `_integrer_trapezes` : l'aire
d'un palier plat ne dépend que de ses deux extrémités, jamais du nombre de
fois où il a été relu. Testé ici sur des traces synthétiques d'énergie
connue -- pas de matériel requis."""
from outils.gpu.mesure import energie as mod


def test_trace_puissance_constante_retrouve_l_energie_connue():
    """Puissance constante 200 W pendant 10 s, échantillonnée toutes les
    2 s : aire = 200 * 10 = 2000 J, avec ou sans dédoublonnage."""
    paires = [(0.0, 200.0), (2.0, 200.0), (4.0, 200.0), (6.0, 200.0), (8.0, 200.0), (10.0, 200.0)]
    assert mod._integrer_trapezes(paires) == 2000.0
    dedup = mod._dedupliquer_puissances(paires)
    assert dedup == [(0.0, 200.0), (10.0, 200.0)]   # le palier entier réduit à ses deux bords
    assert mod._integrer_trapezes(dedup) == 2000.0


def test_trace_avec_doublons_donne_la_meme_energie_que_sans():
    """Même fenêtre 200 W / 10 s, mais chaque valeur relue 3 fois de suite
    (un sondage plus rapide que la période NVML) : après dédoublonnage,
    l'aire doit retomber sur la même énergie -- les doublons ne doivent
    ni la gonfler ni la réduire."""
    sans_doublon = [(0.0, 200.0), (5.0, 200.0), (10.0, 200.0)]
    avec_doublons = [
        (0.0, 200.0), (1.0, 200.0), (2.0, 200.0),
        (5.0, 200.0), (5.5, 200.0),
        (10.0, 200.0),
    ]
    e_sans = mod._integrer_trapezes(sans_doublon)
    e_avec = mod._integrer_trapezes(mod._dedupliquer_puissances(avec_doublons))
    assert e_avec == e_sans == 2000.0


def test_palier_variable_garde_les_deux_bords_du_plateau():
    """100 W tenu de t=0 à t=3 (trois lectures identiques), puis 300 W à
    t=5 -- un dédoublonnage qui ne garderait que le DERNIER horodatage du
    palier (0-3) perdrait toute sa durée réelle : l'aire retomberait sur
    le seul trapèze 100->300 (400 J) au lieu de 700 J. Garder les DEUX
    bords du palier (0,100) et (3,100) restitue les 300 J du plateau
    avant d'ajouter les 400 J de la transition vers 300 W."""
    paires = [(0.0, 100.0), (1.0, 100.0), (3.0, 100.0), (5.0, 300.0)]
    dedup = mod._dedupliquer_puissances(paires)
    assert dedup == [(0.0, 100.0), (3.0, 100.0), (5.0, 300.0)]
    assert mod._integrer_trapezes(dedup) == 100.0 * 3.0 + (100.0 + 300.0) / 2 * 2.0 == 700.0


def test_moins_de_deux_points_rend_none_jamais_zero():
    assert mod._integrer_trapezes([]) is None
    assert mod._integrer_trapezes([(0.0, 150.0)]) is None
    assert mod._dedupliquer_puissances([]) == []


def _fenetre_avec_puissances(paires):
    f = object.__new__(mod.Energie)
    f.debut = {0: 0}
    f.fin = {0: 2_000_000}   # 2000 J au compteur NVML (indépendant du trapèze)
    f.indisponible = None
    f.duree = 10.0
    f.puissances_t = [t for t, _ in paires]
    f.puissances = [w for _, w in paires]
    return f


def test_joules_trapeze_ne_touche_pas_joules_ni_moyenne():
    """Sortie inchangée sur une cellule déjà publiée : `joules` (donc
    `moyenne`) ne dépend que du compteur NVML, jamais des échantillons de
    puissance -- ajouter `joules_trapeze` ne les fait bouger d'un bit."""
    f = _fenetre_avec_puissances([(0.0, 200.0), (2.0, 200.0), (5.0, 200.0), (10.0, 200.0)])
    assert f.joules == 2000.0
    assert f.moyenne == 200.0
    assert f.joules_trapeze == 2000.0   # ici les deux concordent, par construction du test


def test_resume_expose_joules_trapeze_sans_echantillons():
    """Sans thread d'échantillonnage lancé (fenêtre construite à la main,
    comme les autres tests de ce module) : `joules_trapeze` est None, pas
    une fausse valeur, et n'empêche pas la lecture des autres champs."""
    f = object.__new__(mod.Energie)
    f.indisponible = None
    f.debut = {0: 0}
    f.fin = {0: 100000}
    f.pids_debut = {0: ()}
    f.pids_fin = {0: ()}
    f.bridages = set()
    f.duree = 20.0
    f.horloges = []
    f.temperatures = []
    f.charges1 = []
    f.charge_avant = None
    f.charge_apres = None
    f.nproc = 32

    class _FauxNvml:
        cartes = [(0, object())]

        def plafond_w(self, h):
            return 400.0

    mod._nvml = _FauxNvml()
    r = f.resume()
    assert r["joules_trapeze"] is None
    assert r["joules"] == 100.0   # inchangé : 100000 mJ / 1000
