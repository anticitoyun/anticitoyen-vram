"""Pièce 269 b (opt-in `ACVRAM_ADMISSION_GUET=1`) : la fenêtre d'admission s'ouvre à UNE requête en file si une autre est déjà
ENTRÉE dans le service (gestionnaire HTTP commencé, `submit` pas encore fait — compteur `EngineService.en_entree`). Une requête
seule : compteur à 0, porte fermée, aucune attente (179 b tenue). Au défaut (0), rien ne change : la porte reste « ≥ 2 ».
Processeur seul, moteur factice, comme tests/test_fenetre_admission_179.py."""
import time
import types

from acvram.server import app as A


class _Moteur:
    def __init__(self, n):
        self.running, self.waiting, self.max_batch_size = [], list(range(n)), 8


def _service(n, en_entree=0, vision=False):
    s = A.EngineService.__new__(A.EngineService)
    s.engine = _Moteur(n)
    s.engine.loaded = types.SimpleNamespace(manifest={"vision": "oui"} if vision else {})   # 269 d : lu par vision_servie()
    s.en_entree = en_entree
    return s


def _duree(s):
    t0 = time.perf_counter()
    s._attendre_les_arrivees()
    return time.perf_counter() - t0


def test_solo_compteur_nul_la_porte_reste_fermee(monkeypatch):
    """Le cœur de la 179 b, même avec le guet : une requête seule (personne derrière) n'attend pas."""
    monkeypatch.setattr(A, "_FENETRE_ADMISSION_S", 0.005)
    monkeypatch.setattr(A, "_GUET_ADMISSION", True)
    assert _duree(_service(1, en_entree=0)) < 0.002


def test_une_en_file_une_entree_la_porte_s_ouvre(monkeypatch):
    monkeypatch.setattr(A, "_FENETRE_ADMISSION_S", 0.005)
    monkeypatch.setattr(A, "_GUET_ADMISSION", True)
    dt = _duree(_service(1, en_entree=1))
    # la requête entrée n'arrive jamais dans ce test : la fenêtre tient jusqu'au plafond 4 × 5 ms
    assert 0.018 <= dt < 0.06, dt


def test_en_opt_out_zero_le_guet_est_coupe(monkeypatch):
    """ACVRAM_ADMISSION_GUET=0 (opt-out depuis 269 c) : une requête en file + une entrée = comportement d'avant, pas d'attente."""
    monkeypatch.setattr(A, "_FENETRE_ADMISSION_S", 0.005)
    monkeypatch.setattr(A, "_GUET_ADMISSION", False)
    assert _duree(_service(1, en_entree=1)) < 0.002


def test_269c_au_defaut_le_guet_est_ouvert(monkeypatch):
    """Test cassant « défaut 1 » (269 c, décision chef sur la mesure 269 b) : sans variable, le guet est actif — la
    valeur lue par app.py est celle que donne « 1 » par défaut, et le régime déclare « 1 »."""
    import os
    assert A._GUET_ADMISSION is (os.environ.get("ACVRAM_ADMISSION_GUET", "1") == "1")
    from acvram.regime import VARIABLES
    assert next(x for x in VARIABLES if x.nom == "ADMISSION_GUET").defaut == "1"


def test_la_rafale_a_deux_attend_comme_avant(monkeypatch):
    monkeypatch.setattr(A, "_FENETRE_ADMISSION_S", 0.005)
    for guet in (False, True):
        monkeypatch.setattr(A, "_GUET_ADMISSION", guet)
        dt = _duree(_service(3))
        assert 0.005 <= dt < 0.05, (guet, dt)


def test_le_compteur_monte_et_redescend_meme_sur_exception():
    s = _service(0)
    with s.entree():
        assert s.en_entree == 1
        with s.entree():
            assert s.en_entree == 2
    assert s.en_entree == 0
    try:
        with s.entree():
            raise RuntimeError("400 avant submit")
    except RuntimeError:
        pass
    assert s.en_entree == 0, "une requête refusée avant submit doit libérer le compteur"


def test_defaut_un_et_variable_declaree():
    import os
    assert A._GUET_ADMISSION is True or os.environ.get("ACVRAM_ADMISSION_GUET") == "0"
    from acvram.regime import VARIABLES
    v = [x for x in VARIABLES if x.nom == "ADMISSION_GUET"]
    assert len(v) == 1 and v[0].defaut == "1" and v[0].lu_a == ("acvram.server.app", "_GUET_ADMISSION")


def test_269d_alias_vision_le_guet_est_coupe(monkeypatch):
    """Test cassant 269 d (mesure 276, images FAUX) : pour un alias vision, une requête en file + une entrée = pas d'attente,
    comme avant la 269 b ; le même service sans vision attend la fenêtre (le guet reste au défaut en texte)."""
    monkeypatch.setattr(A, "_FENETRE_ADMISSION_S", 0.005)
    monkeypatch.setattr(A, "_GUET_ADMISSION", True)
    assert _duree(_service(1, en_entree=1, vision=True)) < 0.002
    assert _duree(_service(1, en_entree=1, vision=False)) >= 0.004
    assert _service(1, vision=True).guet_actif() is False and _service(1, vision=False).guet_actif() is True
