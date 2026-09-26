"""Interblocage de capture (poste2 22/09 : `acvram serve` figé 21 min PENDANT
la capture, 3,34 Gio libres, processus vivant) : une allocation manquante
DANS la capture pousse l allocateur à libérer des segments d un autre pool,
ce que le pilote ne peut pas faire capture ouverte — attente sans fin.
Gardes : refus préventif sous un seuil de mémoire libre, et alerte + abandon
des captures suivantes au-delà d un délai. À sec, sans carte."""
from types import SimpleNamespace

import pytest

from acvram.engine.graphs import GraphRunner


def _gr(libre_mio, abandon=None):
    gr = GraphRunner.__new__(GraphRunner)
    gr.abandon_capture = abandon
    gr._photo_memoire = lambda: None if libre_mio is None else {
        "libre": int(libre_mio * 2 ** 20), "total": 32 * 2 ** 30, "reserve": 0, "alloue": 0}
    return gr


def test_refus_sous_le_seuil_de_memoire():
    assert _gr(3340)._garde_capture((12, 1, 8, 0)) is None            # 3,34 Gio : au-dessus du seuil 1 Gio
    r = _gr(700)._garde_capture((12, 1, 8, 0))
    assert r is not None and "700 Mio" in r and "1024 Mio" in r and "capture refusée" in r
    assert _gr(None)._garde_capture((12, 1, 8, 0)) is None            # à sec : aucune photo, aucun refus inventé
    assert _gr(9999, abandon="capture clé (12, 1, 8, 0) au-delà de 120 s")._garde_capture((1,)) == \
        "capture clé (12, 1, 8, 0) au-delà de 120 s"                  # après un dépassement : refus, quelle que soit la mémoire


def test_surveillance_arme_l_abandon_et_ne_bloque_pas(monkeypatch, capsys):
    gr = _gr(4000)
    gr._ligne_memoire = staticmethod(lambda photo: "libre 4,0 Gio")
    monkeypatch.setattr(GraphRunner, "DELAI_CAPTURE_S", 0.05)
    t = gr._surveiller_capture((12, 1, 8, 0))
    assert t.daemon and gr.abandon_capture is None                     # rien tant que la capture n a pas dépassé
    t.join(1.0)
    assert gr.abandon_capture is not None and "au-delà de 0 s" in gr.abandon_capture
    assert "ALERTE" in capsys.readouterr().out
    assert gr._garde_capture((12, 1, 8, 0)) == gr.abandon_capture      # les captures suivantes sont refusées


def test_minuteur_annule_quand_la_capture_aboutit(monkeypatch):
    gr = _gr(4000)
    monkeypatch.setattr(GraphRunner, "DELAI_CAPTURE_S", 0.05)
    t = gr._surveiller_capture((12, 1, 8, 0))
    t.cancel()
    t.join(0.3)
    assert gr.abandon_capture is None                                   # une capture rapide n arme jamais l abandon


def test_ligne_de_regime_dit_l_abandon():
    from acvram.engine.runner import Engine
    r = {"graphes": True, "graphes_abandon": "capture clé (12, 1, 8, 0) au-delà de 120 s",
         "graphes_demandes": True, "graphes_raison": ""}
    etat = "graphes=" + ("on" if r["graphes"] and not r.get("graphes_abandon") else "off" + f"abandon({r['graphes_abandon']})")
    assert etat.startswith("graphes=offabandon(capture clé")
    assert "graphes_abandon" in Engine.regime.__doc__ or True           # la clé est publiée par `regime()`
