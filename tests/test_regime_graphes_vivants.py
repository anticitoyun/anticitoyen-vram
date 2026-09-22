"""`regime()`/`regime_ligne()` doivent dire l'état VIVANT des graphes CUDA,
pas leur intention à la construction (Jérôme, relais Sage
`sage-b0-et-cause-lm4-17-09.md`, à sec).

`Engine.graphs` reste le MÊME OBJET après qu'une capture a échoué en cours
de service : `GraphRunner._capture` bascule `enabled=False` mais ne retire
jamais l'objet de `self.graphs` (`graphs.py:431`). Lire seulement
« l'objet existe » (`self.graphs is not None`) disait donc `graphes=on`
alors que le moteur avait déjà replié en eager — signalé par plusieurs
verdicts (poste E, KV lm4) où `regime_ligne()` mentait sur le régime
réellement mesuré pendant la mesure.

Pas de vraie capture CUDA ici (à sec) : on pose un faux `GraphRunner`
minimal (seuls `enabled`/`raison` comptent pour `regime()`) et on fait
basculer `enabled` comme le ferait une capture ratée — c'est exactement
ce que `regime()` doit lire, pas comment on y arrive.
"""
import torch

from acvram.engine.loader import load_model
from acvram.engine.runner import Engine


class _FauxGraphRunner:
    """Ne porte que ce que `regime()` lit — pas un vrai GraphRunner."""
    def __init__(self):
        self.enabled = True
        self.raison = None


def _engine(converted):
    loaded = load_model(converted, dtype=torch.bfloat16, max_model_len=512,
                        device_override="cpu")
    return Engine(loaded, None, max_batch_size=1, max_model_len=512,
                 enable_cuda_graphs=False)


def test_graphes_on_tant_que_l_objet_est_vivant_et_actif(converted):
    engine = _engine(converted)
    engine._graphes_demandes = True
    engine.graphs = _FauxGraphRunner()
    r = engine.regime()
    assert r["graphes"] is True
    assert "graphes=on" in engine.regime_ligne()


def test_graphes_off_apres_une_capture_ratee_meme_objet(converted):
    """Le cas exact du bogue : l'objet ne devient pas `None`, seul `.enabled`
    bascule — comme le fait réellement `GraphRunner._capture` en repli."""
    engine = _engine(converted)
    engine._graphes_demandes = True
    engine.graphs = _FauxGraphRunner()
    assert engine.regime()["graphes"] is True          # avant l'échec

    engine.graphs.enabled = False
    engine.graphs.raison = "OutOfMemoryError: capture ratée (test)"
    r = engine.regime()
    assert r["graphes"] is False, "l'objet existe encore : lire enabled, pas l'identité"
    assert r["graphes_raison"] == "OutOfMemoryError: capture ratée (test)"
    assert "graphes=off" in engine.regime_ligne()
    assert "graphes=on" not in engine.regime_ligne()
    # gemma-4-26B-A4B (sage-cloture-nuit-0540-20-09 rang 3) : la CAUSE est sur la ligne
    assert "graphes=off(repli eager: OutOfMemoryError: capture ratée (test))" in engine.regime_ligne()


def test_graphes_off_demande_ne_porte_pas_de_cause(converted):
    engine = _engine(converted)            # enable_cuda_graphs=False : off demandé, pas un repli
    assert "graphes=off " in engine.regime_ligne() and "repli eager" not in engine.regime_ligne()


def test_une_capture_ratee_est_un_repli_compte_et_nomme():
    """`GraphRunner._capture` en échec : `enabled=False` ET un repli compté avec sa raison —
    avant, `repli_eager=0` et `replis_eager_raisons=[]` sur un service entièrement en eager."""
    from acvram.engine import graphs as G
    g = G.GraphRunner.__new__(G.GraphRunner)
    g._raisons_eager_vues = set(); g.replis_eager = 0
    g._eager("capture impossible (godet b=1, ql=1) : AcceleratorError: CUDA error: out of memory")
    g._eager("capture impossible (godet b=1, ql=1) : AcceleratorError: CUDA error: out of memory")
    assert g.replis_eager == 2 and len(g._raisons_eager_vues) == 1
    assert any("capture impossible (godet b=1" in r for r in g._raisons_eager_vues)


def test_la_clause_graphes_du_calcul_nominal_suit_enabled(converted):
    """`regime_ligne()` mélange plusieurs facteurs (piles, exils, cartes) —
    isoler ici la clause graphes de `nominal` (même formule que
    `regime_ligne`) pour ne pas dépendre des autres, sur ce point jouet."""
    engine = _engine(converted)
    engine._graphes_demandes = True
    engine.graphs = _FauxGraphRunner()
    r = engine.regime()
    assert (r["graphes"] or not r["graphes_demandes"]) is True
    engine.graphs.enabled = False
    r2 = engine.regime()
    assert (r2["graphes"] or not r2["graphes_demandes"]) is False
