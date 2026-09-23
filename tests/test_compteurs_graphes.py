"""Compteurs de graphes CUDA exposés par `/metrics` et par la ligne de régime
(pièce 44, Océane, à sec).

La série b=1 de Manon décroît puis plafonne (354 → 272 t/s, `b3e3fbd5`) et
l'hypothèse H1 — la sonde a franchi un godet de blocs, donc elle a payé une
capture pendant la fenêtre — ne pouvait pas être départagée : `captures`,
`replays` et `len(graphs)` existent dans `GraphRunner` depuis longtemps
(`graphs.py:292-293`, `273`) et ne sortaient NULLE PART. Un compteur qu'on ne
peut pas relever ne réfute rien.

Deux exigences, et le test doit pouvoir rendre faux sur chacune :
  1. les trois compteurs sont dans `EngineStats.to_dict()`, donc dans
     `/metrics` (`server/app.py:940`), et valent les valeurs VIVANTES ;
  2. la ligne de régime les porte sans casser la clé `graphes=on|off` que
     d'autres lecteurs cherchent telle quelle.
"""
import torch

from acvram.engine.loader import load_model
from acvram.engine.runner import Engine, EngineStats


class _FauxGraphRunner:
    """Ce que `regime()` et les compteurs lisent — pas un vrai GraphRunner."""
    def __init__(self, graphes=2, captures=5, replays=203):
        self.enabled = True
        self.raison = None
        self.graphs = {(1, 8 * i): {} for i in range(graphes)}
        self.captures = captures
        self.replays = replays


def _engine(converted):
    loaded = load_model(converted, dtype=torch.bfloat16, max_model_len=512,
                        device_override="cpu")
    return Engine(loaded, None, max_batch_size=1, max_model_len=512,
                 enable_cuda_graphs=False)


def test_stats_sans_source_rend_trois_zeros():
    """À sec ou graphes coupés : zéro est une mesure, pas une clé manquante —
    une clé absente serait relue comme 0 par un client qui l'ignore, et c'est
    exactement la faute qui a coûté « 0,0 ms pour 39 410 jetons »."""
    d = EngineStats().to_dict()
    assert d["graphes_nombre"] == 0
    assert d["graphes_captures"] == 0
    assert d["graphes_replays"] == 0


def test_stats_lit_les_compteurs_vivants():
    s = EngineStats()
    faux = _FauxGraphRunner()
    s.source_graphes = lambda: faux
    d = s.to_dict()
    assert (d["graphes_nombre"], d["graphes_captures"], d["graphes_replays"]) == (2, 5, 203)
    # vivants : une capture de plus pendant le service se voit au relevé suivant
    faux.graphs[(1, 64)] = {}
    faux.captures += 1
    faux.replays += 7
    d2 = s.to_dict()
    assert (d2["graphes_nombre"], d2["graphes_captures"], d2["graphes_replays"]) == (3, 6, 210)


def test_metrics_et_regime_lisent_la_meme_source(converted):
    engine = _engine(converted)
    engine._graphes_demandes = True
    engine.graphs = _FauxGraphRunner(graphes=4, captures=5, replays=1031)
    r = engine.regime()
    d = engine.stats.to_dict()
    assert (r["graphes_nombre"], r["graphes_captures"], r["graphes_replays"]) == (4, 5, 1031)
    assert (d["graphes_nombre"], d["graphes_captures"], d["graphes_replays"]) == (4, 5, 1031)


def test_ligne_de_regime_porte_les_trois_compteurs(converted):
    engine = _engine(converted)
    engine._graphes_demandes = True
    engine.graphs = _FauxGraphRunner(graphes=4, captures=5, replays=1031)
    ligne = engine.regime_ligne()
    assert "graphes_n=4 captures=5 replays=1031" in ligne
    # la clé historique reste lisible telle quelle (capture-godets.py:41)
    assert "graphes=on" in ligne


def test_ligne_de_regime_a_sec_sans_graphes(converted):
    """Graphes coupés : trois zéros sur la ligne, et toujours `graphes=off `."""
    engine = _engine(converted)
    ligne = engine.regime_ligne()
    assert "graphes_n=0 captures=0 replays=0" in ligne
    assert "graphes=off " in ligne
    assert "graphes=on" not in ligne
