"""Pièce 85 (23/09) : effondrement du service à lots successifs. À sec, sans carte.

1. `/metrics` appelle `Engine.regime()`, qui relisait l'horloge SOUS CHARGE à
   chaque appel : un fil de matmuls + `synchronize` dans le processus de
   service, qui invalide une capture en cours (graphes coupés à vie).
2. Le plafond MAX_GRAPHS refusait une forme nouvelle sans compter ni nommer
   le repli (`repli_eager` restait 0)."""
from types import SimpleNamespace

import torch

from acvram import eco
from acvram.engine import graphs as G
from acvram.engine import runner


def test_regime_ne_charge_la_carte_qu_une_fois(monkeypatch):
    appels = []
    monkeypatch.setattr(eco, "etat_eco", lambda relire=False: appels.append(relire) or {})
    monkeypatch.setattr(runner, "_ECO_RELU", False)
    for _ in range(5):                          # ligne de régime du chargement, puis quatre /metrics
        runner._etat_eco()
    assert appels == [True, False, False, False, False]


def _gr_plein():
    gr = G.GraphRunner.__new__(G.GraphRunner)
    gr.enabled, gr.paged_ok, gr.hybrid_layers = True, True, []
    gr.max_model_len = 2304
    gr.graphs = {("occupe", i): {} for i in range(G.MAX_GRAPHS)}
    gr._raisons_eager_vues, gr.replis_eager = set(), 0
    return gr


def test_plafond_max_graphs_compte_et_nomme_le_repli(capsys):
    gr = _gr_plein()
    lot = SimpleNamespace(is_prefill=False, query_lens=[1] * 12, batch_size=12,
                          block_tables=[torch.zeros(70, dtype=torch.int32)] * 12)
    assert gr.preparer(lot) is False and gr.preparer(lot) is False
    assert gr.replis_eager == 2                  # chaque pas refusé compte
    assert any("MAX_GRAPHS" in r for r in gr._raisons_eager_vues)
    assert capsys.readouterr().out.count("repli eager") == 1   # nommé une fois


def test_defaut_couvre_les_cles_denses_d_un_service_a_lots_successifs():
    """b = 1..12 et contexte jusqu'à 2 304 : toutes les clés (godet b, nblk) denses
    tiennent sous le plafond par défaut — à 16, elles n'y tenaient pas (30)."""
    from acvram.memory.kvcache import BLOCK_SIZE, bucket_blocks
    cles = {(G.godet_lot(b), bucket_blocks(n)) for b in range(1, 13)
            for n in range(1, (2304 + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)}
    assert len(cles) == 30 and len(cles) <= G.MAX_GRAPHS
