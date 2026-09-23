"""Pièce 69 (23/09) : le graphe en double (`ACVRAM_GRAPHE_DOUBLE=1`, godet 1) rend, au bit, la même sortie que le
graphe simple — dans le MÊME moteur, le rejeu A puis le rejeu B sur le même état donnent des logits identiques,
sur N pas, et l alternance est reproductible ; le coût mémoire du second graphe est relevé."""
import pytest
import torch

pytestmark = pytest.mark.gpu_requis
from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="pas de peripherique CUDA")


@pytest.fixture(autouse=True)
def _lot_du_test_pas_du_plan(monkeypatch):
    monkeypatch.setenv("ACVRAM_KV_PLAN_OVERRIDE", "1")


@needs_cuda
def test_graphe_double_au_bit_sur_n_pas(converted, monkeypatch):
    monkeypatch.setenv("ACVRAM_GRAPHE_DOUBLE", "1")
    loaded = load_model(converted, dtype=torch.bfloat16, device_override="cuda:0")
    e = Engine(loaded, None, max_batch_size=1, max_model_len=256, enable_cuda_graphs=True)
    assert e.graphs.graphe_double
    seq = e.add_request([7, 3, 9, 1, 4, 8, 2, 5] * 4, SamplingParams(temperature=0.0, max_tokens=40))
    e.step()                                                     # préremplissage
    n_pas, vus = 12, 0
    for _ in range(n_pas):
        dec = e._decodables()
        if not dec:
            break
        for s in dec:
            assert e._grow(s)
        batch = e._build_batch(dec, prefill=False)
        ya = e.graphs.run(batch)
        assert ya is not None, "le godet 1 n est pas capturé"
        ya = ya.clone()
        yb = e.graphs.run(batch).clone()                          # même état, même lot : l autre graphe
        yc = e.graphs.run(batch).clone()                          # retour au premier : reproductible
        assert torch.equal(ya, yb), "graphe B ≠ graphe A au bit"
        assert torch.equal(ya, yc), "rejeu A non reproductible"
        e._emit(yc, dec)
        vus += 1
    assert vus >= 8
    assert e.graphs.captures >= 2
    print(f"[graphe-double] second graphe : +{e.graphs.graphe_double_mio:.1f} Mio réservés, {vus} pas au bit")
    assert e.graphs.graphe_double_mio >= 0
