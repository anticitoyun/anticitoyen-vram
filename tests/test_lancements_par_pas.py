"""Cliquet de lancements par pas de décodage b=12 sous graphes (Sage,
sage-reprise-15-09-b § 2 : ≤ 1 700 avec route+pack ; le routage torch en
faisait 3 677). Exige le modèle Coder-30B et la carte (sous carte.sh) —
sauté sinon. Mesuré le 15/09 : 1 517 (39 noyaux)."""
import os
import pytest

@pytest.fixture(autouse=True)
def _lot_du_test_pas_du_plan(monkeypatch):
    """Le converti du disque porte son plan (kv_planned_seqs) ; le test dimensionne son lot (T4 20/09)."""
    monkeypatch.setenv("ACVRAM_KV_PLAN_OVERRIDE", "1")
import torch
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../outils'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


MODELE = os.path.join(os.environ.get("ACVRAM_MODELES", _RACINE),
                      "Qwen3-Coder-30B-A3B-nvfp4")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not os.path.isdir(MODELE),
                                reason=f"carte requise ; alias absent sous racine_modeles() : {MODELE}")


def test_lancements_par_pas_b12_sous_graphes():
    from torch.profiler import profile, ProfilerActivity
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    loaded = load_model(MODELE, dtype=torch.bfloat16, device_override="cuda:0")
    eng = Engine(loaded, None, max_batch_size=12, max_model_len=1024, enable_cuda_graphs=True,
                 enable_prefix_cache=False)
    eng._eos = set()
    for b in range(12):
        eng.add_request([(1000 + b * 101 + i * 13) % 150000 + 10 for i in range(128)],
                        SamplingParams(temperature=0.0, max_tokens=64))
    while any(not s.prefilled for s in eng.running) or eng.waiting:
        eng.step()
    for _ in range(3):
        eng.step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        eng.step(); torch.cuda.synchronize()
    ev = [e for e in prof.key_averages() if e.self_device_time_total > 0 and not e.key.startswith("aten::")
          and not e.key.startswith("cuda") and "Graph" not in e.key]
    n = sum(e.count for e in ev)
    assert n <= 1700, f"{n} lancements par pas (39 noyaux, 1 517 attendus avec route+pack)"
