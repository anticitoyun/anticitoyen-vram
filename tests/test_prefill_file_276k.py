"""276 k : préfill EN FILE dans le pas groupé (ACVRAM_PREFILL_FILE=1) — cassants à sec (tiny, CPU) :
(1) à b = 2, le premier jeton de chaque séquence est CELUI de la même invite à b = 1 (un forward par séquence : indépendant de
la composition du lot) ; (2) les sorties sont émises PENDANT le pas par `Engine.emettre`, dans l'ordre d'admission, et ne sont
pas rendues une seconde fois ; (3) sans crochet, elles sont rendues à la fin ; (4) n = 1 : chemin groupé, aucune émission ;
(5) EngineService pose le crochet ; (6) régime déclaré."""
from __future__ import annotations

import torch

from acvram.engine import runner as R
from acvram.engine.sampler import SamplingParams


def _moteur(converted, max_batch=2):
    from acvram.engine.loader import load_model
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    return R.Engine(loaded, None, max_batch_size=max_batch, max_model_len=256, enable_cuda_graphs=False)


INVITES = [list(range(3, 30)), list(range(7, 20)) + [3, 3, 3]]


def _premiers(engine, invites, emettre=None):
    engine.emettre = emettre
    seqs = [engine.add_request(ids, SamplingParams(max_tokens=1, temperature=0.0), request_id=f"r{i}")
            for i, ids in enumerate(invites)]
    outs = engine.step()
    return seqs, outs


def test_premier_jeton_b3_egal_b1_et_emission_dans_l_ordre(converted, monkeypatch):
    monkeypatch.setattr(R, "_PREFILL_FILE", True)
    # référence : chaque invite seule (chemin groupé à n = 1)
    ref = {}
    for i, ids in enumerate(INVITES):
        e = _moteur(converted)
        seqs, outs = _premiers(e, [ids])
        ref[i] = outs[0].token_ids[0]
    e = _moteur(converted)
    emis = []
    seqs, outs = _premiers(e, INVITES, emettre=emis.append)
    assert outs == [], f"sorties rendues à la fin alors qu'émises dans le pas : {outs}"
    assert [o.request_id for lot in emis for o in lot] == ["r0", "r1"]         # ordre d'admission
    assert [o.token_ids[0] for lot in emis for o in lot] == [ref[0], ref[1]]
    assert e.stats.prefills_en_file == 2 and e.stats.pas_avec_prefill == 1


def test_sans_crochet_les_sorties_sont_rendues_a_la_fin(converted, monkeypatch):
    monkeypatch.setattr(R, "_PREFILL_FILE", True)
    e = _moteur(converted)
    seqs, outs = _premiers(e, INVITES)
    assert [o.request_id for o in outs] == ["r0", "r1"] and all(len(o.token_ids) == 1 for o in outs)


def test_une_sequence_reste_sur_le_chemin_groupe(converted, monkeypatch):
    monkeypatch.setattr(R, "_PREFILL_FILE", True)
    e = _moteur(converted)
    emis = []
    seqs, outs = _premiers(e, INVITES[:1], emettre=emis.append)
    assert emis == [] and len(outs) == 1 and e.stats.prefills_en_file == 0


def test_temoin_chemin_groupe_inchange(converted, monkeypatch):
    monkeypatch.setattr(R, "_PREFILL_FILE", False)
    e = _moteur(converted)
    emis = []
    seqs, outs = _premiers(e, INVITES, emettre=emis.append)
    assert emis == [] and len(outs) == 2 and e.stats.prefills_en_file == 0


def test_engine_service_pose_le_crochet():
    from acvram.server.app import EngineService

    class _E:
        emettre = None
    e = _E()
    svc = EngineService(e, None, "m")
    assert callable(e.emettre)
    vus = []
    svc._deliver = vus.append
    e.emettre([1, 2])
    assert vus == [1, 2]


def test_variable_de_regime_declaree():
    from acvram import regime
    assert "PREFILL_FILE" in {v.nom for v in regime.VARIABLES}
