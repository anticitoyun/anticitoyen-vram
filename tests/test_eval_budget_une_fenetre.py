"""verdict-ppl-31b-ab-22-09 : `acvram eval` sur le 31B exilait 11 MLP alors
que le modèle tient seul — `perplexity` chargeait sans `max_model_len` ni
`max_concurrent_seqs`, et `_replanifier` dimensionnait le cache KV et la
réserve de préfill comme un serveur (kv_planned_seqs × kv_max_tokens du
manifeste). Reproduction à sec sur un faux 31B (manifeste : kv_max_tokens
9 791, kv_planned_seqs 8, spec Gemma 4 31B) : les options que le planificateur
reçoit, et la réserve de préfill, avant et après ; et `perplexity` passe
bien 1 séquence × (window + bloc) au chargeur."""
import types

import pytest
import torch


def _spec_31b():
    from acvram.engine.config import ModelSpec
    return ModelSpec(name="faux-31b", model_type="gemma4_text", architecture="gemma4", hidden_size=5376,
                     intermediate_size=21504, num_layers=60, num_attention_heads=32, num_key_value_heads=16,
                     head_dim=256, vocab_size=262144, max_position_embeddings=262144, rope_theta=1e6)


_MANIFEST = {"plan": {"tiers": [{"name": "cuda:0", "kind": "gpu"}], "kv_max_tokens": 9791, "kv_planned_seqs": 8,
                      "kv_bytes_per_token": 495360}}


def _options_recues(monkeypatch, **kw):
    """Ce que `_replanifier` demande au planificateur pour ces arguments."""
    from acvram.engine import loader
    from acvram.hardware import detect
    from acvram.memory import tiering
    recu = {}
    monkeypatch.setattr(detect, "detect_rig", lambda: types.SimpleNamespace(gpus=[types.SimpleNamespace(index=0)]))

    def faux_plan(spec, rig, opts):
        recu["ctx"], recu["seqs"] = opts.max_model_len, opts.max_concurrent_seqs
        raise RuntimeError("stop")            # `_replanifier` rend None sur exception : suffit ici
    monkeypatch.setattr(tiering, "auto_plan", faux_plan)
    monkeypatch.delenv("ACVRAM_PLAN_FIGE", raising=False)
    monkeypatch.delenv("ACVRAM_SANS_REPLAN", raising=False)
    assert loader._replanifier(_MANIFEST, _spec_31b(), **kw) is None
    return recu


def test_sans_arguments_le_plan_est_celui_d_un_serveur(monkeypatch):
    from acvram.engine.loader import _reserve_prefill
    spec = _spec_31b()
    recu = _options_recues(monkeypatch)
    assert recu == {"ctx": 9791, "seqs": 8}                       # 8 × 9 791 × 495 360 o = 36 Gio demandés au cache KV
    assert _reserve_prefill(spec, None, _MANIFEST) > 2 * 2 ** 30   # réserve de préfill sur 8 192 : > 2 Gio
    assert _reserve_prefill(spec, 9791, _MANIFEST) > 2.5 * 2 ** 30


def test_avec_une_fenetre_le_budget_est_une_fenetre(monkeypatch):
    from acvram.engine.loader import _reserve_prefill
    spec = _spec_31b()
    recu = _options_recues(monkeypatch, max_model_len=2048 + 16, max_concurrent_seqs=1)
    assert recu == {"ctx": 2064, "seqs": 1}                       # 1 × 2 064 × 495 360 o = 0,95 Gio
    assert _reserve_prefill(spec, 2064, _MANIFEST) < 1 * 2 ** 30    # 0,78 Gio
    # ordre de grandeur de la falaise : serveur ≈ 4,5 + 2,9 Gio de budget hors poids, évaluation ≈ 0,95 + 0,78
    serveur = min(8 * 9791 * 495360, int(4.5 * 2 ** 30)) + _reserve_prefill(spec, 9791, _MANIFEST)
    evaluation = 2064 * 495360 + _reserve_prefill(spec, 2064, _MANIFEST)
    assert serveur > 7 * 2 ** 30 and evaluation < 2 * 2 ** 30


def test_perplexity_charge_avec_une_fenetre(monkeypatch, tmp_path):
    from acvram import evaluate
    from acvram.engine import loader
    recu = {}

    def faux_load(model_dir, **kw):
        recu.update(kw)
        raise SystemExit("stop")
    monkeypatch.setattr(loader, "load_model", faux_load)
    with pytest.raises(SystemExit):
        evaluate.perplexity(str(tmp_path), str(tmp_path / "corpus.txt"), window=2048, stride=2048, device="cpu")
    assert recu["max_model_len"] == 2048 + 16 and recu["max_concurrent_seqs"] == 1
