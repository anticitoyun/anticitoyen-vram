"""lic (01/10, preuve S1 d'poste6) : sous ACVRAM_KV_FORMAT=bf16, « 640 blocs KV nécessaires, 322-325 en tout » à
max_model_len 10 240. Le planificateur budgétait les octets d'un jeton avec `kv_bits` = 8 quel que soit le format
(tiering.py, `kv_per_tok = spec.kv_bytes_per_token(kv_bits, fmt=…)` : seuls lm4 et k8v4 ajustaient), et le chargeur
découpait ce budget en blocs du format RÉEL (loader.py, `_kv_blocks_per_device`, `KVCacheConfig(dtype=kv_fmt)`) :
10 240 × 8,125 / 16 = 5 200 jetons = 325 blocs. Contrat vérifié ici, format par format : les blocs construits tiennent
les jetons que le plan annonce, et un plan qui a la place tient la demande. Processeur seul."""
from __future__ import annotations

import json
import os

import pytest

from acvram.engine.config import ModelSpec
from acvram.engine.loader import _kv_blocks_per_device
from acvram.hardware.profiles import load_profile
from acvram.memory import tiering
from acvram.memory.kvcache import BLOCK_SIZE, KVCacheConfig
from acvram.memory.tiering import PlannerOptions, _subset_rig, auto_plan

FORMATS = ["int8", "fp8_e4m3", "bf16", "fp16", "k8v4", "lm4"]
_SPECS = json.load(open(os.path.join(os.path.dirname(__file__), "specs_planificateur_146.json"), encoding="utf-8"))


def _spec():
    s = _SPECS["qwen32"]      # 64 couches, 8 têtes KV × 128 : un dense comme Devstral-24B, en plus lourd
    return ModelSpec(**{k: v for k, v in s.items() if k in ModelSpec.__dataclass_fields__})


@pytest.fixture(scope="module")
def rig():
    return _subset_rig(load_profile("rig-14900k-5090-3080ti"), 1)


@pytest.mark.parametrize("fmt", FORMATS)
def test_octets_par_jeton_du_plan_egalent_le_bloc_du_format(monkeypatch, fmt):
    """Ce que le planificateur budgète pour un jeton (kv_bits du défaut des options, comme tiering) × BLOCK_SIZE
    = les octets d'un bloc construit dans ce format, à l'octet près par couche."""
    monkeypatch.setattr(tiering, "_KV_FORMAT", fmt)
    spec = _spec()
    bits = tiering.kv_lm4.bits(fmt) if fmt in tiering.kv_lm4.FORMATS else PlannerOptions().kv_bits
    par_couche = spec.kv_bytes_per_token(bits, fmt=fmt) / spec.couches_avec_kv
    bloc = KVCacheConfig(num_layers=1, num_kv_heads=spec.num_key_value_heads, head_dim=spec.head_dim,
                         num_blocks=1, dtype=fmt).bytes_per_block()
    assert par_couche * BLOCK_SIZE == pytest.approx(bloc, abs=BLOCK_SIZE), (fmt, par_couche * BLOCK_SIZE, bloc)


@pytest.mark.parametrize("fmt", FORMATS)
def test_les_blocs_construits_tiennent_les_jetons_du_plan(monkeypatch, rig, fmt):
    """Le cas S1 : un seul tenant, 10 240 jetons, une 5090 qui a la place. Avant le correctif, bf16 et fp16 rendaient
    la moitié des blocs annoncés (5 200 jetons pour 10 240 planifiés)."""
    monkeypatch.setattr(tiering, "_KV_FORMAT", fmt)
    spec = _spec()
    p, _ = auto_plan(spec, rig, PlannerOptions(max_model_len=10240, max_concurrent_seqs=1))
    assert p.kv_max_tokens >= 10240, f"{fmt} : le plan ne tient que {p.kv_max_tokens} jetons"
    blocs = _kv_blocks_per_device(p, spec, 10240)
    jetons = sum(blocs.values()) * BLOCK_SIZE
    assert jetons >= p.kv_max_tokens - BLOCK_SIZE * len(blocs), \
        f"{fmt} : {sum(blocs.values())} blocs = {jetons} jetons pour {p.kv_max_tokens} planifiés"
