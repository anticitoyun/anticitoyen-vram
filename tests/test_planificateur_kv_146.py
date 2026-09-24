"""Pièce 146 (2) : le planificateur dimensionne le cache KV pour la DEMANDE (max_model_len × max_concurrent_seqs), pas
pour une seule séquence. Avant : toutes les fractions essayées étaient faisables dès qu'UNE séquence tenait, le débit
estimé était ex æquo, et `max` gardait la première, 0,06 de la VRAM — 14 256 jetons sur DeepSeek-R1-Distill-Qwen-32B à
8 × 2 560 (20 480 demandés), 9,6 Gio restant libres après chargement (mesure 142, 24/09). Specs : sections `model` des
manifestes des trois modèles mesurés (tests/specs_planificateur_146.json), rig du profil de la machine."""

import json
import os

import pytest

from acvram.engine.config import ModelSpec
from acvram.hardware.profiles import load_profile
from acvram.memory.tiering import PlannerOptions, _subset_rig, auto_plan

_SPECS = json.load(open(os.path.join(os.path.dirname(__file__), "specs_planificateur_146.json"), encoding="utf-8"))
DEMANDE = 8 * 2560


def _spec(nom):
    return ModelSpec(**{k: v for k, v in _SPECS[nom].items() if k in ModelSpec.__dataclass_fields__})


def _exil(p):
    return sum((l.mlp_bytes if l.mlp_storage == "cpu" else 0) + (l.attn_bytes if l.attn_storage == "cpu" else 0)
               for l in p.layers)


@pytest.fixture(scope="module", params=["5090", "rig"])
def rig(request):
    r = load_profile("rig-14900k-5090-3080ti")
    return _subset_rig(r, 1) if request.param == "5090" else r


def _plan(nom, rig, seqs=8, ctx=2560):
    p, essais = auto_plan(_spec(nom), rig, PlannerOptions(max_model_len=ctx, max_concurrent_seqs=seqs))
    return p, essais


def test_dense_32b_tient_la_demande(rig):
    p, _ = _plan("qwen32", rig)
    assert p.kv_max_tokens >= DEMANDE, f"{p.kv_max_tokens} jetons KV < {DEMANDE} demandés"
    assert _exil(p) == 0
    assert len([t for t in p.tiers if t.kind == "gpu"]) == 1, "le KV ne doit pas ajouter la seconde carte"


def test_qwen38_inchange(rig):
    p, _ = _plan("qwen38", rig)
    assert p.kv_max_tokens == DEMANDE or p.kv_max_tokens // 1024 == DEMANDE // 1024
    assert _exil(p) == 0


def test_gemma31_au_plus_ce_que_la_vram_laisse(rig):
    p, essais = _plan("gemma31", rig)
    p006 = min((e for e in essais if not e["overflow"]), key=lambda e: e["kv_fraction"])
    assert p.kv_max_tokens > p006["kv_tokens"], "gemma31 : le KV doit dépasser celui de la fraction 0,06"
    assert _exil(p) == 0
    assert len([t for t in p.tiers if t.kind == "gpu"]) == 1


def test_une_sequence_demande_une_sequence(rig):
    """Le départage ne gonfle pas le cache au-delà de la demande : 1 × 2 560 → au plus la fraction 0,06 déjà suffisante."""
    p, _ = _plan("qwen32", rig, seqs=1)
    assert 2560 <= p.kv_max_tokens <= 2 * 2560
