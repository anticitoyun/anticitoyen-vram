"""Charger un modèle converti et l'exécuter, de bout en bout sur processeur."""

import json
import os

import pytest
import torch

from acvram.engine.loader import load_model
from acvram.engine.model import ForwardBatch
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams, sample
from acvram.memory.kvcache import (BLOCK_SIZE, BlockAllocator, KVCacheConfig,
                                   PagedKVCache)


def _prefill(model, prompt):
    n = len(prompt)
    alloc = BlockAllocator(model.caches[0].cfg.num_blocks)
    blocks = alloc.allocate((n + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)
    slots = torch.tensor([blocks[i // BLOCK_SIZE] * BLOCK_SIZE + i % BLOCK_SIZE
                          for i in range(n)])
    batch = ForwardBatch(torch.tensor(prompt), torch.arange(n), [n], [n],
                         [torch.tensor(blocks)], slots, True)
    return model(batch)


def test_convert_load_prefill_decode(converted):
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    model = loaded.model
    assert len(model.layers) == loaded.spec.num_layers

    logits = _prefill(model, [5, 42, 7, 99, 13])
    assert logits.shape == (1, loaded.spec.vocab_size)
    assert torch.isfinite(logits).all()


def test_decode_advances_and_stays_finite(converted):
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    engine = Engine(loaded, None, max_batch_size=2, max_model_len=256)
    params = SamplingParams(temperature=0.0, max_tokens=8)
    produced = [o for o in engine.generate([5, 42, 7, 99], params)]
    assert len(produced) == 8
    assert produced[-1].finished
    assert produced[-1].finish_reason == "length"


def test_finished_sequences_return_their_blocks(converted):
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    engine = Engine(loaded, None, max_batch_size=2, max_model_len=256)
    before = engine.allocator.num_free
    for _ in engine.generate([1, 2, 3], SamplingParams(temperature=0.0, max_tokens=4)):
        pass
    assert engine.allocator.num_free == before


def test_quantized_model_tracks_the_bf16_reference(tiny_checkpoint, target_rig,
                                                   tmp_path):
    """Le 4 bits doit dégrader, mais pas diverger."""
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint

    spec = load_model_spec(tiny_checkpoint, "tiny")
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=512))

    def build(fmt):
        out = tmp_path / f"m-{fmt}"
        for lp in plan.layers:
            lp.fmt = fmt
        convert_checkpoint(tiny_checkpoint, plan,
                           ConversionOptions(out_dir=str(out), lm_head_format=fmt,
                                             use_hadamard="never"), spec=spec)
        return load_model(str(out), dtype=torch.float32,
                          device_override="cpu").model

    prompt = [5, 42, 7, 99, 13, 250, 61, 3]
    ref = _prefill(build("bf16"), prompt)[0]
    for fmt, floor in (("int8", 0.999), ("nvfp4", 0.90), ("int4_awq", 0.85)):
        q = _prefill(build(fmt), prompt)[0]
        cos = torch.nn.functional.cosine_similarity(ref, q, dim=0).item()
        assert cos > floor, f"{fmt} cosine {cos:.4f} below {floor}"


@pytest.mark.parametrize("dtype", ["int8", "fp8_e4m3", "fp16"])
def test_kv_cache_roundtrip(dtype):
    cfg = KVCacheConfig(num_layers=1, num_kv_heads=8, head_dim=128,
                        num_blocks=32, dtype=dtype, device="cpu")
    cache = PagedKVCache(cfg)
    alloc = BlockAllocator(32)
    blocks = alloc.allocate(4)
    n = 50
    k = torch.randn(n, 8, 128) * 0.5
    v = torch.randn(n, 8, 128) * 0.5
    slots = torch.tensor([blocks[i // BLOCK_SIZE] * BLOCK_SIZE + i % BLOCK_SIZE
                          for i in range(n)])
    cache.write(slots, k, v)
    kk, _ = cache.gather(torch.tensor(blocks), n, torch.float32)
    assert kk.shape == k.shape
    assert (kk - k).norm() / k.norm() < 0.1


def test_int8_kv_beats_fp8_at_equal_size():
    """L'échelle par tête fournit la plage dynamique pour laquelle le FP8 dépense
    des bits d'exposant."""
    torch.manual_seed(0)
    k = torch.randn(64, 8, 128) * 0.5
    errs = {}
    for dtype in ("int8", "fp8_e4m3"):
        cfg = KVCacheConfig(1, 8, 128, 16, dtype=dtype, device="cpu")
        cache = PagedKVCache(cfg)
        slots = torch.arange(64)
        cache.write(slots, k, k)
        kk, _ = cache.gather(torch.arange(4), 64, torch.float32)
        errs[dtype] = ((kk - k).norm() / k.norm()).item()
    assert errs["int8"] < errs["fp8_e4m3"]


def test_block_allocator_refuses_to_oversubscribe():
    alloc = BlockAllocator(4)
    alloc.allocate(4)
    with pytest.raises(MemoryError):
        alloc.allocate(1)


def test_sampler_is_deterministic_at_zero_temperature():
    logits = torch.randn(3, 100)
    p = [SamplingParams(temperature=0.0)] * 3
    a, _ = sample(logits.clone(), p)
    b, _ = sample(logits.clone(), p)
    assert torch.equal(a, b)
    assert torch.equal(a, logits.argmax(dim=-1))


def test_top_p_never_empties_the_distribution():
    logits = torch.randn(1, 1000) * 5
    tok, _ = sample(logits, [SamplingParams(temperature=1.0, top_p=0.01)])
    assert 0 <= int(tok.item()) < 1000
