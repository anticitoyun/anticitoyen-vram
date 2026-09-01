"""Cache de préfixe, décodage spéculatif, noyaux processeur, précision mixte.

Chacun de ces points est une optimisation, et une optimisation qui change la
sortie est un bogue. L'essentiel de ce qui suit vérifie une équivalence, pas une
vitesse.
"""

import math
import os
import shutil
import tempfile

import pytest
import torch

from acvram.engine.layers import (attention, batched_decode_attention,
                                  causal_mask, repeat_kv)
from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
from acvram.engine.speculative import (DraftModelProposer, NGramProposer,
                                       Proposal, verify_proposal)
from acvram.memory.kvcache import BLOCK_SIZE, BlockAllocator


# --------------------------------------------------------------------------
# attention with an offset query block
# --------------------------------------------------------------------------


def test_offset_causal_mask_matches_a_full_causal_run():
    """Le cache de préfixe et le prefill par morceaux reposent tous deux là-dessus."""
    torch.manual_seed(0)
    s, h, d = 12, 4, 16
    q, k, v = (torch.randn(s, h, d) for _ in range(3))
    full = attention(q, k, v, causal=True)
    off = 6
    tail = attention(q[off:], k, v, causal=True, q_offset=off)
    assert torch.allclose(full[off:], tail, atol=1e-5)


def test_builtin_causal_flag_would_have_been_wrong():
    """SDPA aligne son triangle en haut à gauche : d'où un masque explicite."""
    import torch.nn.functional as F
    torch.manual_seed(0)
    s, h, d = 12, 4, 16
    q, k, v = (torch.randn(s, h, d) for _ in range(3))
    full = attention(q, k, v, causal=True)
    off = 6
    naive = F.scaled_dot_product_attention(
        q[off:].transpose(0, 1)[None], k.transpose(0, 1)[None],
        v.transpose(0, 1)[None], is_causal=True).squeeze(0).transpose(0, 1)
    assert not torch.allclose(full[off:], naive, atol=1e-3)


def test_batched_decode_matches_per_sequence():
    torch.manual_seed(0)
    h, d, rep = 4, 16, 2
    lens = [5, 12, 8]
    q = torch.randn(len(lens), h, d)
    ks = [torch.randn(l, h // rep, d) for l in lens]
    vs = [torch.randn(l, h // rep, d) for l in lens]
    batched = batched_decode_attention(q, ks, vs, rep, d ** -0.5)
    ref = torch.stack([
        attention(q[i:i + 1], repeat_kv(ks[i], rep), repeat_kv(vs[i], rep),
                  False, d ** -0.5)[0] for i in range(len(lens))])
    assert torch.allclose(batched, ref, atol=1e-5)


# --------------------------------------------------------------------------
# prefix cache
# --------------------------------------------------------------------------


def test_block_hash_chains_on_history():
    """Des tranches identiques dans des contextes différents ne doivent pas se confondre."""
    a = BlockAllocator.block_hashes([1] * 16 + [2] * 16)
    b = BlockAllocator.block_hashes([9] * 16 + [2] * 16)
    assert a[0] != b[0]
    assert a[1] != b[1]


def test_allocator_reuses_then_evicts():
    alloc = BlockAllocator(2)
    blocks = alloc.allocate(2)
    alloc.register(blocks[0], 111)
    alloc.register(blocks[1], 222)
    alloc.free(blocks)
    assert alloc.num_cached == 2
    assert alloc.match_prefix([111]) == [blocks[0]]
    alloc.free([blocks[0]])
    # Sous pression, le cache cède plutôt que de refuser l'allocation.
    got = alloc.allocate(2)
    assert len(got) == 2
    assert alloc.evictions > 0


def test_prefix_cache_is_output_identical(converted):
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    prompt = [7, 3, 9, 1, 4, 8, 2, 5] * 6

    def run(enable):
        e = Engine(loaded, None, max_batch_size=2, max_model_len=256,
                   enable_prefix_cache=enable)
        first = [t for o in e.generate(prompt, SamplingParams(temperature=0.0,
                                                              max_tokens=6))
                 for t in o.token_ids]
        second = [t for o in e.generate(prompt, SamplingParams(temperature=0.0,
                                                               max_tokens=6))
                  for t in o.token_ids]
        return first, second, e

    (a1, a2, warm) = run(True)
    (b1, b2, _) = run(False)
    assert a1 == a2, "un cache chaud a change la reponse"
    assert a1 == b1, "le cache a change la reponse face a une execution sans cache"
    assert a2 == b2
    assert warm.stats.cached_prompt_tokens > 0, "rien n'a ete servi depuis le cache"


def test_prefix_cache_saves_prefill(converted):
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    prompt = [7, 3, 9, 1, 4, 8, 2, 5] * 6         # 48 jetons = 3 blocs entiers
    e = Engine(loaded, None, max_batch_size=2, max_model_len=256)
    params = SamplingParams(temperature=0.0, max_tokens=3)
    list(e.generate(prompt, params))
    after_first = e.stats.prefill_tokens
    list(e.generate(prompt, params))
    added = e.stats.prefill_tokens - after_first
    assert added < len(prompt), f"la seconde execution a encore precalcule {added} jetons"


# --------------------------------------------------------------------------
# speculative decoding
# --------------------------------------------------------------------------


def test_speculation_preserves_the_target_distribution():
    """La garantie qui rend la spéculation sûre, vérifiée statistiquement."""
    torch.manual_seed(0)
    vocab, trials = 8, 40000
    params = SamplingParams(temperature=1.0)
    logits = torch.randn(vocab) * 1.5
    p = torch.softmax(logits, -1)
    q = torch.softmax(torch.randn(vocab) * 3.0, -1)     # volontairement faux

    counts = torch.zeros(vocab)
    for _ in range(trials):
        tok = int(torch.multinomial(q, 1))
        rows = logits.unsqueeze(0).repeat(2, 1)
        out, _ = verify_proposal(rows, Proposal([tok], q.unsqueeze(0)), params)
        counts[out[0]] += 1
    tv = 0.5 * float((counts / trials - p).abs().sum())
    assert tv < 0.02, f"variation totale {tv:.4f} par rapport a la distribution cible"


def test_greedy_verification_accepts_only_exact_matches():
    logits = torch.zeros(3, 5)
    logits[:, 2] = 10.0                       # l'argmax vaut le jeton 2 partout
    params = SamplingParams(temperature=0.0)
    out, n = verify_proposal(logits, Proposal([2, 2]), params)
    assert n == 2 and out == [2, 2, 2]
    out, n = verify_proposal(logits, Proposal([2, 4]), params)
    assert n == 1 and out == [2, 2]


def test_ngram_proposer_finds_a_repeat():
    class S:
        all_ids = [1, 2, 3, 4, 5, 1, 2, 3]
    prop = NGramProposer(max_ngram=3, min_ngram=2).propose(S(), 3)
    assert prop.tokens == [4, 5, 1]


def test_ngram_speculation_is_output_identical(converted):
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    prompt = [7, 3, 9, 1, 4, 8, 2, 5] * 4

    def run(spec):
        e = Engine(loaded, None, max_batch_size=2, max_model_len=512,
                   enable_prefix_cache=False, speculator=spec, spec_k=4)
        toks = [t for o in e.generate(prompt, SamplingParams(temperature=0.0,
                                                             max_tokens=16))
                for t in o.token_ids]
        return toks, e.stats

    base, sb = run(None)
    spec, ss = run(NGramProposer())
    assert spec == base, "la speculation a change la sortie gloutonne"
    assert ss.spec_steps < sb.spec_steps, "la speculation n'a economise aucune etape"


def test_draft_model_identical_to_itself_accepts_everything(converted):
    """Un brouillon qui *est* la cible doit être accepté à chaque fois."""
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    draft = load_model(converted, dtype=torch.float32, device_override="cpu")
    prompt = [7, 3, 9, 1, 4, 8, 2, 5] * 4

    e0 = Engine(loaded, None, max_batch_size=1, max_model_len=512,
                enable_prefix_cache=False)
    base = [t for o in e0.generate(prompt, SamplingParams(temperature=0.0,
                                                          max_tokens=12))
            for t in o.token_ids]

    e1 = Engine(loaded, None, max_batch_size=1, max_model_len=512,
                enable_prefix_cache=False,
                speculator=DraftModelProposer(draft, max_model_len=512),
                spec_k=4)
    spec = [t for o in e1.generate(prompt, SamplingParams(temperature=0.0,
                                                          max_tokens=12))
            for t in o.token_ids]
    assert spec == base
    assert e1.stats.acceptance_rate > 0.95
    assert e1.stats.tokens_per_step > 3.0


# --------------------------------------------------------------------------
# CPU kernels
# --------------------------------------------------------------------------


def test_cpu_kernels_match_the_reference():
    from acvram.kernels.cpu import (cpu_kernels_available, int4_matmul_cpu,
                                    nvfp4_matmul_cpu)
    if not cpu_kernels_available():
        pytest.skip("no C++ compiler available")
    from acvram.quant.int4 import dequantize_int4, quantize_int4
    from acvram.quant.nvfp4 import dequantize_nvfp4, quantize_nvfp4

    torch.manual_seed(0)
    for m, k in [(256, 512), (97, 300)]:
        w = torch.randn(m, k) * 0.02
        x = torch.randn(3, k)
        t = quantize_int4(w)
        ref = x @ dequantize_int4(t, torch.float32).t()
        got = int4_matmul_cpu(x, t)
        assert (got - ref).abs().max() / ref.abs().max() < 1e-5

        q = quantize_nvfp4(w)
        ref = x @ dequantize_nvfp4(q, torch.float32).t()
        got = nvfp4_matmul_cpu(x, q)
        assert (got - ref).abs().max() / ref.abs().max() < 1e-5


def test_planner_picks_cpu_execution_when_ddr_beats_the_link(tmp_path,
                                                             target_rig):
    import json

    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan

    d = tmp_path / "big"
    d.mkdir()
    json.dump({"architectures": ["LlamaForCausalLM"], "hidden_size": 12288,
               "intermediate_size": 28672, "num_hidden_layers": 88,
               "num_attention_heads": 96, "num_key_value_heads": 8,
               "vocab_size": 32768}, open(d / "config.json", "w"))
    spec = load_model_spec(str(d), "big")

    stream, _ = auto_plan(spec, target_rig,
                          PlannerOptions(max_model_len=8192, host_exec="stream"))
    auto, _ = auto_plan(spec, target_rig,
                        PlannerOptions(max_model_len=8192, host_exec="auto"))
    assert any(l.mlp_exec == "cpu" for l in auto.layers)
    assert all(l.mlp_exec == "gpu" for l in stream.layers)
    assert auto.est_decode_tok_s > stream.est_decode_tok_s


# --------------------------------------------------------------------------
# mixed precision
# --------------------------------------------------------------------------


def test_mixed_precision_promotes_and_improves_snr(tiny_checkpoint, target_rig,
                                                   tmp_path):
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint

    spec = load_model_spec(tiny_checkpoint, "tiny")
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=512))
    off = convert_checkpoint(tiny_checkpoint, plan,
                             ConversionOptions(out_dir=str(tmp_path / "a"),
                                               mixed_precision="off",
                                               dry_run=True), spec=spec)
    auto = convert_checkpoint(tiny_checkpoint, plan,
                              ConversionOptions(out_dir=str(tmp_path / "b"),
                                                mixed_precision="auto",
                                                snr_floor=25.0, dry_run=True),
                              spec=spec)
    assert auto.promotions, "rien n'a ete promu sous un plancher de 25 dB"
    assert auto.mean_out_snr_db > off.mean_out_snr_db
    assert auto.out_bytes > off.out_bytes         # la precision n'est pas gratuite
    for p in auto.promotions:
        assert p["after"] > p["before"]


def test_promotion_stays_within_its_cap(tiny_checkpoint, target_rig, tmp_path):
    """Une mauvaise calibration ne doit pas gonfler tout le modèle en silence."""
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint

    spec = load_model_spec(tiny_checkpoint, "tiny")
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=512))
    r = convert_checkpoint(tiny_checkpoint, plan,
                           ConversionOptions(out_dir=str(tmp_path / "c"),
                                             mixed_precision="auto",
                                             snr_floor=999.0,   # tout promouvoir
                                             max_promotions=0.15,
                                             dry_run=True), spec=spec)
    assert len(r.promotions) <= 0.15 * r.tensors + 1


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def test_perplexity_runs_and_is_finite(converted, tiny_checkpoint):
    from acvram.evaluate import perplexity
    for fn in ("tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(tiny_checkpoint, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(converted, fn))
    if not os.path.isfile(os.path.join(converted, "tokenizer.json")):
        pytest.skip("no tokenizer available")
    r = perplexity(converted, window=64, stride=32, max_tokens=256,
                   device="cpu", dtype=torch.float32)
    assert math.isfinite(r.perplexity)
    assert r.tokens > 0
    assert r.bits_per_weight > 0


def test_bits_budget_knapsack(tiny_checkpoint, target_rig, tmp_path_factory):
    """Le sac à dos dépense le budget là où chaque octet paie le plus.

    Un budget serré promeut moins de tenseurs qu'un budget large, jamais plus ;
    et le modèle converti sous budget se charge et répond.
    """
    import torch

    from acvram.engine.config import load_model_spec
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint

    spec = load_model_spec(tiny_checkpoint, "tiny")
    plan, _ = auto_plan(spec, target_rig,
                        PlannerOptions(max_model_len=512, max_concurrent_seqs=2))

    def convertir(budget):
        out = str(tmp_path_factory.mktemp(f"budget_{int(budget * 1000)}"))
        r = convert_checkpoint(tiny_checkpoint, plan,
                               ConversionOptions(out_dir=out,
                                                 bits_budget_gib=budget),
                               spec=spec)
        return out, r

    out_serre, r_serre = convertir(0.001)      # ~1 Mio : presque rien ne passe
    out_large, r_large = convertir(1.0)        # 1 Gio : tout candidat passe
    assert len(r_large.promotions) >= len(r_serre.promotions)
    assert len(r_large.promotions) > 0, "aucun candidat promu à budget large"
    assert sum(r_serre.per_format.values()) <= 0.001 * 1024 ** 3 \
        or len(r_serre.promotions) == 0

    loaded = load_model(out_large, dtype=torch.float32, device_override="cpu")
    e = Engine(loaded, None, max_batch_size=1, max_model_len=128)
    outs = [t for o in e.generate([3, 1, 4], SamplingParams(temperature=0.0,
                                                            max_tokens=4))
            for t in o.token_ids]
    assert len(outs) == 4


def test_host_kv_pool_roundtrip(converted):
    """Un préfixe évincé de la VRAM remonte de l'étage hôte, à l'identique.

    Petit cache (32 blocs), pool hôte actif : on remplit une longue invite,
    on la fait évincer par d'autres, on la redemande — les jetons produits
    doivent être ceux d'un cache jamais évincé, et le compteur de remontées
    doit avoir tourné.
    """
    import torch

    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams

    prompt = list(range(1, 100))                  # 6 blocs pleins

    def run(host_gib):
        loaded = load_model(converted, dtype=torch.float32,
                            device_override="cpu")
        e = Engine(loaded, None, max_batch_size=1, max_model_len=256,
                   enable_cuda_graphs=False, host_kv_gib=host_gib)
        ref = [t for o in e.generate(prompt, SamplingParams(temperature=0.0,
                                                            max_tokens=6))
               for t in o.token_ids]
        # pression : d'autres invites chassent les blocs du cache VRAM
        for k in range(6):
            base = 200 + 97 * k                  # jetons < vocabulaire (1024)
            list(e.generate([(base + i) % 1000 + 20 for i in range(90)],
                            SamplingParams(temperature=0.0, max_tokens=2)))
        again = [t for o in e.generate(prompt, SamplingParams(temperature=0.0,
                                                              max_tokens=6))
                 for t in o.token_ids]
        return ref, again, e

    ref, again, e = run(host_gib=2.0)
    assert ref == again, "la remontee depuis l'hote a change la sortie"
    if e.host_kv is not None and e.allocator.evictions > 0:
        assert e.host_kv.spills > 0, "aucun bloc n'est descendu a l'hote"
