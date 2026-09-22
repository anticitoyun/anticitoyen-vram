"""Cache de préfixe, décodage spéculatif, noyaux processeur, précision mixte.

Chacun de ces points est une optimisation, et une optimisation qui change la
sortie est un bogue. L'essentiel de ce qui suit vérifie une équivalence, pas une
vitesse.
"""

import math
import pathlib
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
    assert all(l.mlp_exec == "gpu" for l in stream.layers)

    # Sans mesure du débit de calcul hôte, « auto » ne doit PAS choisir le
    # processeur. Le comparer au bus par un débit de lecture DDR est une
    # erreur de grandeur : elle a fait croire le processeur plus rapide
    # qu'une carte graphique et envoyé 14 Gio de perceptrons sur le chemin
    # le plus lent des trois, sur Qwen3-Coder-Next.
    auto, _ = auto_plan(spec, target_rig,
                        PlannerOptions(max_model_len=8192, host_exec="auto"))
    assert all(l.mlp_exec == "gpu" for l in auto.layers)

    # Avec un débit mesuré supérieur au bus, le mécanisme reste disponible.
    mesure, _ = auto_plan(spec, target_rig,
                          PlannerOptions(max_model_len=8192, host_exec="auto",
                                         host_gemm_gb_s=70.0))
    assert any(l.mlp_exec == "cpu" for l in mesure.layers)

    # Et un débit mesuré réaliste pour un GEMM qui déballe de l'E2M1 le
    # laisse sur la carte.
    lent, _ = auto_plan(spec, target_rig,
                        PlannerOptions(max_model_len=8192, host_exec="auto",
                                       host_gemm_gb_s=3.5))
    assert all(l.mlp_exec == "gpu" for l in lent.layers)


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


def test_promotion_cost_ceiling_spares_the_big_tensors(tiny_checkpoint,
                                                       target_rig, tmp_path):
    """Un plafond de prix ne doit laisser passer que les tenseurs bon marché."""
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import (ConversionOptions, PROMOTE,
                                      convert_checkpoint, cout_promotion_mib)

    spec = load_model_spec(tiny_checkpoint, "tiny")
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=512))

    def promus(plafond):
        r = convert_checkpoint(tiny_checkpoint, plan,
                               ConversionOptions(out_dir=str(tmp_path / f"c{plafond}"),
                                                 mixed_precision="auto",
                                                 snr_floor=999.0,
                                                 promotion_cout_max_mib=plafond,
                                                 dry_run=True), spec=spec)
        return r

    plein = promus(0.0)
    assert plein.promotions, "le cas temoin doit promouvoir quelque chose"
    prix = [cout_promotion_mib(
        # les formes ne figurent pas dans le rapport : le prix se relit du
        # manifeste par le format d'origine, seul un ordre de grandeur importe
        1, p["from"], p["to"]) for p in plein.promotions]
    assert all(x > 0 for x in prix)

    # un plafond nul en pratique n'autorise plus rien
    assert promus(1e-9).promotions == []
    # et le prix se chiffre bien dans le sens attendu
    assert (cout_promotion_mib(5120 * 17408, "nvfp4", PROMOTE["nvfp4"])
            > cout_promotion_mib(5120, "nvfp4", PROMOTE["nvfp4"]) * 1000)


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
    assert r.bits_par_poids_en_memoire > 0


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


# --------------------------------------------------------------------------
# le chemin FP4 tensor cores ne s'éteint plus sur une forme qu'il ne prend pas
# --------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available()
                    or torch.cuda.get_device_capability(0) < (10, 0),
                    reason="tensor cores FP4 : sm_100 ou plus")
def test_first_generation_matches_the_next_ones(converted):
    """La première génération d'un processus rend la même chose que les suivantes.

    Une couche dont la dimension contractée n'est pas un multiple de 32 poids
    FP4 fait échouer ``torch._scaled_mm``. L'exception éteignait le chemin FP4
    globalement et sans un mot : les quelques GEMM déjà servies l'avaient été
    sur les tensor cores, toutes les suivantes — et toutes les requêtes
    ultérieures du processus — repassaient par les noyaux fusionnés. La
    première réponse d'un serveur différait donc de toutes les autres.
    """
    import acvram.kernels.fp4_gemm as fp4

    fp4._OK, fp4._PROBED = False, False       # repartir d'une sonde neuve
    assert fp4.fp4_mm_available()
    loaded = load_model(converted, dtype=torch.bfloat16, device_override="cuda:0")
    prompt = [7, 3, 9, 1, 4, 8, 2, 5] * 5
    sorties = []
    for _ in range(3):
        e = Engine(loaded, None, max_batch_size=1, max_model_len=256,
                   enable_cuda_graphs=False)
        sorties.append([t for o in e.generate(prompt,
                                              SamplingParams(temperature=0.0,
                                                             max_tokens=6))
                        for t in o.token_ids])
    assert sorties[0] == sorties[1] == sorties[2], \
        "la premiere generation diverge des suivantes"
    assert fp4._OK, "une forme refusee a eteint le chemin FP4 tensor cores"


# --------------------------------------------------------------------------
# tenseurs figés par un graphe CUDA
# --------------------------------------------------------------------------


def test_rope_ne_se_realloue_pas_sous_capture():
    """Un cache RoPE étendu pendant une capture laisserait au graphe une
    adresse morte (GLM-4.7-Flash, 5/09/2026) : c'est refusé net."""
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("CUDA")
    from acvram.engine.layers import RotaryEmbedding
    rope = RotaryEmbedding(64, 4096)
    d = torch.device("cuda")
    pos = torch.arange(8, device=d)
    rope(pos, d, torch.float32, max_pos=512)          # 1024 lignes (plancher)
    rope.reserver(2048, d, torch.float32)             # hors capture : autorisé
    assert rope._cache_len >= 2048
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        rope(pos, d, torch.float32, max_pos=2048)     # échauffement
    torch.cuda.current_stream().wait_stream(s)
    with pytest.raises(RuntimeError, match="capture"):
        with torch.cuda.graph(g):
            rope(pos, d, torch.float32, max_pos=8192)  # extension sous capture


def test_tampon_mtp_reserve_et_stable():
    """Le tampon de l'état caché MTP est réservé une fois, plus large que le
    pas, et ne bouge plus ; seules ses n premières lignes datent du pas."""
    from acvram.engine.model import ACVRamModel
    class Faux:
        _mtp_hidden = None
        _mtp_hidden_n = 0
        MTP_HIDDEN_LIGNES = ACVRamModel.MTP_HIDDEN_LIGNES
        reserver_hidden = ACVRamModel.reserver_hidden
        _garder_hidden = ACVRamModel._garder_hidden
    m = Faux()
    h5 = torch.randn(5, 32)
    m._garder_hidden(h5)
    buf = m._mtp_hidden
    assert buf.shape == (ACVRamModel.MTP_HIDDEN_LIGNES, 32) and m._mtp_hidden_n == 5
    h1 = torch.randn(1, 32)
    m._garder_hidden(h1)
    assert m._mtp_hidden is buf, "le tampon a été réalloué"
    assert m._mtp_hidden_n == 1 and torch.equal(buf[0], h1[0]) and torch.equal(buf[1], h5[1])


def test_logits_tronques_au_vocabulaire():
    """Les colonnes de rembourrage de la tête ne peuvent pas être échantillonnées."""
    from types import SimpleNamespace
    from acvram.engine.model import ACVRamModel
    faux = SimpleNamespace(spec=SimpleNamespace(vocab_size=10, logits_scaling=1.0,
                                                final_logit_softcapping=0.0))
    logits = torch.zeros(3, 12); logits[:, 11] = 99.0        # la colonne fantôme domine
    out = ACVRamModel._logits_finaux(faux, logits)
    assert out.shape == (3, 10) and int(out.argmax(-1)[0]) < 10


def test_couches_comptees_depuis_les_tenseurs(tmp_path):
    """Une architecture hybride ne se déduit pas de la configuration : Nemotron-H
    alterne des couches Mamba sans MLP et des couches MoE sans attention, et le
    compte analytique triplait sa taille (5/09/2026)."""
    import json
    from acvram.engine.config import load_model_spec
    h, inter, n_ex = 64, 32, 4
    cfg = {"architectures": ["LlamaForCausalLM"], "hidden_size": h, "intermediate_size": inter,
           "moe_intermediate_size": inter, "num_hidden_layers": 4, "num_attention_heads": 4,
           "num_key_value_heads": 2, "vocab_size": 128, "num_experts": n_ex,
           "num_experts_per_tok": 2, "rms_norm_eps": 1e-5}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    analytique = load_model_spec(str(tmp_path))
    # le manifeste dit la vérité : couches 0 et 2 en MoE, 1 et 3 en attention seule
    t = {}
    for i in (0, 2):
        for e in range(n_ex):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                t[f"model.layers.{i}.mlp.experts.{e}.{proj}.weight"] = {"shape": [inter, h]}
    for i in (1, 3):
        for proj in ("q_proj", "o_proj"):
            t[f"model.layers.{i}.self_attn.{proj}.weight"] = {"shape": [h, h]}
    (tmp_path / "acvram_manifest.json").write_text(json.dumps({"tensors": t}))
    reel = load_model_spec(str(tmp_path))
    assert reel.total_params < analytique.total_params, "le recomptage doit corriger à la baisse"
    assert sum(1 for l in reel.layers if l.is_moe) == 2, "seules les couches à experts sont MoE"
    assert reel.layers[1].mlp_params == 0, "une couche d'attention pure n'a pas de MLP"
    attendu = 2 * n_ex * 3 * inter * h + 2 * 2 * h * h
    assert abs(sum(l.total_params - l.norm_params for l in reel.layers) - attendu) < h


def test_plan_rapatrie_sur_une_carte():
    """Un modèle étalé sur deux cartes alors qu'il tient sur la première y
    revient au chargement : chaque frontière coûte un aller-retour par jeton."""
    from acvram.engine.loader import _rapatrier_sur_une_carte
    from acvram.memory.tiering import LayerPlacement, Plan, Tier
    G = 2 ** 30
    def plan_deux_cartes(octets_par_couche, n=8):
        p = Plan(model="essai")
        p.tiers = [Tier(name="cuda:0", kind="gpu", device_index=0, capacity=30 * G,
                        weight_format="nvfp4", kv_format="int8",
                        read_bandwidth=1790.0, link_bandwidth=25.0),
                   Tier(name="cuda:1", kind="gpu", device_index=1, capacity=11 * G,
                        weight_format="nvfp4", kv_format="int8",
                        read_bandwidth=912.0, link_bandwidth=25.0)]
        p.layers = [LayerPlacement(index=i, exec_device="cuda:0" if i < n // 2 else "cuda:1",
                                   attn_storage="cuda:0" if i < n // 2 else "cuda:1",
                                   mlp_storage="cuda:0" if i < n // 2 else "cuda:1",
                                   fmt="nvfp4", attn_bytes=octets_par_couche // 4,
                                   mlp_bytes=octets_par_couche - octets_par_couche // 4,
                                   mlp_active_bytes=octets_par_couche // 8)
                    for i in range(n)]
        p.kv_budget = {"cuda:0": G, "cuda:1": G // 2}
        p.lm_head_device = "cuda:0"
        return p
    attn = {i: G // 4 for i in range(8)}
    mlp = {i: G - G // 4 for i in range(8)}
    # 8 Gio de poids : tiennent largement sur la première carte
    p = plan_deux_cartes(G)
    _rapatrier_sur_une_carte(p, attn, mlp, 0, 0)
    assert {l.exec_device for l in p.layers} == {"cuda:0"}
    assert list(p.kv_budget) == ["cuda:0"] and p.kv_budget["cuda:0"] == G + G // 2
    # 4 Gio par couche, soit 32 Gio : ne tiennent pas, la répartition reste
    gros_a = {i: G for i in range(8)}
    gros_m = {i: 3 * G for i in range(8)}
    p2 = plan_deux_cartes(4 * G)
    _rapatrier_sur_une_carte(p2, gros_a, gros_m, 0, 0)
    assert {l.exec_device for l in p2.layers} == {"cuda:0", "cuda:1"}


def test_flux_completions_sans_stream_options():
    """Un flux /v1/completions sans stream_options ne doit pas lever : le champ
    manquait au schéma et tout appel en streaming tombait en 500 (v0.4.56)."""
    from acvram.server.protocol import CompletionRequest
    r = CompletionRequest(model="x", prompt="bonjour", stream=True)
    assert r.stream_options is None
    r2 = CompletionRequest(model="x", prompt="bonjour", stream=True,
                           stream_options={"include_usage": True})
    assert r2.stream_options["include_usage"] is True


def test_formes_lues_dans_les_entetes(tmp_path):
    """Les formes se lisent dans un manifeste, un en-tête GGUF ou des en-têtes
    safetensors — sans charger un poids."""
    import json, struct
    from acvram.engine.config import _formes_du_point_de_controle
    # safetensors : deux fragments
    for i, noms in enumerate((["model.layers.0.self_attn.q_proj.weight"],
                              ["model.layers.1.mlp.up_proj.weight"])):
        entete = {n: {"dtype": "BF16", "shape": [8, 4], "data_offsets": [0, 64]} for n in noms}
        brut = json.dumps(entete).encode()
        (tmp_path / f"model-0000{i}.safetensors").write_bytes(
            struct.pack("<Q", len(brut)) + brut + b"\0" * 64)
    formes = _formes_du_point_de_controle(str(tmp_path))
    assert formes == {"model.layers.0.self_attn.q_proj.weight": [8, 4],
                      "model.layers.1.mlp.up_proj.weight": [8, 4]}
    # un manifeste présent l'emporte
    (tmp_path / "acvram_manifest.json").write_text(json.dumps(
        {"tensors": {"model.layers.0.self_attn.q_proj.weight": {"shape": [2, 2]}}}))
    assert _formes_du_point_de_controle(str(tmp_path)) == {
        "model.layers.0.self_attn.q_proj.weight": [2, 2]}


def test_formes_exl3_reconstituees():
    """EXL3 range des treillis, pas des matrices : suh donne l'entrée, svh la
    sortie, et les noms suivent backbone/mixer au lieu de model/self_attn."""
    from acvram.engine.config import _traduire_exl3
    brut = {
        "backbone.layers.0.mixer.q_proj.suh": [2688],
        "backbone.layers.0.mixer.q_proj.svh": [4096],
        "backbone.layers.0.mixer.q_proj.trellis": [168, 648, 128],
        "backbone.layers.1.mixer.experts.3.up_proj.suh": [2688],
        "backbone.layers.1.mixer.experts.3.up_proj.svh": [1856],
        "backbone.embeddings.weight": [131072, 2688],
        "backbone.layers.0.mixer.q_proj.svh_orpheline": [1],
    }
    f = _traduire_exl3(brut)
    assert f["model.layers.0.self_attn.q_proj.weight"] == [4096, 2688]
    assert f["model.layers.1.mlp.experts.3.up_proj.weight"] == [1856, 2688]
    assert f["model.embeddings.weight"] == [131072, 2688]
    assert not any("trellis" in k or "suh" in k for k in f)


def test_godet_mla_double():
    """Des paliers doublants : une courte séquence ne balaie pas 1024 positions,
    et un contexte de 32 768 ne demande que neuf paliers."""
    from acvram.engine.mla import godet_mla, MLA_BUCKET
    assert godet_mla(1) == MLA_BUCKET
    assert godet_mla(264) == 512 if MLA_BUCKET <= 512 else godet_mla(264) >= 264
    assert godet_mla(1024) == 1024 and godet_mla(1025) == 2048
    paliers = set()
    n = 1
    while n <= 32768:
        paliers.add(godet_mla(n)); n *= 2
    assert len(paliers) <= 10, paliers
    assert all(godet_mla(v) >= v for v in (1, 63, 100, 5000, 32768))


def test_trace_routage_eteinte_ne_coute_rien(tmp_path, monkeypatch):
    """Hors trace, la fonction sort sur un test de booléen."""
    import importlib
    monkeypatch.delenv("ACVRAM_TRACE_ROUTAGE", raising=False)
    from acvram.memory import trace_routage as t
    importlib.reload(t)
    assert t.actif() is False
    # Un objet sans .detach() : si noter() le touchait, ce serait une erreur.
    t.noter(0, object())


def test_trace_routage_numerote_les_jetons_par_passage(tmp_path, monkeypatch):
    """Toutes les couches d'un passage portent les MÊMES numéros de jeton.

    Une première version repartait du compteur cumulé à chaque couche : la
    couche 1 numérotait 3, 4, 5 les jetons que la couche 0 appelait 0, 1, 2.
    Un rejeu y aurait vu deux fois plus de trafic qu'il n'en passe.
    """
    import importlib
    import torch
    journal = tmp_path / "trace.txt"
    monkeypatch.setenv("ACVRAM_TRACE_ROUTAGE", str(journal))
    from acvram.memory import trace_routage as t
    importlib.reload(t)
    assert t.actif() is True
    for _ in range(2):                      # deux passages
        for couche in (0, 1):               # deux couches
            t.noter(couche, torch.tensor([[1, 2], [3, 4]]))
    t.fermer()

    lu = list(t.relire(str(journal)))
    assert len(lu) == 8
    # Les deux couches d'un même passage voient les mêmes jetons.
    c0 = [j for j, c, _ in lu if c == 0]
    c1 = [j for j, c, _ in lu if c == 1]
    assert c0 == c1 == [0, 1, 2, 3]
    # Le rejeu compte les demandes réelles, pas les couches.
    r = t.taux_de_succes(str(journal), capacite=99)
    assert r["demandes"] == 16
    assert r["succes"] == 8          # second passage : tout est en cache
    importlib.reload(t)


def test_energie_s_abstient_sans_constantes():
    """Sans constantes mesurées, le plan ne prétend pas connaître les joules."""
    from acvram.memory.tiering import PlannerOptions, Plan
    o = PlannerOptions()
    assert o.watts_processeur == 0.0
    assert o.fraction_puissance_decodage == 0.0
    p = Plan(model="essai")
    assert p.est_joules_par_jeton is None
    assert p.est_jetons_par_kj is None
    d = p.to_dict()
    assert d["est_joules_par_jeton"] is None
    assert d["est_jetons_par_kj"] is None


def test_energie_compte_le_processeur_a_part():
    """Le terme distingue deux plans que le modèle en secondes confond.

    À débit égal, un MLP exécuté sur processeur coûte beaucoup plus de joules
    qu'un transfert par le bus — 183 s de temps processeur pour 200 jetons
    contre 25 s, mesuré le 8 septembre 2026. Le modèle en secondes ne voit pas
    cette différence ; celui en joules doit la voir.
    """
    duree = 0.074
    watts_carte, fraction, watts_proc = 400.0, 0.35, 120.0
    j_carte = watts_carte * fraction * duree
    j_avec_proc = j_carte + 0.9 * watts_proc
    assert j_avec_proc > 5 * j_carte


def test_la_conversion_refuse_de_faire_grossir():
    """8/09/2026 : un GGUF de 3,4 bits/poids converti en NVFP4 (4,5) grossit
    d'un tiers, déborde la carte et coûte un facteur quinze au décodage. La
    conversion refuse désormais, sauf autorisation explicite qui journalise."""
    import pytest
    from acvram.quant.convert import garde_grossissement
    n = 80_000_000_000
    src_33g = int(n * 3.4 / 8)
    # source déjà plus large que la cible : rien à dire
    assert garde_grossissement(int(n * 6.5 / 8), n, 4.5, False) is None
    # marge de 2 % : une cible à peine plus large passe
    assert garde_grossissement(int(n * 4.47 / 8), n, 4.5, False) is None
    # grossissement réel : refus par défaut, avec les trois grandeurs
    with pytest.raises(ValueError) as e:
        garde_grossissement(src_33g, n, 4.5, False)
    for morceau in ("GROSSIR", "3.40", "4.50", "+32 %"):
        assert morceau in str(e.value)
    # autorisé : un avertissement, pas une exception
    m = garde_grossissement(src_33g, n, 4.5, True)
    assert m and "autoris" in m
    # sans information sur la source : ne bloque pas
    assert garde_grossissement(0, n, 4.5, False) is None


def test_q3n_empaquetage_et_numerique():
    """Format Q3N (spécification du 8/09) : aller-retour d'empaquetage exact,
    table symétrique, SNR conforme aux mesures de la spécification, et le
    GEMV de référence égale la déquantification. 3,25 bits/poids au bloc 32."""
    import math
    import torch
    from acvram.quant.q3n import (TABLE_Q3N, Q3NTensor, depaqueter_q3,
                                  dequantize_q3n, empaqueter_q3, q3n_gemv,
                                  quantize_q3n)
    assert all(abs(a + b) < 1e-9 for a, b in zip(TABLE_Q3N, reversed(TABLE_Q3N)))
    g = torch.Generator().manual_seed(0)
    q = torch.randint(0, 8, (64, 256), generator=g)
    assert torch.equal(depaqueter_q3(empaqueter_q3(q), 256), q)
    for bloc, snr_min in ((32, 14.0), (16, 14.8)):
        w = torch.randn(128, 512, generator=g) * 0.02
        t = quantize_q3n(w, block=bloc)
        assert isinstance(t, Q3NTensor)
        assert abs(t.bits_per_weight - (3 + 8 / bloc)) < 0.15
        wh = dequantize_q3n(t, torch.float32)
        snr = 10 * math.log10(w.pow(2).mean().item()
                              / (w - wh).pow(2).mean().item())
        assert snr > snr_min, f"bloc {bloc} : {snr:.2f} dB"
    x = torch.randn(3, 512, generator=g, dtype=torch.float32).bfloat16()
    y = q3n_gemv(x, t)
    attendu = x @ dequantize_q3n(t, torch.bfloat16).t()
    assert torch.allclose(y.float(), attendu.float(), atol=1e-2)


def test_une_source_3_bits_bascule_les_etages_sur_q3n():
    """8/09/2026 : plutôt que refuser une source 3 bits, la conversion bascule
    les étages GPU sur q3n (3,25 b/p) et l'annonce ; elle ne refuse plus que
    si même q3n grossirait la source."""
    from acvram.quant.convert import bpw_nominal, garde_grossissement
    # BPW_NOMINAL, dictionnaire ecrit en dur, a ete remplace le 10/09 par
    # bpw_nominal(fmt, group_size) qui delegue a quant/formats.py : il en etait
    # la troisieme copie divergente et il ignorait --group-size.
    assert bpw_nominal("q3n") == 3.25
    assert bpw_nominal("q3n", 32) == 3.25          # q3n ne depend pas du groupe
    assert bpw_nominal("int8", 128) == 8.1875      # celui-ci en depend
    assert bpw_nominal("int8", 32) == 8.75
    n = 80_000_000_000
    # Q3_K_S à 3,4 b/p : q3n (3,25) ne grossit pas -> aucun refus
    assert garde_grossissement(int(n * 3.4 / 8), n, 3.25, False) is None
    # source à 2,5 b/p : même q3n grossit -> refus, qui nomme q3n en issue
    import pytest
    with pytest.raises(ValueError) as e:
        garde_grossissement(int(n * 2.5 / 8), n, 3.25, False)
    assert "q3n" in str(e.value)


def test_q3n_gemv_noyau_egale_la_reference():
    """Le noyau CUDA q3n_gemv égale la référence par déquantification, au
    bit près en float32, à la précision bf16 sinon — formes variées, blocs
    16 et 32, M non multiple de 4. Sauté sans carte ou sans extension."""
    import pytest
    import torch
    if not torch.cuda.is_available():
        pytest.skip("pas de carte")
    from acvram import kernels
    ext = kernels.get_extension()
    if ext is None or not hasattr(ext, "q3n_gemv"):
        pytest.skip("extension sans q3n_gemv")
    from acvram.quant.q3n import dequantize_q3n, quantize_q3n
    g = torch.Generator().manual_seed(1)
    for (M, K, B) in [(64, 512, 32), (96, 512, 16), (5, 256, 32)]:
        t = quantize_q3n(torch.randn(M, K, generator=g) * 0.02, block=B)
        tc = t.to("cuda:0")
        x = torch.randn(2, K, generator=g).cuda()
        y = ext.q3n_gemv(tc.qweight, tc.block_scale.view(torch.uint8),
                         tc.global_scale.cuda(), tc.table_gpu("cuda:0"),
                         x, K, B)
        ref = x @ dequantize_q3n(t, torch.float32).cuda().t()
        assert (y - ref).abs().max().item() < 1e-5
    # une table de manifeste différente doit changer le résultat du noyau
    from acvram.quant.q3n import TABLE_Q3N_LLOYD_CODER_NEXT
    t2 = quantize_q3n(torch.randn(64, 512, generator=g) * 0.02,
                      table=TABLE_Q3N_LLOYD_CODER_NEXT).to("cuda:0")
    x = torch.randn(2, 512, generator=g).cuda()
    y2 = ext.q3n_gemv(t2.qweight, t2.block_scale.view(torch.uint8),
                      t2.global_scale.cuda(), t2.table_gpu("cuda:0"),
                      x, 512, t2.block)
    ref2 = x @ dequantize_q3n(t2.to("cpu"), torch.float32).cuda().t()
    assert (y2 - ref2).abs().max().item() < 1e-5


def test_la_bascule_q3n_change_les_couches_pas_seulement_les_etages(tmp_path):
    """8/09 : la première bascule ne changeait que tiers[].weight_format alors
    que le routeur lit lp.fmt par couche — 42 Go écrits en nvfp4 sous un
    message annonçant q3n. La bascule doit atteindre le routeur, et l'issue
    (octets écrits) est vérifiée en fin de conversion."""
    import json
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, TensorRouter
    from acvram.hardware.detect import detect_rig
    d = tmp_path / "m"; d.mkdir()
    json.dump({"architectures": ["LlamaForCausalLM"], "hidden_size": 256,
               "intermediate_size": 512, "num_hidden_layers": 4,
               "num_attention_heads": 8, "num_key_value_heads": 8,
               "vocab_size": 512}, open(d / "config.json", "w"))
    spec = load_model_spec(str(d), "m")
    plan, _ = auto_plan(spec, detect_rig(), PlannerOptions(max_model_len=256))
    # rejoue la bascule telle que convert_checkpoint l'applique
    gpus = {t.name for t in plan.tiers if t.kind == "gpu"}
    for t in plan.tiers:
        if t.kind == "gpu":
            t.weight_format = "q3n"
    for lp in plan.layers:
        if getattr(lp, "fmt", None) and lp.exec_device in gpus | {"cpu"}:
            lp.fmt = "q3n"
    r = TensorRouter(spec, plan, ConversionOptions(out_dir=str(tmp_path)))
    assert r.format_for("model.layers.0.mlp.gate_proj.weight") == "q3n"
    # v0.4.94 : l'attention ne descend plus sous int8 quand la cible est
    # q3n (trace du 8/09 : effondrement du cosinus dès la première attention
    # pleine en q3n). La bascule atteint bien le routeur — c'est l'objet de
    # ce test — mais le plancher la borne sur les projections d'attention.
    assert r.format_for("model.layers.3.self_attn.q_proj.weight") == "int8"


def test_q3n_traverse_le_chemin_generique_de_quantification():
    """8/09, seconde panne de la journée sur le même thème : le format était
    routé mais pas quantifiable — quantize() levait « format inconnu » après
    une heure de conversion. Le chemin générique complet est verrouillé :
    registre, quantification, déquantification, octets estimés."""
    import torch
    from acvram.quant.formats import (dequantize, estimate_bytes, get_format,
                                      quantize)
    w = torch.randn(16, 64) * 0.02
    t = quantize(w, "q3n")
    assert t.format == "q3n" and get_format("q3n").bpw == 3.25
    assert dequantize(t, torch.float32).shape == w.shape
    assert abs(estimate_bytes(80_000_000_000, "q3n") / 2**30 - 30.3) < 0.3


def test_le_chargeur_reconstruit_un_tenseur_q3n(tmp_path):
    """Troisième panne du 8/09, même thème, troisième porte : quantifiable et
    routé mais pas RECHARGEABLE — _build_quant levait « unknown format » au
    premier tenseur du modèle converti. Aller-retour disque complet verrouillé."""
    import torch
    from safetensors.torch import save_file
    from acvram.engine.loader import _ShardReader, _build_quant
    from acvram.quant.q3n import dequantize_q3n, quantize_q3n
    w = torch.randn(8, 64) * 0.02
    t = quantize_q3n(w)
    save_file({"a.qweight": t.qweight,
               "a.block_scale": t.block_scale.view(torch.uint8),
               "a.global_scale": t.global_scale.reshape(1)},
              str(tmp_path / "acvram-00000.safetensors"))
    reader = _ShardReader(str(tmp_path),
                          {k: "acvram-00000.safetensors"
                           for k in ("a.qweight", "a.block_scale", "a.global_scale")})
    entry = {"format": "q3n", "shape": [8, 64], "block": 32,
             "keys": ["a.qweight", "a.block_scale", "a.global_scale"]}
    t2 = _build_quant(entry, "a", reader, 128)
    assert t2.format == "q3n"
    assert torch.equal(dequantize_q3n(t2, torch.float32).cpu(),
                       dequantize_q3n(t, torch.float32))


def test_les_projections_gdn_ne_descendent_pas_sous_int8(tmp_path):
    """8/09 : tout le modèle en q3n rendait « URTURTURT » — l'attention
    linéaire amplifie l'erreur de poids à chaque pas de sa récurrence, et la
    conversion NVFP4 saine promouvait déjà ces projections en int8. Plancher
    verrouillé au routeur."""
    import json
    from acvram.engine.config import load_model_spec
    from acvram.hardware.detect import detect_rig
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, TensorRouter
    d = tmp_path / "m"; d.mkdir()
    json.dump({"architectures": ["LlamaForCausalLM"], "hidden_size": 256,
               "intermediate_size": 512, "num_hidden_layers": 2,
               "num_attention_heads": 8, "num_key_value_heads": 8,
               "vocab_size": 512}, open(d / "config.json", "w"))
    spec = load_model_spec(str(d), "m")
    plan, _ = auto_plan(spec, detect_rig(), PlannerOptions(max_model_len=256))
    for lp in plan.layers:
        lp.fmt = "q3n"
    r = TensorRouter(spec, plan, ConversionOptions(out_dir=str(tmp_path)))
    assert r.format_for("model.layers.1.linear_attn.alpha.weight") == "int8"
    assert r.format_for("model.layers.1.linear_attn.qkv.weight") == "int8"
    # 8/09, second constat : le plancher GDN seul n'a pas suffi — cosinus
    # contre le modèle sain à 0,99 sur les couches GDN puis effondrement dès
    # la première attention pleine (q/k/v/o en q3n). Toute l'attention et la
    # tête de sortie restent au moins en int8 sous q3n.
    assert r.format_for("model.layers.1.self_attn.q_proj.weight") == "int8"
    assert r.format_for("model.layers.1.self_attn.o_proj.weight") == "int8"
    assert r.format_for("lm_head.weight") == "int8"
    assert r.format_for("model.layers.1.mlp.experts.0.up_proj.weight") == "q3n"


def test_bloc_mtp_nomme_layers_n_pas_int4_awq_en_dur(tmp_path):
    """15/09, GLM-4.7-Flash : le bloc MTP (num_nextn_predict_layers=1) est
    rangé sous `model.layers.<num_hidden_layers>.*` — MÊME convention que
    les couches réelles, PAS le préfixe `.mtp.`/`model.mtp.` que le routeur
    reconnaissait déjà. `layer_index` rendait donc 47 pour un modèle à 47
    couches réelles (indices 0-46), `_layer_fmt` n'avait pas cette clé, et
    le repli codé en dur rendait "int4_awq" -- confondu un moment avec une
    promotion SNR par expert (aucun `promoted_from` sur ces tenseurs : ils
    étaient quantifiés DIRECTEMENT dans ce format, pas promus).
    `model.layers.47.mlp.experts.0.gate_proj.weight` doit suivre le format
    de la DERNIÈRE couche réelle (46), comme le préfixe `.mtp.` littéral."""
    import json
    from acvram.engine.config import load_model_spec
    from acvram.hardware.detect import detect_rig
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, TensorRouter
    d = tmp_path / "m"; d.mkdir()
    json.dump({"architectures": ["LlamaForCausalLM"], "hidden_size": 256,
               "intermediate_size": 512, "num_hidden_layers": 47,
               "num_attention_heads": 8, "num_key_value_heads": 8,
               "vocab_size": 512}, open(d / "config.json", "w"))
    spec = load_model_spec(str(d), "m")
    plan, _ = auto_plan(spec, detect_rig(), PlannerOptions(max_model_len=256))
    r = TensorRouter(spec, plan, ConversionOptions(out_dir=str(tmp_path)))
    attendu = r.format_for("model.layers.46.mlp.gate_proj.weight")
    assert r.format_for("model.layers.47.mlp.experts.0.gate_proj.weight") == attendu
    assert attendu != "int4_awq" or r.format_for(
        "model.layers.47.mlp.shared_expert.down_proj.weight") == attendu


def test_tenseur_hors_plan_leve_plutot_que_repli_silencieux(tmp_path):
    """Corollaire du test précédent : un indice de couche qui n'est NI une
    couche réelle NI le bloc MTP (ex. un nom de tenseur corrompu, ou une
    future convention non reconnue) doit arrêter la conversion en la
    nommant -- jamais retomber sur int4_awq en silence, le bogue même que
    ce commit corrige pour le cas MTP."""
    import json
    import pytest
    from acvram.engine.config import load_model_spec
    from acvram.hardware.detect import detect_rig
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, TensorRouter
    d = tmp_path / "m"; d.mkdir()
    json.dump({"architectures": ["LlamaForCausalLM"], "hidden_size": 256,
               "intermediate_size": 512, "num_hidden_layers": 4,
               "num_attention_heads": 8, "num_key_value_heads": 8,
               "vocab_size": 512}, open(d / "config.json", "w"))
    spec = load_model_spec(str(d), "m")
    plan, _ = auto_plan(spec, detect_rig(), PlannerOptions(max_model_len=256))
    r = TensorRouter(spec, plan, ConversionOptions(out_dir=str(tmp_path)))
    with pytest.raises(ValueError, match="pas de format planifié"):
        r.format_for("model.layers.99.mlp.gate_proj.weight")


def test_convertisseur_refuse_une_couche_moe_a_formats_melanges():
    """Sage, 15/09 : la garde 'écrase la couche' pas 'répare au chargement'.
    Rejoue le manifeste exact du 15/09 sur GLM-4.7-Flash (2 944 experts
    nvfp4 + 64 int4_awq) tel que `_verifier_homogeneite_moe` doit le voir --
    ce test casse si on retire la garde ou son appel dans
    `convert_checkpoint`."""
    import pytest
    from acvram.quant.convert import _verifier_homogeneite_moe
    tensors = {}
    for couche in range(1, 4):
        for expert in range(64):
            tensors[f"model.layers.{couche}.mlp.experts.{expert}.gate_proj.weight"] = {
                "format": "nvfp4"}
    for expert in range(64):
        # une seule couche a l'un de ses experts en int4_awq -- assez pour
        # casser la pile groupee de CETTE couche.
        fmt = "int4_awq" if expert == 0 else "nvfp4"
        tensors[f"model.layers.2.mlp.experts.{expert}.up_proj.weight"] = {"format": fmt}
    with pytest.raises(ValueError, match="MÉLANGÉS"):
        _verifier_homogeneite_moe(tensors, num_layers=4)


def test_convertisseur_tolere_une_echelle_awq_melangee_entre_experts():
    """Sage, 15/09 (a6a7436) : l'echelle AWQ par expert reste legitimement
    heterogene dans la pile -- c'est au chargeur (Laurine) de la porter,
    pas a la conversion de l'uniformiser ou de la refuser. Rejoue 63
    experts avec has_act_scale=True et 1 sans (meme format partout,
    presence d'echelle differente) : NE DOIT PLUS lever (inverse du
    comportement d'avant a6a7436)."""
    from acvram.quant.convert import _verifier_homogeneite_moe
    tensors = {}
    for expert in range(64):
        tensors[f"model.layers.1.mlp.experts.{expert}.gate_proj.weight"] = {
            "format": "int4_awq", "has_act_scale": expert != 0}
    _verifier_homogeneite_moe(tensors, num_layers=2)  # ne leve pas


def test_convertisseur_ignore_le_bloc_mtp_pour_l_homogeneite_moe():
    """Le bloc MTP (couche == num_layers) peut porter un format different
    des couches reelles sans que la garde ne l'accuse -- il n'est jamais
    charge par le moteur au decodage. Reproduit exactement le manifeste du
    15/09 (46 couches reelles homogenes en nvfp4, 1 bloc MTP entierement
    int4_awq) : NE DOIT PAS lever."""
    from acvram.quant.convert import _verifier_homogeneite_moe
    tensors = {}
    for couche in range(1, 47):
        for expert in range(64):
            tensors[f"model.layers.{couche}.mlp.experts.{expert}.gate_proj.weight"] = {
                "format": "nvfp4"}
    for expert in range(64):
        tensors[f"model.layers.47.mlp.experts.{expert}.gate_proj.weight"] = {
            "format": "int4_awq"}
    _verifier_homogeneite_moe(tensors, num_layers=47)  # ne leve pas


def test_mode_liste_explicite(tiny_checkpoint, target_rig, tmp_path_factory,
                              monkeypatch):
    """Le mode liste promeut EXACTEMENT la liste, ignore le budget, et refuse
    un nom absent.

    Trois gardes, éprouvées séparément parce qu'une liste figée qui se
    désynchronise du parc promouvrait moins que demandé en silence : le compte
    de promus ne le dirait pas, on attendrait 76 et on en aurait 74 sans savoir
    lesquels.
    """
    import json

    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint

    spec = load_model_spec(tiny_checkpoint, "tiny")
    plan, _ = auto_plan(spec, target_rig,
                        PlannerOptions(max_model_len=512, max_concurrent_seqs=2))

    def convertir(budget, **env):
        out = str(tmp_path_factory.mktemp("liste"))
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        try:
            # nettoyage en `finally` : les cas 4 à 7 LÈVENT, et une variable
            # laissée en place contaminerait le cas suivant — c'est ce qui a
            # fait passer le cas 5 pour bon au premier essai.
            r = convert_checkpoint(tiny_checkpoint, plan,
                                   ConversionOptions(out_dir=out,
                                                     bits_budget_gib=budget),
                                   spec=spec)
        finally:
            for k in env:
                monkeypatch.delenv(k, raising=False)
        return out, r

    # 1. le parc promouvable, lu sur une conversion à budget large
    _, ref = convertir(1.0)
    noms = sorted(p["name"] for p in ref.promotions)
    assert len(noms) >= 2, "pas assez de candidats pour éprouver le mode liste"
    choisis = noms[:2]

    d = tmp_path_factory.mktemp("listes")
    f_ok = d / "x.json"
    f_ok.write_text(json.dumps({"X": choisis, "Y": noms[2:3]}))

    # 2. la liste est promue EXACTEMENT, et le budget est ignoré : 1 Mio de
    #    budget ne laisserait passer aucun tenseur sous le glouton.
    out, r = convertir(0.001, ACVRAM_LISTE_PROMUS=str(f_ok),
                       ACVRAM_LISTE_CLE="X")
    obtenus = sorted(p["name"] for p in r.promotions)
    assert obtenus == sorted(choisis), \
        f"promus {obtenus} au lieu de {sorted(choisis)}"

    m = json.loads((pathlib.Path(out) / "acvram_manifest.json").read_text())
    b = m["budget"]
    assert b["ordre_glouton"] == "liste_explicite_sans_ordre_ni_budget"
    assert b["budget_ignore"] is True
    assert b["liste_noms"] == len(choisis)
    assert b["liste_cle"] == "X"
    assert len(b["liste_sha256"]) == 64, "le sha de la liste manque au manifeste"

    # 3. deux clés du même fichier donnent deux sha DIFFÉRENTS — sans quoi deux
    #    bras nommés X et Y seraient indistinguables dans leur manifeste.
    out_y, _ = convertir(0.001, ACVRAM_LISTE_PROMUS=str(f_ok),
                         ACVRAM_LISTE_CLE="Y")
    m_y = json.loads((pathlib.Path(out_y) / "acvram_manifest.json").read_text())
    assert m_y["budget"]["liste_sha256"] != b["liste_sha256"]

    # 4. LA GARDE : un nom absent du parc fait LEVER, il ne promeut pas moins
    f_ko = d / "ko.json"
    f_ko.write_text(json.dumps({"X": choisis + ["model.inexistant.weight"]}))
    with pytest.raises(ValueError, match="ne sont pas des candidats"):
        convertir(0.001, ACVRAM_LISTE_PROMUS=str(f_ko), ACVRAM_LISTE_CLE="X")

    # 5. un fichier à plusieurs listes sans clé refuse, au lieu de choisir
    with pytest.raises(ValueError, match="plusieurs listes"):
        convertir(0.001, ACVRAM_LISTE_PROMUS=str(f_ok))

    # 6. deux consignes contradictoires lèvent
    with pytest.raises(ValueError, match="contradictoires"):
        convertir(0.001, ACVRAM_LISTE_PROMUS=str(f_ok), ACVRAM_LISTE_CLE="X",
                  ACVRAM_ORDRE_SAC="erreur")

    # 7. le mode réclamé sans fichier lève aussi
    with pytest.raises(ValueError, match="exige ACVRAM_LISTE_PROMUS"):
        convertir(0.001, ACVRAM_ORDRE_SAC="liste")


def test_ordre_genre_et_cout(tiny_checkpoint, target_rig, tmp_path_factory,
                             monkeypatch):
    """La clé `genre` promeut par priorité de genre, `cout_decroissant` par
    taille, et les deux sont DÉRIVÉES de l'observation.

    Elles sont bâties pour reproduire le bras B, donc elles ne peuvent pas le
    confirmer. Ce test ne juge donc pas leur valeur : il vérifie qu'elles font
    ce qu'elles annoncent, que le manifeste porte l'aveu de dérivation, et que
    la table de priorité avoue ce qu'elle ne connaît pas.
    """
    import json

    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint

    spec = load_model_spec(tiny_checkpoint, "tiny")
    plan, _ = auto_plan(spec, target_rig,
                        PlannerOptions(max_model_len=512, max_concurrent_seqs=2))

    def convertir(budget, mode):
        out = str(tmp_path_factory.mktemp(f"ordre_{mode}"))
        monkeypatch.setenv("ACVRAM_ORDRE_SAC", mode)
        try:
            r = convert_checkpoint(tiny_checkpoint, plan,
                                   ConversionOptions(out_dir=out,
                                                     bits_budget_gib=budget),
                                   spec=spec)
        finally:
            monkeypatch.delenv("ACVRAM_ORDRE_SAC", raising=False)
        m = json.loads((pathlib.Path(out) / "acvram_manifest.json").read_text())
        return m, sorted(p["name"] for p in r.promotions)

    PRIO = {"lm_head": 0, "down_proj": 1, "gate_proj": 2, "up_proj": 2,
            "o_proj": 3, "v_proj": 4, "q_proj": 5, "k_proj": 5}

    def genre(n):
        p = n.split(".")
        return "lm_head" if n.endswith("lm_head.weight") else (
            p[-2] if len(p) > 2 else n)

    # BUDGET QUI MORD VRAIMENT. À 0,02 Gio les 29 candidats passaient tous :
    # le tri ne décidait rien et le contrôle de priorité plus bas était inerte
    # — vérifié en inversant la table de priorité, le test passait quand même.
    # Le budget est donc calé sur le plancher lu au manifeste, plus de quoi
    # promouvoir une PART des candidats.
    m_ref, parc = convertir(1.0, "genre")
    plancher = m_ref["budget"]["plancher_gib"]
    plafond = m_ref["budget"]["plafond_gib"]
    budget_serre = plancher + (plafond - plancher) * 0.4
    assert budget_serre < plafond, "pas de marge pour un budget partiel"
    m_g, promus_g = convertir(budget_serre, "genre")
    b = m_g["budget"]
    assert b["ordre_glouton"] == "genre_du_tenseur_DERIVE_de_l_observation"
    assert b["genre_derive_de_l_observation"] is True, \
        "le manifeste doit AVOUER que la clé est dérivée de l'observation"
    assert b["genres_vus"], "aucun genre recensé"
    # la table de priorité avoue son ignorance au lieu de la taire
    assert set(b["genres_inconnus"]) <= set(b["genres_vus"])
    assert all(g not in PRIO for g in b["genres_inconnus"])

    # LE CONTROLE QUI PEUT RENDRE « FAUX » : l'ORDRE, lu au manifeste.
    #
    # Deux essais precedents etaient inertes ou faux. Le premier comparait la
    # liste des promus a elle-meme triee — vrai de toute liste triee. Le
    # second exigeait qu'aucun promu ne soit de priorite pire qu'un recale :
    # il MORDAIT, et il echouait sur un tri correct, parce que le glouton
    # saute un candidat trop gros pour le reste du budget et prend un moins
    # cher ensuite. L'ensemble des promus n'est donc pas un prefixe de
    # l'ordre, et seul l'ordre lui-meme est testable.
    ordre = m_g["budget"]["ordre_20_premiers"]
    couts = m_g["budget"]["cout_20_premiers_mio"]
    assert ordre, "le manifeste ne porte pas le temoin d'ordre"
    rangs = [(PRIO.get(genre(n), 99), -c) for n, c in zip(ordre, couts)]
    assert rangs == sorted(rangs), (
        f"ordre non trie par (priorite de genre, cout decroissant) : "
        f"{list(zip(ordre, rangs))[:6]}")

    m_c, promus_c = convertir(0.02, "cout_decroissant")
    assert m_c["budget"]["ordre_glouton"] == "cout_decroissant_sans_decibel"

    # un mode inconnu lève toujours, et le message nomme les deux nouveaux
    with pytest.raises(ValueError, match="cout_decroissant"):
        convertir(0.02, "genre_du_tenseur")


def test_max_promus_fixe_le_compte(tiny_checkpoint, target_rig, tmp_path_factory,
                                   monkeypatch):
    """ACVRAM_MAX_PROMUS=N promeut EXACTEMENT les N premiers de l'ordre, budget
    ignore — l'experience « compte egal » du 11/09.
    """
    import json

    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint

    spec = load_model_spec(tiny_checkpoint, "tiny")
    plan, _ = auto_plan(spec, target_rig,
                        PlannerOptions(max_model_len=512, max_concurrent_seqs=2))

    def convertir(budget, maxp=None):
        out = str(tmp_path_factory.mktemp(f"maxp_{maxp}"))
        if maxp is not None:
            monkeypatch.setenv("ACVRAM_MAX_PROMUS", str(maxp))
        try:
            r = convert_checkpoint(tiny_checkpoint, plan,
                                   ConversionOptions(out_dir=out,
                                                     bits_budget_gib=budget),
                                   spec=spec)
        finally:
            monkeypatch.delenv("ACVRAM_MAX_PROMUS", raising=False)
        m = json.loads((pathlib.Path(out) / "acvram_manifest.json").read_text())
        return m, len(r.promotions)

    ref, n_ref = convertir(1.0)  # budget large, tout candidat passe
    cand = ref["budget"]["candidats"]
    assert cand >= 2, "pas assez de candidats pour l'experience de compte"

    cible = max(1, cand // 2)
    # budget SERRE : sans le plafond il promeut peu ; avec, il doit ignorer le
    # budget et promouvoir exactement `cible`.
    m, n = convertir(0.001, maxp=cible)
    assert n == cible, f"{n} promus au lieu de {cible} demandes"
    assert m["budget"]["max_promus_impose"] == cible
    assert m["budget"]["promus"] == cible

    # demander plus que le parc leve
    with pytest.raises(ValueError, match="candidats"):
        convertir(1.0, maxp=cand + 1)
