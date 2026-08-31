"""Le rejeu en graphe CUDA doit être invisible dans la sortie.

L'invariant s'énonce précisément : *au sein d'un même moteur*, le pas capturé
rend exactement les logits que le chemin eager rend sur le même état. C'est ce
que ces tests affirment, pas à pas, y compris en traversant une frontière de
godet. Comparer deux moteurs séparés par l'argmax ne teste rien de tel : sur un
modèle-jouet aux logits quasi plats, cuBLAS et SDPA choisissent leurs
algorithmes selon l'état du contexte, et l'argmax devient une loterie
d'epsilon — le chemin eager seul n'est déjà pas reproductible d'un processus à
l'autre à ce grain-là.
"""

import pytest
import torch

from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="pas de peripherique CUDA")


def _engine(converted, n_batch=2):
    loaded = load_model(converted, dtype=torch.bfloat16,
                        device_override="cuda:0")
    return Engine(loaded, None, max_batch_size=n_batch, max_model_len=256,
                  enable_cuda_graphs=True)


def _prefill(e, prompt, max_tokens=32):
    seq = e.add_request(prompt, SamplingParams(temperature=0.0,
                                               max_tokens=max_tokens))
    e.step()
    return seq


def _decode_both(e):
    """Un pas de décodage : logits eager puis logits du graphe, même état."""
    dec = [s for s in e.running if s.prefilled and not s.finished]
    for s in dec:
        assert e._grow(s)
    batch = e._build_batch(dec, prefill=False)
    eager = e.model(batch).float()
    par_graphe = e.graphs.run(batch)
    assert par_graphe is not None, "le graphe a refuse un pas de decodage pur"
    return eager, par_graphe.float(), batch, dec


@needs_cuda
def test_graph_logits_equal_eager(converted):
    e = _engine(converted)
    assert e.graphs is not None, "graphes refuses sur un modele eligible"
    _prefill(e, [7, 3, 9, 1, 4, 8, 2, 5] * 4)
    for _ in range(4):                       # capture au 1er pas, rejeux ensuite
        eager, graphe, batch, dec = _decode_both(e)
        assert torch.equal(eager, graphe), \
            "le graphe et l'eager divergent sur le meme etat"
        e._emit(graphe, dec)
    assert e.graphs.captures == 1
    assert e.graphs.replays >= 4


@needs_cuda
def test_graph_across_bucket_boundary(converted):
    """130 jetons : 7 blocs deviennent 9, le godet passe de 8 a 16, on recapture."""
    e = _engine(converted, n_batch=1)
    _prefill(e, list(range(1, 101)), max_tokens=60)   # 100 jetons -> 7 blocs
    seen = set()
    for _ in range(30):                                # traverse 128 jetons
        eager, graphe, batch, dec = _decode_both(e)
        assert torch.equal(eager, graphe)
        seen.add(e.graphs._last_key)
        e._emit(graphe, dec)
    assert len(seen) >= 2, "la frontiere de godet n'a pas change de graphe"
    assert e.graphs.captures == len(seen)


@needs_cuda
def test_graph_qknorm_model(converted_qknorm):
    """Le chemin fixe applique aussi les normes de Q et K (forme Qwen3)."""
    e = _engine(converted_qknorm)
    _prefill(e, [5, 9, 2, 7, 1, 8, 3, 6])
    eager, graphe, _, _ = _decode_both(e)
    assert torch.equal(eager, graphe)


@needs_cuda
def test_graph_end_to_end_finishes(converted):
    """Une génération complète sous graphes se termine et remplit ses jetons."""
    e = _engine(converted)
    outs = [t for o in e.generate([1, 2, 3, 4, 5],
                                  SamplingParams(temperature=0.0,
                                                 max_tokens=12))
            for t in o.token_ids]
    assert len(outs) == 12
    assert e.graphs.replays >= 10


@needs_cuda
def test_paged_attention_matches_reference(converted):
    """Le noyau d'attention paginée reproduit le chemin déquantifier-puis-SDPA."""
    from acvram import kernels
    from acvram.engine.layers import decode_attention_fixed

    loaded = load_model(converted, dtype=torch.bfloat16,
                        device_override="cuda:0")
    e = Engine(loaded, None, max_batch_size=2, max_model_len=256,
               enable_cuda_graphs=False)
    e.add_request(list(range(1, 40)), SamplingParams(temperature=0.0,
                                                     max_tokens=4))
    e.step()
    cache = loaded.model.caches[0]
    if cache.k_scale is None:
        pytest.skip("cache non quantifie")
    dec = [s for s in e.running if not s.finished]
    for s in dec:
        e._grow(s)
    batch = e._build_batch(dec, prefill=False)
    tables, lens = batch.fixed_decode_views(torch.device("cuda:0"))
    hq = loaded.spec.num_attention_heads
    hd = loaded.spec.head_dim
    torch.manual_seed(0)
    q = torch.randn(len(dec), hq, hd, device="cuda:0")
    n_rep = hq // loaded.spec.num_key_value_heads

    fusionne = kernels.paged_attention(q, cache, tables, lens, n_rep,
                                       hd ** -0.5)
    assert fusionne is not None, "noyau pagine indisponible"
    kk, vv = cache.gather_fixed(tables, torch.float32)
    reference = decode_attention_fixed(q, kk, vv, lens, n_rep, hd ** -0.5)
    err = (fusionne - reference).abs().max().item()
    assert err < 5e-3, f"attention paginee : ecart {err:.2e} avec la reference"
