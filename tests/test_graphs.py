"""Le rejeu en graphe CUDA doit être invisible dans la sortie.

L'invariant s'énonce précisément : *au sein d'un même moteur*, le pas capturé
rend les mêmes logits que le chemin eager sur le même état. Comparer deux
moteurs séparés par l'argmax ne teste rien de tel : sur un modèle-jouet aux
logits quasi plats, cuBLAS et SDPA choisissent leurs algorithmes selon l'état
du contexte, et l'argmax devient une loterie d'epsilon — le chemin eager seul
n'est déjà pas reproductible d'un processus à l'autre à ce grain-là.

« Les mêmes », pas « les mêmes bits » : depuis que le chemin eager peut lui
aussi emprunter le noyau fusionné ou batcher plusieurs séquences (10/09),
graphe et eager sont deux ordres de calcul légitimement différents, chacun
capable de produire un résultat correct sans reproduire l'autre au bit près.
`assert_logits_proches` (conftest) compare par KL plutôt que par
`torch.equal` — voir `KL_LOGITS_MAX` pour la justification du seuil.
"""

import pytest

@pytest.fixture(autouse=True)
def _lot_du_test_pas_du_plan(monkeypatch):
    """Le converti du disque porte son plan (kv_planned_seqs) ; le test dimensionne son lot (T4 20/09)."""
    monkeypatch.setenv("ACVRAM_KV_PLAN_OVERRIDE", "1")
import torch

pytestmark = pytest.mark.gpu_requis

from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
from conftest import assert_logits_proches

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
    dec = e._decodables()
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
        assert_logits_proches(eager, graphe,
                              "le graphe et l'eager divergent sur le meme etat")
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
        assert_logits_proches(eager, graphe,
                              "le graphe et l'eager divergent apres la frontiere de godet")
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
    assert_logits_proches(eager, graphe,
                          "le graphe et l'eager divergent (modele qknorm)")


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


@needs_cuda
def test_paged_attention_speculative_matches_loop(converted):
    """q_len > 1 : chaque position de requête voit son propre préfixe causal.

    Le chemin de référence est la boucle attention() par séquence avec
    q_offset ; le noyau paginé doit rendre la même chose, position par
    position, dans la marge de la déquantification du cache.
    """
    import os

    from acvram.engine.sampler import SamplingParams as SP

    loaded = load_model(converted, dtype=torch.bfloat16,
                        device_override="cuda:0")
    e = Engine(loaded, None, max_batch_size=2, max_model_len=256,
               enable_cuda_graphs=False)
    e.add_request(list(range(1, 40)), SP(temperature=0.0, max_tokens=30))
    e.step()

    # un lot de vérification artificiel : 4 positions par séquence
    dec = [s for s in e.running if not s.finished]
    for s in dec:
        assert e._grow(s, extra=3)
    from acvram.engine.speculative import Proposal
    props = {s.id: Proposal([5, 6, 7]) for s in dec}
    batch = e._build_spec_batch(dec, props)
    logits_noyau = e.model(batch,
                           logits_positions=batch.all_token_indices()).float()
    os.environ["ACVRAM_DISABLE_PAGED_ATTN"] = "1"
    try:
        logits_boucle = e.model(batch,
                                logits_positions=batch.all_token_indices()).float()
    finally:
        del os.environ["ACVRAM_DISABLE_PAGED_ATTN"]
    err = (logits_noyau - logits_boucle).abs().max().item()
    assert err < 5e-2, f"verification speculative : ecart {err:.3e}"


@needs_cuda
def test_graph_speculative_step_equals_eager(converted):
    """Le pas de vérification spéculative capturé rend les logits de l'eager."""
    from acvram.engine.sampler import SamplingParams as SP
    from acvram.engine.speculative import Proposal

    loaded = load_model(converted, dtype=torch.bfloat16,
                        device_override="cuda:0")
    e = Engine(loaded, None, max_batch_size=2, max_model_len=256,
               enable_cuda_graphs=True)
    assert e.graphs is not None and e.graphs.paged_ok
    e.add_request(list(range(1, 30)), SP(temperature=0.0, max_tokens=30))
    e.step()
    dec = [s for s in e.running if not s.finished]
    for s in dec:
        assert e._grow(s, extra=3)
    props = {s.id: Proposal([5, 6, 7]) for s in dec}
    batch = e._build_spec_batch(dec, props)
    eager = e.model(batch, logits_positions=batch.all_token_indices()).float()
    graphe = e.graphs.run(batch)
    assert graphe is not None, "pas de graphe pour le pas speculatif"
    assert_logits_proches(eager, graphe.float(),
                          "pas speculatif : graphe et eager divergent")


@needs_cuda
def test_speculative_generation_under_graphs(converted):
    """Une génération spéculative complète sous graphes reste exacte.

    À température nulle, l'acceptation exacte garantit la même sortie que le
    décodage ordinaire — c'est l'invariant historique du projet, désormais
    affirmé aussi avec les graphes actifs des deux côtés.
    """
    from acvram.engine.sampler import SamplingParams as SP
    from acvram.engine.speculative import NGramProposer

    prompt = [7, 3, 9, 1, 4, 8, 2, 5] * 5

    def run(spec):
        loaded = load_model(converted, dtype=torch.bfloat16,
                            device_override="cuda:0")
        e = Engine(loaded, None, max_batch_size=1, max_model_len=256,
                   enable_cuda_graphs=True,
                   speculator=NGramProposer() if spec else None, spec_k=3)
        return [t for o in e.generate(prompt, SP(temperature=0.0,
                                                 max_tokens=24))
                for t in o.token_ids]

    assert run(True) == run(False), \
        "la speculation sous graphes a change la sortie"


@needs_cuda
def test_graph_bucket_padding_equivalence(converted):
    """b=3 dans un godet de 4 : le rembourrage ne change pas les logits.

    Le bucket_batch(3) == 4, donc le graphe capture un tenseur de taille 4
    mais seuls les 3 premiers slots portent de vraies séquences. L'invariant :
    les logits des 3 séquences réelles sont identiques entre eager et graphe,
    et l'état des séquences existantes n'est pas corrompu par le slot fantôme.
    """
    e = _engine(converted, n_batch=4)
    assert e.graphs is not None
    prompts = [[7, 3, 9, 1, 4, 8], [2, 5, 6, 1, 3, 8], [4, 1, 7, 2, 9, 5]]
    for p in prompts:
        _prefill(e, p)
    assert len([s for s in e.running if not s.finished]) == 3

    for pas in range(4):
        eager, graphe, batch, dec = _decode_both(e)
        assert batch.batch_size == 3, f"b_reel attendu 3, obtenu {batch.batch_size}"
        assert_logits_proches(eager, graphe,
                              f"pas {pas} : graphe diverge avec b=3 dans godet 4")
        e._emit(graphe, dec)

    assert e.graphs.replays >= 4
