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


def test_la_ligne_du_moteur_nomme_le_format_kv_servi(converted, monkeypatch):
    """20/09 (Jérôme, prise (b) 12B vision) : la ligne de régime n'imprimait aucun kv= — le palier cuda:0 du
    manifeste servait le KV en int8 sans le dire (regime.py ne nommait que int8-canal16). Le mot `kv=` vient des
    caches CONSTRUITS (format servi), jamais d'une variable : int8 par le plan, bf16 sous ACVRAM_KV_FORMAT=bf16."""
    import torch
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.memory import tiering
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    eng = Engine(loaded, None, max_batch_size=1, max_model_len=64, enable_cuda_graphs=False)
    ligne = eng.regime_ligne()
    assert " kv=" in ligne, ligne
    assert f" kv={eng.kv_format_servi()} " in ligne
    fmt = {str(c.cfg.dtype) for c in loaded.model.caches.values()}
    assert eng.kv_format_servi() in fmt or not loaded.model.caches, (eng.kv_format_servi(), fmt)
    monkeypatch.setattr(tiering, "_KV_FORMAT", "bf16")
    loaded2 = load_model(converted, dtype=torch.float32, device_override="cpu")
    eng2 = Engine(loaded2, None, max_batch_size=1, max_model_len=64, enable_cuda_graphs=False)
    assert " kv=bf16 " in eng2.regime_ligne(), eng2.regime_ligne()
    assert eng.kv_format_servi() != "bf16", "le défaut du plan est déjà bf16 : le témoin ne distingue rien"


def _engine_cpu(converted, max_model_len=64):
    import torch
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu", max_model_len=max_model_len, max_concurrent_seqs=2)
    return Engine(loaded, None, max_batch_size=1, max_model_len=max_model_len, enable_cuda_graphs=False)


def test_la_chauffe_prouve_le_contexte_et_la_ligne_le_dit(converted, monkeypatch):
    """Sage sage-3b-lanceur-contexte-20-09 (ii) : max_model_len se prouve au chargement par un prefill plein dans
    le régime servi ; tenu → ctx_tenu=N sur la ligne ; la séquence de chauffe n'est pas gardée par le cache de
    préfixe (un vrai préfixe de « 1 » n'y trouverait rien) ; avant la chauffe la ligne dit non-chauffe."""
    monkeypatch.delenv("ACVRAM_CHAUFFE_CTX", raising=False)
    eng = _engine_cpu(converted)
    assert " ctx_tenu=non-chauffe" in eng.regime_ligne()
    assert eng.chauffer_contexte(pas=8) == 64
    assert " ctx_tenu=64 " in eng.regime_ligne() + " "
    assert eng.allocator.num_free == eng.allocator.num_blocks, "les blocs de la chauffe n'ont pas été rendus"
    assert eng.stats.cached_prompt_tokens == 0 and not eng.running and not eng.waiting


def test_un_contexte_non_tenu_est_refuse_nomme_avec_la_longueur_tenue(converted, monkeypatch):
    """OOM simulé au-delà de 40 jetons (torch.OutOfMemoryError) : dichotomie (≤ 5 pas, multiples de 8) → 40 tenus,
    ContexteNonTenu(64, 40) nommé, blocs rendus, moteur réutilisable."""
    import torch
    import pytest
    from acvram.engine.runner import ContexteNonTenu
    monkeypatch.delenv("ACVRAM_CHAUFFE_CTX", raising=False)
    eng = _engine_cpu(converted)
    vrai = eng.generate

    def faux(prompt_ids, params, images=None):
        if len(prompt_ids) + 2 > 40:
            raise torch.OutOfMemoryError("CUDA out of memory (simulé : 384 Mio demandés, 354 libres)")
        return vrai(prompt_ids, params, images=images)
    monkeypatch.setattr(eng, "generate", faux)
    with pytest.raises(ContexteNonTenu, match="max_model_len=64 demandé, 40 jetons tenus") as e:
        eng.chauffer_contexte(pas=8)
    assert (e.value.demande, e.value.tenu) == (64, 40) and eng.ctx_tenu == 40
    assert " ctx_tenu=40(demandé 64) " in eng.regime_ligne() + " "     # refus : la ligne porte demandé et tenu
    assert eng.allocator.num_free == eng.allocator.num_blocks and not eng.running


def test_opt_out_nomme_de_la_chauffe_jamais_en_service(converted, monkeypatch):
    monkeypatch.setenv("ACVRAM_CHAUFFE_CTX", "0"); monkeypatch.delenv("ACVRAM_TYPE", raising=False)
    eng = _engine_cpu(converted)
    assert eng.chauffer_contexte(pas=8) is None and " ctx_tenu=non-verifie" in eng.regime_ligne()
    monkeypatch.setenv("ACVRAM_TYPE", "service")
    eng2 = _engine_cpu(converted)
    assert eng2.chauffer_contexte(pas=8) == 64, "un service prouve son contexte malgré l'opt-out"


def test_la_sequence_de_chauffe_est_pseudo_aleatoire_graine_fixe(converted):
    """Sage sage-s2-k48-feu-vert-21-09 § 2 (a) : la chauffe ne passe plus ``[1]×N`` mais une séquence sur le
    vocabulaire, graine 0, reproductible d'un chargement à l'autre (même séquence = même crête mesurée)."""
    eng = _engine_cpu(converted)
    a, b = eng.sequence_de_chauffe(62), eng.sequence_de_chauffe(62)
    assert a == b and len(a) == 62 and a != [1] * 62 and len(set(a)) > 8
    assert all(0 <= t < eng.spec.vocab_size for t in a)


def test_un_contexte_non_tenu_est_clampe_et_la_ligne_le_dit(converted, monkeypatch):
    """§ 2 (c) : OOM simulé au-delà de 40 → ctx_tenu=40, max_model_len clampé à 40, ligne
    ``ctx_tenu=40(demandé 64)`` exacte, pas de refus (CTX_TENU_MIN abaissé pour le jouet) ; --ctx-strict refuse."""
    import torch
    import pytest
    from acvram.engine import runner
    monkeypatch.delenv("ACVRAM_CHAUFFE_CTX", raising=False)
    monkeypatch.setattr(runner, "CTX_TENU_MIN", 8)
    eng = _engine_cpu(converted)
    vrai = eng.generate

    def faux(prompt_ids, params, images=None):
        if len(prompt_ids) + 2 > 40:
            raise torch.OutOfMemoryError("CUDA out of memory (simulé)")
        return vrai(prompt_ids, params, images=images)
    monkeypatch.setattr(eng, "generate", faux)
    assert eng.chauffer_contexte(pas=8) == 40
    assert eng.max_model_len == 40 and eng.ctx_demande == 64
    assert " ctx_tenu=40(demandé 64) " in eng.regime_ligne() + " ", eng.regime_ligne()
    eng2 = _engine_cpu(converted)
    monkeypatch.setattr(eng2, "generate", faux)
    with pytest.raises(runner.ContexteNonTenu):
        eng2.chauffer_contexte(pas=8, strict=True)


def test_un_contexte_tenu_sans_reserve_n_est_pas_tenu(converted, monkeypatch):
    """§ 2 (b) : la passe réussit mais laisse moins de max(5 %, 64 Mio) libres au-delà de 40 jetons → ctx_tenu=40 ;
    le témoin (réserve pleine) rend 64 : le seuil de réserve seul fait la différence."""
    from acvram.engine import runner
    monkeypatch.delenv("ACVRAM_CHAUFFE_CTX", raising=False)
    monkeypatch.setattr(runner, "CTX_TENU_MIN", 8)
    total = 32 << 30
    eng = _engine_cpu(converted)
    vu = []
    vrai = eng.generate

    def note(prompt_ids, params, images=None):
        vu.append(len(prompt_ids) + 2)
        return vrai(prompt_ids, params, images=images)
    monkeypatch.setattr(eng, "generate", note)
    monkeypatch.setattr(eng, "_libre_apres_chauffe", lambda: ((total * 4 // 100) if vu[-1] > 40 else total // 2, total))
    assert eng.chauffer_contexte(pas=8) == 40 and eng.reserve_chauffe[1] == total * 5 // 100
    eng2 = _engine_cpu(converted)
    monkeypatch.setattr(eng2, "_libre_apres_chauffe", lambda: (total // 2, total))
    assert eng2.chauffer_contexte(pas=8) == 64


def test_la_capture_des_graphes_vient_apres_le_clamp_et_a_sa_taille(converted, monkeypatch):
    """Maîtresse 21/09 (GLM k48 : OOM à la capture au ctx demandé, avant la chauffe) : demarrer_service = chauffe
    (graphes masqués : aucun run pendant les essais) → clamp → capture au contexte TENU. Faux graphes : ctx demandé
    64 > tenu 40 ⇒ graphs.max_model_len == 40, run jamais appelé avant le clamp, chargement réussi."""
    import torch
    from types import SimpleNamespace
    from acvram.engine import runner
    monkeypatch.delenv("ACVRAM_CHAUFFE_CTX", raising=False)
    monkeypatch.setattr(runner, "CTX_TENU_MIN", 8)
    eng = _engine_cpu(converted)
    eng.pipeline_actif = False                     # les faux graphes n ont que `run` (chemin sans recouvrement) ; rouge depuis ACVRAM_PIPELINE=1 par défaut
    journal = []
    faux_graphes = SimpleNamespace(max_model_len=64, captures=0, enabled=True, run=lambda batch: journal.append(("run", eng.max_model_len, None)) or None)
    eng.graphs = faux_graphes
    vrai = eng.generate

    def faux(prompt_ids, params, images=None):
        if len(prompt_ids) + 2 > 40:
            raise torch.OutOfMemoryError("CUDA out of memory (simulé)")
        journal.append(("chauffe", len(prompt_ids) + 2, eng.graphs))
        return vrai(prompt_ids, params, images=images)
    monkeypatch.setattr(eng, "generate", faux)
    tenu, captures = eng.demarrer_service(warm_max_len=32, pas=8, pas_confirmation=8)
    assert tenu == 40 and eng.max_model_len == 40 and faux_graphes.max_model_len == 40
    passes = [(L, g) for (k, L, g) in journal if k == "chauffe"]
    assert all(g is None for _, g in passes[:-1]), "graphes visibles pendant la dichotomie"
    assert passes[-1] == (40, faux_graphes), "la confirmation se fait AU TENU, graphes actifs, après la capture"
    assert eng.graphs is faux_graphes and " ctx_tenu=40(demandé 64) " in eng.regime_ligne() + " "


def test_la_confirmation_avec_graphes_baisse_le_tenu_de_deux_pas_puis_charge(converted, monkeypatch):
    """Maîtresse 21/09 : des graphes qui « coûtent » 2 pas (OOM avec graphes au-dessus de tenu − 16, pas 8) ⇒
    confirmation échouée deux fois, recapture à chaque baisse, tenu final = 40 − 16 = 24, chargement réussi ;
    des graphes qui coûtent 3 pas ⇒ refus nommé."""
    import torch
    import pytest
    from types import SimpleNamespace
    from acvram.engine import runner
    monkeypatch.delenv("ACVRAM_CHAUFFE_CTX", raising=False)
    monkeypatch.setattr(runner, "CTX_TENU_MIN", 8)

    def moteur(cout_pas: int):
        eng = _engine_cpu(converted)
        eng.pipeline_actif = False                     # idem : faux graphes à `run` seul
        faux_graphes = SimpleNamespace(max_model_len=64, captures=0, enabled=True, run=lambda batch: None)
        eng.graphs = faux_graphes
        vrai = eng.generate
        recaptures = []

        def faux(prompt_ids, params, images=None):
            L = len(prompt_ids) + 2
            if L > 40 or (eng.graphs is not None and L > 40 - 8 * cout_pas):
                raise torch.OutOfMemoryError("CUDA out of memory (simulé)")
            return vrai(prompt_ids, params, images=images)

        def recapturer(warm_max_len):
            recaptures.append(eng.max_model_len); faux_graphes.max_model_len = eng.max_model_len; eng.graphs = faux_graphes
            return 0
        monkeypatch.setattr(eng, "generate", faux); monkeypatch.setattr(eng, "_recapturer", recapturer)
        return eng, faux_graphes, recaptures
    eng, fg, rec = moteur(2)
    tenu, _ = eng.demarrer_service(warm_max_len=32, pas=8, pas_confirmation=8)
    assert tenu == 24 and eng.max_model_len == 24 and fg.max_model_len == 24 and rec == [32, 24], (tenu, rec)
    assert " ctx_tenu=24(demandé 64) " in eng.regime_ligne() + " "
    eng3, _, rec3 = moteur(3)
    with pytest.raises(runner.ContexteNonTenu):
        eng3.demarrer_service(warm_max_len=32, pas=8, pas_confirmation=8)
    assert rec3 == [32, 24], "deux baisses au plus avant le refus"


def test_chaque_pas_de_chauffe_part_d_un_allocateur_vide(converted, monkeypatch):
    """Maîtresse 21/09 (3) : sans libération entre deux pas, les fragments du pas précédent s additionnent et chaque
    pas voit moins de libre — allocateur simulé : réservé cumulé tant que `_avant_essai_de_chauffe` ne le vide pas.
    Avec la libération : 64 tenu (chaque pas laisse ≥ 5 %) ; sans (méthode neutralisée) : la même chauffe descend."""
    from acvram.engine import runner
    monkeypatch.delenv("ACVRAM_CHAUFFE_CTX", raising=False)
    monkeypatch.setattr(runner, "CTX_TENU_MIN", 8)
    total = 100 << 20
    etat = {"reserve": 0, "vidages": 0, "essais": 0}

    def moteur(vider: bool):
        eng = _engine_cpu(converted)
        vrai = eng.generate

        def note(prompt_ids, params, images=None):
            etat["essais"] += 1
            etat["reserve"] += (len(prompt_ids) + 2) * (total // 100) // 2     # un pas de L jetons réserve L/2 % du total
            return vrai(prompt_ids, params, images=images)
        monkeypatch.setattr(eng, "generate", note)
        monkeypatch.setattr(eng, "_libre_apres_chauffe", lambda: (total - etat["reserve"], total))
        if vider:
            def vidage():
                etat["vidages"] += 1; etat["reserve"] = 0
            monkeypatch.setattr(eng, "_avant_essai_de_chauffe", vidage)
        else:
            monkeypatch.setattr(eng, "_avant_essai_de_chauffe", lambda: None)
        return eng
    e1 = moteur(vider=True)
    assert e1.chauffer_contexte(pas=8) == 64 and etat["vidages"] == etat["essais"] == 1
    etat.update(reserve=0, vidages=0, essais=0)
    e2 = moteur(vider=False)
    assert e2.chauffer_contexte(pas=8) == 64, "un seul pas : rien à cumuler, le témoin doit tenir aussi"
    # même simulation, pas de 4 % par jeton : 64 ne tient pas (32 % + 5 % ... ) → la dichotomie fait plusieurs pas
    etat.update(reserve=0, vidages=0, essais=0)
    e3 = moteur(vider=False)
    monkeypatch.setattr(e3, "_libre_apres_chauffe", lambda: (total - etat["reserve"] * 3, total))
    try:
        sans = e3.chauffer_contexte(pas=8)
    except runner.ContexteNonTenu as e:                                   # descend jusqu à 0 : refus nommé
        sans = e.tenu
    etat.update(reserve=0, vidages=0, essais=0)
    e4 = moteur(vider=True)
    monkeypatch.setattr(e4, "_libre_apres_chauffe", lambda: (total - etat["reserve"] * 3, total))
    avec = e4.chauffer_contexte(pas=8)
    assert avec > sans, (avec, sans)
    assert etat["vidages"] == etat["essais"]


def test_le_pipeline_est_actif_par_defaut_et_la_ligne_dit_l_effectif(converted, monkeypatch):
    """0.6.34 (Maîtresse 21/09) : ACVRAM_PIPELINE=1 par défaut, 0 = témoin ; la ligne porte `pipeline=` EFFECTIF
    (demandé ET graphes) : sans graphes (CPU) elle dit 0 même demandé, pour ne pas nommer un recouvrement absent."""
    monkeypatch.delenv("ACVRAM_PIPELINE", raising=False)
    eng = _engine_cpu(converted)
    assert eng.pipeline_actif is True and " pipeline=0 " in eng.regime_ligne() + " "     # graphes off à sec
    monkeypatch.setenv("ACVRAM_PIPELINE", "0")
    assert _engine_cpu(converted).pipeline_actif is False
    from acvram import regime
    v = next(x for x in regime.VARIABLES if x.env == "ACVRAM_PIPELINE")
    assert v.defaut == "1"


def test_jamais_plus_de_blocs_que_le_contexte(converted):
    """P3 (4) 30B (Manon 21/09) : la marge d admission +BLOCK_SIZE donnait 257 blocs à une invite de max_model_len − 2
    quand le graphe est dimensionné au contexte (256) → `size of tensor a (256) must match b (257)` à la confirmation
    avec graphes. Invariant : une séquence ne tient jamais plus de ceil(max_model_len / BLOCK_SIZE) blocs, à
    l admission comme au dernier jeton — c est ce que le godet de graphe suppose."""
    from acvram.engine.runner import SamplingParams
    from acvram.memory.kvcache import BLOCK_SIZE
    eng = _engine_cpu(converted, max_model_len=64)
    plafond = (64 + BLOCK_SIZE - 1) // BLOCK_SIZE
    pic = {"n": 0}
    vrai_finish = eng._finish

    def finish(seq, raison):                       # la table est libérée à la fin : on la lit juste avant
        pic["n"] = max(pic["n"], len(seq.blocks)); return vrai_finish(seq, raison)
    eng._finish = finish
    seq = eng.add_request(eng.sequence_de_chauffe(62), SamplingParams(max_tokens=2, temperature=0.0))
    while not seq.finished:
        list(eng.step()); pic["n"] = max(pic["n"], len(seq.blocks))
    assert seq.length == 64 and pic["n"] == plafond, f"{pic['n']} blocs pour max_model_len=64 (plafond {plafond})"
