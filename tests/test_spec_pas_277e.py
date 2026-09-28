"""Pièce 277e : le ngram corrigé (277fix) faisait 24 à 55 pas de `step()` pour 32 jetons contre 32 pour none — un
proposeur muet retombait sur une amorce du pipeline (0 jeton livré), vidée seule au pas suivant. `_pas_speculatif`
(pipeline.py) fait partager le pas au vidage et à la spéculation, enchaîne la suite après un repli qui ne livre rien,
et reste `spec_repos_max` pas en recouvrement après un repli. Chaque pas livre donc au moins un jeton.

Même générateur que `test_spec_pipeline_277.py` (5 invites × 32, gloutons, graphes). Exigé, sur les deux alias :
pas ngram ≤ pas none à chaque invite, et strictement moins au total. Sur main 1b59ed0bd (277fix seule), le Coder en
fait 211 contre 160 (départage 277fix) : ce test y est ROUGE. Le volet sortie (mixte au bit, Coder quasi-égalités)
reste celui de `test_spec_pipeline_277.py`, inchangé. À lancer dans son propre processus (un modèle à la fois)."""
import os

import pytest
import torch

pytestmark = pytest.mark.gpu_requis
MIXTE, CODER = "Qwen3.8-27B-unsloth-mixte-i8c", "Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c"
N = 32
_UN = {}


def _modele(alias):
    if alias not in _UN:
        _UN.clear()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        import sys
        racine = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(racine, "outils"))
        from racine_modeles import racine_modeles
        chemin = os.path.join(racine_modeles(), alias)
        if not torch.cuda.is_available() or not os.path.isdir(chemin):
            pytest.skip(f"carte ou modèle absent ({alias})")
        if torch.cuda.mem_get_info()[0] < 26 << 30:
            pytest.skip("moins de 26 Gio libres")
        from acvram.engine.loader import load_model
        from acvram.server.chat import load_tokenizer
        tok = load_tokenizer(chemin)
        loaded = load_model(chemin, dtype=torch.bfloat16, max_model_len=1024, max_concurrent_seqs=2)
        ids = tok.encode(open(os.path.join(racine, "README.md"), encoding="utf-8").read())
        _UN[alias] = loaded, tok, [ids[400 * i: 400 * i + 200] for i in range(5)]
    return _UN[alias]


def _pas(loaded, tok, invites, k):
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    from acvram.engine.speculative import NGramProposer
    eng = Engine(loaded, tok, max_batch_size=2, max_model_len=1024, enable_cuda_graphs=True,
                 speculator=NGramProposer() if k else None, spec_k=max(k, 1))
    eng._eos = set()
    pas = []
    for p in invites:
        seq = eng.add_request(list(p), SamplingParams(max_tokens=N, temperature=0.0))
        n = 0
        while (eng.running or eng.waiting) and n < 4 * N:
            eng.step()
            n += 1
        assert len(seq.output_ids) == N
        pas.append(n)
    return pas, eng.stats.proposed_tokens


@pytest.mark.parametrize("alias", [MIXTE, CODER])
def test_ngram_fait_moins_de_pas_que_none(alias):
    loaded, tok, invites = _modele(alias)
    ref, _ = _pas(loaded, tok, invites, 0)
    spec, proposes = _pas(loaded, tok, invites, 4)
    assert proposes > 0
    assert all(s <= r for s, r in zip(spec, ref)) and sum(spec) < sum(ref), f"ngram {spec} contre none {ref}"
