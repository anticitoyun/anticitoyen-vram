"""Pièce 277 (correctif 277fix) : `step()` passait au pas spéculatif alors qu'un pas simple, lancé en recouvrement
(`_pipeline_pendiente`), était encore en vol — la vérification relisait le dernier jeton, puis le pas simple suivant
livrait le résultat périmé : jetons répétés. SANS correctif, sur le mixte-i8c (ngram k = 4, défaut de `serve`
jusqu'à la 283), 4 invites sur 5 divergeaient du décodage simple dès le 5e-17e jeton, et le jeton émis était
5,4 à 13,5 logits de marge sous le premier choix de la cible (13 à 24 logits sous le jeton de référence)
(revue/poste5-piece277abis-verdict-26-09.md). `step()` vide désormais le pipeline avant de spéculer
(`_pipeline_vider`).

Deux volets (décision chef, 277cm) :
* **mixte (hybride GDN)** : égalité EXACTE, ngram k = 4 et k = 1 contre none (vraie au bit après correctif) ;
* **Coder (dense MoE)** : la vérification MoE à q_len 5 change l'ordre des sommes — égalité impossible par
  construction ; à CHAQUE divergence, le jeton de référence doit être le premier choix de la cible, le jeton
  spéculatif le second, marge ≤ 0,5 (seuil du scellé 277a-bis, jamais relevé après coup). Mesuré après
  correctif : 2 divergences, marges 0,0152 et 0,0154.
Condition d'entrée, TENUE (prises poste5-p277fin et poste5-p277finc, même fichier lancé dans l'arbre 8d5c5580c
sans correctif) : mixte ROUGE (k = 4 et k = 1) ; Coder ROUGE — invite 0, j = 8, réf 79 / spéc 397, marge 12,19. Carte et modèles requis (sous carte.sh) ; ignoré sinon."""
import os

import pytest
import torch

pytestmark = pytest.mark.gpu_requis
MIXTE, CODER = "Qwen3.8-27B-unsloth-mixte-i8c", "Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c"
N, SEUIL = 32, 0.5


def _charger(alias):
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
    return loaded, tok, [ids[400 * i: 400 * i + 200] for i in range(5)]


_UN = {}                                                 # un seul modèle chargé à la fois (deux 27-30B ne tiennent pas)


def _modele(alias):
    if alias not in _UN:
        _UN.clear()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        _UN[alias] = _charger(alias)
    return _UN[alias]


@pytest.fixture
def mixte():
    return _modele(MIXTE)


@pytest.fixture
def coder():
    return _modele(CODER)


def _generer(loaded, tok, invites, k, graphes=True, n=N):
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    from acvram.engine.speculative import NGramProposer
    eng = Engine(loaded, tok, max_batch_size=2, max_model_len=1024, enable_cuda_graphs=graphes,
                 speculator=NGramProposer() if k else None, spec_k=max(k, 1))
    eng._eos = set()
    vidages = [0]
    if hasattr(eng, "_pipeline_vider"):                   # arbre corrigé ; absent sur la base (condition d'entrée)
        vrai = eng._pipeline_vider

        def compte():
            vidages[0] += eng._pipeline_pendiente is not None
            return vrai()
        eng._pipeline_vider = compte
    sorties = []
    for p in invites:
        seq = eng.add_request(list(p), SamplingParams(max_tokens=n, temperature=0.0))
        while eng.running or eng.waiting:
            eng.step()
        sorties.append(list(seq.output_ids))
    return sorties, eng.stats.proposed_tokens, vidages[0]


@pytest.mark.parametrize("k", [4, 1])
def test_mixte_ngram_egal_au_decodage_simple_au_bit(mixte, k):
    loaded, tok, invites = mixte
    ref, _, _ = _generer(loaded, tok, invites, 0)
    spec, proposes, _ = _generer(loaded, tok, invites, k)
    assert proposes > 0
    assert spec == ref


def test_coder_ngram_ne_diverge_que_par_quasi_egalite(coder):
    loaded, tok, invites = coder
    ref, _, _ = _generer(loaded, tok, invites, 0)
    spec, proposes, _ = _generer(loaded, tok, invites, 4)
    assert proposes > 0
    vus = []
    vrai = loaded.model._logits_finaux

    def espion(x):
        y = vrai(x)
        vus.append(y[-1].detach().float().clone())
        return y
    loaded.model._logits_finaux = espion
    try:
        for i, (a, b) in enumerate(zip(ref, spec)):
            if a == b:
                continue
            j = next(t for t, (x, y) in enumerate(zip(a, b)) if x != y)
            vus.clear()
            rejeu, _, _ = _generer(loaded, tok, [invites[i]], 0, graphes=False, n=j + 1)
            assert rejeu[0] == a[:j + 1], f"invite {i} : le rejeu eager du témoin n'est pas fidèle jusqu'à j = {j}"
            v, ix = vus[j].topk(2)
            marge = float(v[0] - v[1])
            assert (int(ix[0]), int(ix[1])) == (a[j], b[j]) and marge <= SEUIL, \
                f"invite {i}, j = {j} : top1/top2 {int(ix[0])}/{int(ix[1])} pour réf/spéc {a[j]}/{b[j]}, marge {marge:.4f}"
    finally:
        loaded.model._logits_finaux = vrai
