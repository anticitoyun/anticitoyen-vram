"""Pièce 277 (correctif) : spéculer avec un pas simple ENCORE EN VOL (pipeline, `_pipeline_pendiente`) faisait relire le
dernier jeton à la vérification puis livrer le résultat périmé — jetons répétés, sortie gloutonne ≠ décodage simple,
jetons que la cible rejette de 13 à 24 logits (277a-bis, mixte-i8c, ngram k = 4, défaut de `serve`). `step()` vide
désormais le pipeline avant de spéculer (`_pipeline_vider`).
(1) ngram k = 4 et k = 1 : sortie gloutonne = sans spéculation, vidages > 0 (le chemin corrigé a été pris) ;
(2) témoin cassant : l'ancien comportement (spéculer sans vider) DOIT rendre une sortie différente.
Carte et modèle requis (sous carte.sh) ; ignoré sinon. ACVRAM_TEST_SPEC_ALIAS choisit l'alias."""
import os

import pytest
import torch

pytestmark = pytest.mark.gpu_requis
ALIAS = os.environ.get("ACVRAM_TEST_SPEC_ALIAS", "Qwen3.8-27B-unsloth-mixte-i8c")
N = 32


@pytest.fixture(scope="module")
def modele():
    import sys
    racine = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(racine, "outils"))
    from racine_modeles import racine_modeles
    chemin = os.path.join(racine_modeles(), ALIAS)
    if not torch.cuda.is_available() or not os.path.isdir(chemin):
        pytest.skip(f"carte ou modèle absent ({ALIAS})")
    if torch.cuda.mem_get_info()[0] < 26 << 30:
        pytest.skip("moins de 26 Gio libres")
    from acvram.engine.loader import load_model
    from acvram.server.chat import load_tokenizer
    tok = load_tokenizer(chemin)
    loaded = load_model(chemin, dtype=torch.bfloat16, max_model_len=1024, max_concurrent_seqs=2)
    ids = tok.encode(open(os.path.join(racine, "README.md"), encoding="utf-8").read())
    yield loaded, tok, [ids[400 * i: 400 * i + 200] for i in range(5)]
    del loaded
    torch.cuda.empty_cache()


def _generer(loaded, tok, invites, k):
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams
    from acvram.engine.speculative import NGramProposer
    eng = Engine(loaded, tok, max_batch_size=2, max_model_len=1024, enable_cuda_graphs=True,
                 speculator=NGramProposer() if k else None, spec_k=max(k, 1))
    eng._eos = set()
    vidages = [0]
    vrai = eng._pipeline_vider

    def compte():
        vidages[0] += eng._pipeline_pendiente is not None
        return vrai()
    eng._pipeline_vider = compte
    sorties = []
    for p in invites:
        seq = eng.add_request(list(p), SamplingParams(max_tokens=N, temperature=0.0))
        while eng.running or eng.waiting:
            eng.step()
        sorties.append(list(seq.output_ids))
    return sorties, eng.stats.proposed_tokens, vidages[0]


@pytest.mark.parametrize("k", [4, 1])
def test_ngram_glouton_egal_au_decodage_simple(modele, k):
    loaded, tok, invites = modele
    ref, _, _ = _generer(loaded, tok, invites, 0)
    spec, proposes, vidages = _generer(loaded, tok, invites, k)
    assert proposes > 0 and vidages > 0, (proposes, vidages)
    assert spec == ref


def test_temoin_speculer_sans_vider_change_la_sortie(modele, monkeypatch):
    from acvram.engine.runner import Engine

    def ancien(self):                     # l'avant-277 : spéculer avec le pas simple toujours en vol
        pend = self._pipeline_pendiente
        return self._speculative_decode([s for s in pend["seqs"] if not s.finished])
    loaded, tok, invites = modele
    ref, _, _ = _generer(loaded, tok, invites, 0)
    monkeypatch.setattr(Engine, "_pipeline_vider", ancien)
    faux, _, _ = _generer(loaded, tok, invites, 4)
    assert faux != ref
