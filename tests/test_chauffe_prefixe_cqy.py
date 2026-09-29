"""cqy (29/09) : la chauffe prouve aussi le préfill d'une invite qui réutilise un préfixe en cache (q_offset > 0). 8fx :
« 34 816 tenus » était vrai depuis q_offset = 0 seulement ; une requête claude qui répétait son début faisait 2 Gio de
masque dense et un OOM en service. Contrôle : un OOM simulé au préfill à q_offset > 0 au-delà de 40 jetons fait descendre
le tenu à 40 (refus nommé), alors que la passe depuis 0 tenait 64 ; sans OOM, 64 tenus et aucun cache compté."""
import pytest
import torch

from tests.test_engine import _engine_cpu


def _oom_si_prefixe(eng, limite):
    vrai = eng.model.forward

    def forward(batch, *a, **k):
        if getattr(batch, "is_prefill", False) and int(batch.positions[0]) > 0 and int(batch.positions[-1]) + 1 > limite:
            raise torch.OutOfMemoryError("CUDA out of memory (simulé : masque dense à q_offset > 0)")
        return vrai(batch, *a, **k)
    eng.model.forward = forward


def test_le_tenu_descend_si_le_prefixe_en_cache_ne_tient_pas(converted, monkeypatch):
    from acvram.engine.runner import ContexteNonTenu
    monkeypatch.delenv("ACVRAM_CHAUFFE_CTX", raising=False)
    eng = _engine_cpu(converted)
    assert eng.allocator.enable_prefix_cache
    _oom_si_prefixe(eng, 40)
    with pytest.raises(ContexteNonTenu, match="max_model_len=64 demandé, 40 jetons tenus"):
        eng.chauffer_contexte(pas=8)
    assert eng.allocator.num_free == eng.allocator.num_blocks and not eng.running


def test_sans_oom_le_prefixe_tient_et_rien_n_est_compte(converted, monkeypatch):
    monkeypatch.delenv("ACVRAM_CHAUFFE_CTX", raising=False)
    eng = _engine_cpu(converted)
    vus = []
    vrai = eng.model.forward
    eng.model.forward = lambda batch, *a, **k: (vus.append(int(batch.positions[0])) if getattr(batch, "is_prefill", False)
                                                else None) or vrai(batch, *a, **k)
    assert eng.chauffer_contexte(pas=8) == 64
    assert any(p > 0 for p in vus), vus                    # la passe à préfixe en cache a bien eu lieu
    assert eng.stats.cached_prompt_tokens == 0 and eng.allocator.num_free == eng.allocator.num_blocks
