"""D2 v2 — _garde_contexte borne max_tokens à ctx − prompt au lieu de refuser.

Cas de ce soir : prompt ~9 814 jetons (tokenizer kimi), max_tokens=32 768,
ctx=32 768 → total 42 582 → le moteur refusait. Correctif : bornage silencieux
à ctx − len(prompt_ids) = 22 954, comme llama.cpp.

Cassure : commenter le bloc `if params is not None` dans _garde_contexte
→ params.max_tokens non borné → le moteur voit 32 768 + prompt > ctx et refuse.
"""
import pytest
from unittest.mock import MagicMock

from acvram.server.app import _garde_contexte
from acvram.engine.sampler import SamplingParams


def _engine(ctx: int = 32768):
    e = MagicMock()
    e.max_model_len = ctx
    e.ctx_demande = None
    return e


def _params(max_tokens: int) -> SamplingParams:
    return SamplingParams(max_tokens=max_tokens)


def test_max_tokens_borne_quand_depasse():
    """prompt 9814 + max_tokens 32768 > 32768 → max_tokens borné à 22954."""
    engine = _engine(32768)
    params = _params(32768)
    prompt_ids = list(range(9814))
    _garde_contexte(engine, prompt_ids, params=params)
    assert params.max_tokens == 32768 - 9814, (
        f"attendu {32768 - 9814}, obtenu {params.max_tokens}"
    )


def test_max_tokens_inchange_quand_dans_fenetre():
    """prompt 100 + max_tokens 512 < 32768 → max_tokens intact."""
    engine = _engine(32768)
    params = _params(512)
    _garde_contexte(engine, list(range(100)), params=params)
    assert params.max_tokens == 512


def test_invite_trop_longue_leve_400():
    """prompt ≥ max_model_len → HTTPException 400 (inchangé)."""
    from fastapi import HTTPException
    engine = _engine(32768)
    params = _params(1)
    with pytest.raises(HTTPException) as exc_info:
        _garde_contexte(engine, list(range(32768)), params=params)
    assert exc_info.value.status_code == 400


def test_sans_params_comportement_inchange():
    """Sans params, _garde_contexte ne touche rien (appels existants préservés)."""
    engine = _engine(32768)
    _garde_contexte(engine, list(range(100)))   # ne doit pas lever
