"""Priorité 1 (chef, 24/09, gemma4 PPL=334991) : `token_budget()` traitait `max_tokens=0` comme
absent (`x or default`, 0 est faux en Python). `max_tokens=0` est un contrat OpenAI légitime avec
`echo=True` (lm-eval-harness et consorts : ne lire que les logprobs de l'invite, aucune
génération) — traité en silence comme "non posé", il générait un texte inattendu (contamination
constatée : ~255 jetons générés en plus dans un test `echo`, cf. verdict-echo.md).
Casse si `or` réapparaît."""
from acvram.server.protocol import CompletionRequest


def _req(**kw):
    return CompletionRequest(model="x", prompt="y", **kw)


def test_max_tokens_zero_explicite_rend_zero():
    assert _req(max_tokens=0).token_budget() == 0


def test_max_tokens_absent_rend_le_defaut():
    assert _req().token_budget() == 512
    assert _req().token_budget(default=256) == 256


def test_max_completion_tokens_prime_sur_max_tokens_meme_a_zero():
    assert _req(max_completion_tokens=0, max_tokens=99).token_budget() == 0


def test_max_tokens_non_nul_inchange():
    assert _req(max_tokens=42).token_budget() == 42


def test_bras_cassant_or_est_faux_sur_zero():
    """Bras cassant : la ligne `or` initiale, rejouée telle quelle."""
    req = _req(max_tokens=0)
    casse = req.max_completion_tokens or req.max_tokens or 512
    assert casse == 512          # exactement le bogue : 0 disparaît
    assert req.token_budget() == 0    # le correctif ne s'y trompe pas
