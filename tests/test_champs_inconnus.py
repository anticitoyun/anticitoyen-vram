"""Un champ de requête que le serveur ne lit pas est NOMMÉ, jamais avalé.

Avant : « temprature » (faute de frappe) ou « reasoning_effort » (option d'un
autre serveur) passaient sans bruit, le moteur servait le défaut et le client
croyait avoir réglé quelque chose (MECANISMES, « champs inconnus acceptés en
silence »). Le contrat OpenAI reste le même : la requête passe.
"""
import logging

import pytest

pytest.importorskip("pydantic")
from acvram.server.protocol import ChatCompletionRequest, CompletionRequest  # noqa: E402

_MSG = [{"role": "user", "content": "x"}]


def test_les_champs_inconnus_sont_nommes_et_sans_effet():
    req = ChatCompletionRequest.model_validate({
        "model": "m", "messages": _MSG, "temprature": 0.2, "reasoning_effort": "low"})
    assert req.champs_inconnus() == ["reasoning_effort", "temprature"]
    assert req.temperature == 1.0            # la faute de frappe n'a PAS réglé la température


def test_temoin_un_champ_connu_n_est_pas_signale_et_agit():
    req = ChatCompletionRequest.model_validate({"model": "m", "messages": _MSG, "temperature": 0.2})
    assert req.champs_inconnus() == []
    assert req.temperature == 0.2


def test_completions_herite_du_meme_controle():
    req = CompletionRequest.model_validate({"model": "m", "prompt": "x", "top_kk": 3})
    assert req.champs_inconnus() == ["top_kk"] and req.top_k == 0


def test_le_serveur_journalise_une_fois_par_nom(caplog):
    from acvram.server import app as serveur
    serveur._CHAMPS_SIGNALES.clear()
    req = ChatCompletionRequest.model_validate({"model": "m", "messages": _MSG, "temprature": 0.2})
    with caplog.at_level(logging.WARNING, logger="acvram.server"):
        assert serveur.signaler_champs_inconnus(req, "/v1/chat/completions") == ["temprature"]
        assert serveur.signaler_champs_inconnus(req, "/v1/chat/completions") == ["temprature"]
    avert = [r for r in caplog.records if "champs ignorés" in r.getMessage()]
    assert len(avert) == 1 and "temprature" in avert[0].getMessage() and "/v1/chat/completions" in avert[0].getMessage()


def test_temoin_sans_champ_inconnu_aucun_avertissement(caplog):
    from acvram.server import app as serveur
    serveur._CHAMPS_SIGNALES.clear()
    req = ChatCompletionRequest.model_validate({"model": "m", "messages": _MSG})
    with caplog.at_level(logging.WARNING, logger="acvram.server"):
        assert serveur.signaler_champs_inconnus(req, "/v1/chat/completions") == []
    assert not [r for r in caplog.records if "champs ignorés" in r.getMessage()]
