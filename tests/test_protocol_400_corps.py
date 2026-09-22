"""D3 — les réponses 400 du serveur portent toujours un corps JSON explicatif.

Requête invalide (champ obligatoire absent) → 400, pas 422, corps JSON avec
le champ refusé nommé. Cassure : supprimer le handler RequestValidationError
→ FastAPI renvoie 422 sans ce corps structuré.
"""
import pytest
from fastapi.testclient import TestClient

from acvram.server.app import create_app
from acvram.server.protocol import ChatCompletionRequest


@pytest.fixture(scope="module")
def client():
    app = create_app(engine=None, tokenizer=None, model_name="test-model")
    return TestClient(app, raise_server_exceptions=False)


def test_requete_sans_messages_retourne_400(client):
    """Sans 'messages' (obligatoire), le serveur doit renvoyer 400, pas 422."""
    resp = client.post("/v1/chat/completions", json={"model": "test-model"})
    assert resp.status_code == 400, (
        f"attendu 400, reçu {resp.status_code} — "
        "handler RequestValidationError absent ?"
    )


def test_corps_400_est_json(client):
    """Le corps de la réponse 400 doit être JSON parsable."""
    resp = client.post("/v1/chat/completions", json={"model": "test-model"})
    body = resp.json()
    assert "error" in body or "message" in body or "detail" in body, (
        f"corps inattendu : {body}"
    )


def test_corps_400_nomme_le_champ(client):
    """Le message d'erreur doit mentionner 'messages' (le champ manquant)."""
    resp = client.post("/v1/chat/completions", json={"model": "test-model"})
    text = resp.text
    assert "messages" in text, f"le champ manquant 'messages' absent du corps : {text}"


def test_requete_kimi_prompt_cache_key_passe(client):
    """prompt_cache_key (champ inconnu de kimi v2) doit être accepté sans 400."""
    resp = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "ping"}],
        "prompt_cache_key": "abc123",
        "stream": False,
    })
    # Le serveur peut renvoyer 500 (pas de tokeniseur) mais pas 400 ni 422
    assert resp.status_code not in (400, 422), (
        f"prompt_cache_key refusé à tort : {resp.status_code} {resp.text[:200]}"
    )
