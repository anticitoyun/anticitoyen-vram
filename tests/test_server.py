"""La surface OpenAI, exercée à travers la véritable application ASGI."""

import json
import os

import pytest
import torch

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient   # noqa: E402


@pytest.fixture(scope="module")
def client(converted):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.server.app import create_app
    from acvram.server.chat import load_tokenizer

    vocab = {f"tok{i}": i for i in range(1024)}
    for i, w in enumerate(["<|im_start|>", "<|im_end|>", "hello", "world",
                           "the", "a", "of", "and", "</s>"]):
        vocab[w] = 1000 + i
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="tok0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.WordPiece(prefix="")
    tok.save(os.path.join(converted, "tokenizer.json"))
    json.dump({"eos_token": "</s>", "bos_token": "<|im_start|>",
               "chat_template": "{% for m in messages %}<|im_start|>{{m['role']}}\n"
                                "{{m['content']}}<|im_end|>\n{% endfor %}"
                                "{% if add_generation_prompt %}"
                                "<|im_start|>assistant\n{% endif %}"},
              open(os.path.join(converted, "tokenizer_config.json"), "w"))

    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    tokenizer = load_tokenizer(converted)
    engine = Engine(loaded, tokenizer, max_batch_size=4, max_model_len=256)
    with TestClient(create_app(engine, tokenizer, "tiny")) as c:
        yield c


def test_health_and_models(client):
    assert client.get("/health").json()["status"] == "ok"
    card = client.get("/v1/models").json()["data"][0]
    assert card["id"] == "tiny"
    assert card["acvram"]["formats"]


def test_chat_completion(client):
    r = client.post("/v1/chat/completions", json={
        "model": "tiny", "messages": [{"role": "user", "content": "hello world"}],
        "max_tokens": 5, "temperature": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"]["completion_tokens"] == 5
    assert body["choices"][0]["finish_reason"] == "length"


def test_chat_accepts_content_parts(client):
    r = client.post("/v1/chat/completions", json={
        "model": "tiny",
        "messages": [{"role": "user",
                      "content": [{"type": "text", "text": "hello"}]}],
        "max_tokens": 2, "temperature": 0})
    assert r.status_code == 200


def test_streaming_emits_sse_and_terminates(client):
    with client.stream("POST", "/v1/chat/completions", json={
            "model": "tiny", "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4, "stream": True, "temperature": 0,
            "stream_options": {"include_usage": True}}) as s:
        lines = [l for l in s.iter_lines() if l.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    first = json.loads(lines[0][6:])
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"]["role"] == "assistant"
    final = json.loads(lines[-2][6:])
    assert final["choices"][0]["finish_reason"]
    assert final["usage"]["completion_tokens"] == 4


def test_legacy_completions(client):
    r = client.post("/v1/completions", json={
        "model": "tiny", "prompt": "the world of", "max_tokens": 3,
        "temperature": 0})
    assert r.status_code == 200
    assert r.json()["object"] == "text_completion"


def test_completions_accepts_token_ids(client):
    r = client.post("/v1/completions", json={
        "model": "tiny", "prompt": [5, 42, 7], "max_tokens": 2, "temperature": 0})
    assert r.status_code == 200


def test_embeddings_are_normalised(client):
    r = client.post("/v1/embeddings",
                    json={"model": "tiny", "input": ["hello world", "the a of"]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["data"]) == 2
    for item in body["data"]:
        norm = sum(x * x for x in item["embedding"]) ** 0.5
        assert abs(norm - 1.0) < 1e-4


def test_embeddings_honour_dimensions(client):
    r = client.post("/v1/embeddings",
                    json={"model": "tiny", "input": "hello", "dimensions": 32})
    assert len(r.json()["data"][0]["embedding"]) == 32


def test_kv_blocks_are_returned_after_traffic(client):
    metrics = client.get("/metrics").json()["engine"]
    assert metrics["kv_blocks_free"] == metrics["kv_blocks_total"]


def test_trim_at_stop():
    """La séquence d'arrêt est exclue de la sortie, comme l'API s'y engage."""
    from acvram.engine.runner import _trim_at_stop

    # le stop tombe dans le delta courant
    texte, coupe = _trim_at_stop("1, 2, 3, 4, 5, 6, 7", ", 6, 7", ["7"])
    assert coupe and texte == ", 6, "
    # le stop commence avant le delta courant : rien à livrer de plus
    texte, coupe = _trim_at_stop("abcSTOPdef", "def", ["STOP"])
    assert coupe and texte == ""
    # pas de stop : delta intact
    texte, coupe = _trim_at_stop("abcdef", "def", ["ZZZ"])
    assert not coupe and texte == "def"
    # plusieurs stops : le plus précoce gagne
    texte, coupe = _trim_at_stop("aXbYc", "aXbYc", ["Y", "X"])
    assert coupe and texte == "a"
