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

    loaded = load_model(converted, dtype=torch.float32, device_override="cpu",
                        max_concurrent_seqs=4)
    # `_replanifier` (loader.py) n'agit que si `detect_rig()` voit un GPU —
    # absent ici, le kwarg ci-dessus est ignoré et le plan garde le
    # `kv_planned_seqs=2` du manifeste `converted` (conftest.py). Le porter
    # explicitement au lot réellement servi, comme le ferait un vrai
    # rechargement sur carte (sage-reprise-ordre-18-09 §Suite).
    loaded.plan.kv_planned_seqs = 4
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


def test_anthropic_messages_route(client):
    """/v1/messages : format Anthropic, non-stream et stream."""
    r = client.post("/v1/messages", json={
        "model": "m", "max_tokens": 8,
        "system": "Réponds brièvement.",
        "messages": [{"role": "user",
                      "content": [{"type": "text", "text": "Bonjour"}]}],
        "temperature": 0})
    assert r.status_code == 200
    d = r.json()
    assert d["type"] == "message" and d["role"] == "assistant"
    assert d["content"][0]["type"] == "text"
    assert d["stop_reason"] in ("end_turn", "max_tokens")
    assert d["usage"]["input_tokens"] > 0

    with client.stream("POST", "/v1/messages", json={
            "model": "m", "max_tokens": 8, "stream": True,
            "messages": [{"role": "user", "content": "Bonjour"}]}) as r:
        assert r.status_code == 200
        evenements = [l for l in r.iter_lines() if l.startswith("event: ")]
    noms = [e.split(": ", 1)[1] for e in evenements]
    assert noms[0] == "message_start"
    assert "content_block_delta" in noms
    assert noms[-1] == "message_stop"


def test_chat_template_kwargs_atteint_le_gabarit():
    """{"enable_thinking": false} doit parvenir au gabarit Jinja, comme chez
    vLLM et llama.cpp — sans quoi un Qwen3 réfléchit toujours, et un modèle
    qui répond vide en mode réflexion n'a aucun recours."""
    from acvram.server.chat import Tokenizer, render_chat
    tmpl = ("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>"
            "{% if enable_thinking is defined and not enable_thinking %}<think></think>"
            "{% else %}<think>{% endif %}{% endif %}")
    tk = Tokenizer(backend=None, config={}, template=tmpl, template_source="test")
    msgs = [{"role": "user", "content": "Bonjour"}]
    assert render_chat(tk, msgs, True).endswith("<|assistant|><think>")
    assert render_chat(tk, msgs, True, {"enable_thinking": False}).endswith("<think></think>")
    assert render_chat(tk, msgs, True, {"enable_thinking": True}).endswith("<|assistant|><think>")


def test_chat_template_kwargs_dans_la_requete():
    from acvram.server.protocol import ChatCompletionRequest
    r = ChatCompletionRequest(model="x", messages=[{"role": "user", "content": "a"}],
                              chat_template_kwargs={"enable_thinking": False})
    assert r.chat_template_kwargs == {"enable_thinking": False}


def test_extraire_appels_qwen():
    from acvram.server.chat import extraire_appels
    txt = ('Je regarde.\n<tool_call>\n{"name": "meteo", "arguments": {"ville": "Lyon"}}\n</tool_call>')
    reste, appels = extraire_appels(txt)
    assert reste == "Je regarde."
    assert len(appels) == 1 and appels[0]["type"] == "function"
    assert appels[0]["function"]["name"] == "meteo"
    assert appels[0]["function"]["arguments"] == '{"ville": "Lyon"}'


def test_extraire_appels_json_casse_reste_du_texte():
    from acvram.server.chat import extraire_appels
    txt = "<tool_call>{pas du json}</tool_call>"
    assert extraire_appels(txt) == (txt, [])


def test_extraire_appels_plusieurs():
    from acvram.server.chat import extraire_appels
    txt = ('<tool_call>{"name": "a", "arguments": {}}</tool_call>'
           '<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>')
    reste, appels = extraire_appels(txt)
    assert reste == "" and [a["function"]["name"] for a in appels] == ["a", "b"]
    assert appels[0]["id"] != appels[1]["id"]


def test_outils_rendus_par_le_gabarit():
    """``tools`` doit atteindre le gabarit : c'est lui qui les décrit au modèle."""
    from acvram.server.chat import Tokenizer, render_chat
    tmpl = ("{% if tools %}<|tools|>{% for t in tools %}{{ t.function.name }};{% endfor %}{% endif %}"
            "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}")
    tk = Tokenizer(backend=None, config={}, template=tmpl, template_source="test")
    outils = [{"type": "function", "function": {"name": "meteo", "parameters": {}}}]
    r = render_chat(tk, [{"role": "user", "content": "?"}], True, {"tools": outils})
    assert r.startswith("<|tools|>meteo;")


def test_messages_pour_gabarit_garde_les_champs_outil():
    from acvram.server.chat import messages_pour_gabarit
    from acvram.server.protocol import ChatMessage
    ms = [ChatMessage(role="assistant", content=None,
                      tool_calls=[{"id": "c1", "type": "function",
                                   "function": {"name": "meteo", "arguments": "{}"}}]),
          ChatMessage(role="tool", content="12 °C", tool_call_id="c1", name="meteo")]
    d = messages_pour_gabarit(ms)
    assert d[0]["tool_calls"][0]["function"]["name"] == "meteo" and d[0]["content"] == ""
    assert d[1] == {"role": "tool", "content": "12 °C", "name": "meteo", "tool_call_id": "c1"}


def test_extraire_appels_xml_qwen3_coder():
    from acvram.server.chat import extraire_appels
    txt = ("<tool_call>\n<function=meteo>\n<parameter=ville>\nLyon\n</parameter>\n"
           "<parameter=jours>\n3\n</parameter>\n</function>\n</tool_call>")
    reste, appels = extraire_appels(txt)
    assert reste == "" and appels[0]["function"]["name"] == "meteo"
    import json
    assert json.loads(appels[0]["function"]["arguments"]) == {"ville": "Lyon", "jours": 3}


def test_un_champ_inconnu_passe_et_est_journalise(client, caplog):
    """Le contrat OpenAI tient (200, même réponse) et le journal nomme le champ."""
    import logging
    from acvram.server import app as serveur
    serveur._CHAMPS_SIGNALES.discard("temprature")
    with caplog.at_level(logging.WARNING, logger="acvram.server"):
        r = client.post("/v1/chat/completions", json={
            "model": "tiny", "messages": [{"role": "user", "content": "hello world"}],
            "max_tokens": 5, "temperature": 0, "temprature": 0.7})
    assert r.status_code == 200 and r.json()["usage"]["completion_tokens"] == 5
    assert any("temprature" in rec.getMessage() for rec in caplog.records)
