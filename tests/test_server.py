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
    # rechargement sur carte (poste7-reprise-ordre-18-09 §Suite).
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


def test_completions_sans_logprobs_identique(client):
    """Pièce 36 : sans logprobs, la réponse est inchangée (champ logprobs None) et
    déterministe à température 0 — la sortie par défaut ne bouge pas."""
    j = {"model": "tiny", "prompt": "hello world", "max_tokens": 4, "temperature": 0}
    a = client.post("/v1/completions", json=j).json()
    b = client.post("/v1/completions", json=j).json()
    assert a["choices"][0]["logprobs"] is None
    assert a["choices"][0]["text"] == b["choices"][0]["text"]


def test_completions_logprobs_echo(client):
    """Pièce 36 : logprobs=N + echo → CompletionChoice.logprobs (tokens, token_logprobs,
    top_logprobs, text_offset) sur l'invite PUIS la génération ; 1er jeton d'invite None."""
    r = client.post("/v1/completions", json={
        "model": "tiny", "prompt": "hello world", "max_tokens": 3,
        "temperature": 0, "logprobs": 3, "echo": True})
    assert r.status_code == 200
    lp = r.json()["choices"][0]["logprobs"]
    assert lp is not None
    for k in ("tokens", "token_logprobs", "top_logprobs", "text_offset"):
        assert k in lp and len(lp[k]) == len(lp["tokens"])
    assert lp["token_logprobs"][0] is None          # echo : 1er jeton d'invite sans prédécesseur
    assert len(lp["tokens"]) >= 3                    # au moins les 3 jetons générés (+ invite)
    assert lp["text_offset"] == sorted(lp["text_offset"])   # offsets croissants


def test_completions_logprobs_top_cpu(client):
    """Sur CPU (sampler non-graphe), les top-K logprobs sont servis (non None)."""
    r = client.post("/v1/completions", json={
        "model": "tiny", "prompt": "the world", "max_tokens": 2,
        "temperature": 0, "logprobs": 2})
    lp = r.json()["choices"][0]["logprobs"]
    gen_top = [t for t in lp["top_logprobs"] if t is not None]
    assert gen_top and all(isinstance(d, dict) and d for d in gen_top)


def test_speculation_visible_dans_metrics(client):
    """Pièce 49 : /metrics expose 'speculation' avec les clés attendues —
    le régime se porte par le nom, pas par la vigilance (REGLES §6)."""
    r = client.get("/metrics")
    assert r.status_code == 200
    j = r.json()
    # champ présent et non-None
    assert "speculation" in j, "/metrics doit avoir la clé 'speculation'"
    s = j["speculation"]
    assert s is not None, "speculation ne doit pas être None"
    # structure minimale : mode (str), garde_active (bool), lot_max (int)
    assert isinstance(s.get("mode"), str), "speculation.mode doit être une str"
    assert isinstance(s.get("garde_active"), bool), "speculation.garde_active doit être bool"
    assert isinstance(s.get("lot_max"), int), "speculation.lot_max doit être int"


def test_speculation_texte_dans_regime_ligne():
    """Pièce 49 : le fragment 'speculation=' est présent dans engine.regime_ligne()
    — le régime se porte par le nom (REGLES §6). Test direct sur le helper."""
    from acvram.engine.runner import _speculation_texte
    assert _speculation_texte(None) == " speculation=off"
    assert _speculation_texte({"mode": "off", "garde_active": False,
                               "gain_moyen": None, "lot_max": 0}) == " speculation=off"
    actif = {"mode": "ngram", "garde_active": True, "gain_moyen": 1.5, "lot_max": 4}
    frag = _speculation_texte(actif)
    assert frag.startswith(" speculation=ngram(")
    assert "on" in frag and "lot_max=4" in frag and "gain=1.5" in frag


# -- Pièce 210b : contrats OpenAI de /v1/completions --------------------------------------------------------------
def _sorties_210b(monkeypatch, sorties):
    """Le service rend ces `GenerationOutput` fabriquées (un jeton à texte vide, puis un jeton ordinaire), sans moteur."""
    import asyncio
    from acvram.server import app as A

    async def submit(self, prompt_ids, params, *a, **kw):
        return "req-210b", asyncio.Queue()

    async def collect(self, request_id, q):
        for s in sorties:
            yield s
    monkeypatch.setattr(A.EngineService, "submit", submit)
    monkeypatch.setattr(A.EngineService, "collect", collect)


def _deux_jetons():
    from acvram.engine.runner import GenerationOutput
    return [GenerationOutput(sequence_id=1, request_id="req-210b", token_ids=[7], text_delta="", logprob=-1.5,
                             completion_tokens=1),
            GenerationOutput(sequence_id=1, request_id="req-210b", token_ids=[8], text_delta="hello", logprob=-0.25,
                             completion_tokens=2, finished=True, finish_reason="length")]


def test_jeton_a_texte_vide_garde_son_logprob_210b(client, monkeypatch):
    """(a) tokens/token_logprobs alignés sur completion_tokens : le jeton à texte vide y reste. Témoin : sous l'ancienne
    condition (`out.text_delta`), tokens = ["hello"] pour 2 jetons générés — ce test rougit."""
    _sorties_210b(monkeypatch, _deux_jetons())
    r = client.post("/v1/completions", json={"model": "tiny", "prompt": [5, 6, 7], "max_tokens": 2, "logprobs": 1})
    j = r.json(); lp = j["choices"][0]["logprobs"]
    assert j["usage"]["completion_tokens"] == 2
    assert lp["tokens"] == ["", "hello"] and lp["token_logprobs"] == [-1.5, -0.25], lp
    assert lp["text_offset"] == [0, 0]


def test_flux_sans_include_usage_ne_porte_pas_d_usage_210b(client, monkeypatch):
    """(b) Sans `stream_options.include_usage`, aucun fragment ne porte `usage` (avant : {0, 0, 0} sur chacun) ; avec,
    le dernier seulement, et juste."""
    def fragments(corps):
        _sorties_210b(monkeypatch, _deux_jetons())
        r = client.post("/v1/completions", json=corps)
        return [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: ") and "[DONE]" not in l]
    base = {"model": "tiny", "prompt": [5, 6, 7], "max_tokens": 2, "stream": True}
    sans = fragments(base)
    assert len(sans) == 2 and all("usage" not in f for f in sans), sans
    avec = fragments({**base, "stream_options": {"include_usage": True}})
    assert "usage" not in avec[0], avec[0]
    assert avec[-1]["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}, avec[-1]


# -- sf2 : outils sur /v1/messages (API Anthropic, Claude Code) ---------------
_APPEL_LIRE = ('Je lis le fichier.\n<tool_call>\n{"name": "Read", "arguments": '
               '{"file_path": "/tmp/a.txt"}}\n</tool_call>')
_OUTIL_LIRE = {"name": "Read", "description": "Lit un fichier.",
               "input_schema": {"type": "object",
                                "properties": {"file_path": {"type": "string"}},
                                "required": ["file_path"]}}


def _moteur_factice(monkeypatch, texte):
    """Le modèle minuscule ne sait pas appeler d'outil : on remplace la
    génération par un texte fixé, la route (gabarit, relecture, blocs) reste vraie."""
    from acvram.engine.runner import GenerationOutput
    from acvram.server.app import EngineService
    vu = {}

    async def submit(self, prompt_ids, params, *a, **k):
        vu["prompt_ids"] = prompt_ids
        return "req-sf2", None

    async def collect(self, request_id, q):
        for i, morceau in enumerate((texte[:7], texte[7:30], texte[30:])):
            yield GenerationOutput(0, request_id, [], text_delta=morceau,
                                   completion_tokens=i + 1)
        yield GenerationOutput(0, request_id, [], finished=True,
                               finish_reason="stop", completion_tokens=4)
    monkeypatch.setattr(EngineService, "submit", submit)
    monkeypatch.setattr(EngineService, "collect", collect)
    return vu


def test_messages_tools_rend_tool_use(client, monkeypatch):
    """Requête avec tools -> bloc tool_use et stop_reason tool_use (avant sf2 :
    texte brut « <tool_call>… », end_turn — Claude Code n'avait aucun outil)."""
    _moteur_factice(monkeypatch, _APPEL_LIRE)
    d = client.post("/v1/messages", json={
        "model": "m", "max_tokens": 64, "tools": [_OUTIL_LIRE],
        "messages": [{"role": "user", "content": "Lis /tmp/a.txt"}]}).json()
    assert d["stop_reason"] == "tool_use"
    assert d["content"][0] == {"type": "text", "text": "Je lis le fichier."}
    appel = d["content"][1]
    assert appel["type"] == "tool_use" and appel["name"] == "Read"
    assert appel["input"] == {"file_path": "/tmp/a.txt"}
    assert appel["id"].startswith("toolu_")
    # sans tools : le même texte reste du texte (rien d'inventé)
    d = client.post("/v1/messages", json={
        "model": "m", "max_tokens": 64,
        "messages": [{"role": "user", "content": "Lis /tmp/a.txt"}]}).json()
    assert d["stop_reason"] == "end_turn" and d["content"] == [{"type": "text", "text": _APPEL_LIRE}]


def test_messages_tools_flux(client, monkeypatch):
    """En flux : le texte avant l'appel sort, la balise jamais ; le tool_use
    arrive en bloc 1 (input_json_delta), message_delta porte tool_use."""
    _moteur_factice(monkeypatch, _APPEL_LIRE)
    with client.stream("POST", "/v1/messages", json={
            "model": "m", "max_tokens": 64, "stream": True, "tools": [_OUTIL_LIRE],
            "messages": [{"role": "user", "content": "Lis /tmp/a.txt"}]}) as r:
        data = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ")]
    texte = "".join(e["delta"]["text"] for e in data
                    if e["type"] == "content_block_delta" and e["delta"]["type"] == "text_delta")
    assert texte.strip() == "Je lis le fichier." and "<tool_call>" not in texte
    debut = [e for e in data if e["type"] == "content_block_start" and e["index"] == 1]
    assert debut and debut[0]["content_block"]["type"] == "tool_use"
    assert debut[0]["content_block"]["name"] == "Read"
    json_partiel = "".join(e["delta"]["partial_json"] for e in data
                           if e["type"] == "content_block_delta" and e["index"] == 1)
    assert json.loads(json_partiel) == {"file_path": "/tmp/a.txt"}
    fin = [e for e in data if e["type"] == "message_delta"][0]
    assert fin["delta"]["stop_reason"] == "tool_use"
    assert data[-1]["type"] == "message_stop"


def test_messages_tool_result_suite(client, monkeypatch):
    """tool_result -> la suite : l'appel et son résultat atteignent le gabarit
    (message tool après l'appel), la réponse redevient du texte."""
    import acvram.server.app as app_mod
    _moteur_factice(monkeypatch, "Le fichier contient 42.")
    gabarit = {}
    vrai_rendu = app_mod.render_chat

    def rendu(tk, messages, agp=True, extra=None):
        gabarit.update(messages=messages, extra=extra)
        return vrai_rendu(tk, messages, agp, extra)
    monkeypatch.setattr(app_mod, "render_chat", rendu)
    d = client.post("/v1/messages", json={
        "model": "m", "max_tokens": 64, "tools": [_OUTIL_LIRE],
        "messages": [
            {"role": "user", "content": "Lis /tmp/a.txt"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Je lis."},
                {"type": "tool_use", "id": "toolu_1", "name": "Read",
                 "input": {"file_path": "/tmp/a.txt"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": [{"type": "text", "text": "42"}]}]}]}).json()
    assert d["stop_reason"] == "end_turn"
    assert d["content"] == [{"type": "text", "text": "Le fichier contient 42."}]
    roles = [m["role"] for m in gabarit["messages"]]
    assert roles == ["user", "assistant", "tool"]
    assert gabarit["messages"][1]["tool_calls"][0]["id"] == "toolu_1"
    assert gabarit["messages"][2]["content"] == "42"
    assert gabarit["extra"]["tools"][0]["function"]["name"] == "Read"


def test_messages_anthropic_vers_gabarit():
    from acvram.server.chat import Tokenizer, messages_anthropic, outils_anthropic, render_chat
    req = {"system": [{"type": "text", "text": "Sois bref."}],
           "messages": [
               {"role": "user", "content": [{"type": "text", "text": "Lis a"}]},
               {"role": "assistant", "content": [
                   {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "a"}}]},
               {"role": "user", "content": [
                   {"type": "tool_result", "tool_use_id": "toolu_1", "content": "42"},
                   {"type": "text", "text": "Et alors ?"}]}]}
    m = messages_anthropic(req)
    assert [x["role"] for x in m] == ["system", "user", "assistant", "tool", "user"]
    assert m[2]["tool_calls"][0]["function"] == {"name": "Read", "arguments": {"file_path": "a"}}
    assert m[3] == {"role": "tool", "content": "42", "tool_call_id": "toolu_1", "name": "Read"}
    assert m[4] == {"role": "user", "content": "Et alors ?"}
    # sans bloc d'outil : exactement la forme d'avant sf2
    assert messages_anthropic({"messages": [{"role": "user", "content": "x"}]}) == [
        {"role": "user", "content": "x"}]
    outils = outils_anthropic([_OUTIL_LIRE, {"type": "web_search_20250305", "name": "web_search"}])
    assert [o["function"]["name"] for o in outils] == ["Read"]
    assert outils[0]["function"]["parameters"] == _OUTIL_LIRE["input_schema"]
    tmpl = ("{% if tools %}<T>{% for t in tools %}{{ t.function.name }}{% endfor %}</T>{% endif %}"
            "{% for x in messages %}<{{ x.role }}>{{ x.content }}"
            "{% for c in x.tool_calls or [] %}[{{ c.function.name }} {{ c.function.arguments | tojson }}]"
            "{% endfor %}{% endfor %}")
    tk = Tokenizer(backend=None, config={}, template=tmpl, template_source="test")
    r = render_chat(tk, m, False, {"tools": outils})
    assert r.startswith("<T>Read</T>") and '[Read {"file_path": "a"}]<tool>42<user>Et alors ?' in r


def test_filtre_appels_bloc_illisible_reste_texte():
    """Un « <tool_call> » dont le JSON ne se lit pas ressort en texte à la fin,
    jamais perdu (le flux OpenAI le perdait : pend vidé une fois retenu)."""
    from acvram.server.chat import FiltreAppels, blocs_anthropic
    f = FiltreAppels()
    sortie = "".join(f.pousser(x) for x in ("avant ", "<tool", "_call>{pas du json}</tool_call>"))
    blocs, a_appel = blocs_anthropic(f.total, True)
    assert not a_appel
    assert sortie + f.reste() == "avant <tool_call>{pas du json}</tool_call>"


def test_messages_system_hors_tete_joint_au_user():
    """iqm : Claude Code envoie [user, system] dans ``messages`` (crochets SessionStart). Un gabarit qui
    n'accepte le système qu'en tête (KAT) levait → repli ChatML SANS outils. Le système tardif est joint au user."""
    from acvram.server.chat import Tokenizer, messages_anthropic, outils_anthropic, render_chat
    req = {"system": "Agent.", "tools": [_OUTIL_LIRE],
           "messages": [{"role": "user", "content": [{"type": "text", "text": "Lis a"}]},
                        {"role": "system", "content": [{"type": "text", "text": "Contexte du crochet."}]}]}
    m = messages_anthropic(req)
    assert m == [{"role": "system", "content": "Agent."},
                 {"role": "user", "content": "Contexte du crochet.\n\nLis a"}]     # claude-2 : la question en dernier
    tmpl = ("{% for x in messages %}{% if x.role == 'system' and not loop.first %}"
            "{{ raise_exception('System message must be at the beginning.') }}{% endif %}"
            "<{{ x.role }}>{{ x.content }}{% endfor %}"
            "{% if tools %}<T>{% for t in tools %}{{ t.function.name }}{% endfor %}</T>{% endif %}")
    tk = Tokenizer(backend=None, config={}, template=tmpl, template_source="test")
    r = render_chat(tk, m, True, {"tools": outils_anthropic(req["tools"])})
    assert "<T>Read</T>" in r and "Contexte du crochet." in r
    # système tardif après un assistant : joint au user suivant ; en queue : devient un user
    m = messages_anthropic({"messages": [
        {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
        {"role": "system", "content": "s1"}, {"role": "user", "content": "c"}, {"role": "system", "content": "s2"},
        {"role": "assistant", "content": "d"}, {"role": "system", "content": "s3"}]})
    assert m == [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                 {"role": "user", "content": "s1\n\ns2\n\nc"}, {"role": "assistant", "content": "d"},
                 {"role": "user", "content": "s3"}]


def test_messages_system_hors_tete_la_question_reste_en_dernier():
    """claude-2 (poste6, 29/09) : edz liste-2, Devstral/Nemo-12B-Claude/Qwen3-0.6B « sans Paris » depuis iqm — le
    crochet SessionStart (17 Ko) était joint APRÈS la question, le modèle répondait au crochet. Le texte de
    l'utilisateur termine le message ; le crochet le précède, entier."""
    from acvram.server.chat import messages_anthropic
    crochet = "SessionStart:startup hook success: état du projet, .deb, tests…\n" * 40
    req = {"system": [{"type": "text", "text": "Agent."}],
           "messages": [{"role": "user", "content": [{"type": "text", "text": "Capitale de la France ?"}]},
                        {"role": "system", "content": [{"type": "text", "text": crochet}]}]}
    m = messages_anthropic(req)
    assert [x["role"] for x in m] == ["system", "user"]
    assert m[1]["content"].endswith("Capitale de la France ?") and m[1]["content"].startswith(crochet)


def test_gabarit_tojson_comme_transformers():
    """iqm : GLM-4.7 écrit ``tools | tojson(ensure_ascii=False)`` ; le filtre de Jinja levait TypeError → repli
    ChatML sans outils. Et le rendu doit être celui de transformers (sans échappement HTML)."""
    from acvram.server.chat import Tokenizer, render_chat
    tmpl = ("{% for t in tools %}{{ t | tojson(ensure_ascii=False) }}{% endfor %}"
            "{% for x in messages %}{{ x.content }}{% endfor %}")
    tk = Tokenizer(backend=None, config={}, template=tmpl, template_source="test")
    outil = {"name": "grep", "description": "motif <regex> & 'é'"}
    r = render_chat(tk, [{"role": "user", "content": "?"}], False, {"tools": [outil]})
    assert r == json.dumps(outil, ensure_ascii=False) + "?"
