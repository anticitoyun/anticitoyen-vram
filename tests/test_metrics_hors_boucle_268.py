"""Pièce 268 : /metrics ne gèle plus la boucle HTTP (262 : 343 ms par appel, TTFT à 12 de 0,242 à 0,785 s sous un lecteur
à 20 Hz). Trois cassants, chacun ROUGE sur l'ancien code (vérifié en l'y rejouant) :
* BOUCLE : /metrics lent (regime() retardé de 0,3 s) n'empêche pas /health de répondre — `async def` le bloquait ;
* COMPTE : regime() une seule fois par /metrics (ancien : sept) ;
* PARCOURS : `model.modules()` parcouru une seule fois par regime() (ancien : quatre).
Plus l'identité : mêmes clés qu'avant, champs issus de regime() égaux à `engine.regime()`."""
import json
import os
import threading
import time

import pytest
import torch

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient   # noqa: E402

# clés de /metrics avant la 268 (app.py:996-1031 à e3c345bf7), plus celles de `app.state.info`
CLES_AVANT = {"engine", "kv_max_tokens", "kv_planned_seqs", "kv_tokens_par_sequence_planifiee", "cartes", "repli_eager",
              "replis_eager_raisons", "graphes", "graphes_refus_n", "graphes_refus_principale", "energie", "regime_ligne",
              "depaquetage", "int8_chemins", "speculation", "version"}
DE_REGIME = {"cartes": None, "repli_eager": 0, "replis_eager_raisons": [], "graphes": None, "graphes_refus_n": 0,
             "graphes_refus_principale": None, "speculation": None}


@pytest.fixture(scope="module")
def client(converted):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.server.app import create_app
    from acvram.server.chat import load_tokenizer

    vocab = {f"tok{i}": i for i in range(1024)}
    for i, w in enumerate(["<|im_start|>", "<|im_end|>", "hello", "world", "</s>"]):
        vocab[w] = 1000 + i
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="tok0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.WordPiece(prefix="")
    tok.save(os.path.join(converted, "tokenizer.json"))
    json.dump({"eos_token": "</s>", "bos_token": "<|im_start|>",
               "chat_template": "{% for m in messages %}{{m['content']}}{% endfor %}"},
              open(os.path.join(converted, "tokenizer_config.json"), "w"))
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu", max_concurrent_seqs=4)
    loaded.plan.kv_planned_seqs = 4
    tokenizer = load_tokenizer(converted)
    engine = Engine(loaded, tokenizer, max_batch_size=4, max_model_len=256)
    with TestClient(create_app(engine, tokenizer, "tiny")) as c:
        yield c, engine


def test_cassant_boucle_libre_pendant_metrics(client, monkeypatch):
    c, engine = client
    vrai = engine.regime

    def lent():
        time.sleep(0.3)
        return vrai()
    monkeypatch.setattr(engine, "regime", lent)
    fil = threading.Thread(target=lambda: c.get("/metrics"))
    fil.start()
    time.sleep(0.05)                                   # /metrics est en cours
    t0 = time.perf_counter()
    assert c.get("/health").status_code == 200
    attente = time.perf_counter() - t0
    fil.join()
    assert attente < 0.1, f"/health a attendu {attente:.3f} s derrière /metrics : la boucle HTTP est bloquée"


def test_cassant_regime_une_fois_par_metrics(client, monkeypatch):
    c, engine = client
    vrai, n = engine.regime, [0]

    def compte():
        n[0] += 1
        return vrai()
    monkeypatch.setattr(engine, "regime", compte)
    c.get("/metrics")
    assert n[0] == 1, f"regime() appelé {n[0]} fois par /metrics"


def test_cassant_un_parcours_des_modules_par_regime(client, monkeypatch):
    _, engine = client
    vrai, n = engine.model.modules, [0]

    def compte(*a, **k):
        n[0] += 1
        return vrai(*a, **k)
    monkeypatch.setattr(engine.model, "modules", compte)
    engine.regime()
    assert n[0] == 1, f"model.modules() parcouru {n[0]} fois par regime()"


def test_identite_des_champs(client):
    c, engine = client
    m = c.get("/metrics").json()
    assert CLES_AVANT <= set(m), sorted(CLES_AVANT - set(m))
    assert set(m) - CLES_AVANT == set(c.app.state.info), sorted(set(m) - CLES_AVANT)
    r = engine.regime()
    for cle, defaut in DE_REGIME.items():
        attendu = r[cle] if cle == "cartes" else r.get(cle, defaut)
        assert m[cle] == json.loads(json.dumps(attendu)), cle


# -- étape 2 : gabarit et tokeniseur hors de la boucle HTTP ------------------------------------------------------------

def test_cassant_gabarit_hors_boucle(client, monkeypatch):
    """Gabarit retardé de 0,3 s : /health doit répondre pendant ce temps (ancien code : boucle bloquée → ROUGE)."""
    from acvram.server import app as A
    c, _ = client
    vrai = A.render_chat

    def lent(*a, **k):
        time.sleep(0.3)
        return vrai(*a, **k)
    monkeypatch.setattr(A, "render_chat", lent)
    fil = threading.Thread(target=lambda: c.post("/v1/chat/completions", json={
        "model": "tiny", "max_tokens": 1, "messages": [{"role": "user", "content": "hello world"}]}))
    fil.start()
    time.sleep(0.05)
    t0 = time.perf_counter()
    assert c.get("/health").status_code == 200
    attente = time.perf_counter() - t0
    fil.join()
    assert attente < 0.1, f"/health a attendu {attente:.3f} s derrière le gabarit : la boucle HTTP est bloquée"


def test_memes_jetons_au_bit(client, monkeypatch):
    """Les jetons soumis au moteur = gabarit puis tokeniseur appelés directement, dans le même ordre, au jeton près."""
    from acvram.server.app import _encode
    from acvram.server.chat import messages_pour_gabarit, render_chat
    from acvram.server.protocol import ChatCompletionRequest
    c, engine = client
    vus, vrai = [], engine.add_request

    def espion(prompt_ids, *a, **k):
        vus.append(list(prompt_ids))
        return vrai(prompt_ids, *a, **k)
    monkeypatch.setattr(engine, "add_request", espion)
    corps = {"model": "tiny", "max_tokens": 1,
             "messages": [{"role": "system", "content": "the a of"}, {"role": "user", "content": "hello world and the"}]}
    assert c.post("/v1/chat/completions", json=corps).status_code == 200
    req = ChatCompletionRequest(**corps)
    tok = engine.tokenizer                              # le tokeniseur que create_app a reçu
    attendu = _encode(tok, render_chat(tok, messages_pour_gabarit(req.messages, avec_images=False),
                                       req.add_generation_prompt, {}))
    assert vus == [attendu]
