"""Ajout GUI n°3 (poste7-gui-ajouts-18-09 § 3) : `/metrics` doit porter le
J/jeton en direct, intégré côté serveur sur la fenêtre glissante entre deux
appels — compteur NVML monotone (`capteurs.energie_mj`), jamais une moyenne
de puissances. Recette de poste7 : « à vide "—", jamais 0 ; sous charge, ± 10 %
de energie.py sur la même fenêtre. »"""

import json
import os

import pytest
import torch

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient   # noqa: E402

from acvram.server import app as app_mod   # noqa: E402


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
        yield c, engine


def test_absent_sans_nvml(client, monkeypatch):
    c, engine = client
    monkeypatch.setattr(app_mod._capteurs, "energie_mj", lambda: {})
    monkeypatch.setattr(app_mod._capteurs, "nvidia", lambda: [])
    c.app.state.energie_precedente = None
    m = c.get("/metrics").json()
    assert m["energie"]["j_par_jeton"] is None
    assert m["energie"]["cartes"] == []


def test_absent_au_premier_appel(client, monkeypatch):
    """Le tout premier appel n'a pas de fenetre : jamais 0, toujours None."""
    c, engine = client
    monkeypatch.setattr(app_mod._capteurs, "energie_mj", lambda: {0: 1_000_000})
    monkeypatch.setattr(app_mod._capteurs, "nvidia",
                         lambda: [{"index": 0, "horloge_sm": 1500, "watts_max": 400}])
    c.app.state.energie_precedente = None
    m = c.get("/metrics").json()
    assert m["energie"]["j_par_jeton"] is None
    assert m["energie"]["cartes"] == [{"index": 0, "horloge_sm": 1500, "watts_plafond": 400}]


def test_ratio_calcule_sur_la_fenetre(client, monkeypatch):
    c, engine = client
    engine.stats.decode_tokens = 100
    monkeypatch.setattr(app_mod._capteurs, "energie_mj", lambda: {0: 1_000_000})
    monkeypatch.setattr(app_mod._capteurs, "nvidia",
                         lambda: [{"index": 0, "horloge_sm": 1500, "watts_max": 400}])
    c.app.state.energie_precedente = None
    c.get("/metrics")   # amorce la fenetre

    engine.stats.decode_tokens = 150
    monkeypatch.setattr(app_mod._capteurs, "energie_mj", lambda: {0: 1_050_000})
    m = c.get("/metrics").json()
    # delta 50 000 mJ = 50 J, delta 50 jetons -> 1,0 J/jeton
    assert m["energie"]["j_par_jeton"] == 1.0


def test_ratio_absent_sans_nouveaux_jetons(client, monkeypatch):
    """Diviser par zero doit rendre None, jamais une division fausse."""
    c, engine = client
    engine.stats.decode_tokens = 200
    monkeypatch.setattr(app_mod._capteurs, "energie_mj", lambda: {0: 2_000_000})
    monkeypatch.setattr(app_mod._capteurs, "nvidia", lambda: [])
    c.app.state.energie_precedente = None
    c.get("/metrics")

    monkeypatch.setattr(app_mod._capteurs, "energie_mj", lambda: {0: 2_100_000})
    m = c.get("/metrics").json()   # decode_tokens inchange -> delta_tok = 0
    assert m["energie"]["j_par_jeton"] is None
