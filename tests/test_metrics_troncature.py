"""Ajout GUI n°2 (sage-gui-ajouts-18-09 § 2) : `/metrics` doit porter le
compteur de troncature par budget KV (deja compte par `EngineStats`, jamais
affiche avant ce jour) ACCOMPAGNE du budget reel par sequence planifiee
(`kv_max_tokens / kv_planned_seqs`) — sans quoi un compteur > 0 ne dit pas
si le budget est structurellement sous-dimensionne ou accidentel."""

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
    loaded.plan.kv_planned_seqs = 4   # cf. test_server.py : pas de GPU ici, le kwarg est ignoré
    tokenizer = load_tokenizer(converted)
    engine = Engine(loaded, tokenizer, max_batch_size=4, max_model_len=256)
    with TestClient(create_app(engine, tokenizer, "tiny")) as c:
        yield c, engine


def test_compteur_a_zero_sans_troncature(client):
    c, engine = client
    engine.stats.sequences_tronquees_budget = 0
    m = c.get("/metrics").json()
    assert m["engine"]["sequences_tronquees_budget"] == 0


def test_compteur_et_budget_par_sequence_planifiee(client, monkeypatch):
    """« 12 requêtes ⇒ compteur > 0 » (recette de Sage) : simule directement
    l'etat que l'engine aurait produit — le mecanisme de troncature lui-meme
    est deja couvert par test_budget_kv_slots_chargeur.py, ce test verifie
    seulement que /metrics EXPOSE ce que l'engine sait deja."""
    c, engine = client
    engine.stats.sequences_tronquees_budget = 3
    monkeypatch.setattr(engine.loaded.plan, "kv_max_tokens", 4096, raising=False)
    monkeypatch.setattr(engine.loaded.plan, "kv_planned_seqs", 8, raising=False)

    m = c.get("/metrics").json()
    assert m["engine"]["sequences_tronquees_budget"] == 3
    assert m["kv_max_tokens"] == 4096
    assert m["kv_planned_seqs"] == 8
    assert m["kv_tokens_par_sequence_planifiee"] == 512.0

    engine.stats.sequences_tronquees_budget = 0   # ne pas polluer les tests suivants


def test_ratio_absent_si_aucune_sequence_planifiee(client, monkeypatch):
    """Diviser par zero doit rendre None, jamais planter la route ni
    produire une division silencieuse fausse."""
    c, engine = client
    monkeypatch.setattr(engine.loaded.plan, "kv_planned_seqs", 0, raising=False)
    m = c.get("/metrics").json()
    assert m["kv_tokens_par_sequence_planifiee"] is None


def test_metrics_porte_les_cartes(client):
    """sage-profil-verdict-18-09 §4 : sans ce champ, un client HTTP de
    mesure (hors carte.sh) ne peut pas savoir quelle carte est réellement
    servie et somme celles qu'il voit lui-même (18/09, acvram [0,1] contre
    llama.cpp [0], énergie faussée)."""
    c, engine = client
    m = c.get("/metrics").json()
    assert m["cartes"] == engine.regime()["cartes"]
