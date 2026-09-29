"""8fx (29/09) : après une exception dans un pas moteur (OOM de préfill : masque causal dense de 2,10 Gio sur une invite
claude longue, skyfall 31B en 0.7.13), la requête SUIVANTE (kimi) restait sans réponse 300 s. Contrôle : une exception
injectée une fois au préfill de A ; A reçoit une erreur, B est servie."""
import json
import os
import threading

import pytest
import torch

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient   # noqa: E402


@pytest.fixture
def app_et_moteur(converted):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.server.app import create_app
    from acvram.server.chat import load_tokenizer

    vocab = {f"tok{i}": i for i in range(1024)}
    for i, w in enumerate(["<|im_start|>", "<|im_end|>", "hello", "world", "the", "a", "of", "and", "</s>"]):
        vocab[w] = 1000 + i
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="tok0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.WordPiece(prefix="")
    tok.save(os.path.join(converted, "tokenizer.json"))
    json.dump({"eos_token": "</s>", "bos_token": "<|im_start|>",
               "chat_template": "{% for m in messages %}<|im_start|>{{m['role']}}\n{{m['content']}}<|im_end|>\n{% endfor %}"
                                "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"},
              open(os.path.join(converted, "tokenizer_config.json"), "w"))
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu", max_concurrent_seqs=4)
    loaded.plan.kv_planned_seqs = 4
    engine = Engine(loaded, load_tokenizer(converted), max_batch_size=4, max_model_len=256)
    return create_app(engine, load_tokenizer(converted), "tiny"), engine


def test_requete_suivante_servie_apres_exception_de_prefill(app_et_moteur):
    import gc
    import weakref
    app, engine = app_et_moteur
    vrai = engine.model.forward
    lance = {"n": 0, "ref": None}
    libres = engine.allocator.num_free

    def forward(batch, *a, **k):
        if getattr(batch, "is_prefill", False) and lance["n"] == 0:
            lance["n"] += 1
            gros = torch.empty(1 << 20)                   # tenseur local au pas qui échoue (masque, activations)
            lance["ref"] = weakref.ref(gros)
            raise torch.OutOfMemoryError("OOM simulé au préfill (8fx)")
        return vrai(batch, *a, **k)
    engine.model.forward = forward
    corps = {"model": "tiny", "max_tokens": 4, "temperature": 0,
             "messages": [{"role": "user", "content": "hello world the a of"}]}
    with TestClient(app, raise_server_exceptions=False) as c:
        ra = c.post("/v1/chat/completions", json=corps)
        assert lance["n"] == 1 and ra.status_code >= 400, ra.text[:300]
        gc.collect()
        assert not engine.running, "la séquence du pas en échec reste au moteur"
        assert engine.allocator.num_free == libres, "ses blocs KV ne sont pas rendus"
        assert lance["ref"]() is None, "l'erreur diffusée retient les tenseurs du pas (trace gardée)"
        res = {}
        t = threading.Thread(target=lambda: res.update(r=c.post("/v1/chat/completions", json=corps)))
        t.start(); t.join(timeout=30)
        assert not t.is_alive(), "requête B sans réponse en 30 s après l'exception de A (silence kimi 8fx)"
        assert res["r"].status_code == 200, res["r"].text[:300]


def test_echec_ne_publie_pas_les_blocs_d_un_prefill_interrompu(app_et_moteur):
    """Un préfill interrompu n'a pas écrit son KV : ses blocs ne doivent pas entrer au cache de préfixe."""
    _, engine = app_et_moteur
    appels = []
    engine._register_complete_blocks = lambda seq: appels.append(seq)
    from acvram.engine.sampler import SamplingParams
    engine.add_request(list(range(40)), SamplingParams(max_tokens=2), request_id="r-8fx")
    engine._admit()
    assert engine.running, "séquence non admise"
    ids = engine.echec_du_pas()
    assert ids == ["r-8fx"] and not engine.running and appels == []
