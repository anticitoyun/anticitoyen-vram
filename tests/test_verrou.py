"""`/verrou` et le champ `legitime` de `/moteurs` (sage-gui-ajouts-18-09 § 2,
ajout n°1) : le verrou `outils/carte.sh` est la seule verite sur l'etat du
GPU (REGLES § 2), cette route le rend visible a la console. Motif de fichiers
substituable (ACVRAM_VERROU_GLOB) pour ne jamais lire les vrais verrous
`/tmp/acvram-carte-*.lock` du circuit en service pendant les tests."""

import json
import os
import time

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
        yield c


def test_carte_libre_rend_liste_vide_pas_absente(client, tmp_path, monkeypatch):
    """« carte libre ⇒ "libre", jamais vide » (sage-gui-ajouts-18-09) : la
    route repond quand meme, avec une liste sans verrou tenu — pas un champ
    manquant ni une erreur."""
    monkeypatch.setenv("ACVRAM_VERROU_GLOB", str(tmp_path / "aucun-verrou-*.lock"))
    r = client.get("/verrou")
    assert r.status_code == 200
    assert r.json() == {"verrous": []}


def test_verrou_pris_par_temoin_vivant_affiche_qui_et_depuis(client, tmp_path, monkeypatch):
    """« verrou pris par un shell temoin ⇒ affiche < 5 s » : un `.qui` frais,
    PID reellement vivant (nous-meme), est rendu avec pid/nom/type/depuis."""
    monkeypatch.setenv("ACVRAM_VERROU_GLOB", str(tmp_path / "acvram-carte-*.lock"))
    (tmp_path / "acvram-carte-0.lock").touch()
    pris_a = int(time.time()) - 3
    (tmp_path / "acvram-carte-0.lock.qui").write_text(
        f"{os.getpid()} {pris_a} temoin mesure\n")

    r = client.get("/verrou").json()
    assert len(r["verrous"]) == 1
    v = r["verrous"][0]
    assert v["carte"] == 0
    assert v["tenu"] is True
    assert v["pid"] == os.getpid()
    assert v["nom"] == "temoin"
    assert v["type"] == "mesure"
    assert 0 <= v["depuis_secondes"] < 5


def test_verrou_pid_mort_rend_non_tenu(client, tmp_path, monkeypatch):
    """Un `.qui` dont le PID a disparu (kill -9, verrou perime) ne doit pas
    etre affiche comme tenu — sinon un verrou mort bloque indefiniment la
    lecture, exactement ce que `qui_tient()` de carte.sh evite deja cote
    shell (INFO peut survivre a un kill -9)."""
    monkeypatch.setenv("ACVRAM_VERROU_GLOB", str(tmp_path / "acvram-carte-*.lock"))
    (tmp_path / "acvram-carte-1.lock").touch()
    # PID improbable : le plus grand PID Linux usuel est 4194304 (pid_max par
    # defaut), 999999999 n'existera jamais.
    (tmp_path / "acvram-carte-1.lock.qui").write_text("999999999 1 fantome mesure\n")

    r = client.get("/verrou").json()
    assert r["verrous"] == [{"carte": 1, "pid": None, "nom": None,
                              "type": None, "depuis_secondes": None,
                              "tenu": False}]


def test_moteurs_pid_du_verrou_est_legitime(client, tmp_path, monkeypatch):
    """Un PID GPU qui detient le verrou de SA carte est legitime meme sans
    port permanent — ce champ est calcule cote serveur (jamais recalcule
    dans la console, principe ecrit en tete de console.py)."""
    from acvram.server import app as app_mod

    monkeypatch.setenv("ACVRAM_VERROU_GLOB", str(tmp_path / "acvram-carte-*.lock"))
    (tmp_path / "acvram-carte-0.lock").touch()
    moi = os.getpid()
    (tmp_path / "acvram-carte-0.lock.qui").write_text(f"{moi} {int(time.time())} moi mesure\n")

    def faux_run(cmd, **kw):
        class R:
            stdout = ""
        r = R()
        if "--query-compute-apps=pid,used_memory,gpu_uuid" in cmd[1]:
            r.stdout = f"{moi}, 1024, uuid-0\n"
        elif "--query-gpu=index,uuid" in cmd[1]:
            r.stdout = "0, uuid-0\n"
        elif cmd[0] == "ss":
            r.stdout = "State  Recv-Q Send-Q Local Address:Port\n"   # aucun port ecoute
        return r

    monkeypatch.setattr(app_mod.subprocess, "run", faux_run)

    r = client.get("/moteurs").json()
    moteur = next(m for m in r["moteurs"] if m["pid"] == moi)
    assert moteur["permanent"] is False       # aucun port permanent
    assert moteur["legitime"] is True         # mais detient le verrou de sa carte


def test_moteurs_pid_intrus_sans_verrou_ni_port_permanent(client, tmp_path, monkeypatch):
    """« charge-gpu.py sans verrou ⇒ "intrus" rouge » : un PID GPU present,
    sans verrou pose ET sans port permanent, n'est pas legitime."""
    from acvram.server import app as app_mod

    monkeypatch.setenv("ACVRAM_VERROU_GLOB", str(tmp_path / "aucun-verrou-*.lock"))
    moi = os.getpid()

    def faux_run(cmd, **kw):
        class R:
            stdout = ""
        r = R()
        if "--query-compute-apps=pid,used_memory,gpu_uuid" in cmd[1]:
            r.stdout = f"{moi}, 2048, uuid-0\n"
        elif "--query-gpu=index,uuid" in cmd[1]:
            r.stdout = "0, uuid-0\n"
        elif cmd[0] == "ss":
            r.stdout = "State  Recv-Q Send-Q Local Address:Port\n"
        return r

    monkeypatch.setattr(app_mod.subprocess, "run", faux_run)

    r = client.get("/moteurs").json()
    moteur = next(m for m in r["moteurs"] if m["pid"] == moi)
    assert moteur["permanent"] is False
    assert moteur["legitime"] is False        # ni port permanent, ni verrou
