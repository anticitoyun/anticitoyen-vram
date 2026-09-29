"""Pièce ECONNREFUSED (poste6, 29/09, bd 5xw) : deux services edz morts d'un SIGTERM propre pendant la première
requête de Claude Code, tueur inconnu. Journal des arrêts (`acvram/server/arrets.py`) : le lanceur acvram-serveur
signe le serveur qu'il remplace, la console signe `POST /moteurs/arreter`, le serveur signe son propre arrêt.
Hermétique : ACVRAM_JOURNAL_ARRETS dans tmp, faux serveur HTTP sur un port libre, lanceur à sec (ACVRAM_SERVEUR_A_SEC)."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_lanceur_source import LANCEUR, poste  # noqa: E402,F401  (fixture réutilisée)

FAUX = r'''import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        b = json.dumps({"data": [{"id": "autre-modele"}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
'''


def _port_libre():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def test_journal_arret_ecrit_pid_cmdline_et_motif(tmp_path, monkeypatch):
    monkeypatch.setenv("ACVRAM_JOURNAL_ARRETS", str(tmp_path / "j" / "arrets.journal"))
    from acvram.server.arrets import journal_arret
    journal_arret(os.getpid(), "essai\tavec tabulation", signal="SIGTERM")
    l = (tmp_path / "j" / "arrets.journal").read_text(encoding="utf-8").splitlines()
    assert len(l) == 1
    ch = l[0].split("\t")
    assert ch[1] == str(os.getpid()) and ch[2] == "SIGTERM" and "python" in ch[3] and ch[4] == "essai avec tabulation"


def test_le_lanceur_signe_le_serveur_qu_il_remplace(poste, tmp_path):
    port = _port_libre()
    (tmp_path / "faux.py").write_text(FAUX)
    srv = subprocess.Popen([sys.executable, str(tmp_path / "faux.py"), str(port)])
    try:
        for _ in range(50):
            s = socket.socket()
            if s.connect_ex(("127.0.0.1", port)) == 0:
                s.close(); break
            s.close(); time.sleep(0.1)
        src = poste["lanceur"].read_text().replace("PORT=1", f"PORT={port}")
        assert f"PORT={port}" in src
        poste["lanceur"].write_text(src)
        journal = tmp_path / "arrets.journal"
        r = subprocess.run(["bash", str(poste["lanceur"]), "acvram-essai"], capture_output=True, text=True, timeout=60,
                           env={**poste["env"], "ACVRAM_JOURNAL_ARRETS": str(journal), "ACVRAM_SESSION": "essai-econn"})
        assert r.returncode == 0, r.stdout + r.stderr
        l = journal.read_text(encoding="utf-8").splitlines()
        assert len(l) == 1, l
        ch = l[0].split("\t")
        assert ch[1] == str(srv.pid) and ch[2] == "SIGTERM" and "faux.py" in ch[3], ch
        assert "remplacé par acvram-essai (servait autre-modele) session=essai-econn" in ch[4], ch
        assert srv.wait(timeout=10) is not None                               # le faux serveur a bien reçu le SIGTERM
    finally:
        if srv.poll() is None:
            srv.kill()


def test_le_serveur_signe_son_arret_et_la_console_ne_signe_pas_un_404(converted, tmp_path, monkeypatch):
    import torch
    from fastapi.testclient import TestClient
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.server.app import create_app
    from acvram.server.chat import load_tokenizer

    monkeypatch.setenv("ACVRAM_JOURNAL_ARRETS", str(tmp_path / "arrets.journal"))
    vocab = {f"tok{i}": i for i in range(1024)}
    for i, w in enumerate(["<|im_start|>", "<|im_end|>", "hello", "</s>"]):
        vocab[w] = 1000 + i
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="tok0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace(); tok.decoder = decoders.WordPiece(prefix="")
    tok.save(os.path.join(converted, "tokenizer.json"))
    json.dump({"eos_token": "</s>", "chat_template": "{% for m in messages %}{{m['content']}}{% endfor %}"},
              open(os.path.join(converted, "tokenizer_config.json"), "w"))
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu", max_concurrent_seqs=4)
    loaded.plan.kv_planned_seqs = 4
    tokenizer = load_tokenizer(converted)
    engine = Engine(loaded, tokenizer, max_batch_size=4, max_model_len=256)
    with TestClient(create_app(engine, tokenizer, "tiny-econn")) as c:
        r = c.post("/moteurs/arreter", json={"pid": 1})
        assert r.status_code == 404                                           # PID non vu sur une carte : jamais tué
        assert not (tmp_path / "arrets.journal").exists()                     # … et rien de signé
    l = (tmp_path / "arrets.journal").read_text(encoding="utf-8").splitlines()
    assert len(l) == 1 and l[0].split("\t")[1] == str(os.getpid()) and "servait tiny-econn" in l[0], l
