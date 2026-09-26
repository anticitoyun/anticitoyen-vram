"""Pièce 272 : depuis la 268 (étape 2), le gabarit jinja est rendu dans des fils. La construction paresseuse de
l'Environment (`Tokenizer._render_jinja`) publiait `_env` avant ses globales et ses politiques : une requête de la
première rafale pouvait rendre avec `ensure_ascii` actif (JSON échappé) ou tomber, via un AttributeError rattrapé, sur le
repli ChatML — en silence. Cassant : l'écriture des globales est ralentie (0,2 s) ; huit fils rendent en même temps ;
tous doivent rendre EXACTEMENT le rendu en série, gabarit effectif « jinja ». ROUGE sur l'ancien code (vérifié)."""
import threading
import time

import jinja2

from acvram.server.chat import Tokenizer

GABARIT = "{% for m in messages %}{{ m['content'] | tojson }}{% endfor %}"
MESSAGES = [{"role": "user", "content": "déjà vu — ü"}]


class _Lent(dict):
    def __setitem__(self, k, v):
        time.sleep(0.2)
        super().__setitem__(k, v)


def test_cassant_premiere_rafale_de_fils(monkeypatch):
    attendu = Tokenizer(None, {}, GABARIT, "test").apply_chat_template(MESSAGES, False)
    assert "déjà" in attendu                                      # ensure_ascii=False en série

    vrai = jinja2.Environment

    class EnvLent(vrai):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.globals = _Lent(self.globals)
    monkeypatch.setattr(jinja2, "Environment", EnvLent)
    tok = Tokenizer(None, {}, GABARIT, "test")
    rendus, depart = [], threading.Barrier(8)

    def un():
        depart.wait()
        rendus.append(tok.apply_chat_template(MESSAGES, False))
    fils = [threading.Thread(target=un) for _ in range(8)]
    for f in fils:
        f.start()
    for f in fils:
        f.join()
    assert rendus == [attendu] * 8, f"rendus divergents en fils : {set(rendus)}"
    assert tok.gabarit_effectif == "jinja"
