"""evp (poste2, 02/10, ordre chef) : diagnostic d'poste1 (poste1-evp-parc 795f858b5) —
`servies_max_dans_l_ordre` valait 0 pour TOUT moteur dans `outils/gpu/mesure/duel-moteurs.py`,
pas seulement acvram. Cause : le relevé `/metrics` (`guet`) n'était échantillonné QUE pendant
le tour `max_tokens=1` (TTFT), trop court pour qu'un seul pas de décodage mette à jour
`running` côté serveur (`runner.py:2043`, fin de pas). Correctif : le relevé se fait pendant
le tour `max_tokens=N` (décodage, plusieurs pas, assez long pour que la concurrence réellement
servie se voie).

Faux serveur HTTP : `running` ne monte QUE pendant une requête `max_tokens` élevé (simule un
décodage multi-pas) ; une requête `max_tokens=1` (TTFT) répond trop vite pour y contribuer,
exactement le mécanisme réel."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ICI = Path(__file__).resolve().parent.parent
DUEL = ICI / "outils" / "gpu" / "mesure" / "duel-moteurs.py"
PY = "/usr/bin/python3"

CONC = 4
DUREE_DECODAGE_S = 0.25  # assez long pour qu'un pas de /metrics le voie, court pour le test


class _FauxServeurConcurrence(BaseHTTPRequestHandler):
    """`running` = nombre de requêtes DÉCODAGE (max_tokens > 1) en vol. Une requête TTFT
    (max_tokens == 1) répond immédiatement, sans jamais incrémenter — comme un vrai moteur
    dont un seul pas ne laisse pas le temps à `/metrics` de voir quoi que ce soit."""

    verrou = threading.Lock()
    en_vol = 0

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            self._json(200, {"data": [{"id": "faux-modele-concurrence"}]})
        elif self.path == "/metrics":
            with self.verrou:
                running = self.en_vol
            self._json(200, {"engine": {"running": running, "waiting": 0,
                                        "prefill_seconds": 0.0, "kv_blocks_free": 100,
                                        "cached_prompt_tokens": 0}})
        else:
            self._json(404, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        corps = json.loads(self.rfile.read(n))
        maxtok = corps.get("max_tokens", 1)
        if maxtok > 1:
            with self.verrou:
                type(self).en_vol += 1
            time.sleep(DUREE_DECODAGE_S)
            with self.verrou:
                type(self).en_vol -= 1
            texte = "x " * maxtok
        else:
            texte = "x"
        self._json(200, {"choices": [{"message": {"content": texte}, "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 10, "completion_tokens": maxtok}})

    def _json(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def faux_serveur():
    _FauxServeurConcurrence.en_vol = 0
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FauxServeurConcurrence)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield port
    srv.shutdown()
    t.join(timeout=5)


def test_servies_max_releve_pendant_le_decodage_pas_le_ttft(faux_serveur):
    """CASSE sur l'ancien code (relevé pendant le tour max_tokens=1 seul) : avec ce faux
    serveur, l'ancien relevé donnerait `servies_max_dans_l_ordre` = [0, 0] (aucune requête
    TTFT ne fait jamais monter `running`). Le correctif doit voir CONC servies pendant le
    tour de décodage."""
    r = __import__("subprocess").run(
        [PY, str(DUEL), f"http://127.0.0.1:{faux_serveur}/v1/chat/completions", "x",
         "faux-modele-concurrence", "2", str(CONC)],
        capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout.strip().splitlines()[-1])
    assert d["servies_max_dans_l_ordre"], "aucun relevé — le correctif n'échantillonne rien"
    assert max(d["servies_max_dans_l_ordre"]) == CONC, (
        f"servies_max={d['servies_max_dans_l_ordre']} — devrait voir {CONC} servies pendant "
        "le tour de décodage (régression : relevé repassé au tour TTFT ?)"
    )
