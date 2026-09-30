"""syy (29/09, après poste5-edz2) : le contrôle « Paris » d'edz juge le CONTENU FINAL (hors raisonnement), avec un plafond
qui laisse finir un modèle à raisonnement (2 048 jetons, 64 avant). TÉMOIN : « Paris » dans le raisonnement seulement →
FAUX (avant : compté OK, le contrôle lisait content ou reasoning_content). Raisonnement non fini → panne nommée."""
import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

OUTIL = Path(__file__).resolve().parent.parent / "outils" / "test-menus-reels.py"
spec = importlib.util.spec_from_file_location("tmr", OUTIL)
tmr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tmr)

# alias -> (content, reasoning_content, finish_reason)
REPONSES = {
    "acvram-raisonne-puis-paris": ("Je pèse les options…\n</think>\n\nParis", None, "stop"),
    "acvram-paris-dans-le-raisonnement": ("<think>Paris, évidemment.</think>Lyon", None, "stop"),     # témoin FAUX
    "acvram-paris-en-reasoning-content": ("", "Paris", "stop"),                                      # témoin FAUX
    "acvram-raisonnement-coupe": ("<think>Voyons, la France, sa capitale est", None, "length"),
    "acvram-direct": ("Paris.", None, "stop"),
    # vLLM 0.29 : raisonnement coupé rendu dans « reasoning » (plus « reasoning_content ») — pas une réponse vide
    "acvram-reasoning-champ-vllm": ("", None, "length", "Voyons, la France"),
}
SERVEUR = r'''import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port, ident, trace = int(sys.argv[1]), sys.argv[2], sys.argv[3]
REP = json.loads(sys.argv[4])
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _j(self, d):
        b = json.dumps(d).encode(); self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._j({"data": [{"id": ident}]})
    def do_POST(self):
        corps = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        open(trace, "a").write(f"{ident} {corps.get('max_tokens')}\n")
        c, r, f, *g = REP[ident]
        msg = {"role": "assistant", "content": c}
        if r is not None: msg["reasoning_content"] = r
        if g: msg["reasoning"] = g[0]
        self._j({"choices": [{"message": msg, "finish_reason": f}]})
HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def _port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


@pytest.fixture
def parc(tmp_path):
    port = _port()
    b = tmp_path / "bin"; b.mkdir(); tsv = tmp_path / "TSV"; tsv.mkdir(); kimi = tmp_path / "kimi"; kimi.mkdir()
    (tmp_path / "serveur.py").write_text(SERVEUR)
    rep = json.dumps(REPONSES).replace("'", "'\\''")
    (b / "acvram-serveur").write_text(f'''#!/bin/sh
setsid {sys.executable} {tmp_path}/serveur.py {port} "$1" {tmp_path}/trace.txt '{rep}' >/dev/null 2>&1 < /dev/null &
for i in $(seq 1 50); do {sys.executable} -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(('127.0.0.1',{port})))" && exit 0; sleep 0.1; done
exit 1
''')
    # clients : raisonnement puis réponse ; « acvram-paris-dans-le-raisonnement » : Paris dans le raisonnement seul
    client = '#!/bin/sh\ncase "$1" in acvram-paris-dans-le-raisonnement) printf "<think>Paris ?</think>\\nLyon\\n" ;; *) printf "réflexion</think>\\n\\nParis\\n" ;; esac\n'
    for nom in ("kimi-modele", "claude-modele"):
        (b / nom).write_text(client)
    for f in b.iterdir():
        f.chmod(0o755)
    (kimi / "config.toml").write_text("".join(f'[models.{a}]\nprovider = "acvram"\nmodel = "{a}"\nmax_context_size = 32768\n\n'
                                              for a in REPONSES))
    cfg = tmp_path / "parc.toml"
    cfg.write_text(f'[chemins]\nkimi_dir = "{kimi}"\ntsv_dir = "{tsv}"\nsecrets = "{tmp_path}/secrets.env"\nbin = "{b}"\n\n'
                   f"[moteurs.acvram]\npresent = true\nport = {port}\n")
    env = {**os.environ, "ACVRAM_PARC_CONFIG": str(cfg), "TMR_ETAT": str(tmp_path / "etat"),
           "TMR_RESULTATS": str(tsv / "menus-reels.tsv"), "ACVRAM_SERVEUR_LOG": str(tmp_path / "serveur.log"),
           "TMR_CARTE": ""}     # pas d attente « carte vide » (249674e39) : sous une prise GPU, elle attendait 120 s par bras
    yield {"env": env, "tsv": tsv / "menus-reels.tsv", "tmp": tmp_path}


def test_contenu_final():
    cf = tmr.contenu_final
    assert cf("<think>Paris</think>Lyon") == "Lyon"
    assert cf("réflexion…\n</think>\n\nParis") == "Paris"          # <think> ouvert dans l invite (gabarit)
    assert cf("<think>pas fini") == ""
    assert cf("Paris.") == "Paris."
    assert cf("a<think>x</think>b<think>y</think>c") == "abc"


def test_juge_le_contenu_final_avec_un_plafond_suffisant(parc):
    r = subprocess.run([sys.executable, str(OUTIL), "--pour-de-vrai", "--delai-client", "30", "--attente", "30"],
                       env=parc["env"], capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    l = {c.split("\t")[0]: c.split("\t") for c in parc["tsv"].read_text().splitlines() if c and not c.startswith("#")}
    # colonnes : alias date moteur verdict etape cause prechargement_s models completion kimi_rc claude_rc arret detail
    assert l["acvram-raisonne-puis-paris"][3] == "OK" and l["acvram-raisonne-puis-paris"][8] == "ok"
    assert l["acvram-direct"][3] == "OK"
    temoin = l["acvram-paris-dans-le-raisonnement"]
    assert temoin[3] == "PANNE" and temoin[8] == "sans Paris" and "complétion+claude+kimi" == temoin[4], temoin
    assert l["acvram-paris-en-reasoning-content"][3] == "PANNE"
    assert l["acvram-paris-en-reasoning-content"][8] == "sans contenu final"
    coupe = l["acvram-raisonnement-coupe"]
    assert coupe[3] == "PANNE" and coupe[8] == "raisonnement non fini" and "2048 jetons (fin=length)" in coupe[5], coupe
    vl = l["acvram-reasoning-champ-vllm"]
    assert vl[3] == "PANNE" and vl[8] == "raisonnement non fini" and "réponse vide" not in vl[5], vl
    plafonds = {int(x.split()[1]) for x in (parc["tmp"] / "trace.txt").read_text().splitlines()}
    assert plafonds == {tmr.JETONS_COMPLETION} and tmr.JETONS_COMPLETION >= 2048, plafonds
