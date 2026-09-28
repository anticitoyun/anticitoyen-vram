"""outils/test-menus-reels.py à sec (bd edz) : faux parc (parc.toml, TSV, config kimi), faux lanceurs qui démarrent un
VRAI petit serveur HTTP sur le port du moteur, faux clients kimi/claude. Contrôles : une ligne par alias avec l'étape et
la cause de chaque panne ; serveur arrêté par son PID (plus rien n'écoute) ; reprise sans rejouer ; mode liste sans écrire.
Fautes construites : un lanceur qui échoue, un serveur qui sert un autre id, un client claude qui répond vide, un client kimi qui répond rc 0 sans « Paris » (sortie dégénérée du GLM, 28/09)."""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

OUTIL = Path(__file__).resolve().parent.parent / "outils" / "test-menus-reels.py"

SERVEUR = r'''import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port, ident = int(sys.argv[1]), sys.argv[2]
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _j(self, d):
        b = json.dumps(d).encode(); self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._j({"data": [{"id": ident}]})
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"])); self._j({"choices": [{"message": {"content": "Paris"}}]})
HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def _port_libre():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _ecoute(port):
    s = socket.socket()
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


@pytest.fixture
def parc(tmp_path):
    port = _port_libre()
    b = tmp_path / "bin"; b.mkdir(); tsv = tmp_path / "TSV"; tsv.mkdir(); kimi = tmp_path / "kimi"; kimi.mkdir()
    (tmp_path / "serveur.py").write_text(SERVEUR)
    # acvram-serveur factice : « acvram-panne » échoue ; « acvram-autre » sert un autre id ; sinon sert l'alias
    (b / "acvram-serveur").write_text(f'''#!/bin/sh
[ "$1" = acvram-panne ] && {{ echo "RuntimeError: cause au journal serveur" >> "$ACVRAM_SERVEUR_LOG"; echo "OOM simulé au chargement" >&2; exit 1; }}
id="$1"; [ "$1" = acvram-autre ] && id=un-autre-modele
setsid {sys.executable} {tmp_path}/serveur.py {port} "$id" >/dev/null 2>&1 < /dev/null &
for i in $(seq 1 50); do {sys.executable} -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(('127.0.0.1',{port})))" && exit 0; sleep 0.1; done
exit 1
''')
    (b / "kimi-modele").write_text('#!/bin/sh\n[ "$1" = acvram-degenere ] && { printf "|~|\\n|~|\\n|~|\\n"; exit 0; }\necho Paris\n')
    (b / "claude-modele").write_text('#!/bin/sh\n[ "$1" = acvram-muet ] && exit 0\necho Paris\n')
    for f in b.iterdir():
        f.chmod(0o755)
    (kimi / "config.toml").write_text("".join(f'[models.{a}]\nprovider = "acvram"\nmodel = "{a}"\nmax_context_size = 32768\n\n'
                                              for a in ("acvram-bon", "acvram-panne", "acvram-autre", "acvram-muet", "acvram-degenere")))
    cfg = tmp_path / "parc.toml"
    cfg.write_text(f'[chemins]\nkimi_dir = "{kimi}"\ntsv_dir = "{tsv}"\nsecrets = "{tmp_path}/secrets.env"\nbin = "{b}"\n\n'
                   f"[moteurs.acvram]\npresent = true\nport = {port}\n")
    env = {**os.environ, "ACVRAM_PARC_CONFIG": str(cfg), "TMR_ETAT": str(tmp_path / "etat"),
           "TMR_RESULTATS": str(tsv / "menus-reels.tsv"),
           "ACVRAM_SERVEUR_LOG": str(tmp_path / "serveur.log")}
    yield {"env": env, "port": port, "tsv": tsv / "menus-reels.tsv", "tmp": tmp_path}
    # un test rouge ne laisse pas de faux serveur derrière lui (orphelins du 28/09 : arrêt sauté, port lu à None)
    r = subprocess.run(["ss", "-tlnpH"], capture_output=True, text=True)
    for l in r.stdout.splitlines():
        if f":{port} " in l and "pid=" in l:
            pid = int(l.split("pid=")[1].split(",")[0])
            if "serveur.py" in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace"):
                os.kill(pid, 9)


def _lancer(parc, *args):
    return subprocess.run([sys.executable, str(OUTIL), *args], env=parc["env"], capture_output=True, text=True, timeout=180)


def _lignes(parc):
    return {l.split("\t")[0]: l.split("\t") for l in parc["tsv"].read_text().splitlines() if l and not l.startswith("#")}


def test_liste_sans_rien_lancer(parc):
    r = _lancer(parc)
    assert r.returncode == 0 and "5 alias au menu" in r.stdout and "rien lancé" in r.stdout, r.stdout + r.stderr
    assert not parc["tsv"].exists() and not _ecoute(parc["port"])


def test_une_ligne_par_alias_cause_de_chaque_panne_et_arret(parc):
    r = _lancer(parc, "--pour-de-vrai", "--delai-client", "30", "--attente", "30")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    l = _lignes(parc)
    assert set(l) == {"acvram-bon", "acvram-panne", "acvram-autre", "acvram-muet", "acvram-degenere"}
    # colonnes : alias date moteur verdict etape cause prechargement_s models completion kimi_rc claude_rc arret detail
    assert l["acvram-bon"][3] == "OK" and l["acvram-bon"][7:11] == ["ok", "ok", "0", "0"] and l["acvram-bon"][11] == "ok"
    assert l["acvram-panne"][3:5] == ["PANNE", "préchargement"] and "OOM simulé" in l["acvram-panne"][5]
    assert "cause au journal serveur" in (parc["tmp"] / "etat" / "journaux" / "acvram-panne.log").read_text()
    assert l["acvram-autre"][3:5] == ["PANNE", "/v1/models"] and "un-autre-modele" in l["acvram-autre"][5]
    assert l["acvram-muet"][3:5] == ["PANNE", "claude"] and "vide" in l["acvram-muet"][5] and l["acvram-muet"][9] == "0"  # kimi joué quand même
    assert l["acvram-degenere"][3:5] == ["PANNE", "kimi"] and "sans « Paris »" in l["acvram-degenere"][5], l["acvram-degenere"]
    assert all(v[11].startswith("ok") for v in l.values()), l                      # chaque serveur arrêté par son PID
    time.sleep(0.3)
    assert not _ecoute(parc["port"]), "un serveur écoute encore"
    r2 = _lancer(parc, "--pour-de-vrai")                                             # reprise : rien à rejouer
    assert "ce passage : 0" in r2.stdout and len(_lignes(parc)) == 5


def test_rejuger_passe_en_panne_un_ok_sans_paris(parc):
    """Lignes OK écrites avant le contrôle : --rejuger relit les journaux ; seule la réponse sans « Paris » tombe."""
    j = parc["tmp"] / "etat" / "journaux"; j.mkdir(parents=True)
    ok = "\t".join(["", "28/09 16:00", "acvram", "OK", "", "", "5", "ok", "ok", "0", "0", "ok", ""])
    parc["tsv"].write_text("# en-tête\n" + "acvram-bon" + ok + "\n" + "acvram-degenere" + ok + "\n")
    (j / "acvram-bon.log").write_text("x\n--- claude-modele rc=0\nParis\n\n--- kimi-modele rc=0\nparis.\n")
    (j / "acvram-degenere.log").write_text("x\n--- claude-modele rc=0\nParis\n\n--- kimi-modele rc=0\n``|\n``|\n")
    r = _lancer(parc, "--rejuger")
    assert r.returncode == 0 and "1 ligne(s)" in r.stdout, r.stdout + r.stderr
    l = _lignes(parc)
    assert l["acvram-bon"][3] == "OK" and l["acvram-degenere"][3:5] == ["PANNE", "kimi"], l
    assert parc["tsv"].with_name("menus-reels.tsv.avant-rejuger").exists() and not _ecoute(parc["port"])
