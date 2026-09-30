"""kv31b (poste6, 30/09) — acvram-serveur relit « [acvram] fenêtre qui tient : N jetons » quand le moteur meurt sur
« budget KV insuffisant » : (1) N ≥ ctx_client_min (ou pas de minimum) → relance UNE fois à N, le serveur répond, rc 0,
la commande porte `--max-model-len N` ; (2) N < CTX_CLIENT_MIN → refus nommé avec les deux chiffres, rc 1, une seule
tentative ; (3) une seconde mort à la relance ne reboucle pas (ACVRAM_FENETRE_RELANCEE). À sec : faux carte.sh (lance
la commande en fond, rend son PID), faux acvram (écrit le refus au journal et meurt si la fenêtre dépasse ce qui tient,
sinon sert /v1/models), faux carte_rendue, port libre. Cassure : retirer la relance → (1) rouge (rc 1)."""
import os
import socket
import stat
import subprocess
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
SERVEUR = RACINE / "parc" / "bin" / "acvram-serveur"

FAUX_ACVRAM = r'''#!/bin/bash
# faux acvram : « --version », ou « serve … --max-model-len N » : meurt avec le refus si N > TIENT, sert sinon
[ "$1" = --version ] && { echo "acvram 0.0.0-essai"; exit 0; }
N=""; PORT=8090; a=("$@")
for i in "${!a[@]}"; do [ "${a[$i]}" = --max-model-len ] && N="${a[$((i+1))]}"; [ "${a[$i]}" = --port ] && PORT="${a[$((i+1))]}"; done
echo "$N" >> "$TEST_FENETRES"
if [ "$N" -gt "${TEST_SERT:-$TEST_TIENT}" ]; then
  echo "[acvram] budget KV de cuda:0 borné par la VRAM libre : 15.12 → 10.28 Gio (libre 28.1, poids 5.8, marge 12.0 dont préfill 10.42)"
  echo "[acvram] fenêtre qui tient : $TEST_TIENT jetons (plancher KV d'une séquence + activations de préfill ≤ VRAM libre − poids résidents − marge, par pas de 1 024 ; 0 = aucune)"
  echo "RuntimeError: refus : budget KV insuffisant après 4 tours d'exil — {'cuda:0': 10.28} Gio pour un plancher d'une séquence de $N jetons (4957 Mio manquants) ; fenêtre qui tient : $TEST_TIENT jetons — réduire max_model_len ou forcer l'exil (ACVRAM_EXIL_COUCHES)"
  exit 1
fi
exec python3 -c "
import http.server, json, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        b = json.dumps({'data': [{'id': 'gemma-essai'}]}).encode(); self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
    def log_message(self, *a): pass
http.server.HTTPServer(('127.0.0.1', $PORT), H).serve_forever()"
'''
FAUX_CARTE = '#!/bin/bash\nsetsid nohup "$@" >> "$ACVRAM_SERVICE_LOG" 2>&1 < /dev/null &\necho $!\n'


def _exe(p: Path, corps: str) -> Path:
    p.write_text(corps); p.chmod(p.stat().st_mode | stat.S_IEXEC); return p


def _port_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


def _lancer(tmp_path, tient: int, ctx: int, client_min: str | None, sert: int | None = None):
    home = tmp_path / "home"; tsv = home / "TSV"; tsv.mkdir(parents=True)
    dossier = tmp_path / "gemma-essai"; dossier.mkdir(); (dossier / "acvram_manifest.json").write_text("{}")
    (tsv / "acvram-chemins.tsv").write_text(f"gemma-essai\t{dossier}\t{ctx}\n")
    b = tmp_path / "bin"; b.mkdir()
    _exe(b / "faux-acvram", FAUX_ACVRAM); _exe(b / "carte.sh", FAUX_CARTE); _exe(b / "carte_rendue.py", "import sys; sys.exit(0)\n")
    log = tmp_path / "serveur.log"; fenetres = tmp_path / "fenetres.txt"; port = _port_libre()
    env = {**os.environ, "HOME": str(home), "ACVRAM_PAQUET_BIN": str(b / "faux-acvram"), "ACVRAM_CARTE_SH": str(b / "carte.sh"),
           "PARC_CARTE_RENDUE": str(b / "carte_rendue.py"), "ACVRAM_SERVEUR_LOG": str(log), "PARC_PORT_ACVRAM": str(port),
           "TEST_TIENT": str(tient), "TEST_SERT": str(sert or tient), "TEST_FENETRES": str(fenetres), "ACVRAM_DELAI_DEMARRAGE": "40", "CUDA_VISIBLE_DEVICES": ""}
    env.pop("ACVRAM_ARBRE", None); env.pop("ACVRAM_FENETRE_RELANCEE", None); env.pop("CTX", None)
    if client_min is None:
        env.pop("CTX_CLIENT_MIN", None)
    else:
        env["CTX_CLIENT_MIN"] = client_min
    r = subprocess.run(["bash", str(SERVEUR), "gemma-essai"], capture_output=True, text=True, env=env, timeout=120)
    vues = [int(x) for x in fenetres.read_text().split()] if fenetres.exists() else []
    subprocess.run(["pkill", "-f", f"127.0.0.1', {port}"], capture_output=True)      # le faux serveur en fond
    return r, vues


def test_relance_a_la_fenetre_qui_tient(tmp_path):
    r, vues = _lancer(tmp_path, tient=25600, ctx=32768, client_min=None)
    assert r.returncode == 0, r.stdout[-600:] + r.stderr[-600:]
    assert vues == [32768, 25600], vues
    assert "relance à 25600 jetons (fenêtre qui tient)" in r.stdout and "--max-model-len 25600" in r.stdout


def test_refus_nomme_quand_le_client_exige_plus(tmp_path):
    r, vues = _lancer(tmp_path, tient=25600, ctx=32768, client_min="29120")
    assert r.returncode == 1 and vues == [32768], (r.stdout[-400:], r.stderr[-400:], vues)
    assert "fenêtre qui tient 25600 jetons < ctx≥29120 requis par le client" in r.stderr, r.stderr[-400:]


def test_pas_de_boucle_si_la_relance_meurt_aussi(tmp_path):
    # le moteur annonce 25 600 mais ne sert que ≤ 20 480 : la relance meurt à son tour → pas de 2e relance, refus nommé
    r, vues = _lancer(tmp_path, tient=25600, ctx=32768, client_min=None, sert=20480)
    assert r.returncode == 1 and vues == [32768, 25600], (vues, r.stderr[-300:])
    assert "fenêtre qui tient 25600 jetons" in r.stderr
