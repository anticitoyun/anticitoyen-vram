"""zjf (30/09) : kimi-yals et kimi-tabby importés dans le parc (parc/bin), dépersonnalisés, corrigés.
(1) yals 30/09 : `( cd … && setsid nohup X >> log & )` laissait le sous-shell tenir le stdout/stderr de l'appelant
    jusqu'à la mort de X — `kimi -p` capturé restait « muet 300 s » alors que kimi avait fini en 3 s.
(2) 422 « Model context not initialized » (cache KV refusé) : plus d'attente de 120 s « en arrière-plan », cause nommée
    (Mio refusés), réessai immédiat à contexte moitié, échec nommé au plancher.
(3) « unknown model architecture » (gemma4 sur YALS 3610610, load 200 sans modèle) : échec immédiat nommé.
Hermétique : faux YALS / faux TabbyAPI (HTTP, ports libres, démons détachés comme les vrais), faux proxy, faux kimi,
parc.toml et config kimi dans tmp ; la garde de carte est coupée (ACVRAM_ATTENTE_VRAM=0), aucune carte."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import pytest

PARC = Path(__file__).resolve().parent.parent / "parc"

FAUX_SERVEUR = r'''#!/usr/bin/env python3
# faux YALS / TabbyAPI : lit son port dans config.yml, rend /health, /v1/model, /v1/model/load selon FAUX_MODE
import json, os, re, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
ici = os.path.dirname(os.path.abspath(__file__))
port = int(re.search(r"(?m)^\s*port:\s*(\d+)", open(os.path.join(ici, "config.yml")).read()).group(1))
open(os.path.join(ici, os.path.basename(__file__) + ".pid"), "w").write(str(os.getpid()))
mode, charge = os.environ.get("FAUX_MODE", "ok"), {"id": ""}
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def rendre(self, code, corps):
        b = json.dumps(corps).encode(); self.send_response(code)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health":
            return self.rendre(200, {"status": "healthy"})
        return self.rendre(200, {"id": charge["id"]}) if charge["id"] else self.rendre(503, {"detail": "no model"})
    def do_POST(self):
        d = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        if self.path.endswith("/chat/completions"):
            if mode == "gabarit":        # YALS 3610610 : gabarit refusé → trace Jinja au journal, 200 et rien dedans
                print("Error: Unknown test: sequence", flush=True)
                print("    at Interpreter.evaluateTestExpression (file:///jinja/index.js:1854:13)", flush=True)
                return self.rendre(200, {})
            return self.rendre(200, {"choices": [{"message": {"role": "assistant", "content": "o"}}]})
        ctx = d.get("max_seq_len", 0)
        if mode == "arch":
            print("llama_model_load: error loading model: error loading model architecture: unknown model architecture: 'gemma4'", flush=True)
            return self.rendre(200, {})
        if mode == "kv" or (mode == "kv-puis-ok" and ctx > 65536):
            print("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 17408.00 MiB on device 0: cudaMalloc failed: out of memory", flush=True)
            return self.rendre(422, {"detail": "Model context not initialized"})
        charge["id"] = os.path.basename(d["model_name"])
        self.rendre(200, {})
HTTPServer(("127.0.0.1", port), H).serve_forever()
'''

FAUX_PROXY = r'''import os, socket, time
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", int(os.environ["PROXY_PORT"]))); s.listen()
open(__file__ + ".pid", "w").write(str(os.getpid()))
time.sleep(600)
'''


def _port_libre() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


@pytest.fixture
def poste(tmp_path):
    """Un poste complet dans tmp : parc.toml, config kimi, serveur factice, modèles, faux kimi."""
    kimi_dir = tmp_path / "kimi"; (kimi_dir / "bin").mkdir(parents=True)
    faux_kimi = kimi_dir / "bin" / "kimi"
    faux_kimi.write_text('#!/bin/sh\necho "KIMI $*"\n'); faux_kimi.chmod(0o755)
    modeles = tmp_path / "modeles"; (modeles / "Essai").mkdir(parents=True)
    tsv = tmp_path / "TSV"; tsv.mkdir()
    serveurs = {}
    for moteur in ("yals", "tabby"):
        d = tmp_path / moteur; d.mkdir()
        port, proxy = _port_libre(), _port_libre()
        (d / "config.yml").write_text(f"network:\n  host: 127.0.0.1\n  port: {port}\nmodel:\n  model_dir: {modeles}\n")
        (d / "api_tokens.yml").write_text("api_key: essai-api\nadmin_key: essai-admin\n")
        exe = d / "YALS" if moteur == "yals" else d / "main.py"
        exe.write_text(FAUX_SERVEUR); exe.chmod(0o755)
        (d / "kimi-memoire-proxy.py").write_text(FAUX_PROXY)
        if moteur == "tabby":
            (d / ".venv" / "bin").mkdir(parents=True)
            (d / ".venv" / "bin" / "python").symlink_to(subprocess.run(["which", "python3"], capture_output=True,
                                                                        text=True).stdout.strip())
        serveurs[moteur] = {"dir": d, "port": port, "proxy": proxy}
    (tmp_path / "parc.toml").write_text(
        f'[chemins]\nkimi_dir = "{kimi_dir}"\ntsv_dir = "{tsv}"\nbin = "{PARC / "bin"}"\n'
        f'[moteurs.yals]\npresent = true\nport = {serveurs["yals"]["proxy"]}\nchemin = "{serveurs["yals"]["dir"] / "YALS"}"\n'
        f'[moteurs.tabby]\npresent = true\nport = {serveurs["tabby"]["port"]}\nchemin = "{serveurs["tabby"]["dir"] / "main.py"}"\n')

    def config(ctx):
        (kimi_dir / "config.toml").write_text(
            f'[models.yals-essai]\nprovider = "yals"\nmodel = "Essai/essai.gguf"\nmax_context_size = {ctx}\n'
            f'[models.tabby-essai]\nprovider = "tabby"\nmodel = "Essai"\nmax_context_size = {ctx}\n')

    def lancer(lanceur, alias, mode="ok", ctx=32768, **sup):
        config(ctx)
        env = {**os.environ, "ACVRAM_PARC_CONFIG": str(tmp_path / "parc.toml"), "FAUX_MODE": mode,
               "ACVRAM_ATTENTE_VRAM": "0", "KIMI_TABBY_PORT_PROXY": str(serveurs["tabby"]["proxy"]), **sup}
        env.pop("CUDA_VISIBLE_DEVICES", None)
        t = time.monotonic()
        r = subprocess.run(["bash", str(PARC / "bin" / lanceur), alias, "-p", "question"], capture_output=True,
                           text=True, env=env, timeout=60, stdin=subprocess.DEVNULL)
        return r, time.monotonic() - t

    yield {"lancer": lancer, "serveurs": serveurs}
    for f in tmp_path.rglob("*.pid"):
        try:
            os.kill(int(f.read_text()), signal.SIGKILL)
        except (ValueError, ProcessLookupError):
            pass


def _vivant(fichier_pid: Path) -> bool:
    try:
        os.kill(int(fichier_pid.read_text()), 0)
        return True
    except (OSError, ValueError):
        return False


@pytest.mark.parametrize("lanceur,alias,exe", [("kimi-yals", "yals-essai", "YALS"), ("kimi-tabby", "tabby-essai", "main.py")])
def test_le_lanceur_rend_la_main_quand_kimi_a_fini(poste, lanceur, alias, exe):
    """Cassant : avec l'ancien `( cd && setsid nohup X >> log & )`, subprocess.run attend la mort du faux serveur et
    du proxy (jamais) → TimeoutExpired à 60 s."""
    r, dt = poste["lancer"](lanceur, alias)
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"KIMI " in r.stdout and f"-m {alias} -p question" in r.stdout
    assert dt < 20, f"{lanceur} a rendu la main en {dt:.0f} s"
    d = poste["serveurs"][lanceur.split("-")[1]]["dir"]
    assert _vivant(d / f"{exe}.pid") and _vivant(d / "kimi-memoire-proxy.py.pid")     # démons détachés, vivants


def test_422_kv_echec_immediat_et_nomme(poste):
    r, dt = poste["lancer"]("kimi-yals", "yals-essai", mode="kv", ctx=32768)
    assert r.returncode == 1 and "KIMI" not in r.stdout
    assert ("Cache KV trop grand au contexte 32768 (HTTP 422 : allocating 17408.00 MiB on device 0: "
            "cudaMalloc failed: out of memory)") in r.stderr, r.stderr
    assert dt < 20, f"422 : {dt:.0f} s (l'ancien lanceur attendait 120 s un chargement « en arrière-plan »)"


def test_422_kv_reessai_immediat_a_contexte_moitie(poste):
    r, dt = poste["lancer"]("kimi-yals", "yals-essai", mode="kv-puis-ok", ctx=131072)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Cache KV trop grand au contexte 131072" in r.stderr and "(contexte 65536)" in r.stdout
    assert (poste["serveurs"]["yals"]["dir"] / "ctx-actuel").read_text().strip() == "65536"
    assert dt < 30, f"{dt:.0f} s"


def test_architecture_inconnue_echec_immediat_nomme(poste):
    r, dt = poste["lancer"]("kimi-yals", "yals-essai", mode="arch", ctx=131072)
    assert r.returncode == 1
    assert "Alias non servable par ce YALS : unknown model architecture: 'gemma4' (HTTP 200)" in r.stderr, r.stderr
    assert dt < 20, f"{dt:.0f} s (l'ancien lanceur faisait 3 essais de 2 min)"


def test_aucun_chemin_personnel_ni_cle(poste):
    for nom in ("kimi-yals", "kimi-tabby"):
        src = (PARC / "bin" / nom).read_text()
        assert "/home/" not in src and "$HOME/.kimi-code" not in src and "~/TSV" not in src, nom
        assert "4TO_SATACMR" not in src and "api_key:" not in src.replace("'/^api_key:/", ""), nom


def test_gabarit_refuse_par_yals_echec_immediat_nomme(poste):
    """nemotron 30/09 : chargement réussi, gabarit refusé à chaque requête (200 vide) ; kimi réessayait 9 fois → 300 s.
    Cassant : sans la sonde, le lanceur passe la main à kimi (rc 0, « KIMI » imprimé)."""
    r, dt = poste["lancer"]("kimi-yals", "yals-essai", mode="gabarit")
    assert r.returncode == 1 and "KIMI" not in r.stdout, r.stdout + r.stderr
    assert "Gabarit de chat refusé par YALS : Unknown test: sequence (gabarit du GGUF)" in r.stderr, r.stderr
    assert dt < 20


# ── bascule : parc-installer remplit [moteurs.yals] chemin depuis le poste ─────────────────────────────────────
import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_paquet_parc import _installer, poste as poste_parc  # noqa: E402,F401  (fixture)


def _yals_present(p, tmp_path, lanceur_texte: str | None):
    import tomllib
    p["cfg"].parent.mkdir(parents=True, exist_ok=True)
    p["cfg"].write_text("[moteurs.yals]\npresent = true\nport = 5011\n")
    d = tmp_path / "IA" / "YALS"; d.mkdir(parents=True)
    (d / "YALS").write_text("#!/bin/sh\n"); (d / "YALS").chmod(0o755)
    if lanceur_texte is not None:
        b = p["home"] / ".local" / "bin"; b.mkdir(parents=True, exist_ok=True)
        (b / "kimi-yals").write_text(lanceur_texte.replace("@DOSSIER@", str(d)))
    p["env"]["PARC_PROC"] = str(tmp_path / "proc-vide")          # aucun YALS vivant (le vrai poste en a peut-être un)
    return d, lambda: tomllib.loads(p["cfg"].read_text()).get("moteurs", {}).get("yals", {})


def test_bascule_chemin_yals_repris_du_lanceur_en_place(poste_parc, tmp_path):
    d, lire = _yals_present(poste_parc, tmp_path, '#!/usr/bin/env bash\nYALS_DIR="@DOSSIER@"\nAPI=x\n')
    r = _installer(poste_parc)
    assert r.returncode == 0, r.stdout + r.stderr
    assert lire().get("chemin") == str(d / "YALS") and f"yals : chemin repris du poste → {d / 'YALS'}" in r.stdout + r.stderr


def test_bascule_chemin_yals_introuvable_refus_nomme(poste_parc, tmp_path):
    _, lire = _yals_present(poste_parc, tmp_path, None)
    r = _installer(poste_parc)
    assert r.returncode == 4, r.stdout + r.stderr
    assert "REFUS : [moteurs.yals] présent sans chemin" in r.stdout + r.stderr
    assert not lire().get("chemin")
