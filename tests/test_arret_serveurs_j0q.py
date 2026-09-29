"""Pièces j0q + dlq (poste6, 29/09) — `outils/test-menus-reels.py` : l'arrêt de fin de bras visait le PID mémorisé
au préchargement. (j0q) Un lanceur qui rend 1 sans port laissait vivre le serveur qu'il avait lancé (qwen3 de
l'arbre, 26,6 Gio → OOM de l'alias suivant). (dlq) claude-modele relance le serveur à la fenêtre du TSV : l'ancien
PID est mort, « ok », le nouveau reste (8fx : « sert déjà » au bras suivant). Voulu : arrêter tout serveur APPARU
depuis le début du bras, sur le port ou sous le verrou carte.sh (`.qui`), PID vérifié par sa ligne de commande ;
jamais un serveur présent avant le bras, jamais un service permanent (8faa01465). Hermétique : verrou factice
(TMR_VERROUS), processus `sleep` renommés, port lu par une fonction remplacée, jamais nvidia-smi."""
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_test_menus_reels import SERVEUR, _ecoute, _lignes, _port_libre  # noqa: E402

OUTIL = Path(__file__).resolve().parent.parent / "outils" / "test-menus-reels.py"


@pytest.fixture
def m(tmp_path, monkeypatch):
    monkeypatch.setenv("TMR_VERROUS", str(tmp_path / "carte-*.lock"))
    spec = importlib.util.spec_from_file_location("tmr_j0q", OUTIL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.vram_de = lambda pid: False                      # jamais nvidia-smi
    mod.pid_ecoute = lambda port: None
    mod._qui = tmp_path / "carte-0.lock.qui"
    mod._procs = []
    yield mod
    for p in mod._procs:
        try:
            p.kill()
        except OSError:
            pass


def _faux(m, argv0: str) -> int:
    """Un `sleep` dont la ligne de commande imite un serveur (ou non)."""
    p = subprocess.Popen(["bash", "-c", f'exec -a "{argv0}" sleep 300'])
    m._procs.append(p)
    time.sleep(0.2)
    return p.pid


def _inscrire(m, pid: int, nom: str, type_="service") -> None:
    with m._qui.open("a") as f:
        f.write(f"{pid} {int(time.time())} {nom} {type_}\n")


def _vivant(pid: int) -> bool:
    """Vivant et pas zombie (les faux serveurs sont des enfants du test, tués mais non moissonnés)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"


def test_j0q_serveur_du_verrou_sans_port_est_arrete(m):
    pid = _faux(m, "acvram serve /m --port 8090 --served-name acvram-x-nvfp4")
    _inscrire(m, pid, "acvram-x-nvfp4")
    assert m.arreter(8090, set(), "acvram-x-nvfp4") == "ok"
    assert not _vivant(pid)


def test_dlq_le_serveur_relance_qui_ecoute_est_arrete_pas_l_ancien_pid(m):
    nouveau = _faux(m, "acvram serve /m --port 8090 --served-name acvram-x-nvfp4")
    m.pid_ecoute = lambda port: nouveau if port == 8090 else None
    assert m.arreter(8090, set(), "acvram-x-nvfp4") == "ok"      # avant : rien (l'ancien PID, mort, n'y est pas)
    assert not _vivant(nouveau)


def test_un_serveur_present_avant_le_bras_est_laisse(m):
    ancien = _faux(m, "acvram serve /m --port 8090 --served-name acvram-avant-nvfp4")
    _inscrire(m, ancien, "acvram-avant-nvfp4")
    m.pid_ecoute = lambda port: ancien
    avant = m.serveurs_vivants(8090)
    assert avant == {ancien}
    assert m.arreter(8090, avant, "acvram-x-nvfp4").startswith("ok (rien d'apparu)")
    assert _vivant(ancien)


def test_un_service_permanent_n_est_jamais_arrete(m):
    appoint = _faux(m, "llama-server -m /x.gguf --port 8081 --alias appoint")
    _inscrire(m, appoint, "llamacpp-appoint")
    r = m.arreter(8090, set(), "")
    assert r.startswith("ok (rien d'apparu)") and str(appoint) in r and _vivant(appoint)
    assert not m.est_serveur_du_parc(appoint)


def test_un_pid_qui_n_est_pas_un_serveur_est_laisse_meme_inscrit(m):
    quelconque = _faux(m, "sleep")
    _inscrire(m, quelconque, "faux")
    r = m.arreter(8090, set(), "acvram-x")
    assert r.startswith("ok (rien d'apparu)") and _vivant(quelconque)
    assert not m.est_serveur_du_parc(quelconque)


def test_lignes_de_verrou_mortes_ou_malformees_ignorees(m):
    _inscrire(m, 999999999, "fantome")
    m._qui.write_text(m._qui.read_text() + "n'importe quoi\n42 abc x service\n")
    assert m.services_du_verrou() == []
    assert m.arreter(8090, set(), "") == "ok (rien d'apparu)"


# ---- de bout en bout, faux parc : lanceur rc 1 qui laisse un serveur, client qui relance le serveur ------------------

SERVEUR_TARDIF = SERVEUR.replace("HTTPServer((", "import time; time.sleep(float(sys.argv[3]))\nHTTPServer((")


@pytest.fixture
def parc(tmp_path):
    port = _port_libre()
    b = tmp_path / "bin"; b.mkdir(); tsv = tmp_path / "TSV"; tsv.mkdir(); kimi = tmp_path / "kimi"; kimi.mkdir()
    (tmp_path / "serveur.py").write_text(SERVEUR_TARDIF)
    qui = tmp_path / "carte-0.lock.qui"
    py = sys.executable
    # acvram-rc1-vivant (j0q) : inscrit son serveur au verrou, le lance (écoute 2 s plus tard), rend 1 sans attendre
    # acvram-relance (dlq) : sert normalement ; c'est claude-modele qui relance le serveur (nouveau PID)
    (b / "acvram-serveur").write_text(f'''#!/bin/sh
if [ "$1" = acvram-rc1-vivant ]; then
  setsid {py} {tmp_path}/serveur.py {port} "$1" 2 >/dev/null 2>&1 < /dev/null &
  echo "$! $(date +%s) $1 service" >> {qui}
  echo "acvram n'a pas démarré" >&2; exit 1
fi
setsid {py} {tmp_path}/serveur.py {port} "$1" 0 >/dev/null 2>&1 < /dev/null &
echo "$! $(date +%s) $1 service" >> {qui}
for i in $(seq 1 50); do {py} -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(('127.0.0.1',{port})))" && exit 0; sleep 0.1; done
exit 1
''')
    (b / "kimi-modele").write_text('#!/bin/sh\necho Paris\n')
    (b / "claude-modele").write_text(f'''#!/bin/sh
if [ "$1" = acvram-relance ]; then
  {py} - <<'PY'
import os, re, signal, subprocess, time
r = subprocess.run(["ss", "-tlnpH"], capture_output=True, text=True).stdout
for l in r.splitlines():
    if re.search(r"[:.]{port}\\s", l):
        os.kill(int(re.search(r"pid=(\\d+)", l).group(1)), signal.SIGKILL)
time.sleep(0.5)
PY
  setsid {py} {tmp_path}/serveur.py {port} "$1" 0 >/dev/null 2>&1 < /dev/null &
  echo "$! $(date +%s) $1 service" >> {qui}
  for i in $(seq 1 50); do {py} -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(('127.0.0.1',{port})))" && break; sleep 0.1; done
fi
echo Paris
''')
    for f in b.iterdir():
        f.chmod(0o755)
    (kimi / "config.toml").write_text("".join(f'[models.{a}]\nprovider = "acvram"\nmodel = "{a}"\nmax_context_size = 32768\n\n'
                                              for a in ("acvram-rc1-vivant", "acvram-relance")))
    cfg = tmp_path / "parc.toml"
    cfg.write_text(f'[chemins]\nkimi_dir = "{kimi}"\ntsv_dir = "{tsv}"\nsecrets = "{tmp_path}/secrets.env"\nbin = "{b}"\n\n'
                   f"[moteurs.acvram]\npresent = true\nport = {port}\n")
    env = {**os.environ, "ACVRAM_PARC_CONFIG": str(cfg), "TMR_ETAT": str(tmp_path / "etat"),
           "TMR_RESULTATS": str(tsv / "menus-reels.tsv"), "ACVRAM_SERVEUR_LOG": str(tmp_path / "serveur.log"),
           "TMR_VERROUS": str(tmp_path / "carte-*.lock")}
    yield {"env": env, "port": port, "tsv": tsv / "menus-reels.tsv", "tmp": tmp_path, "qui": qui}
    for l in qui.read_text().splitlines() if qui.exists() else []:      # un test rouge ne laisse rien derrière lui
        pid = int(l.split()[0])
        if "serveur.py" in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace") if Path(f"/proc/{pid}").exists() else False:
            os.kill(pid, 9)


def test_bout_en_bout_rc1_vivant_et_relance_sont_arretes(parc):
    r = subprocess.run([sys.executable, str(OUTIL), "--pour-de-vrai", "--delai-client", "30", "--attente", "30"],
                       env=parc["env"], capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    l = _lignes(parc)
    assert l["acvram-rc1-vivant"][3:5] == ["PANNE", "préchargement"] and l["acvram-rc1-vivant"][11] == "ok", l["acvram-rc1-vivant"]
    assert l["acvram-relance"][3] == "OK" and l["acvram-relance"][11] == "ok", l["acvram-relance"]
    time.sleep(3)                                                     # le serveur tardif aurait écouté ici
    assert not _ecoute(parc["port"]), "un serveur écoute encore (rc 1 sans port, ou relancé pendant le bras)"
    for ligne in parc["qui"].read_text().splitlines():
        assert not Path(f"/proc/{ligne.split()[0]}").exists(), f"serveur encore vivant : {ligne}"
