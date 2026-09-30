"""kwh (30/09, edz 00:48 : 10 OOM en 3 min) : acvram-serveur démarrait dans la seconde où le serveur précédent était
arrêté, avant que la carte ait rendu sa VRAM. Le lanceur attend désormais la carte rendue (aucun PID de calcul en
sortie, VRAM sans processus ≤ seuil), délai borné, refus nommé ; le serveur remplacé est attendu mort (SIGKILL signé).
Hermétique : faux nvidia-smi dans PATH qui rejoue une suite d'instants ; ACVRAM_EXEC relève l'instant du lancement.
Chaque test casse sur le lanceur d'avant (lancement sur VRAM orpheline, garde qui refuse, « sleep 4 » sans SIGKILL)."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_lanceur_source import poste  # noqa: E402,F401  (fixture réutilisée)
from test_arrets_journal_econn import _port_libre  # noqa: E402

TOTAL = 32607

FAUX_SMI = r'''#!/usr/bin/env python3
import json, sys
f = sys.argv[0] + ".json"
e = json.load(open(f))
q = next(a for a in sys.argv if a.startswith("--query-"))
if q == "--query-gpu=memory.used":
    i = min(e["n"], len(e["instants"]) - 1); e["n"] += 1
    json.dump(e, open(f, "w"))
else:
    i = min(max(e["n"] - 1, 0), len(e["instants"]) - 1)
inst = e["instants"][i]
if q.startswith("--query-compute-apps"):
    print("\n".join(f"{p}, {m}" for p, m in inst["apps"]))
elif q == "--query-gpu=memory.free":
    print(%d - inst["used"])
else:
    print(inst["used"])
''' % TOTAL

RELEVE = r'''#!/usr/bin/env python3
import json, os, sys
e = json.load(open(os.environ["FAUX_SMI_ETAT"]))
inst = e["instants"][min(max(e["n"] - 1, 0), len(e["instants"]) - 1)]
json.dump({"orphelins": inst["used"] - sum(m for _, m in inst["apps"]), "apps": inst["apps"]}, open(os.environ["RELEVE"], "w"))
'''


def _pid_mort() -> int:
    p = subprocess.Popen(["true"]); p.wait()
    return p.pid


@pytest.fixture
def carte(poste, tmp_path):
    b = tmp_path / "bin"; b.mkdir()
    (b / "nvidia-smi").write_text(FAUX_SMI); (b / "nvidia-smi").chmod(0o755)
    (tmp_path / "releve.py").write_text(RELEVE); (tmp_path / "releve.py").chmod(0o755)
    etat = b / "nvidia-smi.json"

    def lancer(instants, lancement=True, **sup):
        etat.write_text(json.dumps({"n": 0, "instants": instants}))
        env = {**poste["env"], "PATH": f"{b}:{os.environ['PATH']}", "FAUX_SMI_ETAT": str(etat),
               "RELEVE": str(tmp_path / "releve.json"), "ACVRAM_ATTENTE_VRAM": "5", "ACVRAM_ATTENTE_PAS": "0.05",
               "ACVRAM_JOURNAL_ARRETS": str(tmp_path / "arrets.journal"), **sup}
        if lancement:
            env.update(ACVRAM_SERVEUR_A_SEC="0", ACVRAM_EXEC=str(tmp_path / "releve.py"))
        return subprocess.run(["bash", str(poste["lanceur"]), "acvram-essai"], capture_output=True, text=True,
                              env=env, timeout=60)

    def releve():
        f = tmp_path / "releve.json"
        return json.loads(f.read_text()) if f.exists() else None

    return {"lancer": lancer, "releve": releve, "etat": etat, "poste": poste}


def test_vram_orpheline_attendue_avant_lancement(carte):
    r = carte["lancer"]([{"used": 28000, "apps": []}] * 3 + [{"used": 20, "apps": []}])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "acvram : carte rendue en" in r.stdout
    assert carte["releve"]()["orphelins"] == 20          # lancé sur la carte rendue, pas sur les 28 000 Mio du mort


def test_pid_en_sortie_attendu(carte):
    mort = _pid_mort()
    r = carte["lancer"]([{"used": 15000, "apps": [[mort, 15000]]}] * 3 + [{"used": 20, "apps": []}])
    assert r.returncode == 0, r.stdout + r.stderr
    assert carte["releve"]()["apps"] == []               # critère des orphelins nul ici : seul le PID mort fait attendre


def test_processus_vivant_ne_fait_pas_attendre(carte):
    r = carte["lancer"]([{"used": 20020, "apps": [[os.getpid(), 20000]]}], ACVRAM_ATTENTE_VRAM="0.5")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "carte rendue" not in r.stdout and carte["releve"]()["apps"] == [[os.getpid(), 20000]]


def test_delai_depasse_refus_nomme(carte):
    mort = _pid_mort()
    r = carte["lancer"]([{"used": 28000, "apps": [[mort, 4000]]}], ACVRAM_ATTENTE_VRAM="0.3")
    assert r.returncode == 1
    assert "carte non rendue en 0 s" in r.stderr and f"PID [{mort}] en sortie" in r.stderr
    assert "24000 Mio sans processus" in r.stderr and carte["releve"]() is None


def test_garde_vram_lit_la_carte_rendue(carte):
    # KV fp16 à 32 768 : 2 × 32 × 8 × 128 × 32 768 × 2 o = 4 Gio ; libre 2 607 Mio pendant la mort, 32 587 après
    cfg = {"num_hidden_layers": 32, "num_key_value_heads": 8, "num_attention_heads": 32, "hidden_size": 4096}
    modele = Path(carte["poste"]["env"]["HOME"]).parent / "Modele-nvfp4"
    (modele / "config.json").write_text(json.dumps(cfg))
    r = carte["lancer"]([{"used": 30000, "apps": []}] * 3 + [{"used": 20, "apps": []}])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "KV estimé" not in r.stderr


def test_serveur_remplace_sourd_au_sigterm_tue_et_signe(carte, tmp_path):
    port = _port_libre()
    (tmp_path / "sourd.py").write_text(
        "import signal, socket, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('127.0.0.1', {port}))\n"
        "s.listen(); time.sleep(120)\n")
    srv = subprocess.Popen([sys.executable, str(tmp_path / "sourd.py")])
    try:
        for _ in range(50):
            if subprocess.run(["ss", "-tln"], capture_output=True, text=True).stdout.count(f":{port} "):
                break
            time.sleep(0.1)
        lanceur = carte["poste"]["lanceur"]
        lanceur.write_text(lanceur.read_text().replace("PORT=1 ", f"PORT={port} "))
        r = carte["lancer"]([{"used": 20, "apps": []}], lancement=False, ACVRAM_DELAI_TERM="1")
        assert r.returncode == 0, r.stdout + r.stderr
        assert srv.wait(timeout=10) == -signal.SIGKILL
        j = (tmp_path / "arrets.journal").read_text(encoding="utf-8").splitlines()
        assert [l.split("\t")[2] for l in j] == ["SIGTERM", "SIGKILL"], j
        assert "SIGTERM sans effet en 1 s" in j[1]
    finally:
        if srv.poll() is None:
            srv.kill()
