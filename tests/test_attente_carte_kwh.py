"""kwh (30/09, edz 00:48 : 10 OOM en 3 min) : un lanceur démarrait dans la seconde où le serveur précédent était
arrêté, avant que la carte ait rendu sa VRAM. Garde partagée `parc/lib/carte_rendue.py`, appelée par acvram-serveur,
vllm-serveur et llamacpp-serveur : aucun PID de calcul en sortie ni de la session d'un remplacé, VRAM sans processus
≤ seuil, sur CHAQUE carte servie ; délai borné, refus nommé ; ne tue rien (l'appoint 8081 de la carte 1 vit).
Hermétique : faux nvidia-smi dans PATH qui rejoue une suite d'instants par carte ; ACVRAM_EXEC relève l'instant du
lancement. Les tests de défaut cassent sur les lanceurs d'avant (lancement sur VRAM orpheline, garde qui refuse,
« sleep 4 » sans SIGKILL, seuil fixe 24 000 Mio aveugle à la carte 1)."""
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

PARC = Path(__file__).resolve().parent.parent / "parc"
GARDE = PARC / "lib" / "carte_rendue.py"
TOTAL = 32607

# instant = {"<carte>": {"used": Mio, "apps": [[pid, Mio], ...]}} ; le temps avance à chaque memory.used de la carte 0
FAUX_SMI = r'''#!/usr/bin/env python3
import json, sys
f = sys.argv[0] + ".json"
e = json.load(open(f))
carte = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--id=")), "0")
q = next(a for a in sys.argv if a.startswith("--query-"))
if q == "--query-gpu=memory.used" and carte == "0":
    i = min(e["n"], len(e["instants"]) - 1); e["n"] += 1
    json.dump(e, open(f, "w"))
else:
    i = min(max(e["n"] - 1, 0), len(e["instants"]) - 1)
inst = e["instants"][i].get(carte, {"used": 20, "apps": []})
if q.startswith("--query-compute-apps"):
    print("\n".join(f"{p}, {m}" for p, m in inst["apps"]))
elif q == "--query-gpu=memory.free":
    print(%d - inst["used"])
else:
    print(inst["used"])
''' % TOTAL

RELEVE = r'''#!/usr/bin/env python3
import json, os
e = json.load(open(os.environ["FAUX_SMI_ETAT"]))
inst = e["instants"][min(max(e["n"] - 1, 0), len(e["instants"]) - 1)]["0"]
json.dump({"orphelins": inst["used"] - sum(m for _, m in inst["apps"]), "apps": inst["apps"]}, open(os.environ["RELEVE"], "w"))
'''


def c0(used, apps=()):
    return {"0": {"used": used, "apps": [list(a) for a in apps]}}


def _pid_mort() -> int:
    p = subprocess.Popen(["true"]); p.wait()
    return p.pid


@pytest.fixture
def smi(tmp_path):
    b = tmp_path / "bin"; b.mkdir()
    (b / "nvidia-smi").write_text(FAUX_SMI); (b / "nvidia-smi").chmod(0o755)
    etat = b / "nvidia-smi.json"

    def env(instants, **sup):
        etat.write_text(json.dumps({"n": 0, "instants": instants}))
        return {**os.environ, "PATH": f"{b}:{os.environ['PATH']}", "FAUX_SMI_ETAT": str(etat),
                "ACVRAM_ATTENTE_VRAM": "5", "ACVRAM_ATTENTE_PAS": "0.05", "PARC_CARTE_RENDUE": str(GARDE), **sup}
    return env


def garde(env, *args):
    return subprocess.run([sys.executable, str(GARDE), "--nom", "essai", *args], capture_output=True, text=True,
                          env=env, timeout=60)


# ── la garde seule ──────────────────────────────────────────────────────────────────────────────────────────────

def test_session_du_remplace_attendue_apres_la_mort_du_chef(smi):
    """vLLM : serveur.pid = chef de session (setsid de carte.sh), la carte est tenue par un enfant (EngineCore) qui
    survit un moment au chef ; vivant et pas en sortie, seul son sid le rattache au remplacé."""
    chef = subprocess.Popen(["sh", "-c", "sleep 30 & echo $!; wait"], stdout=subprocess.PIPE, text=True,
                            start_new_session=True)
    enfant = int(chef.stdout.readline())
    try:
        chef.kill(); chef.wait()
        env = smi([c0(9000, [(enfant, 9000)])] * 3 + [c0(20)])
        r = garde(env, "--cartes", "0", "--remplace", f"{chef.pid}")
        assert r.returncode == 0 and "essai : carte rendue en" in r.stdout, r.stdout + r.stderr
        r = garde(smi([c0(9000, [(enfant, 9000)])], ACVRAM_ATTENTE_VRAM="0.3"), "--cartes", "0",
                  "--remplace", f"123456789 {chef.pid}")
        assert r.returncode == 1 and f"carte 0 : PID [{enfant}] en sortie" in r.stderr, r.stdout + r.stderr
    finally:
        os.kill(enfant, signal.SIGKILL)


def test_pid_perime_du_fichier_pid_ne_fait_pas_attendre(smi):
    """serveur.pid périmé, réattribué à un processus vivant qui ne tient pas la carte : pas d'attente."""
    r = garde(smi([c0(20)], ACVRAM_ATTENTE_VRAM="0.3"), "--cartes", "0", "--remplace", str(os.getpid()))
    assert r.returncode == 0 and r.stdout == "", r.stdout + r.stderr


def test_chaque_carte_servie_est_attendue_et_l_appoint_vivant_ne_gene_pas(smi):
    appoint = [os.getpid(), 5606]                                         # llama-server 8081 sur la 3080 Ti : vivant
    orph1 = {"0": {"used": 20, "apps": []}, "1": {"used": 5606 + 8000, "apps": [appoint]}}
    rendu = {"0": {"used": 20, "apps": []}, "1": {"used": 5606 + 20, "apps": [appoint]}}
    r = garde(smi([orph1] * 3 + [rendu]), "--cartes", "0,1")
    assert r.returncode == 0 and "carte rendue en" in r.stdout, r.stdout + r.stderr
    r = garde(smi([orph1], ACVRAM_ATTENTE_VRAM="0.3"), "--cartes", "0,1")
    assert r.returncode == 1 and "carte 1 : 8000 Mio sans processus" in r.stderr and "carte 0" not in r.stderr
    r = garde(smi([orph1], ACVRAM_ATTENTE_VRAM="0.3"), "--cartes", "0")        # la 5090 seule : la 1 ne regarde pas
    assert r.returncode == 0 and r.stdout == "", r.stdout + r.stderr


def test_nvidia_smi_muet_ne_bloque_pas(tmp_path):
    b = tmp_path / "bin"; b.mkdir(); (b / "nvidia-smi").write_text("#!/bin/sh\nexit 9\n"); (b / "nvidia-smi").chmod(0o755)
    r = garde({**os.environ, "PATH": f"{b}:{os.environ['PATH']}"}, "--cartes", "0")
    assert r.returncode == 0 and "nvidia-smi muet" in r.stdout


# ── acvram-serveur ──────────────────────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def carte(poste, smi, tmp_path):
    (tmp_path / "releve.py").write_text(RELEVE); (tmp_path / "releve.py").chmod(0o755)

    def lancer(instants, lancement=True, **sup):
        s = smi(instants)                                  # HOME, paquet et TSV factices de la fixture poste gardés
        env = {**poste["env"], **{k: s[k] for k in ("PATH", "FAUX_SMI_ETAT", "ACVRAM_ATTENTE_VRAM", "ACVRAM_ATTENTE_PAS",
                                                    "PARC_CARTE_RENDUE")}, "RELEVE": str(tmp_path / "releve.json"),
               "ACVRAM_JOURNAL_ARRETS": str(tmp_path / "arrets.journal"), **sup}
        if lancement:
            env.update(ACVRAM_SERVEUR_A_SEC="0", ACVRAM_EXEC=str(tmp_path / "releve.py"))
        return subprocess.run(["bash", str(poste["lanceur"]), "acvram-essai"], capture_output=True, text=True,
                              env=env, timeout=60)

    def releve():
        f = tmp_path / "releve.json"
        return json.loads(f.read_text()) if f.exists() else None

    return {"lancer": lancer, "releve": releve, "poste": poste}


def test_vram_orpheline_attendue_avant_lancement(carte):
    r = carte["lancer"]([c0(28000)] * 3 + [c0(20)])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "acvram : carte rendue en" in r.stdout
    assert carte["releve"]()["orphelins"] == 20          # lancé sur la carte rendue, pas sur les 28 000 Mio du mort


def test_pid_en_sortie_attendu(carte):
    mort = _pid_mort()
    r = carte["lancer"]([c0(15000, [(mort, 15000)])] * 3 + [c0(20)])
    assert r.returncode == 0, r.stdout + r.stderr
    assert carte["releve"]()["apps"] == []               # critère des orphelins nul ici : seul le PID mort fait attendre


def test_processus_vivant_ne_fait_pas_attendre(carte):
    r = carte["lancer"]([c0(20020, [(os.getpid(), 20000)])], ACVRAM_ATTENTE_VRAM="0.5")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "carte rendue" not in r.stdout and carte["releve"]()["apps"] == [[os.getpid(), 20000]]


def test_delai_depasse_refus_nomme(carte):
    mort = _pid_mort()
    r = carte["lancer"]([c0(28000, [(mort, 4000)])], ACVRAM_ATTENTE_VRAM="0.3")
    assert r.returncode == 1
    assert "carte non rendue en 0 s" in r.stderr and f"PID [{mort}] en sortie" in r.stderr
    assert "24000 Mio sans processus" in r.stderr and carte["releve"]() is None


def test_garde_vram_lit_la_carte_rendue(carte):
    # KV fp16 à 32 768 : 2 × 32 × 8 × 128 × 32 768 × 2 o = 4 Gio ; libre 2 607 Mio pendant la mort, 32 587 après
    cfg = {"num_hidden_layers": 32, "num_key_value_heads": 8, "num_attention_heads": 32, "hidden_size": 4096}
    modele = Path(carte["poste"]["env"]["HOME"]).parent / "Modele-nvfp4"
    (modele / "config.json").write_text(json.dumps(cfg))
    r = carte["lancer"]([c0(30000)] * 3 + [c0(20)])
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
        r = carte["lancer"]([c0(20)], lancement=False, ACVRAM_DELAI_TERM="1")
        assert r.returncode == 0, r.stdout + r.stderr
        assert srv.wait(timeout=10) == -signal.SIGKILL
        j = (tmp_path / "arrets.journal").read_text(encoding="utf-8").splitlines()
        assert [l.split("\t")[2] for l in j] == ["SIGTERM", "SIGKILL"], j
        assert "SIGTERM sans effet en 1 s" in j[1]
    finally:
        if srv.poll() is None:
            srv.kill()


# ── vllm-serveur et llamacpp-serveur : on joue leur bloc (les scripts entiers tuent les ports 8000/8080) ──────────

def _bloc(nom: str, debut: str) -> str:
    src = (PARC / "bin" / nom).read_text()
    assert "-gt 24000" not in src, f"{nom} : l'ancien seuil fixe de mémoire libre est revenu"
    d = src.index(debut)
    f = src.index("démarrage sans attendre la carte\"\nfi\n", d) + len("démarrage sans attendre la carte\"\nfi\n")
    return src[d:f]


def _jouer(bloc: str, env: dict, avant: str = "") -> subprocess.CompletedProcess:
    # $0 = un lanceur de parc/bin : la garde se résout par ../lib comme dans le paquet (PARC_CARTE_RENDUE retiré)
    env = {k: v for k, v in env.items() if k != "PARC_CARTE_RENDUE"}
    return subprocess.run(["bash", "-c", f"set -uo pipefail; err() {{ echo \"$*\" >&2; }}; c_d=; c_0=\n{avant}\n{bloc}\necho LANCE",
                           str(PARC / "bin" / "x")], capture_output=True, text=True, env=env, timeout=60)


def test_vllm_serveur_attend_la_session_qu_il_a_arretee(smi):
    bloc = _bloc("vllm-serveur", "# CARTE RENDUE (kwh 30/09)")
    mort = _pid_mort()
    r = _jouer(bloc, smi([c0(26000, [(mort, 26000)])] * 3 + [c0(20)]), f"_remplaces=({mort})")
    l = r.stdout.splitlines()
    assert r.returncode == 0 and l[-1] == "LANCE" and l[-2].startswith("vllm : carte rendue en"), r.stdout + r.stderr
    r = _jouer(bloc, smi([c0(26000)], ACVRAM_ATTENTE_VRAM="0.3"), "_remplaces=()")
    assert r.returncode == 1 and "LANCE" not in r.stdout and "vllm : refusé : carte non rendue" in r.stderr


def test_llamacpp_serveur_attend_les_deux_cartes_en_repartition(smi):
    bloc = _bloc("llamacpp-serveur", 'if [ "$taille" -gt 40000000000 ]; then')
    appoint = [os.getpid(), 5606]
    orph1 = {"0": {"used": 20, "apps": []}, "1": {"used": 5606 + 8000, "apps": [appoint]}}
    avant = 'taille=33125992448; MODE=""; gpu_args=(); _remplaces=()'
    r = _jouer(bloc, smi([orph1], ACVRAM_ATTENTE_VRAM="0.3", CUDA_VISIBLE_DEVICES=""), avant)
    assert r.returncode == 1 and "carte 1 : 8000 Mio sans processus" in r.stderr, r.stdout + r.stderr
    r = _jouer(bloc, smi([orph1], ACVRAM_ATTENTE_VRAM="0.3", CUDA_VISIBLE_DEVICES=""),
               avant.replace("33125992448", "20000000000"))                    # tient sur la 5090 : la 1 n'est pas lue
    assert r.returncode == 0 and r.stdout.splitlines()[-1] == "LANCE", r.stdout + r.stderr
