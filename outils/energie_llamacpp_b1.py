#!/usr/bin/env python3
"""llama-server b=1 (Coder-30B Q4_K_M, binaire LM Studio) : repos CHAUD modèle
chargé (30 s), puis un décodage b=1 ≥ 20 s au compteur NVML (5090 seule),
mêmes dénominateurs que outils/banc_llamacpp.py (jetons décodés / durée).
Point 0 de reprise, poste7 (revue/poste7-organisation-14-09.md).

    outils/carte.sh .venv/bin/python3 outils/energie_llamacpp_b1.py
"""
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
_ICI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ICI, "gpu", "mesure"))
import httpx  # noqa: E402
from energie import Energie, nvml  # noqa: E402

BIN_DIR = os.environ.get("LLAMACPP_LMSTUDIO_BIN", os.path.expanduser("~/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.22.0"))
GGUF = os.environ.get("GGUF", "/mnt/4TO_SATACMR_2022/Modeles/models_gguf/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M/"
                      "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf")
PORT = int(os.environ.get("PORT", "8092")); HOTE = f"http://127.0.0.1:{PORT}"
N_PREDICT = int(os.environ.get("BANC_JETONS", "8000")); REPOS_S = float(os.environ.get("BANC_REPOS", "30"))
env = dict(os.environ, LD_LIBRARY_PATH=f"{BIN_DIR}:/usr/local/lib/ollama/cuda_v12")
journal = open("/tmp/llamacpp-energie.log", "w")
proc = subprocess.Popen([f"{BIN_DIR}/llama-server", "-m", GGUF, "--host", "127.0.0.1", "--port", str(PORT),
                         "-np", "1", "-c", os.environ.get("BANC_CTX", "16384"), "-ngl", "999", "--no-mmap"],
                        stdout=journal, stderr=subprocess.STDOUT, env=env)
h = nvml().cartes[0][1]
try:
    with httpx.Client(timeout=600.0) as c:
        for _ in range(600):
            if proc.poll() is not None:
                raise RuntimeError("llama-server arrêté, voir /tmp/llamacpp-energie.log")
            try:
                if c.get(f"{HOTE}/health", timeout=5.0).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        # chauffe courte puis repos CHAUD, modèle chargé
        c.post(f"{HOTE}/completion", json={"prompt": [1000 + i for i in range(16)], "n_predict": 32,
                                           "temperature": 0.0, "cache_prompt": False, "ignore_eos": True})
        time.sleep(5)
        with Energie(periode=0.5) as e0:
            time.sleep(REPOS_S)
        print(f"REPOS_CHAUD {e0.moyenne:.1f} W sur {REPOS_S:.0f} s, horloge {nvml().horloge_sm(h)} MHz", flush=True)
        t_debut = time.strftime("%H:%M:%S")
        with Energie(periode=0.25) as e:
            r = c.post(f"{HOTE}/completion", json={"prompt": [2000 + 7 * i for i in range(16)], "n_predict": N_PREDICT,
                                                   "temperature": 0.0, "cache_prompt": False, "ignore_eos": True}).json()
        t = r.get("timings", {})
        n = int(t.get("predicted_n", r.get("tokens_predicted", 0)))
        net = max(e.joules - e0.moyenne * e.duree, 0.0)
        print("RESULTAT " + json.dumps({
            "moteur": "llama.cpp", "mode": "decodage", "slots": 1, "n_jetons_decodes": n, "fenetre_debut": t_debut, "fenetre_fin": time.strftime("%H:%M:%S"),
            "duree_mesure_s": round(e.duree, 3), "fenetre_valide": e.duree >= 20,
            "jetons_s_serveur": round(1000 * n / t["predicted_ms"], 1) if t.get("predicted_ms") else None,
            "jetons_s_fenetre": round(n / e.duree, 1), "ms_par_jeton_serveur": round(t["predicted_ms"] / n, 3) if n and t.get("predicted_ms") else None,
            "W": round(e.moyenne, 1), "W_net": round(e.moyenne - e0.moyenne, 1), "repos_chaud_W": round(e0.moyenne, 1),
            "J_par_jeton_brut": round(e.joules / n, 4) if n else None, "J_par_jeton_net": round(net / n, 4) if n else None,
            "bridages": sorted(e.bridages), **e.resume()}, ensure_ascii=False), flush=True)
finally:
    proc.terminate()
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()
