#!/usr/bin/env python3
"""Duel — llama.cpp, binaire REEL sm_120 (Katy), meme protocole que
vLLM/acvram/TabbyAPI (outils/banc_prefill_chaud.py, banc_prefill_vllm.py,
banc_tabbyapi.py) : prefill = L / duree d'un `n_predict=1` complet, invite
differente par repetition ; decodage = 12 requetes concurrentes (-np 12),
200 jetons chacune, energie NVML monotone (energie.py, protocole de Laure).

Mesure 2c de Sage (14/09) : la reference deja citee (x1,91) venait du
binaire LM Studio (outils/banc_llamacpp.py), jamais mesuree jusqu'au bout
(bloquee par un bogue « Failed to parse input at pos 0 »). Ce script
REFAIT la mesure avec le vrai binaire construit par Katy depuis
`externes/llama.cpp` (commit visible via --version), avant de citer un
facteur quelconque contre lui.

    outils/carte.sh python outils/banc_llamacpp_reel.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent / "gpu" / "mesure"))
from energie import Energie, repos  # noqa: E402

BINAIRE = Path(os.environ.get(
    "LLAMACPP_BIN",
    str(Path(__file__).resolve().parents[1].parent.parent
        / "externes/llama.cpp/build/bin/llama-server")))
GGUF = ("/mnt/4TO_SATACMR_2022/Modeles/models_gguf"
       "/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M"
       "/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf")
HOTE = "http://127.0.0.1:8091"
PP_LEN = 2048
SLOTS = 12
CTX_PAR_SLOT = 2304  # multiple de 256, marge sur 256 invite + 200 decode
CTX_TOTAL = SLOTS * CTX_PAR_SLOT
N_JETONS_DECODE = 200
VOCAB_APPROX = 150000
REP = 7


def invite_ids(rep: int, n: int) -> list:
    return [(1000 + rep * 7919 + i * 13) % (VOCAB_APPROX - 100) + 10 for i in range(n)]


def attendre_pret(client: httpx.Client, proc: subprocess.Popen,
                  timeout_s: float = 300.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server s'est arrete (code {proc.returncode}) — "
                f"voir /tmp/llamacpp-reel-serveur.log")
        try:
            r = client.get(f"{HOTE}/health", timeout=5.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2.0)
    raise RuntimeError("llama-server n'a pas repondu apres attente")


def completion_ids(client: httpx.Client, ids: list, n_predict: int) -> dict:
    # llama.cpp accepte directement des jetons dans `prompt` (liste
    # d'entiers) — pas d'aller-retour par le tokenizer, contrairement a
    # TabbyAPI.
    payload = {"prompt": ids, "n_predict": n_predict, "temperature": 0.0,
              "cache_prompt": False}
    r = client.post(f"{HOTE}/completion", json=payload, timeout=120.0)
    r.raise_for_status()
    return r.json()


def main() -> int:
    print("Mesure 2c Sage — llama.cpp binaire reel sm_120 (Katy), "
          "meme protocole vLLM/acvram/TabbyAPI")
    if not BINAIRE.exists():
        print(f"ECHEC / CAUSE: binaire absent {BINAIRE}")
        return 2
    if not Path(GGUF).exists():
        print(f"ECHEC / CAUSE: GGUF absent {GGUF}")
        return 2

    version = subprocess.run([str(BINAIRE), "--version"],
                             capture_output=True, text=True).stdout.strip()
    print(f"  binaire : {version}", flush=True)

    journal = open("/tmp/llamacpp-reel-serveur.log", "w")
    proc = subprocess.Popen(
        [str(BINAIRE), "-m", GGUF, "--host", "127.0.0.1", "--port", "8091",
         "-np", str(SLOTS), "-c", str(CTX_TOTAL), "-ngl", "999",
         "--no-mmap", "--no-cache-idle-slots"],
        stdout=journal, stderr=subprocess.STDOUT, env=dict(os.environ), text=True)
    try:
        with httpx.Client() as client:
            print("  attente du serveur...", flush=True)
            attendre_pret(client, proc)
            print("  serveur pret", flush=True)

            # --- prefill, meme denominateur ---
            def un_pp(rep: int) -> tuple:
                ids = invite_ids(rep, PP_LEN)
                t0 = time.perf_counter()
                rep_json = completion_ids(client, ids, n_predict=1)
                dt = time.perf_counter() - t0
                n_reel = rep_json.get("tokens_evaluated", PP_LEN)
                return dt, n_reel

            for r in range(2):
                un_pp(100 + r)
            mesures = [un_pp(r) for r in range(REP)]
            jps = sorted(n / d for d, n in mesures)
            med_pp = jps[REP // 2]
            sig_pp = (sum((x - sum(jps) / REP) ** 2 for x in jps) / REP) ** 0.5
            print(f"  pp{PP_LEN} : {med_pp:.0f} j/s (sigma {sig_pp:.0f}), "
                  f"jetons reels {[n for _, n in mesures]}", flush=True)

            # --- decodage 12 sequences concurrentes, energie ---
            prompts = [invite_ids(1000 + k, min(256, CTX_PAR_SLOT // 4))
                      for k in range(SLOTS)]
            completion_ids(client, prompts[0], n_predict=4)  # chauffe

            base = repos(secondes=8.0)
            with Energie() as e:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=SLOTS) as pool:
                    futures = [pool.submit(completion_ids, client, p, N_JETONS_DECODE)
                              for p in prompts]
                    resultats = [f.result() for f in futures]
            duree = e.duree
            n = sum(r.get("tokens_predicted", 0) for r in resultats)
            joules_net = max(e.joules - base.moyenne * duree, 0.0)
            resultat = {
                "moteur": "llamacpp_reel", "binaire_version": version,
                "mode": "decodage", "modele": GGUF,
                "slots": SLOTS, "ctx_par_slot": CTX_PAR_SLOT,
                "n_jetons_decodes": n, "duree_mesure_s": round(duree, 4),
                "jetons_s": n / duree if duree else 0.0,
                "joules": round(e.joules, 1), "joules_net": round(joules_net, 1),
                "j_par_jeton_net": round(joules_net / n, 4) if n else None,
                "watts_repos": round(base.moyenne, 1),
                "pp_len": PP_LEN, "pp_js": round(med_pp), "pp_sigma": round(sig_pp),
                **e.resume(),
            }
            print("RESULTAT " + json.dumps(resultat, ensure_ascii=False))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
