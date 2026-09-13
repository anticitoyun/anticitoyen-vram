#!/usr/bin/env python3
"""Duel A2 — TabbyAPI (EXL3 4.0bpw, Qwen3-Coder-30B-A3B), MÊME dénominateur
que vLLM/acvram (outils/banc_prefill_chaud.py, outils/banc_prefill_vllm.py) :
prefill = L / durée d'un `max_tokens=1` complet, invite différente par
répétition, 2 chauffes + 7 répétitions, médiane ; décodage = 12 requêtes
concurrentes, 200 jetons chacune, énergie NVML monotone (energie.py, même
protocole que poste3).

TabbyAPI n'a pas d'API offline `generate()` — lance le serveur lui-même
(comme un client réel le ferait), l'interroge en HTTP, l'arrête. Le
verrou carte.sh doit envelopper CE script (serveur + mesure), pas
seulement la mesure — sinon le serveur reste hors verrou pendant son
chargement.

    outils/carte.sh /opt/ia/TabbyAPI/.venv/bin/python outils/banc_tabbyapi.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent / "gpu" / "mesure"))
from energie import Energie, repos  # noqa: E402

MODELE = "Qwen3-Coder-30B-A3B-4.0bpw-EXL3"
HOTE = "http://127.0.0.1:5000"
CLE_API = "4864f61e315bfc55c63fb5ea7adb4f36"
PP_LEN = 2048
SLOTS = 12
CTX = 2048
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
                f"TabbyAPI s'est arrete (code {proc.returncode}) pendant le "
                f"chargement — voir /tmp/tabbyapi-serveur.log")
        try:
            r = client.get(f"{HOTE}/v1/models", timeout=5.0)
            if r.status_code == 200 and r.json().get("data"):
                return
        except httpx.HTTPError:
            pass
        time.sleep(2.0)
    raise RuntimeError("TabbyAPI n'a pas repondu apres attente")


def decoder_ids(client: httpx.Client, ids: list) -> str:
    # /v1/completions n'accepte qu'un `prompt` STRING (pas de jetons bruts,
    # contrairement a vLLM/acvram) : aller-retour par /v1/token/decode pour
    # rester sur la MEME formule d'invite numerique que les autres bancs.
    # La retokenisation du texte decode peut legerement deriver du compte
    # de jetons d'origine — completion_ids lit le compte REEL rendu par le
    # serveur (usage.prompt_tokens), jamais L suppose.
    r = client.post(f"{HOTE}/v1/token/decode", json={"tokens": ids}, timeout=30.0)
    r.raise_for_status()
    return r.json()["text"]


def completion_ids(client: httpx.Client, ids: list, max_tokens: int) -> dict:
    texte = decoder_ids(client, ids)
    r = client.post(f"{HOTE}/v1/completions", json={
        "model": MODELE, "prompt": texte, "max_tokens": max_tokens,
        "temperature": 0.0, "stream_options": {"include_usage": True},
    }, timeout=120.0)
    r.raise_for_status()
    return r.json()


def main() -> int:
    print("BEAD audit poste7 A2 — TabbyAPI, meme protocole vLLM/acvram")
    journal = open("/tmp/tabbyapi-serveur.log", "w")
    proc = subprocess.Popen(
        ["/opt/ia/TabbyAPI/.venv/bin/python", "main.py",
         "--model-name", MODELE, "--max-seq-len", str(CTX + 256),
         # cache_size doit couvrir le pire des deux phases : un seul pp2048
         # (max_seq_len suffit) OU 12 sequences concurrentes x (256 invite +
         # 200 decodes) au decodage — multiple de 256 (PAGE_SIZE d'exllamav3).
         "--cache-size", str(((SLOTS * (256 + N_JETONS_DECODE)) // 256 + 1) * 256),
         "--max-batch-size", str(SLOTS)],
        cwd="/opt/ia/TabbyAPI", stdout=journal, stderr=subprocess.STDOUT,
        text=True)
    try:
        with httpx.Client(headers={"x-api-key": CLE_API}) as client:
            print("  attente du serveur...", flush=True)
            attendre_pret(client, proc)
            print("  serveur pret", flush=True)

            # --- prefill, meme denominateur ---
            def un_pp(rep: int) -> tuple:
                ids = invite_ids(rep, PP_LEN)
                t0 = time.perf_counter()
                rep_json = completion_ids(client, ids, max_tokens=1)
                dt = time.perf_counter() - t0
                n_reel = (rep_json.get("usage") or {}).get("prompt_tokens", PP_LEN)
                return dt, n_reel

            for r in range(2):
                un_pp(100 + r)
            mesures = [un_pp(r) for r in range(REP)]
            jps = sorted(n / d for d, n in mesures)
            med_pp = jps[REP // 2]
            sig_pp = (sum((x - sum(jps) / REP) ** 2 for x in jps) / REP) ** 0.5
            print(f"  jetons prompt reels (rendus par le serveur) : "
                  f"{[n for _, n in mesures]}", flush=True)
            print(f"  pp{PP_LEN} : {med_pp:.0f} j/s (sigma {sig_pp:.0f})", flush=True)

            # --- decodage 12 sequences, energie ---
            prompts = [invite_ids(1000 + k, min(256, CTX // 4)) for k in range(SLOTS)]
            completion_ids(client, prompts[0], max_tokens=4)  # chauffe

            base = repos(secondes=8.0)
            with Energie() as e:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=SLOTS) as pool:
                    futures = [pool.submit(completion_ids, client, p, N_JETONS_DECODE)
                              for p in prompts]
                    resultats = [f.result() for f in futures]
            duree = e.duree
            n = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in resultats)
            joules_net = max(e.joules - base.moyenne * duree, 0.0)
            resultat = {
                "moteur": "tabbyapi", "mode": "decodage", "modele": MODELE,
                "slots": SLOTS, "ctx": CTX, "n_jetons_decodes": n,
                "duree_mesure_s": round(duree, 4),
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
