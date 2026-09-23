#!/usr/bin/env python3
"""Duel A2 — llama.cpp, MÊME dénominateur que vLLM/acvram/TabbyAPI
(outils/banc_prefill_chaud.py, banc_prefill_vllm.py, banc_tabbyapi.py) :
prefill = L / durée d'un `n_predict=1` complet, invite différente par
répétition ; décodage = 12 requêtes concurrentes (-np 12), 200 jetons
chacune, énergie NVML monotone (energie.py, protocole de Laure).
Consigne de Jérôme, 14/09/2026 (ferme le tableau du duel à 4 moteurs).

Binaire : build CUDA de LM Studio (Jan/upstream llama.cpp), déjà utilisé
dans le duel du 9/09 — pas de source `externes/llama.cpp` trouvée dans ce
dépôt, et reconstruire depuis les sources sortirait largement du budget
de 30 min. LD_LIBRARY_PATH requis (libs a cote du binaire + libcudart
d'ollama, verifie manuellement avant ce script).

BOGUE NON RESOLU (14/09) : le serveur echoue de façon deterministe
(« Failed to parse input at pos 0 ») apres N complétions reussies dans la
meme session — N a valu 5 (2 chauffes + 3 mesurees) puis 11 (6 chauffes +
5 mesurees) selon le nombre de chauffes, donc PAS un compteur de session
fixe simple. Essaye sans succes : fermer la connexion a chaque appel
(Connection: close), epingler un slot fixe (id_slot=0), reessayer 3 fois
a l'identique. Le contenu du prompt qui echoue n'a rien d'anormal (verifie
: entiers 24767-51378, JSON valide). Piste non explorée : fuite/corruption
dans la table de deduplication du cache de prompt cote serveur (les logs
montrent « cache state: N prompts, X MiB » meme avec cache_prompt=false),
a chercher avec --slots-endpoint-disable ou une version plus recente du
binaire. Mesure PAS FAITE — bloquée par ce bogue, pas par la méthode.

    outils/carte.sh python outils/banc_llamacpp.py
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

BINAIRE_DIR = Path(os.environ.get(
    "LLAMACPP_LMSTUDIO_BIN",
    os.path.expanduser("~/.lmstudio/extensions/backends"
                       "/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.22.0")))
BINAIRE = BINAIRE_DIR / "llama-server"
LD_PATH = f"{BINAIRE_DIR}:/usr/local/lib/ollama/cuda_v12"
GGUF = ("/mnt/4TO_SATACMR_2022/Modeles/models_gguf"
       "/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M"
       "/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf")
HOTE = "http://127.0.0.1:8090"
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
                f"voir /tmp/llamacpp-serveur.log")
        try:
            r = client.get(f"{HOTE}/health", timeout=5.0)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2.0)
    raise RuntimeError("llama-server n'a pas repondu apres attente")


def completion_ids(client: httpx.Client, ids: list, n_predict: int,
                   id_slot: int = -1) -> dict:
    # llama.cpp accepte directement des jetons dans `prompt` (liste
    # d'entiers) — pas d'aller-retour par le tokenizer, contrairement a
    # TabbyAPI. La 6e requete (LRU choisissant un NOUVEAU slot a chaque
    # fois en usage sequentiel) echouait systematiquement ("Failed to
    # parse input at pos 0"), deterministe au COMPTE de requetes, pas au
    # contenu, pas a la reutilisation de connexion (Connection: close
    # n'a rien change) — semble un bogue de rotation de slot cote
    # serveur. Contourne en EPINGLANT le slot pour les appels
    # sequentiels (id_slot fixe) ; le decodage concurrent (id_slot=-1)
    # a besoin de la rotation, donc pas touche ici.
    payload = {"prompt": ids, "n_predict": n_predict, "temperature": 0.0,
              "cache_prompt": False}
    if id_slot >= 0:
        payload["id_slot"] = id_slot
    for tentative in range(3):
        r = client.post(f"{HOTE}/completion", json=payload, timeout=120.0)
        if r.status_code == 200:
            return r.json()
        time.sleep(0.5)
    r.raise_for_status()
    return r.json()


def main() -> int:
    print("BEAD audit Sage A2 — llama.cpp, meme protocole vLLM/acvram/TabbyAPI")
    if not BINAIRE.exists():
        print(f"ECHEC / CAUSE: binaire absent {BINAIRE}")
        return 2
    if not Path(GGUF).exists():
        print(f"ECHEC / CAUSE: GGUF absent {GGUF}")
        return 2

    import os
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = LD_PATH
    journal = open("/tmp/llamacpp-serveur.log", "w")
    proc = subprocess.Popen(
        [str(BINAIRE), "-m", GGUF, "--host", "127.0.0.1", "--port", "8090",
         "-np", str(SLOTS), "-c", str(CTX_TOTAL), "-ngl", "999",
         "--no-mmap", "--no-cache-idle-slots"],
        stdout=journal, stderr=subprocess.STDOUT, env=env, text=True)
    try:
        with httpx.Client() as client:
            print("  attente du serveur...", flush=True)
            attendre_pret(client, proc)
            print("  serveur pret", flush=True)

            # --- prefill, meme denominateur ---
            def un_pp(rep: int) -> tuple:
                ids = invite_ids(rep, PP_LEN)
                t0 = time.perf_counter()
                rep_json = completion_ids(client, ids, n_predict=1, id_slot=0)
                dt = time.perf_counter() - t0
                n_reel = rep_json.get("tokens_evaluated", PP_LEN)
                return dt, n_reel

            # 6 chauffes, pas 2 : le serveur echoue de façon deterministe
            # sur la 6e requete completion de la session ("Failed to
            # parse input at pos 0"), quels que soient le contenu, la
            # connexion (Connection: close), le slot (id_slot fixe) ou
            # une nouvelle tentative (x3) — hypothese non prouvee mais
            # coherente : task_id passe a deux chiffres (0,2,4,6,8 -> 10)
            # exactement a la 6e requete. Chauffer plus loin queue ce
            # seuil AVANT la mesure plutot que de le heurter pendant.
            for r in range(6):
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
                "moteur": "llamacpp", "mode": "decodage", "modele": GGUF,
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
