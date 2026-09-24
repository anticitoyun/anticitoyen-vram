"""Pièce 146 : serveur RÉEL (uvicorn, create_app, boucle moteur, graphes CUDA, pipeline par défaut) sur le modèle-jouet
converti, cache KV réduit à 4 blocs (64 jetons). Une requête de 24 jetons à max_tokens 200 épuise le budget EN DÉCODAGE ;
elle doit se clore (finish_reason "length"). Imprime RESULTAT <json>. Lancé en sous-processus par
test_troncature_kv_bout_en_bout_146.py, qui borne sa durée : une requête jamais close = délai dépassé = rouge.
Usage : serveur_troncature_146.py <converti>"""
import json, os, socket, sys, threading, time, urllib.request

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acvram.engine import loader                                      # noqa: E402

BLOCS = 4
_orig = loader._kv_blocks_per_device
loader._kv_blocks_per_device = lambda plan, spec, mml: {d: min(n, BLOCS) if str(d).startswith("cuda") else n
                                                        for d, n in _orig(plan, spec, mml).items()}
from acvram.engine.runner import Engine                               # noqa: E402
from acvram.server.app import create_app                              # noqa: E402
from acvram.server.chat import load_tokenizer                         # noqa: E402
import uvicorn                                                        # noqa: E402

chemin = sys.argv[1]
loaded = loader.load_model(chemin, dtype=torch.bfloat16, max_model_len=256, max_concurrent_seqs=2)
tok = load_tokenizer(chemin)
eng = Engine(loaded, tok, max_batch_size=2, max_model_len=256)
assert eng.allocator.num_blocks == BLOCS, eng.allocator.num_blocks
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
serveur = uvicorn.Server(uvicorn.Config(create_app(eng, tok, "jouet"), host="127.0.0.1", port=port, log_level="warning"))
threading.Thread(target=serveur.run, daemon=True).start()
while not serveur.started:
    time.sleep(0.05)
corps = json.dumps({"model": "jouet", "prompt": list(range(5, 29)), "max_tokens": 200, "temperature": 0.0,
                    "ignore_eos": True}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=corps,
                             headers={"Content-Type": "application/json"})
t0 = time.time()
rep = json.loads(urllib.request.urlopen(req, timeout=120).read())
print("RESULTAT " + json.dumps({
    "finish_reason": rep["choices"][0]["finish_reason"], "completion_tokens": rep["usage"]["completion_tokens"],
    "duree_s": round(time.time() - t0, 2), "graphes": eng.graphs is not None, "pipeline": bool(eng.pipeline_actif),
    "tronquees": eng.stats.sequences_tronquees_budget}), flush=True)
os._exit(0)
