#!/usr/bin/env python3
"""A/B du décodage MLA batché (bead 6wa) — un bras par processus.

    outils/carte.sh .venv/bin/python3 outils/ab-mla-batch.py A   # boucle par créneau
    outils/carte.sh .venv/bin/python3 outils/ab-mla-batch.py B   # mla_decode_batch

Protocole (spec-batching-mla-6wa-13-09.md §7) : GLM-4.7-Grande-Heretic-42B,
ACVRAM_HYBRID_SLOTS=12, ctx 2048, ACVRAM_CHRONO_SYNC=1, b_reel=12 réel, EOS
neutralisé. N_TOURS fenêtres de décodage concurrent ; médian par fenêtre de
pas_total et de replay ; l'ordre ABBA se fait en lançant A, B, B, A.
Le bras A doit retrouver pas_total_median ≈ 92,80 ms (témoin du 13/09).
"""
import os, sys, time, json, math, hashlib
_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
sys.path.insert(0, _REPO)                       # le code de CET arbre, pas le venv
BRAS = sys.argv[1].upper() if len(sys.argv) > 1 else "A"
N_TOURS = int(os.environ.get("AB_TOURS", "15"))
N_JETONS = int(os.environ.get("AB_JETONS", "150"))
L_INVITE = int(os.environ.get("AB_INVITE", "32"))   # 32 (court) ou 1536 (long : godet MLA plein)
SLOTS = 12
os.environ["ACVRAM_MLA_BATCH"] = "1" if BRAS == "B" else "0"
os.environ.setdefault("ACVRAM_HYBRID_SLOTS", str(SLOTS))
os.environ.setdefault("ACVRAM_CHRONO_SYNC", "1")
import torch
from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
from acvram import kernels
import importlib.util
_s = importlib.util.spec_from_file_location("rm", os.path.join(_ICI, "racine_modeles.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
chemin = os.path.join(_m.MODELES, "GLM-4.7-Grande-Heretic-42B-srcQ4_K_M-nvfp4")
MAX_LEN = 2048

def med(l):
    s = sorted(l); return s[len(s) // 2] if s else float("nan")

ext = kernels.get_extension()
so = getattr(kernels, "_SO_PATH", "") or ""
empreinte = hashlib.sha256(open(so, "rb").read()).hexdigest()[:12] if so and os.path.exists(so) else "?"
print(f"[AB] bras={BRAS} ACVRAM_MLA_BATCH={os.environ['ACVRAM_MLA_BATCH']} mla_decode_batch={hasattr(ext, 'mla_decode_batch')} so={empreinte}", flush=True)

loaded = load_model(chemin, dtype=torch.bfloat16, max_model_len=MAX_LEN)
eng = Engine(loaded, None, max_batch_size=SLOTS, max_model_len=MAX_LEN)
eng._eos = set()
c = {"batch": 0, "boucle": 0}
for _, mod in loaded.model.named_modules():
    la = getattr(mod, "linear_attn", None)
    if la is not None and hasattr(la, "decode_static_batch"):
        f1 = la.decode_static_batch
        def w1(*a, _f=f1, **k): c["batch"] += 1; return _f(*a, **k)
        la.decode_static_batch = w1
        f2 = la.decode_static
        def w2(*a, _f=f2, **k): c["boucle"] += 1; return _f(*a, **k)
        la.decode_static = w2
# Les compteurs ne voient que le code Python EXÉCUTÉ : une capture de graphe
# et ses échauffements, jamais un rejeu. Ils prouvent donc quel chemin a été
# capturé — c'est ce qui compte — et restent à zéro pendant les fenêtres.
print(f"[AB] graphes captures : {eng.warm_graphs()} ; appels pendant la capture : {c}", flush=True)
_alloc = getattr(eng, "allocator", None)
_nb = getattr(_alloc, "n_blocks", None) or getattr(_alloc, "num_blocks", None) or len(getattr(_alloc, "free", []) or [])
print(f"[AB] capacite KV : {_nb} blocs x 16 = {(_nb or 0) * 16} jetons ; 12 x (invite {L_INVITE} + {N_JETONS}) = {12 * (L_INVITE + N_JETONS)}", flush=True)
vocab = getattr(getattr(loaded, "spec", None), "vocab_size", 0) or 32000
SP = SamplingParams(temperature=0.0, max_tokens=N_JETONS)

def fenetre(tour):
    for b in range(SLOTS):
        eng.add_request([((tour + 1) * 104729 + b * 7919 + i * 13) % (vocab - 100) + 10 for i in range(L_INVITE)], SP)
    while any(not s.prefilled for s in eng.running) or eng.waiting:
        eng.step()
    torch.cuda.synchronize()
    eng.temps_pas_total = []; eng.temps_pas_voie = []
    if eng.graphs is not None: eng.graphs.temps_replay = []
    pas = 0
    sorties = []
    b_reels = []
    while eng.running:
        b_reels.append(len(eng.running))
        for o in eng.step():
            sorties.append((getattr(o, "request_id", ""), tuple(getattr(o, "token_ids", ()) or ())))
        pas += 1
    torch.cuda.synchronize()
    # empreinte des jetons produits : A et B doivent la partager (bit-identique)
    emp = hashlib.sha256(repr(sorted(sorties)).encode()).hexdigest()[:12]
    replay = list(eng.graphs.temps_replay) if eng.graphs is not None else []
    voie = eng.temps_pas_voie
    return {"pas": pas, "b_min": min(b_reels), "b_max": max(b_reels), "jetons": emp, "pas_total_med": med(eng.temps_pas_total), "replay_med": med(replay),
            "graphe": sum(1 for v in voie if v == "graphe"), "eager": sum(1 for v in voie if v == "eager"),
            "appels_batch": c["batch"], "appels_boucle": c["boucle"]}

fenetre(-1)                                     # chauffe (capture des godets si besoin)
print(f"[AB] appels cumules apres chauffe : {c}", flush=True)
res = []
for t in range(N_TOURS):
    r = fenetre(t); res.append(r)
    print(f"[TOUR {t:02d}] b={r['b_min']}..{r['b_max']} jetons={r['jetons']} pas={r['pas']} pas_total_med={r['pas_total_med']:.2f} ms replay_med={r['replay_med']:.2f} ms graphe={r['graphe']} eager={r['eager']} batch={r['appels_batch']} boucle={r['appels_boucle']}", flush=True)
if eng.graphs is not None:
    print(f"[AB] cles de graphes capturees (b, ql, nblk, lb) : {sorted(eng.graphs.graphs.keys())}", flush=True)
pt = [r["pas_total_med"] for r in res]; rp = [r["replay_med"] for r in res]
moy = sum(pt) / len(pt); sig = math.sqrt(sum((x - moy) ** 2 for x in pt) / len(pt))
print("RESULTAT " + json.dumps({"bras": BRAS, "so": empreinte, "tours": N_TOURS, "invite": L_INVITE, "pas_total_median_ms": round(med(pt), 2),
      "pas_total_sigma_ms": round(sig, 2), "replay_median_ms": round(med(rp), 2),
      "jetons_tour0": res[0]["jetons"], "b_min": min(r["b_min"] for r in res), "b_max": max(r["b_max"] for r in res), "graphe": sum(r["graphe"] for r in res), "eager": sum(r["eager"] for r in res),
      "appels_batch": res[-1]["appels_batch"], "appels_boucle": res[-1]["appels_boucle"]}))
