#!/usr/bin/env python3
"""Puissance (W) et énergie par poste d'un pas de décodage b=12, chaque
poste en boucle ~DUREE s, compteur d'énergie NVML (5090 seule :
CUDA_VISIBLE_DEVICES=0 posé ici). Deux témoins encadrent : une copie DRAM
pure (octets sans instructions) et un GEMM bf16 (instructions sans octets).
Si un noyau du pas consomme comme la copie, ses instructions ne coûtent rien
de plus que ses octets ; s'il consomme comme le GEMM, elles coûtent.

    outils/carte.sh .venv/bin/python3 outils/energie_par_poste.py [b] [duree_s]

Carte EXCLUSIVE (ACVRAM_TYPE=mesure), repos mesuré avant, 30 s.
"""
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("ACVRAM_TYPE", "mesure")
_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_ICI, "gpu", "mesure")); sys.path.insert(0, _ICI)
B = int(sys.argv[1]) if len(sys.argv) > 1 else 12
DUREE = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
REPOS = float(os.environ.get("BANC_REPOS", "30"))
import torch  # noqa: E402
from energie import Energie, nvml  # noqa: E402
from acvram import kernels  # noqa: E402
from acvram.engine.loader import load_model  # noqa: E402
from acvram.engine.runner import Engine  # noqa: E402
from acvram.engine.sampler import SamplingParams  # noqa: E402
from regime import exiger_regime_nominal  # noqa: E402
import importlib.util  # noqa: E402
_s = importlib.util.spec_from_file_location("rm", os.path.join(_ICI, "racine_modeles.py"))
_m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
chemin = os.path.join(_m.MODELES, os.environ.get("BANC_MODELE", "Qwen3-Coder-30B-A3B-nvfp4"))

loaded = load_model(chemin, dtype=torch.bfloat16, device_override="cuda:0")
model = loaded.model
ext = kernels.get_extension()
dev = torch.device("cuda:0")
h = nvml().cartes[0][1]


def mesure(nom, fn, lots=20):
    """fn() lance un lot de travail ; on boucle DUREE s, énergie NVML."""
    for _ in range(3): fn()
    torch.cuda.synchronize()
    n = 0; horloges = []
    with Energie(periode=0.25) as e:
        t0 = time.time()
        while time.time() - t0 < DUREE:
            for _ in range(lots): fn()
            torch.cuda.synchronize(); n += lots
            horloges.append(nvml().horloge_sm(h))
    w = e.moyenne
    r = {"poste": nom, "W": round(w, 1), "W_net": round(w - repos_w, 1), "J": round(e.joules, 1),
         "s": round(e.duree, 2), "lancements": n, "us_par_lancement": round(1e6 * e.duree / n, 1),
         "mJ_par_lancement": round(1e3 * e.joules / n, 3), "MHz": sorted(horloges)[len(horloges) // 2],
         "bridages": sorted(e.bridages)}
    print("POSTE " + json.dumps(r, ensure_ascii=False), flush=True)
    return r


# Repos.
torch.cuda.synchronize(); time.sleep(2)
with Energie(periode=0.5) as e:
    time.sleep(REPOS)
repos_w = e.moyenne
print(f"REPOS {repos_w:.1f} W sur {REPOS:.0f} s (bridages {sorted(e.bridages)})", flush=True)

# Le moteur d'abord : les piles d'experts (_stacks) et les graphes n'existent
# qu'après lui ; le pas complet se mesure en dernier, dans le même état thermique.
# Pas complet, rejeu de graphes, b=B — le régime réel.
eng = Engine(loaded, None, max_batch_size=B, max_model_len=1024, enable_cuda_graphs=True, enable_prefix_cache=False)
eng._eos = set()
SP = SamplingParams(temperature=0.0, max_tokens=4096)
for b in range(B):
    eng.add_request([(1000 + b * 101 + i * 13) % 150000 + 10 for i in range(128)], SP)
while any(not s.prefilled for s in eng.running) or eng.waiting:
    eng.step()
for _ in range(5): eng.step()
torch.cuda.synchronize()
# Garde-fou d'poste1 : moteur dégradé (exil, graphes coupés, deux cartes) = refus,
# c'est ce qui a rendu invalide la reprise du profil M1 le 14/09.
exiger_regime_nominal(eng, autoriser_piles_inconnues=False)
res = []
# Témoins.
src = torch.empty(1 << 30, dtype=torch.uint8, device=dev); dst = torch.empty_like(src)
res.append(mesure("temoin copie DRAM 1 Gio (octets, ~0 instruction)", lambda: dst.copy_(src), lots=4))
a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=dev); bm = torch.randn(8192, 8192, dtype=torch.bfloat16, device=dev)
res.append(mesure("temoin GEMM bf16 8192^3 (instructions, ~0 octet DRAM)", lambda: torch.matmul(a, bm), lots=2))
del src, dst, a, bm

# Postes du pas, couche 5, tenseurs de la forme réelle du décodage b=B.
couche = model.layers[5]
moe, attn = couche.mlp, couche.self_attn
x = torch.randn(B, model.spec.hidden_size, dtype=torch.bfloat16, device=dev)
g = torch.Generator(device="cpu").manual_seed(7)
topi = torch.stack([torch.randperm(len(moe.experts), generator=g)[:moe.top_k] for _ in range(B)]).to(dev)
topw = torch.softmax(torch.randn(B, moe.top_k, generator=g), -1).to(dev).to(torch.bfloat16)
eid = topi.reshape(-1).to(torch.int32)
tok = torch.arange(B, device=dev, dtype=torch.int32).repeat_interleave(moe.top_k)
pg, pu, pd = moe._stacks["gate_proj"], moe._stacks["up_proj"], moe._stacks["down_proj"]
gateup = lambda: ext.nvfp4_gemv_grouped_gateup(pg[1], pg[2], pg[3], pu[1], pu[2], pu[3], eid, tok, x, pg[4], 0)
act = gateup()[:, :pg[5]].contiguous()
seq = torch.arange(eid.shape[0], device=dev, dtype=torch.int32)
if os.environ.get("BANC_SEULEMENT") != "mma":
    res.append(mesure("MoE gate·up GEMV (nvfp4_gemv_grouped_gateup)", gateup, lots=50))
    res.append(mesure("MoE down GEMV (_grouped down_proj)", lambda: moe._grouped(act, pd, eid, seq), lots=50))
res.append(mesure("MoE complet (_forward_grouped : gate·up + down + reduce)", lambda: moe._forward_grouped(x, topw, topi), lots=50))
# Le même MoE par le chemin GEMM groupée MMA FP4 (celui du prefill) à t=B jetons :
# quant_act + gate + up + moe_act + quant_act + down + reduce_trie — étape (iii) de poste7
# (revue/poste7-moe-mma-decodage-14-09.md). Tuile ACVRAM_MOE_MMA_BT (16 conseillé à M≈3).
SEUL = os.environ.get("BANC_SEULEMENT")          # "mma" : ne mesurer que les postes MMA
if moe._forward_prefill_grouped(x, topw, topi) is not None:
    # Les 3 GEMM MMA seules (gate, up, down), tuiles et activations quantifiées une
    # fois : W du noyau sans la glue hôte (argsort, bincount, .item()) qui borne la
    # boucle complète ci-dessous.
    import torch.nn.functional as F
    flat_e = topi.reshape(-1).to(torch.int64); ordre = torch.argsort(flat_e, stable=True)
    cnt = torch.bincount(flat_e, minlength=len(moe.experts))
    bt = int(os.environ.get("ACVRAM_MOE_MMA_BT", "64")); tiles = moe._tuiles(cnt, bt)
    xs = x[torch.arange(B, device=dev).repeat_interleave(moe.top_k)[ordre]].to(torch.bfloat16).contiguous()
    if xs.shape[1] != pg[4]: xs = F.pad(xs, (0, pg[4] - xs.shape[1])).contiguous()
    xq, xsf = ext.nvfp4_quant_act(xs)
    g = moe._gemm_mma(pg, xq, xsf, tiles, brut=True); u = moe._gemm_mma(pu, xq, xsf, tiles, brut=True)
    a3 = ext.moe_act(g, u, pg[5], pd[4], 0); aq, asf = ext.nvfp4_quant_act(a3)
    def trois_gemm():
        moe._gemm_mma(pg, xq, xsf, tiles, brut=True); moe._gemm_mma(pu, xq, xsf, tiles, brut=True)
        moe._gemm_mma(pd, aq, asf, tiles, brut=True)
    res.append(mesure("MoE 3 GEMM MMA seules (gate+up+down, BT=%d, %d tuiles)" % (bt, int(tiles[0].numel())), trois_gemm, lots=50))
    res.append(mesure("quant_act x2 (E2M1 bloc 16)", lambda: (ext.nvfp4_quant_act(xs), ext.nvfp4_quant_act(a3)), lots=100))
    res.append(mesure("MoE complet MMA (_forward_prefill_grouped, BT=%s)" % os.environ.get("ACVRAM_MOE_MMA_BT", "64"),
                      lambda: moe._forward_prefill_grouped(x, topw, topi), lots=50))
if SEUL != "mma":
    if getattr(attn, "qkv_proj", None) is not None:
        res.append(mesure("projections q/k/v int8 empilées (qkv_proj)", lambda: attn.qkv_proj(x), lots=100))
    else:
        res.append(mesure("projection q int8", lambda: attn.q_proj(x), lots=100))
    xo = torch.randn(B, attn.o_proj.qweight.shape[1] if hasattr(attn.o_proj, "qweight") else 4096, dtype=torch.bfloat16, device=dev)
    res.append(mesure("projection o int8 (o_proj)", lambda: attn.o_proj(xo), lots=100))
    lm = model.lm_head
    res.append(mesure("lm_head int8", lambda: lm(x), lots=50))
    res.append(mesure("norme RMS (input_layernorm)", lambda: couche.input_layernorm(x), lots=200))

res.append(mesure("PAS COMPLET b=%d rejeu (Engine.step)" % B, lambda: eng.step(), lots=10))
print("RESULTAT " + json.dumps({"B": B, "repos_W": round(repos_w, 1), "postes": res}, ensure_ascii=False))
