"""Pièce 71 bis, jalon A au banc : µs par couche du MoE b=12 (20 routages réels, 33 experts distincts en moyenne, 8 jeux
de piles en rotation = L2 froid) — chemin tensor actuel (gate, up : 2 GEMM + moe_act + down) contre w13 (1 GEMM gate‖up
+ moe_act à échelles + down), sur les formes Coder E=128 K=2048 I=768 k=8. Graphe CUDA des 20 appels, rejoué."""
import json, os, sys, time, torch
# Pièce 211 : racine dérivée de __file__, pas du cwd — "." et "tests" en dur cassaient
# nommément (garde a86fa1dd) dès que le script était lancé d'ailleurs que la racine (poste5, 25/09).
_RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_RACINE, "tests")); sys.path.insert(0, _RACINE)
from test_moe_tensor_decodage import _charger
from acvram.engine.moe import gemm_experts_tensor
E, K, I, k, b, PILES = 128, 2048, 768, 8, 12, 8
_, MP, ext, banc = _charger(); dev = torch.device("cuda", 0)
routages = [r.to(dev).to(torch.int32).reshape(-1).contiguous() for r in torch.load("scratchpad/poste5-p62-23-09/routages-cellule-b12-20.pt", weights_only=False)]
assert len(routages) == 20 and routages[0].numel() == b * k
jeux = []
for i in range(PILES):
    qg, bg, gsg, _ = banc.pile(I, K, 100 + i, dev); qu, bu, gsu, _ = banc.pile(I, K, 200 + i, dev); qd, bd, gsd, _ = banc.pile(K, I, 300 + i, dev)
    mg, mu, md = MP.preparer_pile(qg, bg, gsg), MP.preparer_pile(qu, bu, gsu), MP.preparer_pile(qd, bd, gsd)
    sep = {"gate_proj": (*mg, K, I), "up_proj": (*mu, K, I), "down_proj": (*md, I, K)}
    w13 = MP.preparer_pile(torch.cat((qg, qu), 1).contiguous(), torch.cat((bg, bu), 1).contiguous(), torch.ones(E, dtype=torch.float32, device=dev))
    fus = dict(sep); fus["gate_up"] = (*w13, K, 2 * I); fus["gs_gate"] = gsg.reshape(-1).float().contiguous(); fus["gs_up"] = gsu.reshape(-1).float().contiguous()
    jeux.append((sep, fus))
x = torch.randn(b, K, dtype=torch.bfloat16, device=dev, generator=torch.Generator(dev).manual_seed(71))
ws = MP.espace_travail(dev, 4); uns = torch.ones(b * k, 1, dtype=torch.float32, device=dev)
def chrono(mode, rep=40):
    tamp, sort = {}, {}
    def passe():
        for i, eid in enumerate(routages):
            m = jeux[i % PILES][mode]
            gemm_experts_tensor(MP, ext, x, eid, m, k, I, I, 0, ws, uns, tamp, sort)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s): passe(); passe()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): passe()
    torch.cuda.synchronize(); ts = []
    for _ in range(rep):
        a, c = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); c.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(c) * 1e3 / len(routages))
    ts.sort(); return ts[len(ts) // 2]
distincts = sum(len(set(r.tolist())) for r in routages) / len(routages)
ua, ub = chrono(0), chrono(1)
res = {"distincts_moy": round(distincts, 1), "us_couche_gate_up_separes": round(ua, 1), "us_couche_w13": round(ub, 1), "gain_us_couche": round(ua - ub, 1), "gain_ms_pas_48": round((ua - ub) * 48 / 1e3, 3)}
print("RESULTAT", json.dumps(res)); json.dump(res, open(sys.argv[1], "w"), indent=1)
