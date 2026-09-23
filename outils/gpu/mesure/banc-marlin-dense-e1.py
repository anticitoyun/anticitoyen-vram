"""Pièce 71, jalon 1 : GEMM dense W4A16 par la GEMM groupée Marlin du port avec E = 1, contre le gemm_etroit int8 servi,
qkv [5120, 2048] et o [2048, 4096], M = 12, L2 froid (24 copies des poids dans un graphe, harnais p57).
    outils/carte.sh python scratchpad/gaelle-p71-23-09/banc-marlin-dense.py sortie.json"""
import importlib.util, json, os, sys, torch
R = os.path.dirname(os.path.abspath(__file__)) + "/../.."
sys.path.insert(0, R); sys.path.insert(0, R + "/outils/gpu/mesure")
def _mod(nom, chemin):
    sp = importlib.util.spec_from_file_location(nom, chemin); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
occ = _mod("occ", R + "/outils/gpu/mesure/banc-etroites-occupation.py")
lat = _mod("lat", R + "/outils/gpu/mesure/banc-etroites-latence.py")
from acvram.kernels import gemm_etroit as GE, marlin_port as MP
from acvram.quant.nvfp4 import dequantize_nvfp4, quantize_nvfp4
dev = torch.device("cuda", 0); COPIES = 24; REP = 200; M = 12
ws = MP.espace_travail(dev, 4); uns = torch.ones(M, 1, dtype=torch.float32, device=dev)
bloc = 16
sorted_ids = torch.full((bloc,), M, dtype=torch.int32, device=dev); sorted_ids[:M] = torch.arange(M, dtype=torch.int32, device=dev)
expert_ids = torch.zeros(1, dtype=torch.int32, device=dev); num_post = torch.tensor([bloc], dtype=torch.int32, device=dev)
res = {"M": M, "copies": COPIES, "formes": {}}
for nom, (n, k) in occ.FORMES_POIDS.items():
    x = (torch.randn(M, k, generator=torch.Generator().manual_seed(7)) * 0.5).to(torch.bfloat16).to(dev)
    # int8 servi (p57)
    c8 = [occ.poids_int8(n, k, graine=(hash(nom) + i) % 1000, device=dev) for i in range(COPIES)]
    GE.regler_forme(None)
    f8 = lat.chrono_froid([(lambda t=t: GE.gemm_etroit(x, t, compact=True)) for t in c8], 30)
    oct8 = n * k + c8[0].scales.numel() * 2 + c8[0].zeros.numel()
    # nvfp4 → Marlin E = 1
    piles, refs = [], None
    for i in range(COPIES):
        g = torch.Generator().manual_seed(1000 + i)
        w = (torch.randn(n, k, generator=g) * 0.02).to(torch.bfloat16)
        t = quantize_nvfp4(w)
        qw = t.qweight.unsqueeze(0).contiguous().to(dev); bs = t.block_scale.unsqueeze(0).contiguous().to(dev); gs = t.global_scale.float().reshape(1).to(dev)
        wm, sm, gm = MP.preparer_pile(qw, bs, gs)
        piles.append((wm, sm, gm))
        if i == 0: refs = (x.float() @ dequantize_nvfp4(t, torch.float32).to(dev).T)
    def marlin(p):
        return MP.gemm_moe(x, p[0], p[1], p[2], sorted_ids, expert_ids, num_post, uns, bloc, 1, M, n, k, ws)
    y = marlin(piles[0]).float()
    hors = int(((y - refs).abs() > 2 ** -7 * refs.abs().amax(1, keepdim=True)).sum())
    fm = lat.chrono_froid([(lambda p=p: marlin(p)) for p in piles], 30)
    octm = piles[0][0].numel() * 4 + piles[0][1].numel()
    r = {"n_k": [n, k], "int8_us_froid": round(f8[len(f8) // 2], 2), "int8_to_s": round(oct8 / (f8[len(f8) // 2] * 1e-6) / 1e12, 3),
         "marlin_e1_us_froid": round(fm[len(fm) // 2], 2), "marlin_to_s": round(octm / (fm[len(fm) // 2] * 1e-6) / 1e12, 3),
         "gain_pct": round(100 * (1 - fm[len(fm) // 2] / f8[len(f8) // 2]), 1), "hors_2e-7": hors, "sortie_shape": list(y.shape)}
    res["formes"][nom] = r; print("RESULTAT", nom, json.dumps(r), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
