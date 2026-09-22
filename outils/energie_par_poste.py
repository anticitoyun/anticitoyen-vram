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

loaded = load_model(chemin, dtype=torch.bfloat16, device_override="cuda:0", max_model_len=1024, max_concurrent_seqs=B)   # le Plan connaît le lot (garde kv_planned_seqs, 17/09)
model = loaded.model
ext = kernels.get_extension()
dev = torch.device("cuda:0")
h = nvml().cartes[0][1]


def _t_noyau(fn, lancements):
    """Σ durée GPU des noyaux (CUPTI, torch.profiler) sur `lancements` appels de fn(),
    en µs par lancement, et les noyaux par part décroissante. Passe séparée, APRÈS
    la boucle d'énergie : le profileur charge l'hôte et fausserait W et t_mur."""
    from torch.profiler import profile, ProfilerActivity
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(lancements): fn()
        torch.cuda.synchronize()
    par_noyau = {}
    for ev in prof.events():
        if ev.device_type.name == "CUDA" and ev.device_time > 0:
            par_noyau[ev.name] = par_noyau.get(ev.name, 0.0) + ev.device_time
    total = sum(par_noyau.values())
    noyaux = sorted(par_noyau.items(), key=lambda kv: -kv[1])[:4]
    return total / lancements, [(k[:60], round(v / total, 3)) for k, v in noyaux] if total else []


def mesure(nom, fn, lots=20, unite="", octets=None):
    """fn() lance un lot de travail ; on boucle DUREE s, énergie NVML. `unite` dit ce
    qu'est UN lancement (la grandeur que divise mJ_par_lancement). Après la boucle, le
    rapport cyclique Σ t_noyau / t_mur (§ Sage sage-c10-scelle-retire-m1 : < 0,9 → le W
    est celui de la boucle, pas du noyau)."""
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
    us_mur = 1e6 * e.duree / n
    us_noyau, noyaux = _t_noyau(fn, max(lots, min(n, 200)))
    r = {"poste": nom, "unite": unite, "W": round(w, 1), "W_net": round(w - repos_w, 1), "J": round(e.joules, 1),
         "s": round(e.duree, 2), "lancements": n, "us_par_lancement": round(us_mur, 1),
         "us_noyau_par_lancement": round(us_noyau, 1), "rapport_cyclique": round(us_noyau / us_mur, 3),
         "noyaux": noyaux,
         "Go_par_lancement": None if octets is None else round(octets / 1e9, 4),
         "Go_s_noyau": None if octets is None or us_noyau <= 0 else round(octets / us_noyau / 1e3, 1),
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
# Garde-fou d'Océane : moteur dégradé (exil, graphes coupés, deux cartes) = refus,
# c'est ce qui a rendu invalide la reprise du profil M1 le 14/09.
exiger_regime_nominal(eng, autoriser_piles_inconnues=False)
res = []
# Témoins.
src = torch.empty(1 << 30, dtype=torch.uint8, device=dev); dst = torch.empty_like(src)
SEUL = os.environ.get("BANC_SEULEMENT")          # "mma" : postes MMA seuls ; "pas" : le pas complet seul ; "mesure1" : Marlin et mma2 à unité égale
TEMOINS = os.environ.get("BANC_TEMOINS", "1") != "0"     # 0 : sauter les deux témoins (fenêtre courte)
if SEUL != "pas" and TEMOINS:
    res.append(mesure("temoin copie DRAM 1 Gio (octets, ~0 instruction)", lambda: dst.copy_(src), lots=4, unite="1 copie de 1 Gio"))
a = torch.randn(8192, 8192, dtype=torch.bfloat16, device=dev); bm = torch.randn(8192, 8192, dtype=torch.bfloat16, device=dev)
if SEUL != "pas" and TEMOINS:
    res.append(mesure("temoin GEMM bf16 8192^3 (instructions, ~0 octet DRAM)", lambda: torch.matmul(a, bm), lots=2, unite="1 GEMM 8192^3 bf16 (1,1 TFLOP)"))
del src, dst, a, bm

# Postes du pas, couche 5, tenseurs de la forme réelle du décodage b=B.
# P7 (Sage, revue/sage-lancement-14-09.md) : TAMPONS TOURNANTS. Une boucle sur les
# poids d'UNE couche (experts touchés 53 Mo, projection 2-8 Mo) tient dans le L2
# de 96 Mo : le noyau relit le L2, pas la DRAM, et son W n'est pas celui du pas.
# Chaque poste tourne donc sur les 48 couches à tour de rôle (2,5 Go d'experts,
# 0,9 Go de projections) : chaque lancement lit ses poids depuis la DRAM.
# Témoin de la règle : même mesure sur la couche 5 seule (BANC_L2_CHAUD=1).
L2_CHAUD = os.environ.get("BANC_L2_CHAUD") == "1"
couches = [model.layers[5]] if L2_CHAUD else [c for c in model.layers if getattr(c.mlp, "_stacks", None)]
nc = len(couches)
print(f"COUCHES {nc} ({'L2 chaud, couche 5 seule' if L2_CHAUD else 'tampons tournants'})", flush=True)
couche = couches[0]
moe, attn = couche.mlp, couche.self_attn
x = torch.randn(B, model.spec.hidden_size, dtype=torch.bfloat16, device=dev)
g = torch.Generator(device="cpu").manual_seed(7)
topi = torch.stack([torch.randperm(len(moe.experts), generator=g)[:moe.top_k] for _ in range(B)]).to(dev)
topw = torch.softmax(torch.randn(B, moe.top_k, generator=g), -1).to(dev).to(torch.bfloat16)
eid = topi.reshape(-1).to(torch.int32)
U_COUCHE = "1 couche, %d jetons" % B
U_MOE = "1 couche MoE, %d jetons x top_k %d = %d lignes d'expert" % (B, moe.top_k, B * moe.top_k)
tok = torch.arange(B, device=dev, dtype=torch.int32).repeat_interleave(moe.top_k)
seq = torch.arange(eid.shape[0], device=dev, dtype=torch.int32)
piles = [(c.mlp._stacks["gate_proj"], c.mlp._stacks["up_proj"], c.mlp._stacks["down_proj"]) for c in couches]
pg, pu, pd = piles[0]
# Disposition unique Marlin (P1, défaut depuis le 18/09) : la pile naturelle est
# rendue (pg[1] is None), les GEMV du décodage lisent la disposition Marlin —
# le poste MoE se boucle sur les noyaux Marlin (gate·up NW=2, down NW=1)
marlins = [getattr(c.mlp, "_stacks_marlin", None) for c in couches]
MARLIN = marlins[0] is not None and pg[1] is None
def gateup_c(i):
    pg, pu, _ = piles[i % nc]
    if MARLIN:
        mg, mu = marlins[i % nc]["gate_proj"], marlins[i % nc]["up_proj"]
        return ext.nvfp4_gemv_marlin_gateup(mg[0], mg[1], mg[2], mu[0], mu[1], mu[2], eid, tok, x, mg[3], mg[4], 0)
    return ext.nvfp4_gemv_grouped_gateup(pg[1], pg[2], pg[3], pu[1], pu[2], pu[3], eid, tok, x, pg[4], 0)
def down_c(i):
    if MARLIN:
        md = marlins[i % nc]["down_proj"]
        return ext.nvfp4_gemv_marlin(md[0], md[1], md[2], eid, seq, act, md[3], md[4])
    return moe._grouped(act, piles[i % nc][2], eid, seq)
act = gateup_c(0)[:, :pg[5]].contiguous()
class Tour:
    """Un compteur : chaque appel avance d'une couche."""
    def __init__(self): self.i = 0
    def __call__(self):
        self.i += 1; return self.i
t_gu, t_dn, t_moe, t_qkv, t_o, t_norm, t_mma, t_mmac = (Tour() for _ in range(8))
if SEUL not in ("mma", "pas", "mesure1"):
    res.append(mesure("MoE gate·up GEMV (%s)" % ("nvfp4_gemv_marlin_gateup" if MARLIN else "nvfp4_gemv_grouped_gateup"),
                      lambda: gateup_c(t_gu()), lots=48, unite=U_MOE))
    res.append(mesure("MoE down GEMV (%s)" % ("nvfp4_gemv_marlin" if MARLIN else "_grouped down_proj"),
                      lambda: down_c(t_dn()), lots=48, unite=U_MOE))
if SEUL not in ("pas", "mesure1"):
    res.append(mesure("MoE complet (_forward_grouped : gate·up + down + reduce)",
                      lambda: couches[t_moe() % nc].mlp._forward_grouped(x, topw, topi), lots=48, unite=U_MOE))
# Le même MoE par le chemin GEMM groupée MMA FP4 (celui du prefill) à t=B jetons :
# quant_act + gate + up + moe_act + quant_act + down + reduce_trie — étape (iii) de Sage
# (revue/sage-moe-mma-decodage-14-09.md). Tuile ACVRAM_MOE_MMA_BT (16 conseillé à M≈3).
if SEUL not in ("pas", "mesure1") and not MARLIN and moe._forward_prefill_grouped(x, topw, topi) is not None:
    # (sous la disposition unique Marlin, `_gemm_mma` lirait une pile rendue : section sautée)
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
        cpg, cpu_, cpd = piles[t_mma() % nc]
        moe._gemm_mma(cpg, xq, xsf, tiles, brut=True); moe._gemm_mma(cpu_, xq, xsf, tiles, brut=True)
        moe._gemm_mma(cpd, aq, asf, tiles, brut=True)
    res.append(mesure("MoE 3 GEMM MMA seules (gate+up+down, BT=%d, %d tuiles)" % (bt, int(tiles[0].numel())), trois_gemm, lots=48, unite=U_MOE))
    res.append(mesure("quant_act x2 (E2M1 bloc 16)", lambda: (ext.nvfp4_quant_act(xs), ext.nvfp4_quant_act(a3)), lots=100, unite="2 quantifications, %d lignes" % (B * moe.top_k)))
    res.append(mesure("MoE complet MMA (_forward_prefill_grouped, BT=%s)" % os.environ.get("ACVRAM_MOE_MMA_BT", "64"),
                      lambda: couches[t_mmac() % nc].mlp._forward_prefill_grouped(x, topw, topi), lots=48, unite=U_MOE))
if SEUL not in ("mma", "pas", "mesure1"):
    if getattr(attn, "qkv_proj", None) is not None:
        res.append(mesure("projections q/k/v int8 empilées (qkv_proj)", lambda: couches[t_qkv() % nc].self_attn.qkv_proj(x), lots=96, unite=U_COUCHE))
    else:
        res.append(mesure("projection q int8", lambda: couches[t_qkv() % nc].self_attn.q_proj(x), lots=96, unite=U_COUCHE))
    xo = torch.randn(B, attn.o_proj.qweight.shape[1] if hasattr(attn.o_proj, "qweight") else 4096, dtype=torch.bfloat16, device=dev)
    res.append(mesure("projection o int8 (o_proj)", lambda: couches[t_o() % nc].self_attn.o_proj(xo), lots=96, unite=U_COUCHE))
    lm = model.lm_head
    res.append(mesure("lm_head int8", lambda: lm(x), lots=50, unite="%d jetons (vocabulaire entier)" % B))
    res.append(mesure("norme RMS (input_layernorm)", lambda: couches[t_norm() % nc].input_layernorm(x), lots=192, unite=U_COUCHE))

# Mesure 1 à unité égale (sage-nsys-coder-c16-mma2-19-09, addendum 20 h 15) : les deux noyaux
# du MoE — mma2 (GEMM groupée MMA, chemin du prefill, M=B lignes) et Marlin GEMV — sur la
# MÊME liste d'experts, à deux unités de distincts par couche (27 = le harnais synthétique
# de ncu M2, 55 = l'uniforme rabattu), t + W + rapport cyclique par noyau → t et J par octet
# à octets égaux. Règle de décision (Sage) : mma2 gagne si W_mma2 × t_mma2 ≤ 0,85 × 398 × t_marlin
# à la même unité. Exige les DEUX dispositions en mémoire : ACVRAM_DOUBLE_DISPOSITION_DIAG=1
# (régime diagnostic, dit sur la ligne de régime) ; sinon refus nommé.
if SEUL == "mesure1":
    # Les deux dispositions ensemble (ACVRAM_DOUBLE_DISPOSITION_DIAG=1) ne tiennent pas avec un
    # processus étranger de 5,6 Gio sur la carte (OOM 19:51) : chaque disposition seule est
    # acceptée, la liste d'experts est identique (générateur figé) et le bilan se fait hors ligne.
    A_MARLIN, A_NAT = marlins[0] is not None, pg[1] is not None
    if not (A_MARLIN or A_NAT):
        raise SystemExit("mesure1 : aucune disposition d'experts")
    import torch.nn.functional as F
    E = len(moe.experts)
    def liste_experts(u):
        """B jetons × top_k, exactement u experts distincts (u ≤ B·top_k), chaque jeton
        8 experts distincts : les u premiers slots reçoivent 0..u-1, les autres rebouclent."""
        assert moe.top_k <= u <= B * moe.top_k and u <= E
        g2 = torch.Generator(device="cpu").manual_seed(11)
        perm = torch.randperm(E, generator=g2)[:u]
        t = torch.empty(B, moe.top_k, dtype=torch.int64)
        n = 0
        for b in range(B):
            for k in range(moe.top_k):
                t[b, k] = perm[n % u]; n += 1
        assert all(len(set(t[b].tolist())) == moe.top_k for b in range(B))
        assert len(set(t.reshape(-1).tolist())) == u
        return t.to(dev)
    unites = [int(v) for v in os.environ.get("BANC_DISTINCTS", "45,27").split(",")]
    bt = int(os.environ.get("ACVRAM_MOE_MMA_BT", "16"))
    t_m1 = {k: Tour() for k in ("mgu", "mdn", "mma_gu", "mma_dn")}
    def octets_par_expert(pile):
        """poids + échelles de bloc d'UN expert, depuis les tenseurs de la pile (E en tête)."""
        return sum(t.numel() * t.element_size() for t in pile if torch.is_tensor(t) and t.dim() >= 2) / E
    o_nat = {n: octets_par_expert(piles[0][i][1:3]) for i, n in enumerate(("gate", "up", "down"))} if A_NAT else {}
    o_mar = {n: octets_par_expert(marlins[0][n + "_proj"][:2]) for n in ("gate", "up", "down")} if A_MARLIN else {}
    print("OCTETS_PAR_EXPERT " + json.dumps({"naturel_Mo": {k: round(v / 1e6, 3) for k, v in o_nat.items()},
                                             "marlin_Mo": {k: round(v / 1e6, 3) for k, v in o_mar.items()}}), flush=True)
    bilan = {}
    for u in unites:
        topi_u = liste_experts(u)
        eid_u = topi_u.reshape(-1).to(torch.int32)
        unite = "1 couche MoE, %d jetons x top_k %d = %d lignes, %d experts DISTINCTS (meme liste pour les 4 noyaux)" % (B, moe.top_k, B * moe.top_k, u)
        rs = {}
        if A_MARLIN:
            # Marlin GEMV (décodage) : gate·up puis down, sur eid_u
            def m_gu(i, eid_u=eid_u):
                mg, mu = marlins[i % nc]["gate_proj"], marlins[i % nc]["up_proj"]
                return ext.nvfp4_gemv_marlin_gateup(mg[0], mg[1], mg[2], mu[0], mu[1], mu[2], eid_u, tok, x, mg[3], mg[4], 0)
            act_u = m_gu(0)[:, :pg[5]].contiguous()
            def m_dn(i, eid_u=eid_u, act_u=act_u):
                md = marlins[i % nc]["down_proj"]
                return ext.nvfp4_gemv_marlin(md[0], md[1], md[2], eid_u, seq, act_u, md[3], md[4])
            rs["r1"] = mesure("M1 u=%d Marlin gate·up GEMV (nvfp4_gemv_marlin_gateup)" % u, lambda: m_gu(t_m1["mgu"]()), lots=48, unite=unite, octets=u * (o_mar["gate"] + o_mar["up"]))
            rs["r2"] = mesure("M1 u=%d Marlin down GEMV (nvfp4_gemv_marlin)" % u, lambda: m_dn(t_m1["mdn"]()), lots=48, unite=unite, octets=u * o_mar["down"])
        if A_NAT:
            # mma2 (GEMM groupée MMA FP4, M=B·top_k lignes triées par expert) : gate+up puis down
            flat_e = topi_u.reshape(-1).to(torch.int64); ordre = torch.argsort(flat_e, stable=True)
            cnt = torch.bincount(flat_e, minlength=E); tiles = moe._tuiles(cnt, bt)
            xs = x[torch.arange(B, device=dev).repeat_interleave(moe.top_k)[ordre]].to(torch.bfloat16).contiguous()
            if xs.shape[1] != pg[4]: xs = F.pad(xs, (0, pg[4] - xs.shape[1])).contiguous()
            from acvram.engine.model import _qa_compteurs
            cpt = _qa_compteurs(xs.device)
            xq, xsf, gr = ext.nvfp4_quant_act(xs, None, None, cpt, 0)
            g_ = moe._gemm_mma(pg, xq, xsf, tiles, brut=True, grow=gr); u_ = moe._gemm_mma(pu, xq, xsf, tiles, brut=True, grow=gr)
            a3 = ext.moe_act(g_, u_, pg[5], pd[4], 0, None, None); aq, asf, gra = ext.nvfp4_quant_act(a3, None, None, cpt, 0)
            def mma_gu(i, xq=xq, xsf=xsf, tiles=tiles, gr=gr):
                cpg, cpu_, _ = piles[i % nc]
                moe._gemm_mma(cpg, xq, xsf, tiles, brut=True, grow=gr); moe._gemm_mma(cpu_, xq, xsf, tiles, brut=True, grow=gr)
            def mma_dn(i, aq=aq, asf=asf, tiles=tiles, gra=gra):
                moe._gemm_mma(piles[i % nc][2], aq, asf, tiles, brut=True, grow=gra)
            nt = int(tiles[0].numel())
            rs["r3"] = mesure("M1 u=%d mma2 gate+up (2 GEMM groupees MMA, BT=%d, %d tuiles)" % (u, bt, nt), lambda: mma_gu(t_m1["mma_gu"]()), lots=48, unite=unite, octets=u * (o_nat["gate"] + o_nat["up"]))
            rs["r4"] = mesure("M1 u=%d mma2 down (GEMM groupee MMA, BT=%d, %d tuiles)" % (u, bt, nt), lambda: mma_dn(t_m1["mma_dn"]()), lots=48, unite=unite, octets=u * o_nat["down"])
            rs["r5"] = mesure("M1 u=%d quant_act x2 (E2M1 bloc 16, entrees de gate/up et de down)" % u, lambda: (ext.nvfp4_quant_act(xs, None, None, cpt, 0), ext.nvfp4_quant_act(a3, None, None, cpt, 0)), lots=100, unite=unite, octets=float(xs.numel() * 2 + a3.numel() * 2))
        if A_MARLIN and hasattr(moe, "_gemm_mma_marlin") and pg[4] % 64 == 0:
            # C17 : mma2 LISANT LES TUILES MARLIN (chantier-c17-mma2-lit-marlin-19-09) — même
            # disposition que la GEMV Marlin, même liste de lignes, mêmes xq/tuiles que le bras
            # naturel ; la ligne t(C17)/t(Marlin) par u est le scellé (2) de C17 (≤ 0,92 à u=45)
            flat_e = topi_u.reshape(-1).to(torch.int64); ordre = torch.argsort(flat_e, stable=True)
            cnt = torch.bincount(flat_e, minlength=E); tiles = moe._tuiles(cnt, bt)
            xs = x[torch.arange(B, device=dev).repeat_interleave(moe.top_k)[ordre]].to(torch.bfloat16).contiguous()
            if xs.shape[1] != pg[4]: xs = F.pad(xs, (0, pg[4] - xs.shape[1])).contiguous()
            from acvram.engine.model import _qa_compteurs
            cpt = _qa_compteurs(xs.device)
            xq, xsf, gr = ext.nvfp4_quant_act(xs, None, None, cpt, 0)
            g_ = moe._gemm_mma_marlin("gate_proj", xq, xsf, tiles, grow=gr, bt=bt)
            u_ = moe._gemm_mma_marlin("up_proj", xq, xsf, tiles, grow=gr, bt=bt)
            a3 = ext.moe_act(g_, u_, pg[5], pd[4], 0, None, None); aq, asf, gra = ext.nvfp4_quant_act(a3, None, None, cpt, 0)
            t_m1.setdefault("mm_gu", Tour()); t_m1.setdefault("mm_dn", Tour())
            def mm_gu(i, xq=xq, xsf=xsf, tiles=tiles, gr=gr):
                c_ = couches[i % nc].mlp
                c_._gemm_mma_marlin("gate_proj", xq, xsf, tiles, grow=gr, bt=bt); c_._gemm_mma_marlin("up_proj", xq, xsf, tiles, grow=gr, bt=bt)
            def mm_dn(i, aq=aq, asf=asf, tiles=tiles, gra=gra):
                couches[i % nc].mlp._gemm_mma_marlin("down_proj", aq, asf, tiles, grow=gra, bt=bt)
            nt = int(tiles[0].numel())
            rs["r6"] = mesure("M1 u=%d mma2-MARLIN gate+up (C17, 2 GEMM sur tuiles Marlin, BT=%d, %d tuiles)" % (u, bt, nt), lambda: mm_gu(t_m1["mm_gu"]()), lots=48, unite=unite, octets=u * (o_mar["gate"] + o_mar["up"]))
            rs["r7"] = mesure("M1 u=%d mma2-MARLIN down (C17)" % u, lambda: mm_dn(t_m1["mm_dn"]()), lots=48, unite=unite, octets=u * o_mar["down"])
            rs["r8"] = mesure("M1 u=%d quant_act x2 (C17, meme cout que le bras naturel)" % u, lambda: (ext.nvfp4_quant_act(xs, None, None, cpt, 0), ext.nvfp4_quant_act(a3, None, None, cpt, 0)), lots=100, unite=unite, octets=float(xs.numel() * 2 + a3.numel() * 2))
            t_c17 = rs["r6"]["us_noyau_par_lancement"] + rs["r7"]["us_noyau_par_lancement"] + rs["r8"]["us_noyau_par_lancement"]
            if "r1" in rs and "r2" in rs:
                t_mar = rs["r1"]["us_noyau_par_lancement"] + rs["r2"]["us_noyau_par_lancement"]
                bilan[("c17", u)] = {"t_marlin_us_couche": round(t_mar, 1), "t_c17_us_couche": round(t_c17, 1),
                                     "t_c17_sur_t_marlin": round(t_c17 / t_mar, 3) if t_mar else None,
                                     "W_c17_max": max(rs["r6"]["W"], rs["r7"]["W"]), "MHz_c17": rs["r6"]["MHz"], "MHz_marlin": rs["r1"]["MHz"],
                                     "scelle_0_92": bool(t_mar and t_c17 <= 0.92 * t_mar)}
                print("BILAN C17 u=%d %s" % (u, json.dumps(bilan[("c17", u)], ensure_ascii=False)), flush=True)
        res += list(rs.values())
        if not (A_MARLIN and A_NAT):
            continue
        r1, r2, r3, r4, r5 = (rs[k] for k in ("r1", "r2", "r3", "r4", "r5"))
        # t par couche (temps noyau, pas la boucle), W max des noyaux d'experts, règle de Sage
        # (sage-c16bis-puissance-mesure1-19-09 § 2) : Mesure 2 si t_mma2 ≤ 1,15 × t_marlin ET W_mma2 ≤ 350
        t_mar = r1["us_noyau_par_lancement"] + r2["us_noyau_par_lancement"]
        t_mma = r3["us_noyau_par_lancement"] + r4["us_noyau_par_lancement"] + r5["us_noyau_par_lancement"]
        w_mma = max(r3["W"], r4["W"])
        cyc_min = min(r["rapport_cyclique"] for r in (r1, r2, r3, r4))
        bilan[u] = {"t_marlin_us_couche": round(t_mar, 1), "t_mma2_us_couche": round(t_mma, 1),
                    "t_mma2_sur_t_marlin": round(t_mma / t_mar, 3) if t_mar else None,
                    "W_marlin_max": max(r1["W"], r2["W"]), "W_mma2_max": w_mma, "rapport_cyclique_min": cyc_min,
                    "J_couche_marlin_mJ": round(sum(r["mJ_par_lancement"] for r in (r1, r2)), 3),
                    "J_couche_mma2_mJ": round(sum(r["mJ_par_lancement"] for r in (r3, r4, r5)), 3),
                    "mesure2": bool(t_mar and t_mma <= 1.15 * t_mar and w_mma <= 350),
                    "cyclique_ok": cyc_min >= 0.9}
        print("BILAN u=%d %s" % (u, json.dumps(bilan[u], ensure_ascii=False)), flush=True)
    print("RESULTAT " + json.dumps({"B": B, "repos_W": round(repos_w, 1), "mode": "mesure1", "distincts": unites, "bilan": {str(k): v for k, v in bilan.items()}, "postes": res}, ensure_ascii=False))
    raise SystemExit(0)

def pas_complet():
    # Les séquences finissent (max_model_len) : réadmettre un lot dès que le moteur se
    # vide, sinon la boucle mesure un moteur au repos (b=1, 14/09 : 114 W, 3,9 µs/pas).
    if not eng.running:
        for b in range(B):
            eng.add_request([(1000 + b * 101 + i * 13) % 150000 + 10 for i in range(128)], SP)
        while any(not s.prefilled for s in eng.running) or eng.waiting:
            eng.step()
    eng.step()
res.append(mesure("PAS COMPLET b=%d rejeu (Engine.step)" % B, pas_complet, lots=10, unite="1 pas de décodage, %d jetons, %d couches" % (B, nc)))
print("RESULTAT " + json.dumps({"B": B, "repos_W": round(repos_w, 1), "postes": res}, ensure_ascii=False))
