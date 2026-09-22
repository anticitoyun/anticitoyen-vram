"""Porte P1 (sage-reprise-ordre-18-09 § 4 (a), corrigée note-marlin-classe-a-sec-
18-09) : GEMM groupée W4A16 classe Marlin (port vLLM v0.29.0,
acvram/kernels/marlin_port) sur les formes du préfill Coder 2 048 — noyau
SEUL, sans moteur : E = 128 experts, top_k 8, K 2 048, N 768 (gate+up en une
pile N = 1 536, down N = 2 048 / K 768), T = 16 384 lignes routées, 48 couches.

Verdict X = TFLOPS effectifs des trois GEMM (7,43 TFLOP par préfill) en rejeu
de graphe, jugé exact contre la déquantification (2⁻⁷ × Σ|x·w|) ; trois
bandes (Sage) : X ≥ 137 → in situ, scellé parité ≥ 15 700 j/s ; 110 ≤ X < 137
→ in situ une passe, ≥ 0,95 × formule(X), publié « gain sous parité » ;
X < 110 → fermé sans carte. Formule scellée : j/s = 2048 / (0,076 + 7,43/X).
Témoin : déquantification bf16 + GEMM cutlass par expert (la classe B0).
GLM-4.7-Flash : même formule sur ses postes (7,11 TFLOP d'experts routés,
part fixe à donner : --glm-fixe-ms, sinon trois hypothèses).

    CUDA_VISIBLE_DEVICES="" python outils/banc-marlin-p1-18-09.py --compiler-seulement   # AVANT la carte (nvcc seul)
    outils/carte.sh python outils/banc-marlin-p1-18-09.py [--rapide] [--glm-fixe-ms 80]

REGLES § 6 : le JIT ne se fait jamais sous le verrou ni sous capture — le
binaire est compilé à sec, le banc le charge depuis le cache (sinon il se
déclare invalide), chauffe en eager avant tout rejeu, et vérifie que le
binaire n'a pas changé. En-tête (INDEX) : régime + sha256 du .so + version
de la source vLLM portée.
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from acvram.kernels import marlin_port as MP                                # noqa: E402
from acvram.quant.nvfp4 import dequantize_nvfp4, quantize_nvfp4, NVFP4Tensor  # noqa: E402

E, TOPK, T_JETONS, K, N, COUCHES = 128, 8, 2048, 2048, 768, 48
FLOP_PAS = 2 * T_JETONS * TOPK * K * (2 * N) * COUCHES + 2 * T_JETONS * TOPK * N * K * COUCHES   # 7,43e12
FIXE_S = 0.076


def chrono(f, repet):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2):
            f()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repet):
        d, a = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        d.record(); g.replay(); a.record(); torch.cuda.synchronize()
        ts.append(d.elapsed_time(a))
    return statistics.median(ts)


def pile(n, k, seed, dev):
    g = torch.Generator().manual_seed(seed)
    ts = [quantize_nvfp4((torch.randn(n, k, generator=g) * 0.02).to(torch.bfloat16)) for _ in range(E)]
    qw = torch.stack([t.qweight for t in ts]).contiguous().to(dev)
    bs = torch.stack([t.block_scale for t in ts]).contiguous().to(dev)
    gs = torch.stack([t.global_scale.float().reshape(()) for t in ts]).contiguous().to(dev)
    return qw, bs, gs, ts


def dequant_experts(ts, dev):
    return torch.stack([dequantize_nvfp4(t, torch.bfloat16) for t in ts]).to(dev)      # [E, n, k] bf16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapide", action="store_true")
    ap.add_argument("--glm-fixe-ms", type=float, default=None)
    ap.add_argument("--lignes-juge", type=int, default=512)
    ap.add_argument("--compiler-seulement", action="store_true",
                    help="compile l'extension (nvcc, sans carte : CUDA_VISIBLE_DEVICES=\"\") et sort — à faire AVANT la prise de carte")
    args = ap.parse_args()
    if args.compiler_seulement:
        t0 = time.time(); MP.charger(verbose=False)
        print(f"extension Marlin compilée/chargée en {time.time() - t0:.0f} s ; .so {MP.chemin_so()} sha256 {MP.sha_so()} ; "
              f"source {MP.VERSION_SOURCE}")
        return
    repet = 8 if args.rapide else 30
    dev = torch.device("cuda")
    from acvram.regime import regime_ligne
    # en-tête (INDEX) : régime, sha256 du binaire, version de la source portée
    t0 = time.time(); MP.charger(verbose=False); dt = time.time() - t0
    entete = f"{regime_ligne()} marlin_port_so={MP.sha_so()} marlin_source={MP.VERSION_SOURCE}"
    print(entete, flush=True)
    print(f"extension Marlin chargée en {dt:.0f} s ({'COMPILÉE DANS CE PROCESSUS — banc invalide, REGLES § 6' if MP.COMPILE_ICI else 'depuis le cache'})", flush=True)
    sha_avant = MP.sha_so()

    qw13, bs13, gs13, ts13 = pile(2 * N, K, 1, dev)          # gate+up empilés : [E, 1536, 2048]
    qw2, bs2, gs2, ts2 = pile(K, N, 2, dev)                  # down : [E, 2048, 768]
    t0 = time.time()
    w13m, s13m, g13m = MP.preparer_pile(qw13, bs13, gs13)
    w2m, s2m, g2m = MP.preparer_pile(qw2, bs2, gs2)
    print(f"repack Marlin des deux piles : {time.time() - t0:.1f} s ; w13 {tuple(w13m.shape)} {w13m.dtype}, "
          f"s13 {tuple(s13m.shape)} {s13m.dtype}", flush=True)

    g = torch.Generator().manual_seed(18)
    topk_ids = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(T_JETONS)]).to(dev)
    topk_w = torch.rand(T_JETONS, TOPK, generator=g).to(dev)
    topk_w = (topk_w / topk_w.sum(-1, keepdim=True)).float()
    x = (torch.randn(T_JETONS, K, generator=g) * 0.5).to(torch.bfloat16).to(dev)
    block = MP.choisir_block_size(T_JETONS, TOPK, E)
    sorted_ids, expert_ids, num_post = MP.aligner_blocs(topk_ids, block, E)
    ws = MP.espace_travail(dev, 4)
    print(f"routage : {int(torch.unique(topk_ids).numel())}/{E} experts, {T_JETONS * TOPK} paires, "
          f"block_size_m {block}, {int(num_post)} lignes rembourrées", flush=True)
    c1 = torch.empty(T_JETONS * TOPK, 2 * N, dtype=torch.bfloat16, device=dev)
    c3 = torch.empty(T_JETONS * TOPK, K, dtype=torch.bfloat16, device=dev)

    def marlin():
        h = MP.gemm_moe(x, w13m, s13m, g13m, sorted_ids, expert_ids, num_post, topk_w, block, TOPK,
                        T_JETONS, 2 * N, K, ws, mul_topk_weights=False, c=c1)
        h = h.view(-1, 2 * N)
        act = torch.nn.functional.silu(h[:, :N]) * h[:, N:]
        return MP.gemm_moe(act, w2m, s2m, g2m, sorted_ids, expert_ids, num_post, topk_w, block, 1,
                           T_JETONS * TOPK, K, N, ws, mul_topk_weights=True, c=c3)

    # témoin : déquantification bf16 + GEMM cutlass par expert (classe B0), même routage
    w13d, w2d = dequant_experts(ts13, dev), dequant_experts(ts2, dev)
    plat = topk_ids.reshape(-1)
    ordre = torch.argsort(plat, stable=True)
    comptes = torch.bincount(plat, minlength=E).tolist()
    xs = x[ordre // TOPK]
    tw = topk_w.reshape(-1)[ordre].unsqueeze(1)

    def temoin():
        out = torch.empty(T_JETONS * TOPK, K, dtype=torch.bfloat16, device=dev)
        d = 0
        for e in range(E):
            n = comptes[e]
            if n == 0:
                continue
            h = xs[d:d + n] @ w13d[e].T
            act = torch.nn.functional.silu(h[:, :N]) * h[:, N:]
            out[d:d + n] = (act @ w2d[e].T) * tw[d:d + n].to(torch.bfloat16)
            d += n
        return out

    # juge : sortie Marlin contre la référence fp32 (déquant fp32) sur les premières lignes triées
    y = marlin()
    ref_rows = args.lignes_juge
    y_ref = torch.empty(ref_rows, K, dtype=torch.float32, device=dev)
    borne = torch.empty(ref_rows, K, dtype=torch.float32, device=dev)
    d = 0
    for e in range(E):
        n = comptes[e]
        if d >= ref_rows or n == 0:
            d += n; continue
        m = min(n, ref_rows - d)
        w13f = dequantize_nvfp4(ts13[e], torch.float32).to(dev); w2f = dequantize_nvfp4(ts2[e], torch.float32).to(dev)
        xe = xs[d:d + m].float()
        h = xe @ w13f.T
        act = torch.nn.functional.silu(h[:, :N]) * h[:, N:]
        y_ref[d:d + m] = (act @ w2f.T) * tw[d:d + m]
        borne[d:d + m] = (act.abs() @ w2f.abs().T) * tw[d:d + m]
        d += m
    y_tri = y.float()[ordre[:ref_rows]]                       # la sortie Marlin est en ordre de paire
    hors = int(((y_tri - y_ref).abs() > 2 ** -7 * borne).sum())
    ecart_max = ((y_tri - y_ref).abs() / borne.clamp_min(1e-6)).max().item()
    print(f"juge : {hors} valeurs hors 2⁻⁷ × Σ|x·w| sur {ref_rows}×{K} ; écart max {ecart_max:.2e} (2⁻⁷ = 7.8e-3)", flush=True)

    # garde b (REGLES § 6) : appel eager chauffé AVANT tout rejeu (chrono le
    # fait : 3 appels eager, 2 sur flux annexe, puis capture) ; le binaire ne
    # doit pas changer pendant le banc et n'avoir pas été compilé ici
    ms_marlin = chrono(marlin, repet)
    ms_temoin = chrono(temoin, max(3, repet // 3))
    jit_invalide = MP.COMPILE_ICI or MP.sha_so() != sha_avant
    X = FLOP_PAS / (ms_marlin * COUCHES * 1e-3) / 1e12
    X_temoin = FLOP_PAS / (ms_temoin * COUCHES * 1e-3) / 1e12
    js = lambda tf: 2048 / (FIXE_S + 7.43 / tf)
    if jit_invalide:
        verdict = "BANC INVALIDE : extension compilée pendant le banc ou binaire changé (REGLES § 6) — recompiler à sec puis rejouer"
    elif hors:
        verdict = "SORTIE FAUSSE : ne compte pas"
    elif X >= 137:
        verdict = "≥ 137 : in situ, scellé parité ≥ 15 700 j/s"
    elif X >= 110:
        verdict = f"110-137 : in situ une passe, scellé ≥ 0,95 × {js(X):.0f} = {0.95 * js(X):.0f} j/s, « gain sous parité »"
    else:
        verdict = "< 110 : fermé sans carte"
    print(f"\nMarlin : {ms_marlin:.3f} ms/couche × 48 = {ms_marlin * 48:.1f} ms/pas → X = {X:.1f} TFLOPS ; "
          f"formule : {js(X):.0f} j/s (Coder 2048)\n"
          f"témoin déquant+cutlass : {ms_temoin:.3f} ms/couche → {X_temoin:.1f} TFLOPS\n"
          f"verdict : {verdict}", flush=True)
    # GLM-4.7-Flash : 46 couches MoE, E 64, top_k 4, H 2048, I_moe 1536 → 7,11 TFLOP routés
    glm_tflop = 2 * 2048 * 4 * (3 * 1536 * 2048) * 46 / 1e12
    fixes = [args.glm_fixe_ms] if args.glm_fixe_ms else [60.0, 80.0, 100.0]
    for f in fixes:
        print(f"GLM (7,11 TFLOP, part fixe {f:.0f} ms{' hypothèse' if not args.glm_fixe_ms else ''}) : "
              f"{2048 / (f / 1000 + glm_tflop / X):.0f} j/s à X = {X:.0f}")
    out = os.path.join(os.path.dirname(__file__), "..", "scratchpad", "banc-marlin-p1-18-09.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"date": time.strftime("%Y-%m-%d %H:%M"), "carte": torch.cuda.get_device_name(0), "regime": regime_ligne(),
               "entete": entete, "marlin_port_so_sha256": sha_avant, "marlin_source": MP.VERSION_SOURCE,
               "jit_invalide": jit_invalide,
               "repet": repet, "block_size_m": block, "ms_couche_marlin": ms_marlin, "ms_pas_marlin": ms_marlin * 48,
               "X_tflops": X, "X_temoin": X_temoin, "hors_2m7": hors, "ecart_max": ecart_max,
               "formule_js": js(X), "verdict": verdict, "glm_tflop": glm_tflop}, open(out, "w"), indent=1, ensure_ascii=False)
    print("JSON :", os.path.relpath(out))


if __name__ == "__main__":
    main()
