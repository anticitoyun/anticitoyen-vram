"""Disposition unique Marlin servie AUSSI au décodage (poste7-p1-disposition-
unique-18-09 : le chemin vLLM, fused_marlin_moe) — banc des formes de
décodage Coder : b = 12 (96 paires, ~69 experts distincts) et b = 1 (8
paires), E 128, top_k 8, K 2 048, I 768, gate + up + act·up + down, × 48
couches, SOUS GRAPHE CUDA (aligneur capturable en un lancement Triton :
`marlin_port.aligner_blocs_capturable`), contre le GEMV actuel (v1, rpw et
xreg aux défauts). Sortie de chaque bras jugée contre fp32 : |Δ| ≤ 2⁻⁷·max|y|
par ligne (part hors ≤ 5·10⁻⁴), le GEMV aussi (la référence contre elle-même).

Bandes (poste7) : Marlin b=12 ≤ 5,5 ms/pas ET b=1 ≤ 1,0 → disposition unique,
P1 continue à 0 octet ; 5,5-6,5 ou 1,0-1,2 → juge = cellule complète t/s + J,
une passe ; > 6,5 ou > 1,2 → forme (b) GEMV relisant la disposition Marlin
(2 j, bit-exact) ; si (b) échoue, P1 fermé VRAM. Prédiction poste7 : b=12
4,6-5,4 ms, b=1 0,9-1,1. Routages : tirés au sort (top-8 uniformes) — les
routages réels d'un modèle chargé ne sont pas disponibles à sec ; la
répartition (experts distincts) est imprimée.

    CUDA_VISIBLE_DEVICES="" python outils/banc-marlin-p1-18-09.py --compiler-seulement   # .so à sec d'abord
    outils/carte.sh python outils/banc-marlin-decode-18-09.py [--rapide] [--routages fichier.pt]

Bras (b) (poste7-p1-disposition-unique-18-09, GO poste7 18/09 sur le chiffrage
revue/p1-disposition-unique-chiffrage-18-09) : `nvfp4_gemv_marlin[_gateup]`,
le GEMV lisant la disposition Marlin. Scellé (poste7) : (b) ≤ 0,97 × GEMV à
b = 12 ET à b = 1, chaque chemin contre fp32 (part hors 2⁻⁷ ≤ 5·10⁻⁴ ou ≤ celle
du GEMV) ; faux → P1 fermé VRAM, verdict daté. Prédiction poste4 (à sec,
SASS : (b) 8,9 instr/octet contre ≈ 8,5 pour v1, mais 4 LDS par 64 o au lieu
de 128 — le poste ncu) : b = 12 5,0-6,5 ms/pas, b = 1 2,2-2,9 ms.
`--routages fichier.pt` : liste de tenseurs [B, top_k] (ACVRAM_TRACE_ROUTAGE_PT
sur un décodage réel) rejoués à la place du tirage uniforme.
"""
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from acvram.kernels import get_extension, marlin_port as MP                 # noqa: E402
from acvram.quant.nvfp4 import dequantize_nvfp4, quantize_nvfp4              # noqa: E402

E, TOPK, K, I, COUCHES = 128, 8, 2048, 768, 48
REPET = 8 if "--rapide" in sys.argv else 30
ROUTAGES = 5 if "--rapide" in sys.argv else 20


def chrono(f):
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
    for _ in range(REPET):
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


def hors_par_ligne(y, ref):
    return int(((y.float() - ref).abs() > 2 ** -7 * ref.abs().amax(1, keepdim=True)).sum())


# Biais (poste7-p1-situ-verdict-18-09, REGLES § 4 bis : un comptage ne détecte
# pas un biais) : moyenne signée de Δ rapportée à moy|y| (critère poste7 ≤ 1e-4)
# ET gain − 1 = Σ y·ref / Σ ref² − 1 — le biais MULTIPLICATIF, que la moyenne
# signée ne voit pas sur une sortie centrée (un MoE l'est) ; le témoin négatif
# (échelles de bloc tronquées d'un bit : toujours ≤ la vraie) doit le rendre.
SEUIL_BIAIS = 1e-4


def biais(y, ref):
    d = y.float() - ref
    return float(d.mean() / ref.abs().mean()), float((y.float() * ref).sum() / (ref * ref).sum() - 1.0)


def main():
    from acvram.regime import regime_ligne
    ext = get_extension()
    assert ext is not None
    t0 = time.time(); assert MP.charger(compiler=False) is not None, "extension Marlin non compilée à sec"
    print(f"{regime_ligne()} marlin_port_so={MP.sha_so()} marlin_source={MP.VERSION_SOURCE} (chargée en {time.time() - t0:.0f} s)", flush=True)
    dev = torch.device("cuda")
    qg, bg, gsg, tsg = pile(I, K, 1, dev); qu, bu, gsu, tsu = pile(I, K, 2, dev); qd, bd, gsd, tsd = pile(K, I, 3, dev)
    mg = MP.preparer_pile(qg, bg, gsg); mu = MP.preparer_pile(qu, bu, gsu); md = MP.preparer_pile(qd, bd, gsd)
    # le GEMV lit les échelles de bloc en OCTETS (uint8), Marlin en Float8_e4m3fn
    # (« expected scalar type Byte but found Float8_e4m3fn », poste3) : deux vues
    bg8, bu8, bd8 = (b.view(torch.uint8).contiguous() for b in (bg, bu, bd))
    bg_t, bu_t, bd_t = ((b.view(torch.uint8) & 0xFE).contiguous() for b in (bg, bu, bd))    # témoin négatif
    ws = MP.espace_travail(dev, 4)
    wg32 = [dequantize_nvfp4(t, torch.float32).to(dev) for t in tsg]
    wu32 = [dequantize_nvfp4(t, torch.float32).to(dev) for t in tsu]
    wd32 = [dequantize_nvfp4(t, torch.float32).to(dev) for t in tsd]
    gen = torch.Generator().manual_seed(18)
    reels = None
    if "--routages" in sys.argv:
        reels = [t for t in torch.load(sys.argv[sys.argv.index("--routages") + 1]) if t.shape[1] == TOPK]
        print(f"routages réels : {len(reels)} appels rejoués", flush=True)
    forme_b = hasattr(ext, "nvfp4_gemv_marlin_gateup")
    res = {}
    for B in (12, 1):
        G = B * TOPK
        bloc = MP.choisir_block_size(B, TOPK, E)
        lignes = []
        for r in range(ROUTAGES):
            if reels is not None:
                cand = [t for t in reels if t.shape[0] >= B]
                topi = cand[r % len(cand)][:B].to(torch.long).to(dev)
            else:
                topi = torch.stack([torch.randperm(E, generator=gen)[:TOPK] for _ in range(B)]).to(dev)
            topw = torch.rand(B, TOPK, generator=gen).to(dev); topw = (topw / topw.sum(-1, keepdim=True)).float()
            x = (torch.randn(B, K, generator=gen) * 0.5).to(torch.bfloat16).to(dev)
            eid = topi.reshape(-1).to(torch.int32)
            tok = torch.arange(B, dtype=torch.int32, device=dev).repeat_interleave(TOPK)
            seq = torch.arange(G, dtype=torch.int32, device=dev)
            tampons = MP.aligner_blocs_capturable(eid, bloc, E)
            uns = torch.ones(G, 1, dtype=torch.float32, device=dev)
            c1 = torch.empty(G, I, dtype=torch.bfloat16, device=dev); c2 = torch.empty_like(c1); c3 = torch.empty(G, K, dtype=torch.bfloat16, device=dev)

            def marlin():
                s_ids, e_ids, n_post = MP.aligner_blocs_capturable(eid, bloc, E, tampons)     # dans le graphe
                g = MP.gemm_moe(x, mg[0], mg[1], mg[2], s_ids, e_ids, n_post, topw, bloc, TOPK, B, I, K, ws, c=c1)
                u = MP.gemm_moe(x, mu[0], mu[1], mu[2], s_ids, e_ids, n_post, topw, bloc, TOPK, B, I, K, ws, c=c2)
                act = torch.nn.functional.silu(g) * u
                return MP.gemm_moe(act, md[0], md[1], md[2], s_ids, e_ids, n_post, uns, bloc, 1, G, K, I, ws, c=c3)

            def gemv():
                act = ext.nvfp4_gemv_grouped_gateup(qg, bg8, gsg, qu, bu8, gsu, eid, tok, x, K, 0)
                return ext.nvfp4_gemv_grouped(qd, bd8, gsd, eid, seq, act, I)[:, :K]

            def forme_b_fn():
                act = ext.nvfp4_gemv_marlin_gateup(*mg, *mu, eid, tok, x, K, I, 0)
                return ext.nvfp4_gemv_marlin(*md, eid, seq, act, I, K)

            def temoin_negatif():
                act = ext.nvfp4_gemv_grouped_gateup(qg, bg_t, gsg, qu, bu_t, gsu, eid, tok, x, K, 0)
                return ext.nvfp4_gemv_grouped(qd, bd_t, gsd, eid, seq, act, I)[:, :K]

            # référence fp32 par paire
            xf = x.float(); ref = torch.empty(G, K, dtype=torch.float32, device=dev)
            for p in range(G):
                e, t = int(eid[p]), int(tok[p])
                a = torch.nn.functional.silu(xf[t] @ wg32[e].T) * (xf[t] @ wu32[e].T)
                ref[p] = (a.to(torch.bfloat16).float()) @ wd32[e].T          # act bf16 à l'entrée de down, comme les deux bras
            ym, yg = marlin(), gemv()
            hm, hg = hors_par_ligne(ym.float(), ref), hors_par_ligne(yg.float(), ref)
            ms_m, ms_g = chrono(marlin), chrono(gemv)
            ligne = {"routage": r, "distincts": int(torch.unique(eid).numel()), "marlin_ms": ms_m, "gemv_ms": ms_g,
                     "marlin_hors": hm, "gemv_hors": hg, "n": ref.numel(),
                     "marlin_biais": biais(ym, ref), "gemv_biais": biais(yg, ref), "temoin_biais": biais(temoin_negatif(), ref)}
            if forme_b:
                yb = forme_b_fn(); ligne["b_hors"] = hors_par_ligne(yb.float(), ref); ligne["b_ms"] = chrono(forme_b_fn)
                ligne["b_determ"] = bool(torch.equal(yb, forme_b_fn())); ligne["b_biais"] = biais(yb, ref)
            lignes.append(ligne)
            print(f"b={B:2d} routage {r:2d} distincts={ligne['distincts']:3d} marlin {ms_m:.4f} ms gemv {ms_g:.4f} ms  "
                  + (f"(b) {ligne['b_ms']:.4f} ms  " if forme_b else "")
                  + f"hors 2⁻⁷ par ligne : marlin {hm} gemv {hg}" + (f" (b) {ligne['b_hors']}" if forme_b else "")
                  + f" / {ref.numel()}  biais(signé, gain−1) : marlin {ligne['marlin_biais'][0]:+.1e}/{ligne['marlin_biais'][1]:+.1e}"
                  + f" gemv {ligne['gemv_biais'][0]:+.1e}/{ligne['gemv_biais'][1]:+.1e}"
                  + (f" (b) {ligne['b_biais'][0]:+.1e}/{ligne['b_biais'][1]:+.1e}" if forme_b else "")
                  + f" témoin {ligne['temoin_biais'][0]:+.1e}/{ligne['temoin_biais'][1]:+.1e}", flush=True)
        med_m = statistics.median(l["marlin_ms"] for l in lignes) * COUCHES
        med_g = statistics.median(l["gemv_ms"] for l in lignes) * COUCHES
        exact = all(l["marlin_hors"] <= 5e-4 * l["n"] and l["gemv_hors"] <= 5e-4 * l["n"] for l in lignes)
        res[B] = {"marlin_ms_pas": med_m, "gemv_ms_pas": med_g, "exact": exact, "lignes": lignes}
        # témoin négatif : s'il passe le critère de biais, l'instrument est aveugle
        res[B]["temoin_vu"] = all(abs(l["temoin_biais"][1]) > SEUIL_BIAIS for l in lignes)
        res[B]["marlin_sans_biais"] = all(max(abs(v) for v in l["marlin_biais"]) <= SEUIL_BIAIS for l in lignes)
        res[B]["gemv_sans_biais"] = all(max(abs(v) for v in l["gemv_biais"]) <= SEUIL_BIAIS for l in lignes)
        if forme_b:
            res[B]["b_ms_pas"] = statistics.median(l["b_ms"] for l in lignes) * COUCHES
            res[B]["b_exact"] = all(l["b_hors"] <= max(5e-4 * l["n"], l["gemv_hors"]) and l["b_determ"] for l in lignes)
            res[B]["b_sans_biais"] = all(max(abs(v) for v in l["b_biais"]) <= SEUIL_BIAIS for l in lignes)
        print(f"b={B} : biais ≤ 1e-4 (signé et gain) — marlin {res[B]['marlin_sans_biais']} gemv {res[B]['gemv_sans_biais']}"
              + (f" (b) {res[B]['b_sans_biais']}" if forme_b else "") + f" · témoin négatif vu {res[B]['temoin_vu']}", flush=True)
        print(f"b={B} : Marlin {med_m:.2f} ms/pas (48 couches) · GEMV {med_g:.2f} · exact {exact}"
              + (f" · (b) {res[B]['b_ms_pas']:.2f} ms/pas, exact {res[B]['b_exact']}" if forme_b else ""), flush=True)
    m12, m1 = res[12]["marlin_ms_pas"], res[1]["marlin_ms_pas"]
    if not (res[12]["exact"] and res[1]["exact"]):
        verdict = "SORTIE HORS CRITÈRE : ne compte pas"
    elif m12 <= 5.5 and m1 <= 1.0:
        verdict = "disposition unique : P1 continue à 0 octet"
    elif m12 <= 6.5 and m1 <= 1.2:
        verdict = "zone grise (5,5-6,5 ou 1,0-1,2) : juge = cellule complète t/s + J, une passe"
    else:
        verdict = "> 6,5 ou > 1,2 : forme (b), GEMV relisant la disposition Marlin (2 j, bit-exact)"
    print(f"\nverdict Marlin décodage : b=12 {m12:.2f} ms, b=1 {m1:.2f} ms → {verdict}")
    if forme_b:
        b12, b1, g12, g1 = res[12]["b_ms_pas"], res[1]["b_ms_pas"], res[12]["gemv_ms_pas"], res[1]["gemv_ms_pas"]
        if not (res[12]["temoin_vu"] and res[1]["temoin_vu"]):
            vb = "INSTRUMENT AVEUGLE : le témoin négatif passe le critère de biais — ne compte pas"
        elif not (res[12]["b_exact"] and res[1]["b_exact"]):
            vb = "SORTIE HORS CRITÈRE ou non déterministe : ne compte pas"
        elif not (res[12]["b_sans_biais"] and res[1]["b_sans_biais"]):
            vb = "BIAIS (|moy Δ| ou |gain−1| > 1e-4 × moy|y|) : ne compte pas"
        elif b12 <= 0.97 * g12 and b1 <= 0.97 * g1:
            vb = "scellé TENU (≤ 0,97 × GEMV à b=12 et b=1) : disposition unique, P1 continue"
        else:
            vb = "scellé FAUX : P1 fermé VRAM (verdict daté, ligne utilisateur)"
        print(f"verdict forme (b) : b=12 {b12:.2f} ms ({b12 / g12:.3f} × GEMV), b=1 {b1:.2f} ms ({b1 / g1:.3f} × GEMV) → {vb}")
        verdict = verdict + " | (b) : " + vb
    out = os.path.join(os.path.dirname(__file__), "..", "scratchpad", "banc-marlin-decode-18-09.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"date": time.strftime("%Y-%m-%d %H:%M"), "carte": torch.cuda.get_device_name(0), "regime": regime_ligne(),
               "marlin_port_so_sha256": MP.sha_so(), "repet": REPET, "resultats": {str(k): v for k, v in res.items()},
               "verdict": verdict}, open(out, "w"), indent=1, ensure_ascii=False)
    print("JSON :", os.path.relpath(out))


if __name__ == "__main__":
    main()
