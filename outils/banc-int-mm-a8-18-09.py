"""Dernière porte P0-a8 (Sage, après verdict-p0-prefill-a8-colle-18-09 : la GEMM
int8 Triton fait 32,1 ms là où déquant + cutlass bf16 font 30,6) : la voie
cuBLASLt int8 — `torch._int_mm` — sur les quatre formes q/k/v/o du préfill
Coder 2 048 (M = 2 048 ; q [2048→4096], k et v [2048→512], o [4096→2048]),
× 48 couches. **Porte : somme ≤ 16 ms ⇒ in situ a8-cublas (2 min de carte,
scellé ≥ 11 000 inchangé) ; > 16 ms ⇒ P0-a8 fermé, pas de troisième noyau.**

Ce qui est mesuré (rejeu de graphe), par forme :
  (b) voie « par canal » : quantification A8 par jeton (Triton, `gemm_w8a8.
      quantifier_a8`) + UN `torch._int_mm` (int8 × int8 → int32 sur K entier)
      + épilogue fp32 (× s_x[m] × s_w[n] → bf16). C'est la SEULE voie où
      `_int_mm` sert : elle exige des poids int8 SYMÉTRIQUES PAR CANAL (une
      échelle par ligne de sortie sur tout K) — pas notre INT8 affine par
      groupes de 128 : une REQUANTIFICATION des q/k/v/o (au chargement ou à la
      conversion) et donc une porte PPL, et une seconde disposition en VRAM
      si le décodage garde la forme par groupes (+1 o/poids : ≈ 1 Go sur Coder).
  (a) voie « par groupe » : NG `_int_mm` de K = 128 par projection + NG
      épilogues — respecte nos poids tels quels ; attendue hors porte
      (96 GEMM à K = 128 par couche), mesurée pour l'écrire.
  témoin : `torch.matmul` bf16 (cutlass) sur les mêmes formes — le 19,4 ms.
Aucun modèle, aucune sortie de modèle. JSON scratchpad/banc-int-mm-a8-18-09.json.

    outils/carte.sh python outils/banc-int-mm-a8-18-09.py [--rapide]
"""
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from acvram.kernels import gemm_w8a8 as W                               # noqa: E402

M, COUCHES, G = 2048, 48, 128
FORMES = [("q_proj", 2048, 4096), ("k_proj", 2048, 512), ("v_proj", 2048, 512), ("o_proj", 4096, 2048)]
REPET = 30 if "--rapide" not in sys.argv else 8


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


def main():
    from acvram.regime import regime_ligne
    print(regime_ligne(), flush=True)
    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(18)
    res = {}
    tot_b = tot_a = tot_t = 0.0
    for nom, K, N in FORMES:
        x = (torch.randn(M, K, generator=gen) * 0.5).to(torch.bfloat16).to(dev)
        # poids int8 symétriques par canal (voie b) et affine par groupes (voie a), formes seules
        w_canal = torch.randint(-127, 128, (K, N), generator=gen, dtype=torch.int8).to(dev)       # [K, N] pour _int_mm
        s_w = (torch.rand(N, generator=gen) * 0.01).to(dev)
        q_grp = torch.randint(0, 256, (N, K), generator=gen, dtype=torch.uint8).to(dev)
        s_grp = (torch.rand(N, K // G, generator=gen) * 0.01).to(torch.float16).to(dev)
        z_grp = torch.randint(0, 256, (N, K // G), generator=gen, dtype=torch.uint8).to(dev)
        w_bf16 = (torch.randn(N, K, generator=gen) * 0.02).to(torch.bfloat16).to(dev)

        def voie_b():
            a8, sx = W.quantifier_a8(x)
            acc = torch._int_mm(a8, w_canal)                                  # int32 [M, N]
            return (acc.to(torch.float32) * sx[:, None] * s_w[None, :]).to(torch.bfloat16)

        def voie_a():
            a8, sx = W.quantifier_a8(x)
            out = torch.zeros(M, N, dtype=torch.float32, device=dev)
            somme_a = a8.view(M, K // G, G).to(torch.int32).sum(-1)              # [M, NG]
            for gi in range(K // G):
                ks = slice(gi * G, (gi + 1) * G)
                wq = (q_grp[:, ks].to(torch.int16) - 128).to(torch.int8).T.contiguous()   # [G, N]
                acc = torch._int_mm(a8[:, ks].contiguous(), wq)
                z = z_grp[:, gi].to(torch.int32) - 128
                out += (acc - somme_a[:, gi:gi + 1] * z[None, :]).to(torch.float32) * s_grp[:, gi].to(torch.float32)[None, :]
            return (out * sx[:, None]).to(torch.bfloat16)

        def temoin():
            return x @ w_bf16.T

        # _int_mm exige M, N, K multiples de 8 et N > 16 : formes Coder OK
        mb, ma, mt = chrono(voie_b), chrono(voie_a), chrono(temoin)
        flop = 2 * M * K * N
        res[nom] = {"K": K, "N": N, "b_ms": mb, "a_ms": ma, "bf16_ms": mt,
                    "b_TOPS": flop / mb / 1e9, "bf16_TFLOPS": flop / mt / 1e9}
        tot_b += mb; tot_a += ma; tot_t += mt
        print(f"{nom:7s} K={K:4d} N={N:4d}  (b) int_mm par canal {mb:.4f} ms ({flop / mb / 1e9:5.0f} TOPS)  "
              f"(a) par groupe {ma:.4f} ms  témoin bf16 {mt:.4f} ms ({flop / mt / 1e9:4.0f} TFLOPS)", flush=True)
    pb, pa, pt = tot_b * COUCHES, tot_a * COUCHES, tot_t * COUCHES
    verdict = ("≤ 16 ms : in situ a8-cublas (2 min, scellé ≥ 11 000) — sous réserve de la requantification par canal (porte PPL)"
               if pb <= 16 else "> 16 ms : P0-a8 fermé, pas de troisième noyau")
    print(f"\npar préfill (× {COUCHES}) : (b) int_mm par canal {pb:.1f} ms · (a) par groupe {pa:.1f} ms · témoin bf16 {pt:.1f} ms "
          f"(déquant 11,2 + cutlass 19,4 = 30,6 mesurés en situ)\nverdict (b) : {verdict}")
    out = os.path.join(os.path.dirname(__file__), "..", "scratchpad", "banc-int-mm-a8-18-09.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"date": time.strftime("%Y-%m-%d %H:%M"), "carte": torch.cuda.get_device_name(0), "regime": regime_ligne(),
               "repet": REPET, "formes": res, "ms_prefill_b": pb, "ms_prefill_a": pa, "ms_prefill_bf16": pt, "verdict": verdict},
              open(out, "w"), indent=1, ensure_ascii=False)
    print("JSON :", os.path.relpath(out))


if __name__ == "__main__":
    main()
