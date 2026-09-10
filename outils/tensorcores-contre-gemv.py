#!/usr/bin/env python3
"""Que vaut le chemin tensor cores FP4 a b>8, contre nos noyaux fusionnes ?

**Pourquoi ce banc mesure des NOYAUX et non un debit de bout en bout.** Depuis
le correctif du 10/09, le backend `fp4-tensorcores` (priorite 110) est ecarte
**sous capture** : en decodage de production, chaque pas est capture, donc ce
chemin n'est JAMAIS pris, quelle que soit sa valeur. Un banc de bout en bout ne
pourrait le comparer qu'en mode eager — et le mode eager coute jusqu'a
+11,53 Gio de pic (mesure du 10/09), ce qui melangerait la question posee avec
un tout autre effet. Comparer les deux noyaux sur les MEMES formes repond a la
question sans ce confondant.

Ce que le resultat decide, et rien d'autre :

    ecart sous le bruit  ->  la question est close, on ecarte definitivement,
                             et le troisieme correctif (prechauffer cuBLASLt
                             sur le flux de capture) n'a pas d'objet
    le chemin gagne      ->  alors, et alors seulement, mesurer la QUALITE :
                             a 4 bits la quantification de l'activation ne
                             s'amortit pas sur un jeton decode

ABBA : chaque forme est mesuree A-B-B-A pour que la derive thermique porte
egalement sur les deux chemins. Alterner sans ABBA ne corrige rien (mesure du
9/09 : +3 W en douze passages).
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

LOTS = (1, 8, 9, 12, 16, 24, 32)


def _chrono(fn, n=50, chauffe=10):
    for _ in range(chauffe):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        d = torch.cuda.Event(True), torch.cuda.Event(True)
        d[0].record()
        fn()
        d[1].record()
        torch.cuda.synchronize()
        ts.append(d[0].elapsed_time(d[1]) * 1000.0)      # us
    return statistics.median(ts)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=int, default=4096, help="lignes du poids")
    ap.add_argument("--in", dest="k", type=int, default=2560)
    ap.add_argument("--passages", type=int, default=50)
    ns = ap.parse_args(argv[1:])

    if torch.cuda.device_count() != 1:
        raise SystemExit("CUDA_VISIBLE_DEVICES=0 obligatoire")
    if torch.cuda.get_device_capability(0) < (10, 0):
        raise SystemExit("chemin FP4 tensor cores : sm_100+ requis")

    from acvram.quant.nvfp4 import quantize_nvfp4
    from acvram.kernels import nvfp4_matmul
    from acvram.kernels.fp4_gemm import nvfp4_mm_tensorcore, fp4_mm_available

    if not fp4_mm_available(torch.device("cuda:0")):
        raise SystemExit("chemin FP4 indisponible sur cette carte : rien a comparer")

    d = torch.device("cuda:0")
    w = quantize_nvfp4(torch.randn(ns.out, ns.k, device=d, dtype=torch.float32))
    print("lot\ttc_us\tgemv_us\trapport\tverdict")
    for b in LOTS:
        x = torch.randn(b, ns.k, device=d, dtype=torch.bfloat16)
        # Le GEMV fusionne, sans passer par le registre : on veut CE noyau,
        # pas la resolution de backend qui pourrait rendre l autre.
        gemv = lambda: nvfp4_matmul(x, w)                # noqa: E731
        tc = lambda: nvfp4_mm_tensorcore(x, w)           # noqa: E731
        if tc() is None:
            print(f"{b}\t-\t{_chrono(gemv, ns.passages):.2f}\t-\t"
                  f"chemin tc refuse cette forme")
            continue
        # ABBA
        a1 = _chrono(tc, ns.passages)
        b1 = _chrono(gemv, ns.passages)
        b2 = _chrono(gemv, ns.passages)
        a2 = _chrono(tc, ns.passages)
        t_tc = (a1 + a2) / 2
        t_gv = (b1 + b2) / 2
        # Bruit : l ecart entre les deux mesures d un MEME chemin. Un ecart
        # entre chemins qui n excede pas celui-la ne se publie pas comme un gain.
        bruit = max(abs(a1 - a2), abs(b1 - b2))
        ecart = t_gv - t_tc
        verdict = ("tensor cores" if ecart > bruit else
                   "gemv fusionne" if -ecart > bruit else "SOUS LE BRUIT")
        print(f"{b}\t{t_tc:.2f}\t{t_gv:.2f}\t{t_gv/t_tc:.2f}x\t{verdict}"
              f"\t(bruit {bruit:.2f} us)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
