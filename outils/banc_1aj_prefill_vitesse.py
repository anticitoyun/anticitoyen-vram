#!/usr/bin/env python3
"""Bead 1aj, volet A (vitesse seule, qualité ignorée) -- prédiction scellée
dans acvram-memoire/revue/1aj-prefill-w4a4-14-09.md, à lire avant ce script.

Micro-banc sur les VRAIS poids nvfp4 du modèle chargé (pas de tenseurs
synthétiques) : pour q/k/v/o de chaque couche + lm_head, chronomètre le
chemin ACTUEL (kernels.matmul, registre) contre le chemin NOUVEAU
(nvfp4_quant_act + nvfp4_gemm_grouped_mma, E=1 -- le noyau écrit main, PAS
torch._scaled_mm, cf. la note du 10/09 dans kernels/__init__.py).

    outils/carte.sh .venv/bin/python outils/banc_1aj_prefill_vitesse.py
"""
import os
import sys
import time

_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
sys.path.insert(0, _REPO)

import torch

from acvram import kernels
from acvram.engine.loader import load_model
from acvram.engine.model import MoEBlock
# Pas d'Engine construit ici (appel direct aux noyaux sur les poids charges) :
# exiger_regime_nominal ne s'applique pas, rien a regarder.

MODEL = "/mnt/2TO_2023_980PRO/Modeles/models_acvram/Qwen3-Coder-30B-A3B-nvfp4"
G = 2048          # pp2048, comme la mesure de départ (poste4:4346)
REP = 5
BT, ETAGES, KS = 16, 4, 64


def _projections(model):
    """(nom, QuantLinear) pour q/k/v/o de chaque couche + lm_head, nvfp4 SEUL
    (pas int8 promu, pas de pile à échelle par segment -- meme garde que
    nvfp4_mm_tensorcore, kernels/fp4_gemm.py:172)."""
    for i, layer in enumerate(model.layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        for nom in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(attn, nom, None)
            if lin is None:
                continue
            w = lin.qweight
            if getattr(w, "format", None) == "nvfp4" and getattr(w, "global_scale_rows", None) is None:
                yield f"L{i}.{nom}", lin
    lin = getattr(model, "lm_head", None)
    if lin is not None:
        w = getattr(lin, "qweight", None)
        if getattr(w, "format", None) == "nvfp4" and getattr(w, "global_scale_rows", None) is None:
            yield "lm_head", lin


def _chrono(fn, rep=REP):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rep


def main():
    loaded = load_model(MODEL, dtype=torch.bfloat16, device_override="cuda:0")
    ext = kernels.get_extension()
    if ext is None or not hasattr(ext, "nvfp4_gemm_grouped_mma"):
        print("ÉCHEC : extension CUDA indisponible ou nvfp4_gemm_grouped_mma absent")
        sys.exit(1)

    dev = torch.device("cuda:0")
    cnt = torch.tensor([G], dtype=torch.int64, device=dev)
    tiles = MoEBlock._tuiles(cnt, bt=BT)

    n = 0
    t_actuel = t_nouveau = 0.0
    ecarts = []
    for nom, lin in _projections(loaded.model):
        w = lin.qweight
        x = torch.randn(G, lin.in_features, dtype=torch.bfloat16, device=dev)
        x = x * 0.02  # échelle d'activation réaliste (pas de saturation E2M1)

        dt_a = _chrono(lambda: kernels.matmul(x, w))

        table_qw = torch.tensor([w.qweight.data_ptr()], dtype=torch.int64, device=dev)
        table_bs = torch.tensor([w.block_scale.data_ptr()], dtype=torch.int64, device=dev)
        gscales = torch.tensor([w.global_scale_float()], dtype=torch.float32, device=dev)

        def nouveau():
            xq, xsf = ext.nvfp4_quant_act(x)
            return ext.nvfp4_gemm_grouped_mma(
                table_qw, table_bs, gscales, xq, xsf,
                tiles[0], tiles[1], tiles[2],
                lin.out_features, lin.in_features, BT, ETAGES, KS)

        dt_n = _chrono(nouveau)

        y_a = kernels.matmul(x, w)
        xq, xsf = ext.nvfp4_quant_act(x)
        y_n = ext.nvfp4_gemm_grouped_mma(table_qw, table_bs, gscales, xq, xsf,
                                         tiles[0], tiles[1], tiles[2],
                                         lin.out_features, lin.in_features, BT, ETAGES, KS)
        y_n = y_n[:, :lin.out_features]
        ecart = (y_a - y_n).abs().mean().item() / (y_a.abs().mean().item() + 1e-9)
        ecarts.append(ecart)

        t_actuel += dt_a
        t_nouveau += dt_n
        n += 1
        print(f"  {nom:16s} actuel={dt_a*1000:7.3f} ms  nouveau={dt_n*1000:7.3f} ms  "
              f"ecart_relatif={ecart:.4f}", flush=True)

    print(f"\n{n} projections nvfp4 mesurées (q/k/v/o + lm_head)")
    print(f"SOMME actuel  : {t_actuel*1000:.2f} ms")
    print(f"SOMME nouveau : {t_nouveau*1000:.2f} ms")
    print(f"écart relatif moyen (nouveau vs actuel, sanité seulement, pas la mesure de qualité du volet B) : "
          f"{sum(ecarts)/len(ecarts):.4f}")
    print(f"\nDépart (poste4:4346) : 30 ms")
    print(f"Seuil de preuve : <= 8 ms")
    print(f"Seuil de réfutation : >= 20 ms")
    if t_nouveau * 1000 <= 8:
        verdict = "PREUVE"
    elif t_nouveau * 1000 >= 20:
        verdict = "RÉFUTATION"
    else:
        verdict = "NI L'UN NI L'AUTRE"
    print(f"VERDICT : {verdict}")


if __name__ == "__main__":
    main()
