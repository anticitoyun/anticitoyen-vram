#!/usr/bin/env python3
"""Bead 1aj, contrôle de chef (14/09 soir) sur le volet A' réfuté :
« ton verdict porte sur quant torch non fusée + int_mm, pas sur int_mm ».
Décompose les 42,28 ms mesurées en (1) int_mm SEUL, sur des activations
DÉJÀ int8 (préquantifiées hors chrono) et (2) la quantification SEULE
(amax/div/round/clamp/cast), mêmes 192 formes que le banc précédent.

    outils/carte.sh .venv/bin/python outils/banc_1aj_w8a8_decompose.py
"""
import os
import sys
import time
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '.'))
from racine_modeles import racine_modeles as _racine_modeles  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


_ICI = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ICI)
sys.path.insert(0, _REPO)

import torch

from acvram import kernels
from acvram.engine.loader import load_model

MODEL = _RACINE + "/Qwen3-Coder-30B-A3B-nvfp4"
G = 2048
REP = 5


def _projections(model):
    for i, layer in enumerate(model.layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        for nom in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(attn, nom, None)
            if lin is None:
                continue
            if getattr(lin.qweight, "format", None) == "int8":
                yield f"L{i}.{nom}", lin


def _chrono(fn, rep=REP):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rep


def _quantifier(xf):
    x_amax = xf.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
    x_scale = x_amax / 127.0
    return (xf / x_scale).round().clamp(-127, 127).to(torch.int8), x_scale


def main():
    loaded = load_model(MODEL, dtype=torch.bfloat16, device_override="cuda:0")
    dev = torch.device("cuda:0")

    n = 0
    t_intmm = t_quant = 0.0
    for nom, lin in _projections(loaded.model):
        t = lin.qweight
        x = (torch.randn(G, lin.in_features, dtype=torch.bfloat16, device=dev) * 0.02)
        xf = x.to(torch.float32)

        w_bf16 = kernels.int8_dequant(t, torch.bfloat16).to(torch.float32)
        w_amax = w_bf16.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
        w_scale = w_amax / 127.0
        w_i8 = (w_bf16 / w_scale).round().clamp(-127, 127).to(torch.int8)
        w_i8_t = w_i8.t().contiguous()

        # (2) quantification seule -- meme calcul que dans le banc precedent,
        # rien d'autre.
        dt_quant = _chrono(lambda: _quantifier(xf))

        # (1) int_mm seul -- activation prequantifiee HORS chrono.
        x_i8, x_scale = _quantifier(xf)

        def intmm_seul():
            y_i32 = torch._int_mm(x_i8, w_i8_t)
            return y_i32.to(torch.float32) * x_scale * w_scale.t()

        dt_intmm = _chrono(intmm_seul)

        t_intmm += dt_intmm
        t_quant += dt_quant
        n += 1
        print(f"  {nom:16s} int_mm_seul={dt_intmm*1000:7.3f} ms  quant_seule={dt_quant*1000:7.3f} ms",
              flush=True)

    print(f"\n{n} projections (q/k/v/o)")
    print(f"SOMME int_mm seul   : {t_intmm*1000:.2f} ms")
    print(f"SOMME quant seule   : {t_quant*1000:.2f} ms")
    print(f"somme des deux (verif, pas le meme protocole que le banc precedent -- "
          f"pas de recouvrement CPU/GPU ici) : {(t_intmm+t_quant)*1000:.2f} ms")
    print(f"\nSi int_mm seul <= 16 ms : le levier n'est pas mort (quant a fusionner + "
          f"partager entre q/k/v).")
    print(f"Si int_mm seul >= 24 ms : ferme pour de bon.")
    if t_intmm * 1000 <= 16:
        verdict = "LEVIER VIVANT (int_mm seul sous le seuil de preuve)"
    elif t_intmm * 1000 >= 24:
        verdict = "FERMÉ POUR DE BON (int_mm seul au-dessus du seuil de réfutation)"
    else:
        verdict = "NI L'UN NI L'AUTRE"
    print(f"VERDICT : {verdict}")


if __name__ == "__main__":
    main()
