#!/usr/bin/env python3
"""Bead 1aj, repli W8A8, volet A' (vitesse seule) -- prédiction scellée
dans acvram-memoire/revue/1aj-prefill-w4a4-14-09.md § Repli W8A8, à lire
avant ce script.

Chemin ACTUEL (confirmé par lecture, kernels/__init__.py:625-663) : à
pp2048 le poids int8 est déquantifié en bf16 puis passe par cuBLAS bf16
(`F.linear`). Chemin NOUVEAU : poids int8 affine -> int8 signé (décalage
-128, bijection réversible) PAR LIGNE (une seule échelle par ligne de
sortie, simplification EN MÉMOIRE pour ce banc vitesse seule -- le poids
réel est quantifié par groupes de 128, une vraie intégration W8A8
garderait le groupage ; ce test ignore la qualité comme convenu),
activation int8 dynamique par jeton (amax par ligne), produit via
`torch._int_mm` (tensor cores int8).

    outils/carte.sh .venv/bin/python outils/banc_1aj_prefill_w8a8_vitesse.py
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


def main():
    loaded = load_model(MODEL, dtype=torch.bfloat16, device_override="cuda:0")
    dev = torch.device("cuda:0")

    n = 0
    t_actuel = t_nouveau = 0.0
    for nom, lin in _projections(loaded.model):
        t = lin.qweight
        x = (torch.randn(G, lin.in_features, dtype=torch.bfloat16, device=dev) * 0.02)

        dt_a = _chrono(lambda: kernels.matmul(x, t))

        # dequant reel (correct, tient compte du zero-point ET des groupes)
        # puis requantification symetrique PAR LIGNE en memoire -- simplification
        # du volet vitesse, cf. docstring.
        w_bf16 = kernels.int8_dequant(t, torch.bfloat16).to(torch.float32)
        w_amax = w_bf16.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
        w_scale = w_amax / 127.0
        w_i8 = (w_bf16 / w_scale).round().clamp(-127, 127).to(torch.int8)
        w_i8_t = w_i8.t().contiguous()          # [in, out], ce que veut torch._int_mm

        def nouveau():
            xf = x.to(torch.float32)
            x_amax = xf.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
            x_scale = x_amax / 127.0
            x_i8 = (xf / x_scale).round().clamp(-127, 127).to(torch.int8)
            y_i32 = torch._int_mm(x_i8, w_i8_t)
            y = y_i32.to(torch.float32) * x_scale * w_scale.t()
            return y.to(torch.bfloat16)

        dt_n = _chrono(nouveau)

        t_actuel += dt_a
        t_nouveau += dt_n
        n += 1
        print(f"  {nom:16s} actuel={dt_a*1000:7.3f} ms  nouveau={dt_n*1000:7.3f} ms", flush=True)

    print(f"\n{n} projections mesurées (q/k/v/o, int8 -> int8 signé par ligne + torch._int_mm)")
    print(f"SOMME actuel  : {t_actuel*1000:.2f} ms")
    print(f"SOMME nouveau : {t_nouveau*1000:.2f} ms")
    print(f"\nDépart : 30,67 ms (volet A)")
    print(f"Seuil de preuve : <= 16 ms")
    print(f"Seuil de réfutation : >= 24 ms")
    if t_nouveau * 1000 <= 16:
        verdict = "PREUVE"
    elif t_nouveau * 1000 >= 24:
        verdict = "RÉFUTATION"
    else:
        verdict = "NI L'UN NI L'AUTRE"
    print(f"VERDICT : {verdict}")


if __name__ == "__main__":
    main()
