#!/usr/bin/env python3
"""Erreur du chemin W4A8 de prefill (kernels/__init__.py, ACVRAM_PREFILL=w8a8 —
défaut « a8 » jusqu'au 17/09 → kernels/fp4_gemm.py nvfp4_mm_w4a8) sur les poids RÉELS d'un converti,
à sec (processeur, même arithmétique que le noyau : déquant fp32, échelle par
ligne amax/448, E4M3, produit fp32).

Ce chemin ne sert pas les experts routés (pile bf16 + torch._grouped_mm dans
les deux régimes) mais l'expert partagé et la couche dense, à chaque jeton de
chaque couche ; le régime tout-torch (ACVRAM_DISABLE_KERNELS=1, backend
« reference ») fait dequantize_nvfp4 → linear bf16 exact.

Trois bras par projection : A8 seul (activations E4M3, poids exacts), W8 seul
(poids déquantifiés requantifiés E4M3 par ligne — double quantification
FP4 → E4M3 — activations exactes), W8A8 = le noyau. Référence : produit fp32
des mêmes entrées avec le poids déquantifié exact. Activations synthétiques
(gaussienne, et queue lourde t₃ pour les jetons à valeurs aberrantes) : les
chiffres A8 dépendent de la distribution, ceux de W8 seul non.

    python outils/w4a8-expert-partage-17-09.py <converti> [couches=1,10,30] [jetons=512]
"""
import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acvram.kernels.fp4_gemm import _F8_MAX               # noqa: E402
from acvram.quant.nvfp4 import NVFP4Tensor, dequantize_nvfp4  # noqa: E402


def _charge(rep, nom, manif, fichiers):
    ent = manif["tensors"][nom]
    d = {}
    for k in ent["keys"]:
        f = manif["weight_map"][k]
        if f not in fichiers:
            fichiers[f] = safe_open(os.path.join(rep, f), "pt", device="cpu")
        d[k.rsplit(".", 1)[1]] = fichiers[f].get_tensor(k)
    K = d["qweight"].shape[1] * 2
    return NVFP4Tensor(d["qweight"], d["block_scale"].view(torch.float8_e4m3fn),
                       d["global_scale"].reshape(()).float(), tuple(ent["shape"]), K)


def _e4m3_par_ligne(a):
    """fp4_gemm.py nvfp4_mm_w4a8 : s = amax/448 par ligne, clamp, E4M3, retour fp32."""
    s = a.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / _F8_MAX
    return (a / s).clamp(-_F8_MAX, _F8_MAX).to(torch.float8_e4m3fn).float() * s


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def main():
    rep = sys.argv[1]
    couches = [int(c) for c in (sys.argv[2] if len(sys.argv) > 2 else "1,10,30").split(",")]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    manif = json.load(open(os.path.join(rep, "acvram_manifest.json")))
    fichiers = {}
    g = torch.Generator().manual_seed(17)
    print(f"{'tenseur':52s} {'W8 poids':>9s} {'A8 gauss':>9s} {'W8 gauss':>9s} {'W8A8 gauss':>10s} "
          f"{'A8 t3':>9s} {'W8A8 t3':>9s}")
    for c in couches:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            nom = f"model.layers.{c}.mlp.shared_expert.{proj}.weight"
            if nom not in manif["tensors"]:
                nom = f"model.layers.{c}.mlp.{proj}.weight"        # couche dense
            t = _charge(rep, nom, manif, fichiers)
            w = dequantize_nvfp4(t, torch.float32)                  # [M, K] exact
            w8 = _e4m3_par_ligne(w)
            res = [_rel(w8, w)]
            for loi in ("gauss", "t3"):
                x = torch.randn(n, w.shape[1], generator=g)
                if loi == "t3":
                    x = x / (torch.distributions.Chi2(3).sample((n, 1)) / 3).sqrt()
                x8 = _e4m3_par_ligne(x)
                ref = x @ w.T
                res += [_rel(x8 @ w.T, ref), _rel(x @ w8.T, ref), _rel(x8 @ w8.T, ref)] if loi == "gauss" \
                    else [_rel(x8 @ w.T, ref), _rel(x8 @ w8.T, ref)]
            print(f"{nom[len('model.layers.'):]:52s} " + " ".join(f"{v:9.4f}" for v in res[:1])
                  + " " + " ".join(f"{v:9.4f}" for v in res[1:4]) + " " + " ".join(f"{v:9.4f}" for v in res[4:]))


if __name__ == "__main__":
    main()
