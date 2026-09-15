#!/usr/bin/env python3
"""Contrôle pré-échelle 2^k pour down_proj, GLM (16/09) — ordre poste7 §4
(revue/poste7-glm-pile-correctif-16-09.md, main c50f12f), suite au plancher
dénormal confirmé sur down_proj (revue/verdict-glm-saturation-16-09.md,
23-31 % de blocs à zéro).

Propose un correctif bon marché : multiplier `act/s_down` par une puissance
de deux `2^k` avant `nvfp4_quant_act` (et diviser par la même puissance
après déquantification) — un décalage binaire, pas une vraie échelle
flottante par appel comme `quantize_nvfp4` (poids). `k_act` = le plus
grand `k` tel que `max(amax) × 2^k ≤ 672` (marge sous la borne dure de
`nvfp4_quant_act_kernel`, 6×448÷4). Pour k = 6, 8, 10, 12 et k_act :
fraction de blocs dont `amax < 0,0117 / 2^k` — ce que verrait encore le
plancher dénormal après un décalage par `2^k`.

Scellé : fraction restante (à k_act) ≤ 0,1 % → décalage confirmé, pas de
repli. Sinon : repli sur une échelle `amax` par appel, propre à `down_proj`
seul (comme `quantize_nvfp4`, pas un décalage binaire fixe).

À sec, aucun GPU. Réutilise le mini-converti bf16 déjà construit par
`outils/discriminateur-glm-mma0-16-09.py` (cache `/tmp/glm-discriminateur-mma0`)
et les fonctions de `outils/controle-saturation-16-09.py`.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn.functional as F

SCRATCH = Path("/tmp/glm-discriminateur-mma0")


def charger_controle_saturation():
    chemin = Path(__file__).resolve().parent / "controle-saturation-16-09.py"
    spec = importlib.util.spec_from_file_location("controle_saturation", chemin)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    csat = charger_controle_saturation()

    if not csat.CONVERTI_BF16.exists():
        print("[1/2] extraction + conversion mini bf16 (pas de cache)...", flush=True)
        csat.extraire_mini()
        csat.convertir_bf16()
    print("[2/2] forward réel + poids/échelle réels, calcul act/s_down...", flush=True)
    x = csat.capturer_activations()
    w_gate = csat.lire_poids_bf16("gate_proj", csat.EXPERTS)
    w_up = csat.lire_poids_bf16("up_proj", csat.EXPERTS)
    s_down = csat.lire_awq_reel("down_proj", csat.EXPERTS)

    amax_list = []
    for i in csat.EXPERTS:
        g = x @ w_gate[i].t()
        u = x @ w_up[i].t()
        act = F.silu(g) * u
        acts = act / s_down[i]
        rows, k = acts.shape
        amax = acts.reshape(rows, k // csat.BLOCK, csat.BLOCK).abs().amax(dim=-1).reshape(-1)
        amax_list.append(amax)
    amax = torch.cat(amax_list)
    n = amax.numel()
    max_amax = float(amax.max())
    p999 = float(torch.quantile(amax, 0.999))
    print(f"n_blocs={n} max={max_amax:.8f} p999={p999:.8f}")

    k_act = math.floor(math.log2(672.0 / max_amax))
    print(f"k_act = {k_act}  (max×2^k_act={max_amax * 2 ** k_act:.4f} <= 672 ; "
         f"max×2^(k_act+1)={max_amax * 2 ** (k_act + 1):.4f})")

    grille = {}
    for k in sorted({6, 8, 10, 12, k_act}):
        seuil = 0.0117 / (2 ** k)
        frac = float((amax < seuil).float().mean())
        grille[k] = {"seuil": seuil, "fraction": frac, "n_sous_seuil": int((amax < seuil).sum())}
        print(f"k={k:3d}  seuil={seuil:.10f}  fraction={frac:.6%}  "
             f"n={grille[k]['n_sous_seuil']}/{n}")

    frac_k_act = grille[k_act]["fraction"]
    confirme = frac_k_act <= 0.001
    print(f"\nVERDICT (k_act={k_act}) : fraction={frac_k_act:.6%}  "
         f"{'<= 0,1% -> pre-echelle 2^k_act CONFIRMEE' if confirme else '> 0,1% -> repli amax par appel sur down seul'}")

    out = SCRATCH / "resultat-plancher-echelle.json"
    with open(out, "w") as fh:
        json.dump({"max_amax": max_amax, "p999": p999, "k_act": k_act,
                  "grille": grille, "confirme_k_act": confirme}, fh, indent=2)
    print(f"\nécrit: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
