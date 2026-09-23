#!/usr/bin/env python3
"""Trois déquantifications d'UN tenseur d'expert de GLM-4.7-Flash-NVFP4 (source
compressed-tensors de vLLM, lue telle quelle) — ordre Sage (verdict-dequant-
trois-references-17-09) : (a) la référence de vLLM
(`nvfp4_emulation_utils.dequantize_to_dtype`, chemin torch, swizzle=False) sur
weight_packed / weight_scale / 1÷weight_global_scale ; (b) notre référence
`dequantize_nvfp4` (nvfp4.py) sur le NVFP4Tensor du passage direct ; (c) notre
noyau `nvfp4_dequant` (carte requise ; sauté sinon, dit explicitement).
Scellé : max |Δ| ≤ 1 ulp bf16 entre les trois. Prédiction de Sage : (a) et (b)
s'écartent. Contrôle qui peut rendre « faux » : un bras témoin où l'échelle
globale est appliquée après l'arrondi bf16 (l'ordre suspecté) doit différer.

    python outils/dequant-trois-references-17-09.py [--source DIR] [--tenseur NOM] [--n 3]
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import torch

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
SOURCE = "/mnt/4TO_SATACMR_2022/Modeles/models_vllm/GLM-4.7-Flash-NVFP4"


def ulp_bf16(a: torch.Tensor, b: torch.Tensor) -> float:
    """|a−b| en ulp bf16 de |b| élément par élément (max)."""
    af, bf = a.float(), b.float()
    ulp = torch.where(bf == 0, torch.full_like(bf, 2.0 ** -133), 2.0 ** (torch.floor(torch.log2(bf.abs())) - 7))
    return float(((af - bf).abs() / ulp).max())


def vllm_dequant(packed, scale_e4m3, global_scale_src, inverser=True):
    """La référence torch de vLLM, telle quelle ; le global qu'elle reçoit est
    celui que la couche vLLM garde : 1 / weight_global_scale (fp32)."""
    sys.path.insert(0, "/opt/ia/vLLM/.venv/lib/python3.12/site-packages")
    from vllm.model_executor.layers.quantization.utils import nvfp4_emulation_utils as U
    from vllm.platforms import current_platform
    ancien = current_platform.is_cuda_alike
    current_platform.is_cuda_alike = lambda *a, **k: False       # chemin torch même avec CUDA
    try:
        g = global_scale_src.to(torch.float32).reshape(())
        inv = (torch.ones((), dtype=torch.float32) / g) if inverser else g
        return U.dequantize_to_dtype(packed, scale_e4m3.view(torch.uint8), inv, torch.bfloat16,
                                     block_size=16, swizzle=False)
    finally:
        current_platform.is_cuda_alike = ancien


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--tenseur", default="model.layers.3.mlp.experts.7.gate_proj")
    ap.add_argument("--n", type=int, default=3, help="tenseurs supplémentaires (experts suivants)")
    args = ap.parse_args()
    from safetensors import safe_open
    from acvram.quant.hfquant import HFQuantCheckpoint
    from acvram.quant.nvfp4 import dequantize_nvfp4
    from acvram import kernels
    index = json.load(open(os.path.join(args.source, "model.safetensors.index.json")))["weight_map"]

    def lire(k):
        with safe_open(os.path.join(args.source, index[k]), framework="pt", device="cpu") as fh:
            return fh.get_tensor(k)
    noms = [args.tenseur] + [args.tenseur.replace("experts.7.", f"experts.{7 + i}.") for i in range(1, args.n)]
    verdict = True
    for base in noms:
        modelopt = (base + ".weight_scale_2") in index
        if modelopt:      # Coder FP4 : weight / weight_scale / weight_scale_2 (global direct)
            packed = lire(base + ".weight"); scale = lire(base + ".weight_scale"); g = lire(base + ".weight_scale_2")
            a = vllm_dequant(packed, scale, g, inverser=False)
            t = HFQuantCheckpoint.nvfp4_direct(packed, scale, g, inverser_global=False)
        else:             # GLM ct : weight_packed / weight_scale / 1 ÷ weight_global_scale
            packed = lire(base + ".weight_packed"); scale = lire(base + ".weight_scale"); g = lire(base + ".weight_global_scale")
            a = vllm_dequant(packed, scale, g, inverser=True)
            t = HFQuantCheckpoint.nvfp4_direct(packed, scale, g, inverser_global=True)
        ligne0 = f"  (échelle globale effective {float(t.global_scale):.6g})"
        b = dequantize_nvfp4(t, torch.bfloat16)
        d_ab = ulp_bf16(a, b)
        ligne = f"{base}: vLLM vs nvfp4.py {d_ab:.3f} ulp (max), identiques {torch.equal(a, b)}"
        if torch.cuda.is_available():
            c = kernels.nvfp4_dequant(t.to("cuda:0"), torch.bfloat16).cpu()
            d_ac, d_bc = ulp_bf16(a, c), ulp_bf16(b, c)
            ligne += f" ; noyau vs vLLM {d_ac:.3f} ulp, noyau vs nvfp4.py {d_bc:.3f} ulp"
            verdict &= d_ac <= 1.0 and d_bc <= 1.0
        else:
            ligne += " ; noyau : NON MESURÉ (pas de carte)"
        verdict &= d_ab <= 1.0
        # témoin qui doit différer : global appliqué APRÈS l'arrondi bf16 du produit code×bloc
        from acvram.quant.nvfp4 import unpack_e2m1, _levels_tensor
        codes = unpack_e2m1(t.qweight); vals = _levels_tensor(codes.device)[(codes & 7).long()]
        vals = torch.where((codes & 8) != 0, -vals, vals).view(t.shape[0], -1, 16)
        temoin = ((vals * t.block_scale.float().unsqueeze(-1)).to(torch.bfloat16).float()
                  * t.global_scale).to(torch.bfloat16).reshape(t.shape[0], -1)
        d_t = ulp_bf16(temoin, b)
        ligne += f" ; témoin (global après arrondi) vs nvfp4.py {d_t:.3f} ulp"
        print(ligne + ligne0)
        if float(t.global_scale) == 1.0:
            print("  ATTENTION : échelle globale = 1 sur cette source, le témoin d'ordre est inerte (0 ulp par construction)")
    print("VERDICT :", "TENU (≤ 1 ulp entre les références mesurées)" if verdict else "RÉFUTÉ (> 1 ulp)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
