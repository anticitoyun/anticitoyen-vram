#!/usr/bin/env python3
"""1a — sur les blocs divergents ModelOpt (point 1, poste7-convertisseur-formats-16-09
§6), part de blocs SANS code de magnitude 6 (mag=7 dans l'E2M1, nvfp4.py:53).

Contrôle qui rend « faux » : sous amax/6 chaque bloc de 16 porte au moins un
code de magnitude 6 (à l'arrondi près). ≈0 % de blocs sans → (b) instrument
désaligné. Nettement > 20 % → (a) échelle cherchée par ModelOpt.

Réutilise le protocole de outils/verif-requant-nvfp4-bitabit-16-09.py (mêmes
3 tenseurs, même lecture). À sec, aucun GPU.
"""
import torch
from safetensors import safe_open

from acvram.quant.hfquant import HFQuantCheckpoint
from acvram.quant.nvfp4 import quantize_nvfp4, unpack_e2m1

CKPT = "/mnt/4TO_SATACMR_2022/Modeles/models_vllm/Qwen3.8-27B-NVFP4"
TENSEURS = [
    "lm_head.weight",
    "model.language_model.layers.0.mlp.gate_proj.weight",
    "model.language_model.layers.0.mlp.down_proj.weight",
]


def lire(cle: str) -> dict:
    import json
    with open(f"{CKPT}/model.safetensors.index.json") as fh:
        wm = json.load(fh)["weight_map"]
    base = cle[:-len(".weight")]
    fichiers = {wm[cle], wm[base + ".weight_scale"], wm[base + ".weight_scale_2"]}
    out = {}
    for f in fichiers:
        with safe_open(f"{CKPT}/{f}", framework="pt", device="cpu") as fh:
            for k in fh.keys():
                if k in (cle, base + ".weight_scale", base + ".weight_scale_2"):
                    out[k] = fh.get_tensor(k)
    return out


def main() -> int:
    total_divergents = 0
    total_sans_6 = 0
    for cle in TENSEURS:
        t = lire(cle)
        packed = t[cle]
        base = cle[:-len(".weight")]
        w_scale = t[base + ".weight_scale"]
        w_scale_2 = t[base + ".weight_scale_2"]

        bf16 = HFQuantCheckpoint._modelopt_nvfp4(packed, w_scale, w_scale_2)
        nv = quantize_nvfp4(bf16)

        divergent = (nv.block_scale != w_scale)  # [out, k/16]

        # codes ModelOpt, magnitude par element (mag=7 -> niveau 6.0, nvfp4.py:53)
        codes = unpack_e2m1(packed)                # [out, k]
        mag = codes & 0x07
        out_f, k = mag.shape
        mag_blocks = mag.view(out_f, k // 16, 16)
        a_un_6 = (mag_blocks == 7).any(dim=-1)      # [out, k/16]

        n_div = int(divergent.sum())
        n_div_sans_6 = int((divergent & ~a_un_6).sum())

        print(f"{cle}")
        print(f"  blocs divergents        : {n_div}/{divergent.numel()}")
        print(f"  divergents SANS code=6  : {n_div_sans_6}/{n_div} "
              f"({100*n_div_sans_6/max(1,n_div):.2f} %)")

        total_divergents += n_div
        total_sans_6 += n_div_sans_6

    pct = 100 * total_sans_6 / max(1, total_divergents)
    print(f"\nVERDICT: {total_sans_6}/{total_divergents} blocs divergents sans code "
          f"de magnitude 6 ({pct:.2f} %)")
    print("-> (b) instrument desaligne" if pct < 5 else
          "-> (a) echelle cherchee (ModelOpt)" if pct > 20 else
          "-> zone grise, ni <5% ni >20%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
