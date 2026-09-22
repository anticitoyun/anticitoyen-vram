#!/usr/bin/env python3
"""Vérification bit à bit — modelopt NVFP4 déquant/requant, ordre Sage §doute
(revue/sage-convertisseur-formats-16-09.md, main 1cf933b), 16/09/2026.

3 tenseurs réels d'un NVFP4 modelopt (`Qwen3.8-27B-NVFP4`,
`/mnt/4TO_SATACMR_2022/Modeles/models_vllm/`) : lus bruts (`weight` u8,
`weight_scale` e4m3, `weight_scale_2` f32), déquantifiés par
`HFQuantCheckpoint._modelopt_nvfp4` (le chemin réel du convertisseur,
`hfquant.py`), puis requantifiés par `quantize_nvfp4` SANS échelle AWQ
(pas d'appel à `search_channel_scales` — « passage direct »). Codes
(`qweight`) et échelles (`block_scale`, `global_scale`) comparés bit à
bit aux originaux.

Doute de Sage : si `quantize_nvfp4` retrouve exactement les mêmes codes
et échelles (même formule `amax/(448·6)` global, `amax/6` par bloc),
le trou 1 (§3.1) se réduit à une option `--no-awq` + garde manifeste ;
sinon Laurine écrit le passage direct (mapper les tenseurs bruts vers
`NVFP4Tensor` sans repasser par un arrondi).

À sec, aucun GPU.
"""
import torch
from safetensors import safe_open

from acvram.quant.hfquant import HFQuantCheckpoint
from acvram.quant.nvfp4 import quantize_nvfp4

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
    tout_identique = True
    for cle in TENSEURS:
        t = lire(cle)
        packed = t[cle]
        base = cle[:-len(".weight")]
        w_scale = t[base + ".weight_scale"]
        w_scale_2 = t[base + ".weight_scale_2"]

        # 1. Déquant réel (chemin convertisseur)
        bf16 = HFQuantCheckpoint._modelopt_nvfp4(packed, w_scale, w_scale_2)

        # 2. Requant sans AWQ (passage direct testé)
        nv = quantize_nvfp4(bf16)

        codes_ok = torch.equal(nv.qweight, packed)
        # global_scale (scalaire f32) contre weight_scale_2 (scalaire f32)
        gs_ok = torch.equal(nv.global_scale.reshape(()), w_scale_2.to(torch.float32).reshape(()))
        # block_scale (e4m3) contre weight_scale (e4m3)
        bs_ok = torch.equal(nv.block_scale, w_scale)

        print(f"{cle}")
        print(f"  codes (qweight)   : {'IDENTIQUE' if codes_ok else 'DIFFERENT'} "
             f"({int((nv.qweight != packed).sum())}/{packed.numel()} octets differents)")
        print(f"  global_scale      : {'IDENTIQUE' if gs_ok else 'DIFFERENT'} "
             f"(nous={float(nv.global_scale):.10g} modelopt={float(w_scale_2):.10g})")
        print(f"  block_scale (e4m3): {'IDENTIQUE' if bs_ok else 'DIFFERENT'} "
             f"({int((nv.block_scale != w_scale).sum())}/{w_scale.numel()} blocs differents)")
        tout_identique = tout_identique and codes_ok and gs_ok and bs_ok

    print(f"\nVERDICT: {'IDENTIQUE bit a bit sur les 3 tenseurs' if tout_identique else 'DIFFERENT — au moins un tenseur diverge'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
