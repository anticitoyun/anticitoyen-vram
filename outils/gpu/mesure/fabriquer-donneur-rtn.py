#!/usr/bin/env python3
"""Pièce 114 : fabriquer À SEC (processeur) un « donneur » de projections d'attention en NVFP4 par arrondi simple, SANS
AWQ ni calibration, depuis la source bf16 — les mêmes octets qu'une conversion `--no-awq` écrirait pour ces tenseurs
(quantize_with_calibration(use_awq=False), échelle max6, groupe 128), sans les 30 min de carte. Le dossier rendu porte
un manifeste minimal (tensors + weight_map) qu'`assembler-alias.py` lit comme donneur.
    fabriquer-donneur-rtn.py SOURCE_HF SORTIE --motif REGEX [--echelle max6]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, R)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("sortie")
    ap.add_argument("--motif", required=True)
    ap.add_argument("--echelle", default="max6")
    ap.add_argument("--group-size", type=int, default=128)
    a = ap.parse_args()
    from safetensors import safe_open
    from safetensors.torch import save_file
    from acvram.quant import nvfp4
    from acvram.quant.calibrate import quantize_with_calibration
    nvfp4.regler_echelle(a.echelle)
    idx = json.load(open(os.path.join(a.source, "model.safetensors.index.json")))["weight_map"]
    motif = re.compile(a.motif)
    noms = sorted(n for n in idx if motif.search(n))
    assert noms, "aucun tenseur source ne correspond"
    os.makedirs(a.sortie, exist_ok=True)
    tensors, sd_tout, wm = {}, {}, {}
    ouverts = {}
    t0 = time.time()
    for i, n in enumerate(noms):
        fn = idx[n]
        if fn not in ouverts:
            ouverts[fn] = safe_open(os.path.join(a.source, fn), framework="pt", device="cpu")
        w = ouverts[fn].get_tensor(n)
        qt, scaler, metrics = quantize_with_calibration(w.to(torch.float32), "nvfp4", None, group_size=a.group_size,
                                                        use_awq=False)
        sd = qt.state_dict(prefix=f"{n}.")
        sd.update(scaler.state_dict(prefix=f"{n}."))
        sd = {k: v.cpu().contiguous() for k, v in sd.items()}
        entry = {"format": "nvfp4", "shape": list(qt.shape), "keys": list(sd.keys()), "group_size": a.group_size,
                 "hadamard_block": scaler.hadamard_block, "has_act_scale": scaler.scale is not None,
                 "bpw": round(float(metrics["bpw"]), 3), "out_snr_db": round(float(metrics["out_snr_db"]), 2),
                 "snr_db": round(float(metrics.get("snr_db", metrics["out_snr_db"])), 3),
                 **({"ratio_norme": round(float(metrics["ratio_norme"]), 4)} if "ratio_norme" in metrics else {}),
                 "rtn_sans_awq": True, "echelle": {"regle": a.echelle}}
        tensors[n] = entry
        for k in sd:
            wm[k] = "acvram-rtn.safetensors"
        sd_tout.update(sd)
        if i % 24 == 0:
            print(f"[{i + 1}/{len(noms)}] {n} snr {entry['out_snr_db']} dB · {time.time() - t0:.0f} s", flush=True)
    save_file(sd_tout, os.path.join(a.sortie, "acvram-rtn.safetensors"))
    with open(os.path.join(a.sortie, "acvram_manifest.json"), "w") as f:
        json.dump({"donneur_rtn": {"source": os.path.basename(os.path.normpath(a.source)), "motif": a.motif, "echelle": a.echelle,
                                   "group_size": a.group_size, "awq": False}, "tensors": tensors, "weight_map": wm}, f, indent=1)
    snr = sorted(e["out_snr_db"] for e in tensors.values())
    print(json.dumps({"tenseurs": len(tensors), "cles": len(sd_tout), "snr_min_med_max": [snr[0], snr[len(snr) // 2], snr[-1]],
                      "duree_s": round(time.time() - t0)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
