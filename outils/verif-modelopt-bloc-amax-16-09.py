#!/usr/bin/env python3
"""Vérification bloc/amax — ordre poste7 §6 (revue/poste7-*, main 37bb20a),
suite à 98d1e87 (déquant/requant NVFP4 modelopt : DIFFÉRENT, 16/09).

Sous la règle « échelle de bloc = amax_bloc/6 », l'élément qui a défini
l'amax d'un bloc de 16 doit porter le code de magnitude 6 (valeur E2M1
maximale, indice 7 de la table `_E2M1`/`_levels_tensor`, mag = code & 0x7
== 7) — à l'arrondi de l'échelle en E4M3 près. Compte la part de blocs
SANS AUCUN code à magnitude 6, directement sur les tenseurs BRUTS
modelopt (`weight` u8), sans redéquant/requant.

Scellé (poste7) : ≈ 0 % → instrument désaligné (reprendre 98d1e87 avant
d'écrire) ; > 20 % → échelle de bloc réellement cherchée par ModelOpt,
le passage direct de poste4 devra inclure une recherche d'échelle dans
`quantize_nvfp4`.

À sec, aucun GPU.
"""
import json

import torch
from safetensors import safe_open

CKPT = "/mnt/4TO_SATACMR_2022/Modeles/models_vllm/Qwen3.8-27B-NVFP4"
TENSEURS = [
    "lm_head.weight",
    "model.language_model.layers.0.mlp.gate_proj.weight",
    "model.language_model.layers.0.mlp.down_proj.weight",
]
BLOCK = 16
MAG_6 = 7  # indice de la valeur 6.0 dans la table E2M1 (mag = code & 0x7)


def lire_weight(cle: str) -> torch.Tensor:
    with open(f"{CKPT}/model.safetensors.index.json") as fh:
        wm = json.load(fh)["weight_map"]
    with safe_open(f"{CKPT}/{wm[cle]}", framework="pt", device="cpu") as fh:
        return fh.get_tensor(cle)


def fraction_sans_mag6(packed: torch.Tensor) -> tuple[float, int, int]:
    lo = (packed & 0xF)
    hi = (packed >> 4)
    # nibbles dans l'ordre [octet0_lo, octet0_hi, octet1_lo, octet1_hi, ...]
    codes = torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], -1)   # [out, in]
    mag = codes & 0x7
    n_in = mag.shape[1]
    assert n_in % BLOCK == 0
    blocs = mag.reshape(mag.shape[0], n_in // BLOCK, BLOCK)
    a_mag6 = (blocs == MAG_6).any(dim=-1)
    sans = (~a_mag6)
    return float(sans.float().mean()), int(sans.sum()), int(sans.numel())


def main() -> int:
    total_sans, total_blocs = 0, 0
    for cle in TENSEURS:
        packed = lire_weight(cle)
        frac, sans, n = fraction_sans_mag6(packed)
        print(f"{cle}: {frac:.4%} des blocs sans code a magnitude 6 "
             f"({sans}/{n})")
        total_sans += sans
        total_blocs += n

    frac_pool = total_sans / total_blocs
    print(f"\nPoolé (3 tenseurs) : {frac_pool:.4%} ({total_sans}/{total_blocs})")
    if frac_pool < 0.01:
        verdict = "instrument désaligné (≈0 %) — reprendre 98d1e87 avant d'écrire"
    elif frac_pool > 0.20:
        verdict = "échelle de bloc réellement cherchée par ModelOpt (>20 %) — recherche d'échelle nécessaire dans quantize_nvfp4"
    else:
        verdict = "zone grise (entre 1 % et 20 %) — ni instrument désaligné ni recherche confirmée"
    print(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
