"""evp — MXFP4 (gpt-oss) vers NVFP4, EXACTEMENT.

MXFP4 (transformers `integrations/mxfp4.py`) : par bloc de 32 poids, 16 octets de codes E2M1 (quartet bas = élément
pair, la convention de `nvfp4.pack_e2m1`) et une échelle E8M0 `s` (valeur 2^(s − 127)). NVFP4 : codes E2M1 identiques,
une échelle E4M3 par bloc de 16 et une échelle globale fp32. Chaque bloc de 32 devient deux blocs de 16 de même échelle ;
avec une échelle globale 2^(s_max − 127 − 8), l'échelle de bloc vaut 2^(s − s_max + 8) — une puissance de deux que l'E4M3
porte exactement de 2⁻⁹ (sous-normal) à 2⁸. Sous 18 octaves d'étendue par tenseur, poids déquantifiés = poids de la
source au bit (mesuré sur gpt-oss-20b : ≤ 16 octaves par tenseur — scratchpad/poste1-evp-01-10/scelle.md § 2) ; au-delà,
refus nommé, jamais un arrondi silencieux.

La sortie (`iter_gpt_oss`) suit les noms d'un MoE acvram : experts désentrelacés (gate = lignes paires de gate_up, up =
lignes impaires, transformers `GptOssExperts._apply_gate`), un tenseur par expert et par projection, biais compris ;
routeur `mlp.router` → `mlp.gate`. Le reste passe tel quel (attention bf16 et ses biais, puits, normes, plongements).
"""
from __future__ import annotations

import json
import os
from typing import Iterator

import torch

from .nvfp4 import NVFP4Tensor

_E8M0_BIAIS = 127
_MARGE_HAUTE = 8          # 2^8 : plus grande puissance de deux de l'E4M3
_PLANCHER = -9            # 2^-9 : plus petite (sous-normale) de l'E4M3


def mxfp4_vers_nvfp4(blocs: torch.Tensor, echelles: torch.Tensor) -> NVFP4Tensor:
    """``blocs`` u8 [N, nb, 16], ``echelles`` u8 [N, nb] (une matrice [N, 32·nb]) → NVFP4Tensor aux mêmes valeurs."""
    if blocs.dtype != torch.uint8 or echelles.dtype != torch.uint8:
        raise TypeError("MXFP4 : blocs et échelles attendus en uint8")
    n, nb, _ = blocs.shape
    s = echelles.to(torch.int32)
    s_max = int(s.max())
    expo = s - s_max + _MARGE_HAUTE                          # exposant de l'échelle de bloc, ≤ 8
    if int(expo.min()) < _PLANCHER:
        raise ValueError(f"MXFP4 → NVFP4 : étendue d'échelles {s_max - int(s.min())} octaves > "
                         f"{_MARGE_HAUTE - _PLANCHER} (E4M3) — conversion exacte impossible, refus")
    bloc16 = torch.pow(2.0, expo.to(torch.float32)).repeat_interleave(2, dim=-1)        # [N, 2·nb]
    bs = bloc16.to(torch.float8_e4m3fn)
    if not torch.equal(bs.to(torch.float32), bloc16):
        raise AssertionError("MXFP4 → NVFP4 : une échelle de bloc n'est pas exacte en E4M3")
    k = nb * 32
    gs = torch.tensor(2.0 ** (s_max - _E8M0_BIAIS - _MARGE_HAUTE), dtype=torch.float32)
    return NVFP4Tensor(qweight=blocs.reshape(n, nb * 16).contiguous(), block_scale=bs.contiguous(),
                       global_scale=gs, shape=(n, k), padded_in=k)


def mxfp4_dequant(blocs: torch.Tensor, echelles: torch.Tensor) -> torch.Tensor:
    """Référence indépendante (la table de transformers) : [N, 32·nb] fp32."""
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    n, nb, _ = blocs.shape
    v = torch.empty(n, nb, 32)
    v[..., 0::2] = lut[(blocs & 0x0F).long()]
    v[..., 1::2] = lut[(blocs >> 4).long()]
    return torch.ldexp(v, (echelles.to(torch.int32) - _E8M0_BIAIS)[..., None]).reshape(n, nb * 32)


def est_gpt_oss_mxfp4(path: str) -> bool:
    try:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            c = json.load(f)
    except (OSError, ValueError):
        return False
    return (c.get("model_type") == "gpt_oss"
            and (c.get("quantization_config") or {}).get("quant_method") == "mxfp4")


def iter_gpt_oss(path: str, direct_nvfp4: bool = True) -> Iterator[tuple[str, torch.Tensor]]:
    """Tenseurs d'un gpt-oss MXFP4 aux noms acvram. ``direct_nvfp4`` False : experts déquantifiés en bf16 (exact aussi :
    valeurs E2M1 × puissance de deux), pour un plan qui requantifie."""
    from safetensors import safe_open
    with open(os.path.join(path, "model.safetensors.index.json"), encoding="utf-8") as f:
        ou = json.load(f)["weight_map"]
    poignees: dict = {}

    def lire(k: str) -> torch.Tensor:
        fn = ou[k]
        if fn not in poignees:
            poignees[fn] = safe_open(os.path.join(path, fn), framework="pt", device="cpu")
        return poignees[fn].get_tensor(k)

    for key in sorted(ou):
        if key.endswith(("_scales", "_proj_bias")):
            continue                                      # lus avec leurs blocs
        if key.endswith("mlp.experts.gate_up_proj_blocks") or key.endswith("mlp.experts.down_proj_blocks"):
            base = key[: -len("_blocks")]                 # …mlp.experts.gate_up_proj | …down_proj
            pref = base.rsplit(".", 1)[0]                 # …mlp.experts
            blocs, ech, biais = lire(key), lire(base + "_scales"), lire(base + "_bias")
            for e in range(blocs.shape[0]):
                if base.endswith("gate_up_proj"):
                    parts = (("gate_proj", slice(0, None, 2)), ("up_proj", slice(1, None, 2)))
                else:
                    parts = (("down_proj", slice(None)),)
                for nom, lignes in parts:
                    b, s = blocs[e, lignes].contiguous(), ech[e, lignes].contiguous()
                    w = mxfp4_vers_nvfp4(b, s) if direct_nvfp4 else mxfp4_dequant(b, s).to(torch.bfloat16)
                    yield f"{pref}.{e}.{nom}.weight", w
                    yield f"{pref}.{e}.{nom}.bias", biais[e, lignes].contiguous()
            continue
        nom = key.replace(".mlp.router.", ".mlp.gate.")
        yield nom, lire(key)
