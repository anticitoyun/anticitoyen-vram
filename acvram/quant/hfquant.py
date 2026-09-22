"""Lecture des points de contrôle HF déjà quantifiés (AWQ « gemm »,
compressed-tensors ``pack-quantized`` / ``nvfp4-pack-quantized``,
modelopt NVFP4) : chaque projection est déquantifiée en bf16 à la volée,
le reste du pipeline (adaptation d'architecture, quantification NVFP4 /
INT8 maison) est inchangé.

Conventions vérifiées sur les tenseurs :
- AWQ gemm : ``qweight`` int32 [in, out/8] (8 nibbles par mot, ordre
  AWQ : nibble i → colonne [0,4,1,5,2,6,3,7][i]), ``qzeros`` int32 [in/g, out/8], ``scales``
  f16 [in/g, out] ; w = (q − z) · s.
- compressed-tensors pack-quantized : ``weight_packed`` int32 [out, in/8]
  (nibbles séquentiels, bits faibles d'abord), entiers signés si
  symétrique, ``weight_scale`` [out, in/g], ``weight_zero_point``
  optionnel (packé de même) ; w = (q − z) · s.
- compressed-tensors nvfp4-pack-quantized : ``weight_packed`` u8
  [out, in/2] (fp4 e2m1, nibble bas d'abord), ``weight_scale`` e4m3
  [out, in/16], ``weight_global_scale`` f32 ; w = fp4 · s / global.
- modelopt NVFP4 : ``weight`` u8 [out, in/2] + ``weight_scale`` e4m3 +
  ``weight_scale_2`` f32 (w = fp4 · s · s2) ; les couches laissées en bf16
  passent telles quelles.
- fp8 statique (Devstral-Small-2, ``weight_block_size: null`` — PAS de
  blocs, une échelle PAR TENSEUR) : ``weight`` float8_e4m3fn [out, in],
  ``weight_scale`` scalaire f32 ; w = weight.float() · weight_scale.
  ``input_scale`` (côté activation, ``activation_scheme: "static"``) est
  ignorée : le reste du pipeline requantifie depuis des poids bf16, jamais
  depuis une activation statique. Vérifié sur le config.json publié
  (huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512, 18/09), pas
  deviné depuis une convention voisine.
"""
from __future__ import annotations

import json
import os
from typing import Iterator, Optional

import torch
from safetensors import safe_open

__all__ = ["HFQuantCheckpoint", "is_hfquant"]

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
_AWQ_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]        # ordre inverse AutoAWQ (reverse_awq_order)
_SHIFTS = torch.arange(0, 32, 4, dtype=torch.int32)


def _quant_config(path: str) -> dict:
    cfg_path = os.path.join(path, "config.json")
    if not os.path.isfile(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return json.load(fh).get("quantization_config") or {}


def is_hfquant(path: str) -> bool:
    q = _quant_config(path)
    m = q.get("quant_method")
    if m == "awq":
        return True
    if m == "compressed-tensors":
        return q.get("format") in ("pack-quantized", "nvfp4-pack-quantized")
    if m == "fp8":
        return True
    return m == "modelopt"


def _unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """int32 [..., n] → int32 [..., n·8], nibble de poids faible d'abord."""
    shifts = _SHIFTS.to(packed.device)
    return ((packed.unsqueeze(-1) >> shifts) & 0xF).reshape(*packed.shape[:-1], -1)


class HFQuantCheckpoint:
    def __init__(self, path: str) -> None:
        self.path = path
        self.q = _quant_config(path)
        self.method = self.q.get("quant_method")
        self.format = self.q.get("format")
        groups = self.q.get("config_groups") or {}
        g0 = next(iter(groups.values()), {}) if groups else {}
        w = g0.get("weights") or {}
        self.bits = int(self.q.get("bits") or w.get("num_bits") or 4)
        self.group_size = int(self.q.get("group_size") or w.get("group_size") or 128)
        self.symmetric = bool(w.get("symmetric", not self.q.get("zero_point", False)))
        if (self.bits != 4 and self.format != "nvfp4-pack-quantized"
                and self.method not in ("modelopt", "fp8")):
            raise NotImplementedError(f"{self.method} {self.bits} bits non pris en charge")

    # ---------------------------------------------------------------- déquant
    def _awq(self, qweight: torch.Tensor, qzeros: torch.Tensor,
             scales: torch.Tensor) -> torch.Tensor:
        order = torch.tensor(_AWQ_ORDER, device=qweight.device)
        w = _unpack_nibbles(qweight).reshape(qweight.shape[0], -1, 8)[:, :, order]
        w = w.reshape(qweight.shape[0], -1)                     # [in, out]
        z = _unpack_nibbles(qzeros).reshape(qzeros.shape[0], -1, 8)[:, :, order]
        z = z.reshape(qzeros.shape[0], -1)                      # [in/g, out]
        g = w.shape[0] // z.shape[0]
        s = scales.to(torch.float32)
        out = (w.to(torch.float32) - z.to(torch.float32).repeat_interleave(g, 0)) \
            * s.repeat_interleave(g, 0)
        return out.t().contiguous().to(torch.bfloat16)          # [out, in]

    def _ct_int4(self, packed: torch.Tensor, scale: torch.Tensor,
                 zp: Optional[torch.Tensor], shape: Optional[torch.Tensor]) -> torch.Tensor:
        w = _unpack_nibbles(packed)                              # [out, in]
        if shape is not None:
            w = w[:, :int(shape[1])]
        # compressed-tensors pack_to_int32 décale les entiers signés de +8
        # avant l'empaquetage (pas de complément à deux) ; le zero-point
        # subit le même décalage, la différence w − z est donc inchangée
        w = w.to(torch.float32) - 8.0
        g = w.shape[1] // scale.shape[1]
        if zp is not None:
            if zp.dtype == torch.int32 and zp.shape[1] == scale.shape[1]:
                z = _unpack_nibbles(zp.t().contiguous()).t()   # packé sur les lignes
            elif zp.dtype == torch.int32:
                z = _unpack_nibbles(zp)
            else:
                z = zp
            z = z[:w.shape[0], :scale.shape[1]].to(torch.float32) - 8.0
            w = w - z.repeat_interleave(g, 1)
        return (w * scale.to(torch.float32).repeat_interleave(g, 1)).to(torch.bfloat16)

    @staticmethod
    def _ct_nvfp4(packed: torch.Tensor, scale: torch.Tensor,
                  global_scale: torch.Tensor) -> torch.Tensor:
        lo = (packed & 0xF).to(torch.long)
        hi = (packed >> 4).to(torch.long)
        idx = torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], -1)   # [out, in]
        w = _E2M1.to(packed.device)[idx]
        # × (1/global) et non / global : c'est l'arithmétique de vLLM (alpha =
        # 1/(échelle globale) en fp32 multiplié dans l'épilogue) et celle du
        # passage direct (`nvfp4_direct`, global_scale = 1/weight_global_scale) —
        # les deux déquantifient au bit près les mêmes octets.
        inv = torch.ones((), dtype=torch.float32, device=packed.device) / global_scale.to(torch.float32).reshape(())
        s = scale.to(torch.float32) * inv
        blk = w.shape[1] // s.shape[1]
        return (w * s.repeat_interleave(blk, 1)).to(torch.bfloat16)

    @staticmethod
    def nvfp4_direct(packed: torch.Tensor, scale: torch.Tensor, global_scale: torch.Tensor,
                     inverser_global: bool):
        """Passage DIRECT (sage-convertisseur-formats-16-09 § 3.1) : les octets
        NVFP4 de la source deviennent un ``NVFP4Tensor`` sans déquantifier ni
        requantifier — même E2M1 (quartet bas = indice pair, `pack_e2m1`),
        mêmes échelles E4M3 par bloc de 16, échelle globale fp32 :
        modelopt ``weight_scale_2`` telle quelle ; compressed-tensors
        ``1 / weight_global_scale`` (vLLM multiplie par l'inverse). Les poids
        servis sont alors ceux de vLLM, et la colonne PPL du duel compare deux
        moteurs, pas deux quantifications."""
        from .nvfp4 import NVFP4Tensor
        out_f, half = packed.shape
        g = global_scale.to(torch.float32).reshape(())
        if inverser_global:
            g = torch.ones((), dtype=torch.float32) / g
        bs = scale.contiguous()
        if bs.dtype != torch.float8_e4m3fn:
            bs = bs.view(torch.float8_e4m3fn) if bs.dtype == torch.uint8 else bs.to(torch.float8_e4m3fn)
        if bs.shape != (out_f, half * 2 // 16):
            raise ValueError(f"échelles NVFP4 {tuple(bs.shape)} pour un poids {out_f}x{half * 2}")
        return NVFP4Tensor(qweight=packed.contiguous(), block_scale=bs, global_scale=g.clone(),
                           shape=(out_f, half * 2), padded_in=half * 2)

    @staticmethod
    def _fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """weight_block_size=null : une echelle SCALAIRE par tenseur, pas de
        bloc -- w = weight.float() * scale, aucun repli-interleave a faire."""
        return (weight.to(torch.float32) * scale.to(torch.float32).reshape(())).to(torch.bfloat16)

    @staticmethod
    def _modelopt_nvfp4(packed: torch.Tensor, scale: torch.Tensor,
                        scale_2: torch.Tensor) -> torch.Tensor:
        """modelopt : w = fp4 · weight_scale(e4m3) · weight_scale_2(f32)."""
        lo = (packed & 0xF).to(torch.long)
        hi = (packed >> 4).to(torch.long)
        idx = torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], -1)
        w = _E2M1.to(packed.device)[idx]
        s = scale.to(torch.float32) * scale_2.to(torch.float32).reshape(())
        blk = w.shape[1] // s.shape[1]
        return (w * s.repeat_interleave(blk, 1)).to(torch.bfloat16)

    # ---------------------------------------------------------------- lecture
    def _files(self) -> list[str]:
        index_path = os.path.join(self.path, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path, "r", encoding="utf-8") as fh:
                return sorted(set(json.load(fh)["weight_map"].values()))
        return [f for f in sorted(os.listdir(self.path)) if f.endswith(".safetensors")]

    def iter_tensors(self, direct_nvfp4: bool = False) -> Iterator[tuple[str, torch.Tensor]]:
        """``direct_nvfp4`` : les poids NVFP4 (modelopt, compressed-tensors
        nvfp4-pack-quantized) sortent en ``NVFP4Tensor`` tels quels ; les
        autres formats et les couches gardées en clair sortent comme avant."""
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        # les compagnons (scales, qzeros, weight_scale…) peuvent vivre dans un
        # autre fragment que le poids : index global clé → fichier
        index_path = os.path.join(self.path, "model.safetensors.index.json")
        ou: dict[str, str] = {}
        if os.path.isfile(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                ou = dict(json.load(f)["weight_map"])
        poignees: dict[str, object] = {}

        def lire(fh_courant, k: str) -> torch.Tensor:
            f = ou.get(k)
            if f is None or f == fn:
                return fh_courant.get_tensor(k)
            if f not in poignees:
                poignees[f] = safe_open(os.path.join(self.path, f), framework="pt", device="cpu")
            return poignees[f].get_tensor(k)

        for fn in self._files():
            with safe_open(os.path.join(self.path, fn), framework="pt", device="cpu") as fh:
                keys = list(fh.keys())
                have = set(keys) | set(ou)
                for key in keys:
                    base, _, suffix = key.rpartition(".")
                    if self.method == "awq":
                        if suffix in ("qzeros", "scales"):
                            continue
                        if suffix == "qweight":
                            yield base + ".weight", self._awq(
                                fh.get_tensor(key).to(dev),
                                lire(fh, base + ".qzeros").to(dev),
                                lire(fh, base + ".scales").to(dev)).cpu()
                            continue
                    elif self.method == "compressed-tensors":
                        if suffix in ("weight_scale", "weight_zero_point", "weight_shape",
                                      "weight_global_scale", "input_global_scale",
                                      "input_scale"):
                            continue
                        if suffix == "weight_packed":
                            if direct_nvfp4 and self.format == "nvfp4-pack-quantized":
                                yield base + ".weight", self.nvfp4_direct(
                                    fh.get_tensor(key), lire(fh, base + ".weight_scale"),
                                    lire(fh, base + ".weight_global_scale"), inverser_global=True)
                                continue
                            packed = fh.get_tensor(key).to(dev)
                            scale = lire(fh, base + ".weight_scale").to(dev)
                            if self.format == "nvfp4-pack-quantized":
                                t = self._ct_nvfp4(packed, scale,
                                                   lire(fh, base + ".weight_global_scale"))
                            else:
                                zp = base + ".weight_zero_point"
                                shp = base + ".weight_shape"
                                t = self._ct_int4(
                                    packed, scale,
                                    lire(fh, zp).to(dev) if zp in have else None,
                                    lire(fh, shp) if shp in have else None)
                            yield base + ".weight", t.cpu()
                            continue
                    elif self.method == "modelopt":
                        if suffix in ("weight_scale", "weight_scale_2", "input_scale",
                                      "k_scale", "v_scale"):
                            continue
                        if suffix == "weight" and base + ".weight_scale_2" in have:
                            t = fh.get_tensor(key)
                            if t.dtype == torch.uint8 and direct_nvfp4:
                                yield key, self.nvfp4_direct(
                                    t, lire(fh, base + ".weight_scale"),
                                    lire(fh, base + ".weight_scale_2"), inverser_global=False)
                                continue
                            if t.dtype == torch.uint8:      # fp4 empaqueté [out, in/2]
                                yield key, self._modelopt_nvfp4(
                                    t.to(dev), lire(fh, base + ".weight_scale").to(dev),
                                    lire(fh, base + ".weight_scale_2")).cpu()
                                continue
                            yield key, t                    # bf16 (couche gardée en clair)
                            continue
                    elif self.method == "fp8":
                        if suffix in ("weight_scale", "input_scale"):
                            continue
                        if suffix == "weight" and base + ".weight_scale" in have:
                            t = fh.get_tensor(key)
                            if t.dtype == torch.float8_e4m3fn:
                                yield key, self._fp8(
                                    t.to(dev), lire(fh, base + ".weight_scale")).cpu()
                                continue
                            yield key, t   # modules_to_not_convert (vision_tower, lm_head…)
                            continue
                    yield key, fh.get_tensor(key)
