"""Quantification INT4 par groupes, sur les poids seuls (chemin Ampere / sm_86).

La RTX 3080 Ti est une GA102 : ses tensor cores traitent le FP16, le BF16, le
TF32 et l'INT8, mais il n'existe ni chemin FP8 ni chemin FP4. Un format de
*calcul* sur 4 bits n'est donc pas disponible. La façon d'obtenir malgré tout
une empreinte quatre fois moindre est de ne quantifier que les poids : stocker
4 bits, déquantifier une tuile en FP16 à l'intérieur du noyau, et alimenter les
tensor cores FP16 ordinaires. Le trafic mémoire — le véritable goulot pendant
le décodage — est divisé par quatre ; l'arithmétique reste en FP16.

Disposition, asymétrique, compatible AWQ :

    q[i]     uint4                  groupe de 128 le long de K
    échelle  fp16   par groupe
    zéro     uint4  par groupe

    w[i] ≈ (q[i] − zéro) × échelle

Coût de stockage par poids :

    4 + 16/128 + 4/128 = 4,156 bits par poids   (×3,85 plus petit que le FP16)

La partie proprement « AWQ » — la mise à l'échelle des canaux guidée par les
activations — vit dans :mod:`acvram.quant.calibrate` ; ce module-ci n'est que
le codec entier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

__all__ = ["INT4Tensor", "quantize_int4", "dequantize_int4", "pack_uint4", "unpack_uint4"]

GROUP = 128


def pack_uint4(x: torch.Tensor) -> torch.Tensor:
    """Empaquette des valeurs 4 bits, deux par octet, quartet bas en premier."""
    if x.shape[-1] % 2:
        x = torch.nn.functional.pad(x, (0, 1))
    x = x.to(torch.uint8)
    return (x[..., 0::2] & 0x0F) | ((x[..., 1::2] & 0x0F) << 4)


def unpack_uint4(packed: torch.Tensor) -> torch.Tensor:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    return torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1],
                                                 packed.shape[-1] * 2)


@dataclass
class INT4Tensor:
    qweight: torch.Tensor        # uint8 [out, in//2]
    scales: torch.Tensor         # fp16  [out, in//group]
    zeros: torch.Tensor          # uint8 [out, ceil(in/group/2)] packed uint4
    group_size: int
    shape: tuple[int, ...]
    padded_in: int

    format = "int4_awq"

    @property
    def nbytes(self) -> int:
        return self.qweight.numel() + self.scales.numel() * 2 + self.zeros.numel()

    @property
    def bits_per_weight(self) -> float:
        n = 1
        for d in self.shape:
            n *= d
        return self.nbytes * 8 / max(1, n)

    def to(self, device, non_blocking: bool = False) -> "INT4Tensor":
        return INT4Tensor(
            self.qweight.to(device, non_blocking=non_blocking),
            self.scales.to(device, non_blocking=non_blocking),
            self.zeros.to(device, non_blocking=non_blocking),
            self.group_size, self.shape, self.padded_in,
        )

    def state_dict(self, prefix: str = "") -> dict[str, torch.Tensor]:
        return {
            f"{prefix}qweight": self.qweight,
            f"{prefix}scales": self.scales,
            f"{prefix}zeros": self.zeros,
        }

    @staticmethod
    def from_state_dict(sd: dict[str, torch.Tensor], prefix: str,
                        shape: tuple[int, ...], group_size: int = GROUP) -> "INT4Tensor":
        q = sd[f"{prefix}qweight"]
        return INT4Tensor(q, sd[f"{prefix}scales"], sd[f"{prefix}zeros"],
                          group_size, tuple(shape), q.shape[-1] * 2)


def quantize_int4(weight: torch.Tensor, group_size: int = GROUP,
                  symmetric: bool = False) -> INT4Tensor:
    """Quantifie ``[sorties, entrées]`` en uint4 par groupes.

    Asymétrique par défaut : les groupes de poids d'un modèle de langage sont
    rarement centrés sur zéro, et dépenser un point zéro rapporte environ un
    demi-bit de précision effective pour 4 bits de métadonnées par 128 poids.
    """
    if weight.dim() != 2:
        raise ValueError(f"poids 2-D attendu, reçu {tuple(weight.shape)}")
    orig_shape = tuple(weight.shape)
    w = weight.detach().to(torch.float32)
    out_f, k = w.shape
    if k % group_size:
        w = torch.nn.functional.pad(w, (0, group_size - k % group_size))
    k_pad = w.shape[1]
    ng = k_pad // group_size
    wg = w.view(out_f, ng, group_size)

    if symmetric:
        amax = wg.abs().amax(dim=-1, keepdim=True)
        scale = (amax / 7.0).clamp(min=1e-8)
        zero = torch.full_like(scale, 8.0)
    else:
        wmax = wg.amax(dim=-1, keepdim=True)
        wmin = wg.amin(dim=-1, keepdim=True)
        # Un groupe constant ne doit jamais faire tomber l'échelle à zéro.
        scale = ((wmax - wmin) / 15.0).clamp(min=1e-8)
        zero = (-wmin / scale).round().clamp(0, 15)

    q = (wg / scale + zero).round().clamp(0, 15).to(torch.uint8)
    q = q.reshape(out_f, k_pad)

    scales = scale.squeeze(-1).to(torch.float16)          # [sortie, ng]
    zeros = pack_uint4(zero.squeeze(-1).to(torch.uint8))  # [sortie, ceil(ng/2)]

    return INT4Tensor(pack_uint4(q), scales, zeros, group_size,
                      orig_shape, k_pad)


def dequantize_int4(t: INT4Tensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Déquantification de référence ; le noyau CUDA fusionné doit s'y conformer."""
    q = unpack_uint4(t.qweight).to(torch.float32)         # [sortie, k_rempli]
    out_f, k_pad = q.shape
    ng = k_pad // t.group_size
    zeros = unpack_uint4(t.zeros)[:, :ng].to(torch.float32)
    scales = t.scales[:, :ng].to(torch.float32)
    qg = q.view(out_f, ng, t.group_size)
    out = (qg - zeros.unsqueeze(-1)) * scales.unsqueeze(-1)
    out = out.reshape(out_f, k_pad)
    if k_pad != t.shape[-1]:
        out = out[:, : t.shape[-1]]
    return out.to(dtype)
