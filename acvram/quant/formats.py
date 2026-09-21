"""Registre des formats de poids.

Une entrée par format de stockage, avec la comptabilité dont le planificateur
de placement a besoin (les bits par poids) et la capacité de calcul minimale
qui sait l'*exploiter* nativement. C'est ce qui fait de « un format différent
par GPU » une idée de premier plan plutôt qu'un cas particulier dispersé dans
le chargeur.

    format      bits/poids  natif sur   notes
    --------    ----------  ----------  -------------------------------------
    nvfp4          4,50     sm_100+     tensor cores FP4 (RTX 5090)
    int4_awq       4,16     sm_75+      poids seuls, déquantifiés dans le noyau
    int8           8,19     sm_75+      symétrique par groupe, repli
    bf16          16,00     sm_80+      intact, normalisations et plongements
    fp16          16,00     tous        intact
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from .int4 import INT4Tensor, dequantize_int4, quantize_int4
from .nvfp4 import NVFP4Tensor, dequantize_nvfp4, quantize_nvfp4

__all__ = ["FormatSpec", "FORMATS", "get_format", "quantize", "dequantize",
           "bits_per_weight", "estimate_bytes", "PlainTensor"]


@dataclass
class PlainTensor:
    """Passe-plat non compressé, pour que tous les chemins voient la même API."""

    weight: torch.Tensor
    shape: tuple[int, ...]
    format: str = "bf16"

    @property
    def nbytes(self) -> int:
        return self.weight.numel() * self.weight.element_size()

    @property
    def bits_per_weight(self) -> float:
        return self.weight.element_size() * 8.0

    def to(self, device, non_blocking: bool = False) -> "PlainTensor":
        return PlainTensor(self.weight.to(device, non_blocking=non_blocking),
                           self.shape, self.format)

    def state_dict(self, prefix: str = "") -> dict[str, torch.Tensor]:
        return {f"{prefix}weight": self.weight}

    @staticmethod
    def from_state_dict(sd, prefix, shape, dtype="bf16"):
        return PlainTensor(sd[f"{prefix}weight"], tuple(shape), dtype)


def _q_int8(w: torch.Tensor, group_size: int = 128, **_: Any) -> INT4Tensor:
    """L'INT8 réutilise la machinerie affine par groupes, sur une plage plus large."""
    raise NotImplementedError("int8 path is handled by quantize() directly")


@dataclass(frozen=True)
class FormatSpec:
    name: str
    bpw: float                       # nominal, à la taille de groupe par défaut
    min_sm: int                      # plus petite capacité qui l'exécute nativement
    quantize: Optional[Callable[..., Any]]
    dequantize: Optional[Callable[..., torch.Tensor]]
    compute_dtype: torch.dtype
    description: str

    def bits_per_weight(self, in_features: int, group_size: Optional[int] = None) -> float:
        if self.name == "nvfp4":
            return 4.0 + 8.0 / 16.0
        if self.name == "q3n":
            return 3.0 + 8.0 / 32.0
        if self.name == "int4_awq":
            g = group_size or 128
            return 4.0 + 16.0 / g + 4.0 / g
        if self.name == "int8":
            g = group_size or 128
            return 8.0 + 16.0 / g + 8.0 / g
        return self.bpw


FORMATS: dict[str, FormatSpec] = {
    "nvfp4": FormatSpec(
        name="nvfp4", bpw=4.5, min_sm=100,
        quantize=quantize_nvfp4, dequantize=dequantize_nvfp4,
        compute_dtype=torch.bfloat16,
        description="E2M1 FP4 + échelle de bloc FP8 E4M3 (16), tensor cores Blackwell",
    ),
    "int4_awq": FormatSpec(
        name="int4_awq", bpw=4.15625, min_sm=75,
        quantize=quantize_int4, dequantize=dequantize_int4,
        compute_dtype=torch.float16,
        description="uint4 affine par groupes de 128, poids seuls, déquantifié dans le noyau",
    ),
    "int8": FormatSpec(
        name="int8", bpw=8.1875, min_sm=75,
        quantize=None, dequantize=None,
        compute_dtype=torch.float16,
        description="uint8 affine par groupes de 128, repli pour couches sensibles",
    ),
    "q3n": FormatSpec(
        name="q3n", bpw=3.25, min_sm=75,
        quantize=None, dequantize=None,     # chemins dans quant/q3n.py,
        compute_dtype=torch.bfloat16,       # branchés par quantize()/dequantize()
        description="quantiles 3 bits + échelle de bloc FP8 E4M3 (32), "
                    "le seul format sous la source d'un GGUF 3 bits",
    ),
    "bf16": FormatSpec(
        name="bf16", bpw=16.0, min_sm=80,
        quantize=None, dequantize=None, compute_dtype=torch.bfloat16,
        description="bfloat16 non compressé",
    ),
    "fp16": FormatSpec(
        name="fp16", bpw=16.0, min_sm=0,
        quantize=None, dequantize=None, compute_dtype=torch.float16,
        description="float16 non compressé",
    ),
}


def get_format(name: str) -> FormatSpec:
    try:
        return FORMATS[name]
    except KeyError:
        raise KeyError(f"format de poids inconnu {name!r} ; "
                       f"connus : {', '.join(FORMATS)}") from None


def bits_per_weight(name: str, in_features: int = 4096,
                    group_size: Optional[int] = None) -> float:
    return get_format(name).bits_per_weight(in_features, group_size)


def estimate_bytes(n_params: int, name: str, group_size: Optional[int] = None) -> int:
    return int(n_params * bits_per_weight(name, group_size=group_size) / 8)


def quantize(weight: torch.Tensor, fmt: str, group_size: Optional[int] = None,
             **kwargs: Any):
    """Quantifie un poids 2-D dans le format ``fmt``."""
    spec = get_format(fmt)
    if fmt == "nvfp4":
        return spec.quantize(weight, **kwargs)
    if fmt == "int4_awq":
        return spec.quantize(weight, group_size=group_size or 128, **kwargs)
    if fmt == "int8":
        return _quantize_int8(weight, group_size or 128,
                              symmetric=kwargs.get("symmetric", False))
    if fmt == "q3n":
        from .q3n import quantize_q3n
        return quantize_q3n(weight, table=kwargs.get("table"))
    if fmt in ("bf16", "fp16"):
        dtype = torch.bfloat16 if fmt == "bf16" else torch.float16
        return PlainTensor(weight.detach().to(dtype), tuple(weight.shape), fmt)
    raise KeyError(fmt)


def dequantize(t: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    fmt = getattr(t, "format", None)
    if fmt == "nvfp4":
        return dequantize_nvfp4(t, dtype or torch.bfloat16)
    if fmt == "int4_awq":
        return dequantize_int4(t, dtype or torch.float16)
    if fmt == "int8":
        return _dequantize_int8(t, dtype or torch.float16)
    if fmt == "q3n":
        from .q3n import dequantize_q3n
        return dequantize_q3n(t, dtype or torch.bfloat16)
    if isinstance(t, PlainTensor):
        return t.weight.to(dtype) if dtype else t.weight
    if isinstance(t, torch.Tensor):
        return t.to(dtype) if dtype else t
    raise TypeError(f"impossible de déquantifier {type(t)!r}")


# --------------------------------------------------------------------------
# INT8 affine par groupes — ici même, car il partage le conteneur d'INT4Tensor
# --------------------------------------------------------------------------


@dataclass
class INT8Tensor:
    qweight: torch.Tensor        # uint8 [out, in]
    scales: torch.Tensor         # fp16  [out, ng]
    zeros: torch.Tensor          # uint8 [out, ng]
    group_size: int
    shape: tuple[int, ...]
    format: str = "int8"

    @property
    def nbytes(self) -> int:
        return self.qweight.numel() + self.scales.numel() * 2 + self.zeros.numel()

    @property
    def bits_per_weight(self) -> float:
        n = 1
        for d in self.shape:
            n *= d
        return self.nbytes * 8 / max(1, n)

    def to(self, device, non_blocking: bool = False) -> "INT8Tensor":
        return INT8Tensor(self.qweight.to(device, non_blocking=non_blocking),
                          self.scales.to(device, non_blocking=non_blocking),
                          self.zeros.to(device, non_blocking=non_blocking),
                          self.group_size, self.shape)

    def state_dict(self, prefix: str = "") -> dict[str, torch.Tensor]:
        return {f"{prefix}qweight": self.qweight, f"{prefix}scales": self.scales,
                f"{prefix}zeros": self.zeros}


def _quantize_int8(weight: torch.Tensor, group_size: int,
                   symmetric: bool = False) -> INT8Tensor:
    w = weight.detach().to(torch.float32)
    out_f, k = w.shape
    if k % group_size:
        w = torch.nn.functional.pad(w, (0, group_size - k % group_size))
    ng = w.shape[1] // group_size
    wg = w.view(out_f, ng, group_size)
    if symmetric:
        # sage-p2-qkvo-int8-canal-18-09 : torch._int_mm/cuBLASLt attend un
        # int8 signe sans point-zero (une correction sinon a appliquer a
        # chaque produit). Loge dans le MEME conteneur affine (zero fixe a
        # 128) plutot qu'un format separe : le chargeur et les noyaux qui
        # testent `format == "int8"` (model.py, mla.py) n'ont rien a savoir
        # du symetrique. `group_size` egal a la largeur d'entree donne le
        # "par canal" (une echelle par ligne de sortie).
        amax = wg.abs().amax(-1, keepdim=True).clamp(min=1e-9)
        scale = amax / 127.0
        zero = torch.full_like(scale, 128.0)
        q = (wg / scale).round().clamp(-127, 127) + 128.0
    else:
        wmax, wmin = wg.amax(-1, keepdim=True), wg.amin(-1, keepdim=True)
        scale = ((wmax - wmin) / 255.0).clamp(min=1e-9)
        zero = (-wmin / scale).round().clamp(0, 255)
        q = (wg / scale + zero).round().clamp(0, 255)
    q = q.to(torch.uint8).reshape(out_f, -1)
    return INT8Tensor(q, scale.squeeze(-1).to(torch.float16),
                      zero.squeeze(-1).to(torch.uint8), group_size,
                      tuple(weight.shape))


def _dequantize_int8(t: INT8Tensor, dtype: torch.dtype) -> torch.Tensor:
    out_f, k_pad = t.qweight.shape
    ng = k_pad // t.group_size
    q = t.qweight.to(torch.float32).view(out_f, ng, t.group_size)
    out = (q - t.zeros.to(torch.float32).unsqueeze(-1)) * \
          t.scales.to(torch.float32).unsqueeze(-1)
    out = out.reshape(out_f, k_pad)[:, : t.shape[-1]]
    return out.to(dtype)


FORMATS["int8"] = FormatSpec(
    name="int8", bpw=8.1875, min_sm=75,
    quantize=_quantize_int8, dequantize=_dequantize_int8,
    compute_dtype=torch.float16,
    description="uint8 affine par groupes de 128, repli pour couches sensibles",
)
