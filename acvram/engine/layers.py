"""Briques de transformeur conscientes de la quantification et des étages.

Chaque couche linéaire du modèle est un :class:`QuantLinear`. Elle détient son
poids dans le format qu'a choisi son appareil, applique la mise à l'échelle de
calibration produite par le convertisseur, et aiguille vers le noyau fusionné
lorsqu'il en existe un. Une couche dont les poids résident en mémoire vive les
enveloppe dans un :class:`StreamedWeight`, qui les copie vers le GPU sur un flux
annexe, de sorte que le transfert de la couche i+1 recouvre le calcul de la
couche i.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import kernels
from ..quant.calibrate import ChannelScaler
from ..quant.formats import PlainTensor, dequantize
from ..quant.int4 import INT4Tensor
from ..quant.nvfp4 import NVFP4Tensor

__all__ = ["QuantLinear", "StreamedWeight", "RMSNorm", "RotaryEmbedding",
           "apply_rope", "repeat_kv", "repeat_kv_batched", "attention",
           "batched_decode_attention", "causal_mask"]


class StreamedWeight:
    """Un poids qui vit en mémoire hôte épinglée et ne visite le GPU qu'à la demande.

    C'est la mémoire épinglée qui rend la copie asynchrone : une source
    paginable forcerait le pilote à la sérialiser et le recouvrement
    disparaîtrait. Le double tampon fait que le transfert de la couche i+1 est
    déjà en vol pendant que la couche i calcule, si bien qu'une couche
    transférée coûte ``max(copie, calcul)`` et non leur somme — ce que suppose
    exactement le modèle de coût du planificateur.
    """

    def __init__(self, host_tensors: dict[str, torch.Tensor], device: torch.device,
                 n_buffers: int = 2) -> None:
        self.host = {k: (v.pin_memory() if not v.is_pinned() else v)
                     for k, v in host_tensors.items()}
        self.device = device
        self.stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self._buffers: list[dict[str, torch.Tensor]] = []
        self._events: list[Any] = []
        self._slot = 0
        self.n_buffers = n_buffers

    def _ensure(self) -> None:
        if self._buffers:
            return
        for _ in range(self.n_buffers):
            self._buffers.append({
                k: torch.empty_like(v, device=self.device)
                for k, v in self.host.items()})
            self._events.append(
                torch.cuda.Event() if self.device.type == "cuda" else None)

    def prefetch(self) -> int:
        """Lance la copie vers le tampon suivant ; rend son emplacement."""
        if self.device.type != "cuda":
            return 0
        self._ensure()
        slot = self._slot
        self._slot = (self._slot + 1) % self.n_buffers
        with torch.cuda.stream(self.stream):
            for k, dst in self._buffers[slot].items():
                dst.copy_(self.host[k], non_blocking=True)
            self._events[slot].record(self.stream)
        return slot

    def wait(self, slot: int) -> dict[str, torch.Tensor]:
        if self.device.type != "cuda":
            return self.host
        self._events[slot].wait(torch.cuda.current_stream(self.device))
        return self._buffers[slot]

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.host.values())


class QuantLinear(nn.Module):
    """``y = x @ W.T (+ b)`` où W est stocké quantifié.

    La mise à l'échelle s'applique à l'*entrée*, jamais repliée dans le poids :
    le convertisseur l'a choisie précisément pour que le poids se quantifie bien
    une fois mis à l'échelle, et la replier défairait cela.
    """

    def __init__(self, qweight: Any, bias: Optional[torch.Tensor] = None,
                 scaler: Optional[ChannelScaler] = None,
                 out_features: Optional[int] = None,
                 in_features: Optional[int] = None) -> None:
        super().__init__()
        self.qweight = qweight
        self.scaler = scaler
        self.bias = bias
        shape = getattr(qweight, "shape", None)
        self.out_features = out_features or (shape[0] if shape else 0)
        self.in_features = in_features or (shape[1] if shape else 0)
        self.streamed: Optional[StreamedWeight] = None
        self._pending_slot: Optional[int] = None

    # -- placement -------------------------------------------------------
    def to_device(self, device: torch.device, streamed: bool = False) -> "QuantLinear":
        if streamed:
            self.streamed = StreamedWeight(
                dict(self.qweight.state_dict()), device)
        else:
            self.qweight = self.qweight.to(device)
            if self.bias is not None:
                self.bias = self.bias.to(device)
        if self.scaler is not None:
            self.scaler = self.scaler.to(device)
        return self

    def prefetch(self) -> None:
        if self.streamed is not None:
            self._pending_slot = self.streamed.prefetch()

    # -- forward ---------------------------------------------------------
    def _resolved_weight(self) -> Any:
        if self.streamed is None:
            return self.qweight
        slot = self._pending_slot if self._pending_slot is not None \
            else self.streamed.prefetch()
        tensors = self.streamed.wait(slot)
        self._pending_slot = None
        return _rehydrate(self.qweight, tensors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scaler is not None and not self.scaler.is_identity:
            x = self.scaler.apply(x)
        w = self._resolved_weight()
        # Le registre choisit le backend par (format, peripherique) ; voir
        # kernels/backends.py. Un acces de dictionnaire memoise, rien de plus.
        y = kernels.matmul(x, w)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y

    @property
    def nbytes(self) -> int:
        return getattr(self.qweight, "nbytes", 0)

    def extra_repr(self) -> str:
        fmt = getattr(self.qweight, "format", "?")
        return (f"in={self.in_features}, out={self.out_features}, fmt={fmt}"
                f"{', streamed' if self.streamed else ''}")


def _rehydrate(template: Any, tensors: dict[str, torch.Tensor]) -> Any:
    """Reconstruit un objet tenseur quantifié autour de tampons GPU fraîchement copiés."""
    if isinstance(template, NVFP4Tensor):
        return NVFP4Tensor(
            tensors["qweight"], tensors["block_scale"].view(torch.float8_e4m3fn),
            tensors["global_scale"], template.shape, template.padded_in)
    if isinstance(template, INT4Tensor):
        return INT4Tensor(tensors["qweight"], tensors["scales"], tensors["zeros"],
                          template.group_size, template.shape, template.padded_in)
    if isinstance(template, PlainTensor):
        return PlainTensor(tensors["weight"], template.shape, template.format)
    raise TypeError(f"impossible de reconstruire {type(template)!r}")


class RMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        # On accumule la variance en fp32 : avec des poids sur 4 bits, les
        # activations sont déjà bruitées, et une réduction en demi-précision sur
        # 8192 canaux ajoute de l'erreur pour un gain de vitesse dérisoire.
        if dtype == torch.bfloat16 and x.is_cuda \
                and self.weight.dtype == torch.bfloat16:
            ext = kernels.get_extension()
            if ext is not None and hasattr(ext, "rmsnorm_bf16"):
                return ext.rmsnorm_bf16(x, self.weight, self.eps)   # 1 lancement
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return (x32.to(dtype) * self.weight.to(dtype))


class RotaryEmbedding(nn.Module):
    """RoPE, avec les variantes de mise à l'échelle que les modèles actuels embarquent."""

    def __init__(self, head_dim: int, max_position: int, base: float = 10000.0,
                 scaling: Optional[dict] = None, device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_position = max_position
        self.base = base
        self.scaling = scaling or {}
        inv_freq = self._build_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache_len = 0
        self._cos: Optional[torch.Tensor] = None
        self._sin: Optional[torch.Tensor] = None
        self._dtype = dtype

    def _build_inv_freq(self, device) -> torch.Tensor:
        dim = self.head_dim
        inv = 1.0 / (self.base ** (torch.arange(0, dim, 2, device=device,
                                                dtype=torch.float32) / dim))
        rtype = str(self.scaling.get("rope_type") or self.scaling.get("type") or "")
        factor = float(self.scaling.get("factor", 1.0) or 1.0)
        if rtype in ("linear",):
            return inv / factor
        if rtype in ("dynamic", "ntk"):
            base = self.base * (factor ** (dim / (dim - 2)))
            return 1.0 / (base ** (torch.arange(0, dim, 2, device=device,
                                                dtype=torch.float32) / dim))
        if rtype in ("llama3",):
            low = float(self.scaling.get("low_freq_factor", 1.0))
            high = float(self.scaling.get("high_freq_factor", 4.0))
            orig = float(self.scaling.get("original_max_position_embeddings", 8192))
            wavelen = 2 * math.pi / inv
            low_wl, high_wl = orig / low, orig / high
            smooth = ((orig / wavelen) - low) / max(1e-6, (high - low))
            smoothed = (1 - smooth) * (inv / factor) + smooth * inv
            inv = torch.where(wavelen > low_wl, inv / factor, inv)
            inv = torch.where((wavelen <= low_wl) & (wavelen >= high_wl), smoothed, inv)
            return inv
        return inv

    def _ensure(self, seq_len: int, device, dtype) -> None:
        if self._cos is not None and seq_len <= self._cache_len \
                and self._cos.device == device:
            return
        n = max(seq_len, 1024)
        t = torch.arange(n, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos = emb.cos().to(dtype)
        self._sin = emb.sin().to(dtype)
        self._cache_len = n

    def forward(self, positions: torch.Tensor, device, dtype,
                max_pos: Optional[int] = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        # ``positions.max()`` vit sur le GPU : le rapatrier synchronise tout le
        # flux, a chaque couche, a chaque jeton — 40 synchronisations par jeton
        # sur un modele de 40 couches. L'appelant connait deja la longueur de
        # contexte en Python ; qu'il la donne, et le cache s'etend sans jamais
        # attendre le GPU.
        if max_pos is None:
            max_pos = int(positions.max().item()) + 1 if positions.numel() else 1
        self._ensure(max_pos, device, dtype)
        return self._cos[positions], self._sin[positions]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.shape[-1] < q.shape[-1]:
        # RoPE partiel (qwen3-next : 64 dims tournées sur 256) : la tranche
        # au-delà passe telle quelle.
        d = cos.shape[-1]
        q1, q2 = q[..., :d], q[..., d:]
        k1, k2 = k[..., :d], k[..., d:]
        q1, k1 = apply_rope(q1, k1, cos, sin)
        return torch.cat([q1, q2], dim=-1), torch.cat([k1, k2], dim=-1)
    """``q`` et ``k`` valent [jetons, têtes, dim] ; cos et sin valent [jetons, dim]."""
    cos = cos.unsqueeze(1).to(q.dtype)
    sin = sin.unsqueeze(1).to(q.dtype)
    return (q * cos + _rotate_half(q) * sin,
            k * cos + _rotate_half(k) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Étend les têtes clé/valeur de la GQA au nombre de têtes de requête."""
    if n_rep == 1:
        return x
    t, h, d = x.shape
    return x.unsqueeze(2).expand(t, h, n_rep, d).reshape(t, h * n_rep, d)


def causal_mask(q_len: int, kv_len: int, q_offset: int, device,
                dtype: torch.dtype) -> Optional[torch.Tensor]:
    """Masque pour un bloc de requêtes commençant à la position absolue ``q_offset``.

    ``F.scaled_dot_product_attention(is_causal=True)`` aligne le triangle en
    *haut à gauche*, ce qui n'est correct que si la requête couvre toute la
    séquence. Dès qu'un prefill est découpé — ou qu'un préfixe est servi depuis
    le cache et que seule la queue est précalculée — le bloc de requêtes commence
    en cours de séquence et le drapeau intégré masque silencieusement les
    mauvaises cellules. Le cache de préfixe et le prefill par morceaux dépendent
    tous deux de ce détail, d'où un masque construit explicitement dès que la
    requête est décalée.
    """
    if q_len == 1:
        return None                       # le décodage attend sur tout
    if q_offset == 0 and q_len == kv_len:
        return None                       # le drapeau causal intégré est correct
    rows = torch.arange(q_offset, q_offset + q_len, device=device).unsqueeze(1)
    cols = torch.arange(kv_len, device=device).unsqueeze(0)
    allowed = cols <= rows
    mask = torch.zeros(q_len, kv_len, device=device, dtype=dtype)
    return mask.masked_fill(~allowed, float("-inf"))


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              causal: bool = True, scale: Optional[float] = None,
              q_offset: int = 0) -> torch.Tensor:
    """Attention par produit scalaire normalisé sur des tenseurs ``[jetons, têtes, dim]``.

    Délègue au SDPA de PyTorch, qui choisit FlashAttention sur tout GPU qui le
    gère. Les transpositions sont des vues, pas des copies.
    """
    qh = q.transpose(0, 1).unsqueeze(0)          # [1, heads, tq, dim]
    kh = k.transpose(0, 1).unsqueeze(0)
    vh = v.transpose(0, 1).unsqueeze(0)
    q_len, kv_len = q.shape[0], k.shape[0]
    mask = causal_mask(q_len, kv_len, q_offset, q.device, q.dtype) if causal else None
    if mask is not None:
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=scale)
    else:
        use_causal = bool(causal and q_len > 1 and q_offset == 0
                          and q_len == kv_len)
        out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=use_causal,
                                             scale=scale)
    return out.squeeze(0).transpose(0, 1).contiguous()


def batched_decode_attention(q: torch.Tensor, keys: list[torch.Tensor],
                             values: list[torch.Tensor], n_rep: int,
                             scale: float) -> torch.Tensor:
    """Un seul appel SDPA pour tout un lot de décodage, au lieu d'un par séquence.

    Les séquences ont des longueurs de contexte différentes : les clés sont donc
    complétées à droite jusqu'à la plus longue, et ce remplissage est masqué.
    Cela coûte ``lot × (longueur_max − longueur)`` emplacements de clé gâchés ;
    en face, la boucle Python disparaît et le GPU ne voit qu'un lancement au
    lieu de ``lot``. À un lot de 16, le seul surcoût de lancement était déjà le
    plus grand des deux.
    """
    b = len(keys)
    lens = [kk.shape[0] for kk in keys]
    max_len = max(lens)
    hq, d = q.shape[1], q.shape[2]
    hkv = keys[0].shape[1]
    device, dtype = q.device, q.dtype

    kpad = torch.zeros(b, max_len, hkv, d, device=device, dtype=dtype)
    vpad = torch.zeros(b, max_len, hkv, d, device=device, dtype=dtype)
    for i, (kk, vv) in enumerate(zip(keys, values)):
        kpad[i, : lens[i]] = kk
        vpad[i, : lens[i]] = vv

    kh = repeat_kv_batched(kpad, n_rep).permute(0, 2, 1, 3)   # [b, hq, s, d]
    vh = repeat_kv_batched(vpad, n_rep).permute(0, 2, 1, 3)
    qh = q.unsqueeze(2)                                       # [b, hq, 1, d]

    valid = torch.arange(max_len, device=device).unsqueeze(0) < \
        torch.tensor(lens, device=device).unsqueeze(1)        # [b, s]
    mask = torch.zeros(b, 1, 1, max_len, device=device, dtype=dtype)
    mask = mask.masked_fill(~valid[:, None, None, :], float("-inf"))

    out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=scale)
    return out.squeeze(2)                                     # [b, hq, d]


def decode_attention_fixed(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           seq_lens: torch.Tensor, n_rep: int,
                           scale: float) -> torch.Tensor:
    """Attention de décodage à formes fixes, pour la capture en graphe CUDA.

    ``q`` vaut ``[lot, têtes, dim]`` (un jeton par séquence), ``k``/``v``
    ``[lot, S, têtes_kv, dim]`` avec ``S`` fixé par le godet de capture, et
    ``seq_lens`` est un tenseur — jamais une liste Python : la frontière vit
    sur le GPU, seul le masque en dépend.
    """
    b, s = k.shape[0], k.shape[1]
    # enable_gqa laisse SDPA diffuser les têtes KV vers les têtes de requête :
    # l'ancien repeat_kv passait par reshape-sur-expand, qui matérialise une
    # copie ×n_rep de K et de V à chaque couche, à chaque pas.
    kh = k.permute(0, 2, 1, 3)                                # [b, hkv, S, d]
    vh = v.permute(0, 2, 1, 3)
    mask = (torch.arange(s, device=q.device)[None, :]
            < seq_lens[:, None]).view(b, 1, 1, s)
    out = F.scaled_dot_product_attention(
        q.unsqueeze(2), kh, vh,
        attn_mask=mask, scale=scale, enable_gqa=(n_rep > 1))  # [b, hq, 1, d]
    return out.squeeze(2)                                     # [b, hq, d]


def repeat_kv_batched(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """``[lot, s, têtes_kv, d]`` -> ``[lot, s, têtes_kv × n_rep, d]``."""
    if n_rep == 1:
        return x
    b, s, h, d = x.shape
    return x.unsqueeze(3).expand(b, s, h, n_rep, d).reshape(b, s, h * n_rep, d)
