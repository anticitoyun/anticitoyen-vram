"""Cache KV à 4 bits par rotation : référence torch du format « lm4 ».

Conception : revue/conception-kv-4bits-rotation-17-09.md (TurboQuant, Zandieh
et al. 2025). Chaque vecteur de tête ``x`` (D coordonnées) est tourné par une
matrice orthogonale fixe Π = H·diag(±1)/√D (Hadamard de Sylvester × signes
tirés d'une graine fixe) : ses coordonnées deviennent approximativement
gaussiennes de variance ‖x‖²/D, les canaux aberrants sont étalés, et UNE
échelle par vecteur suffit — la norme, pas l'amax. Chaque coordonnée est
ensuite arrondie au plus proche des 2^b centroïdes de Lloyd-Max gaussiens
(b = 4 : MSE 0,0095 σ²), deux codes de 4 bits par octet.

Le produit scalaire ne demande pas de dé-rotation (⟨Πq, Πk⟩ = ⟨q, k⟩) ; cette
référence dé-tourne pourtant à la relecture, pour rendre des k, v dans la base
d'origine à tout consommateur existant (SDPA, noyau paginé futur exclu). Un
noyau qui tournerait q à la place rendrait le même résultat au bruit près.

« lm3 » et « lm2 » partagent le stockage de lm4 (codes sur un quartet) : ce
sont des témoins de mesure — un instrument qui ne casse pas à 2 bits ne
prouve rien à 4 — pas des formats de livraison.
"""
from __future__ import annotations

import math

import torch

FORMATS = ("lm4", "lm3", "lm2")
GRAINE_ROTATION = 1709          # portée par le nom du format : un cache lm4 se relit avec elle

# Centroïdes de Lloyd-Max pour N(0, 1), moitié positive (Max 1960) ; le test
# `test_kv_lm4.py` les recalcule par itération de Lloyd.
_CENTROIDES = {
    4: (0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6181, 2.0690, 2.7326),
    3: (0.2451, 0.7560, 1.3439, 2.1520),
    2: (0.4528, 1.5104),
}
MSE = {4: 0.00950, 3: 0.03454, 2: 0.11750}      # σ² par coordonnée, à ± 1 %

_cache: dict = {}


def bits(fmt: str) -> int:
    if fmt not in FORMATS:
        raise ValueError(f"format KV inconnu : {fmt!r} (attendu {FORMATS})")
    return int(fmt[2])


def table(b: int, device=None) -> torch.Tensor:
    """Les 2^b centroïdes, croissants, en fp32 : le code est l'indice."""
    cle = ("table", b, str(device))
    if cle not in _cache:
        pos = torch.tensor(_CENTROIDES[b], dtype=torch.float32)
        _cache[cle] = torch.cat([-pos.flip(0), pos]).to(device)
    return _cache[cle]


def seuils(b: int, device=None) -> torch.Tensor:
    """Les 2^b − 1 frontières (milieux entre centroïdes voisins)."""
    cle = ("seuils", b, str(device))
    if cle not in _cache:
        t = table(b, device)
        _cache[cle] = (t[1:] + t[:-1]) / 2
    return _cache[cle]


def rotation(d: int, device=None) -> torch.Tensor:
    """Π [d, d] orthogonale : Hadamard de Sylvester (d puissance de 2) × signes
    de graine fixe, normalisée par √d. Π @ Π.T = I."""
    cle = ("rot", d, str(device))
    if cle not in _cache:
        if d & (d - 1):
            raise ValueError(f"dimension de tête {d} : la rotation demande une puissance de 2")
        h = torch.ones(1, 1, dtype=torch.float32)
        while h.shape[0] < d:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        g = torch.Generator().manual_seed(GRAINE_ROTATION)
        signes = torch.randint(0, 2, (d,), generator=g).float() * 2 - 1
        _cache[cle] = ((h * signes) / math.sqrt(d)).to(device)
    return _cache[cle]


def quantifier(x: torch.Tensor, fmt: str = "lm4") -> tuple[torch.Tensor, torch.Tensor]:
    """``x`` [..., D] → (codes emballés uint8 [..., D/2], échelle fp16 [...]).

    Échelle = ‖x‖/√D : l'écart-type des coordonnées tournées. Le code d'un
    indice pair occupe le quartet bas, l'impair le quartet haut (convention
    NVFP4)."""
    b = bits(fmt)
    d = x.shape[-1]
    xr = x.to(torch.float32) @ rotation(d, x.device)
    echelle = (xr.norm(dim=-1, keepdim=True) / math.sqrt(d)).clamp(min=1e-8)
    echelle = echelle.to(torch.float16)
    y = xr / echelle.to(torch.float32)
    codes = torch.bucketize(y, seuils(b, x.device)).to(torch.uint8)
    emballe = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return emballe, echelle.squeeze(-1)


def dequantifier(emballe: torch.Tensor, echelle: torch.Tensor, fmt: str,
                 dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Inverse de ``quantifier`` : [..., D/2] uint8 + [...] fp16 → [..., D]
    dans la base d'origine (dé-rotation comprise)."""
    b = bits(fmt)
    codes = torch.stack([emballe & 15, emballe >> 4], -1).reshape(*emballe.shape[:-1], -1)
    y = table(b, emballe.device)[codes.long()]
    xr = y * echelle.to(torch.float32).unsqueeze(-1)
    return (xr @ rotation(xr.shape[-1], emballe.device).T).to(dtype)
