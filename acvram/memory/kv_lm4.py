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
import os
from typing import Optional

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


# ---------------------------------------------------------------------------
# Deux interrupteurs de diagnostic (poste7-kv-lm4-clos-17-09 § 1) : lm4 clos
# comme réfuté (verdict-kv-lm4-qualite-17-09, écart × 10 au seuil), cause pas
# encore lue. Trois bras à 25 min de carte (poste3) décident seulement si
# tq3+1 s'écrit après le commit B de poste4 — ils ne rouvrent PAS lm4.
# Ni l'un ni l'autre ne touche `kvcache.py` : c'est une substitution
# numérique (aller-retour) pour une mesure de PPL, pas un nouveau format de
# stockage — rouvrir la pile paginée pour une passe de cause d'une heure
# serait le remède disproportionné que REGLES § 9 met en garde.
# ---------------------------------------------------------------------------


def diagnostic_actif() -> bool:
    """L'un des deux interrupteurs est posé (même à sa valeur neutre :
    `ACVRAM_KV_LM4_PUITS=0` seul = lm4 des deux côtés par le diagnostic, le
    bras de contrôle qui doit reproduire le 1,0215 de lm4 stocké)."""
    return ("ACVRAM_KV_LM4_SEUL" in os.environ) or ("ACVRAM_KV_LM4_PUITS" in os.environ)


def actif(cote: str) -> bool:
    """``lm4`` s'applique-t-il à ce côté (``"k"`` ou ``"v"``) ?
    ``ACVRAM_KV_LM4_SEUL`` restreint lm4 à un seul côté pour isoler la
    cause de la perte ; vide (défaut) = les deux côtés. L'autre côté
    retombe sur l'int8 par amax (`_int8_amax`), pas sur lm4 — sinon
    l'interrupteur ne changerait rien."""
    if cote not in ("k", "v"):
        raise ValueError(f"côté inconnu : {cote!r} (attendu 'k' ou 'v')")
    seul = os.environ.get("ACVRAM_KV_LM4_SEUL", "").strip().lower()
    if seul not in ("", "k", "v"):
        raise ValueError(
            f"ACVRAM_KV_LM4_SEUL invalide : {seul!r} (attendu 'k', 'v' ou vide)")
    return seul in ("", cote)


def hors_puits(positions: torch.Tensor) -> torch.Tensor:
    """Masque booléen, même forme que ``positions`` : ``True`` = hors du
    puits d'attention, quantifié lm4 comme d'habitude.
    ``ACVRAM_KV_LM4_PUITS`` (0 = aucun puits, défaut) exempte les jetons
    de position ``< N`` — l'ancre d'attention (position 0, norme
    10-40×, l'erreur relative lm4 y pèse 10-40× sur le logit dominant)
    retombe sur l'int8 par amax."""
    n = int(os.environ.get("ACVRAM_KV_LM4_PUITS", "0") or "0")
    if n < 0:
        raise ValueError(f"ACVRAM_KV_LM4_PUITS invalide : {n} (attendu >= 0)")
    return positions >= n


def _int8_amax(x: torch.Tensor) -> torch.Tensor:
    """Aller-retour int8 par ligne (échelle = amax/127) : le format de
    repli des deux interrupteurs, même convention que
    `PagedKVCache._quantize` (memory/kvcache.py) et
    `test_quatre_bits_par_rotation_bat_int8_par_amax_sur_un_puits` — pas un
    troisième format inventé pour ce diagnostic."""
    amax = x.abs().amax(dim=-1, keepdim=True).to(torch.float32).clamp(min=1e-8)
    scale = amax / 127.0
    q = (x.to(torch.float32) / scale).round().clamp(-127, 127)
    return (q * scale).to(x.dtype)


def quantifier_diagnostic(x: torch.Tensor, cote: str,
                         positions: Optional[torch.Tensor] = None,
                         fmt: str = "lm4") -> torch.Tensor:
    """Aller-retour lm4 sur ``x`` [..., D], SAUF où l'un des deux
    interrupteurs de diagnostic l'exempte — alors aller-retour int8 par
    amax à la place. Rend un tenseur déjà déquantifié (même forme/dtype que
    ``x``), pour une substitution directe dans un montage de PPL en
    décodage (même montage que `verdict-kv-lm4-qualite-17-09`) : ce
    diagnostic ne stocke rien, il ne fait que remplacer les valeurs lues.

    ``positions`` : indices de jeton par ligne de ``x`` (dim 0), ``None``
    si le puits n'est pas mesuré sur cet appel (`ACVRAM_KV_LM4_PUITS` est
    alors sans effet)."""
    if not actif(cote):
        return _int8_amax(x)
    if positions is None:
        q, s = quantifier(x, fmt)
        return dequantifier(q, s, fmt, x.dtype)
    dans_le_puits = ~hors_puits(positions)
    sortie = torch.empty_like(x)
    if dans_le_puits.any():
        sortie[dans_le_puits] = _int8_amax(x[dans_le_puits])
    hors = ~dans_le_puits
    if hors.any():
        q, s = quantifier(x[hors], fmt)
        sortie[hors] = dequantifier(q, s, fmt, x.dtype)
    return sortie
