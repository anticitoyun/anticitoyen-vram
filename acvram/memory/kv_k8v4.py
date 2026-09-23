"""Pièce 104 — cache KV « k8v4 » : clés int8 par (jeton, tête) comme aujourd'hui,
valeurs int4 symétriques par groupe de GROUPE canaux. Jumeau torch des noyaux
`kv_write_k8v4_kernel` et `paged_attn_partial_kernel<…, V4>` (acvram_kernels.cu).

Pourquoi (revue/poste5-piece104-kv-k8v4-23-09.md) : les clés portent deux tiers
du coût de qualité du KV int8 (C5, C5-b), les valeurs un tiers ; réduire V à
4 bits retire 22 % des octets du cache (808 contre 1 040 o par jeton et par
couche sur Coder) là où le lecteur paginé est déjà au plancher HBM (b=12) —
levier de contexte long et de lot, pas de b=1 court. Opt-in
``ACVRAM_KV_FORMAT=k8v4`` ; le défaut reste int8.

Forme, identique au bit entre ce module et les noyaux :

* un groupe = GROUPE canaux contigus d'une tête (D / GROUPE groupes) ; échelle
  ``sv = max(half(amax_groupe / 7), 2⁻²⁴)`` — arrondie en half AVANT de servir
  (l'échelle stockée est exactement celle qui a quantifié : erreur ≤ sv/2 par
  valeur, sans le désaccord float/half du chemin int8 ; 2⁻²⁴ = plus petit half
  non nul, jamais une division par zéro) ; code ``rn(v / sv)`` (division IEEE —
  le noyau emploie ``__fdiv_rn``, jamais la division approchée de
  ``--use_fast_math``) clampé à ± 7 : −8 n'est jamais émis, le quartet garde
  un bit de signe propre ;
* deux codes par octet : canal PAIR dans le quartet bas, canal IMPAIR dans le
  quartet haut ; déballage par ``(quartet ^ 8) − 8`` (extension de signe) ;
* déquantification ``v = code × sv[jeton, tête, groupe]`` (half → float32).
"""
from __future__ import annotations

import torch

FORMAT = "k8v4"
GROUPE = 32          # canaux par échelle : dans le noyau, un warp = un groupe (D threads, D % 32 == 0)
NIVEAU = 7           # codes dans [−7, 7]
PLANCHER = 2.0 ** -24   # plus petit half non nul : l'échelle stockée n'est jamais 0


# Repli 104 (1) — puits d'attention (scellé `scratchpad/poste1-p104s5-23-09/scelle-puits.md`, écrit avant le
# code) : les PUITS premières positions de chaque séquence gardent V en int8 par (jeton, tête) — K l'est déjà —
# dans une réserve indexée comme K (bloc, décalage, tête). Le V k8v4 est écrit PARTOUT quand même : tout lecteur
# qui ignore les puits reste juste (il lit du k8v4). Opt-in ``ACVRAM_KV_PUITS=16`` (0 ou 16 : 16 = un bloc, le
# premier de chaque séquence), sous ``ACVRAM_KV_FORMAT=k8v4`` seulement. v1 DE MESURE : la réserve couvre tous
# les blocs (octets du cache > int8) ; une réserve compacte ne se code que si le scellé tient.
PUITS_VALEURS = (0, 16)


def puits_demandes(env: dict | None = None) -> int:
    import os
    brut = (os.environ if env is None else env).get("ACVRAM_KV_PUITS", "0") or "0"
    try:
        n = int(brut)
    except ValueError:
        n = -1
    if n not in PUITS_VALEURS:
        raise ValueError(f"ACVRAM_KV_PUITS={brut!r} : attendu 0 ou 16 (un bloc de puits)")
    return n


PUITS = puits_demandes()


def groupes(d: int) -> int:
    if d % GROUPE:
        raise ValueError(f"k8v4 : dimension de tête {d} non multiple de {GROUPE}")
    return d // GROUPE


def octets_par_jeton_couche(hkv: int, d: int) -> int:
    """K : D codes + 1 half ; V : D/2 octets + D/GROUPE half — par tête."""
    return hkv * (d + 2) + hkv * (d // 2 + 2 * groupes(d))


def quantifier_k(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """K int8 par (jeton, tête), même forme que le chemin int8 — mais chaque
    division est une VRAIE division (tenseur / tenseur) : sur carte, torch
    remplace ``t / 127.0`` par ``t × (1/127)`` (1 ulp d'écart sur l'échelle,
    d'où ± 1 code aux demi-entiers contre le noyau, mesuré le 23/09 sur 2
    éléments / 18 944) ; le noyau `kv_write_k8v4_kernel` divise en IEEE."""
    x = k.to(torch.float32)
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = (amax / torch.full_like(amax, 127.0)).clamp(min=1e-8)
    q = (x / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale.squeeze(-1).to(torch.float16)


def quantifier_v(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``v`` [..., D] → (codes uint8 [..., D/2], échelles half [..., D/GROUPE])."""
    d = v.shape[-1]
    g = groupes(d)
    x = v.to(torch.float32).reshape(*v.shape[:-1], g, GROUPE)
    amax = x.abs().amax(dim=-1, keepdim=True)
    sv = (amax / float(NIVEAU)).to(torch.float16).to(torch.float32).clamp(min=PLANCHER)
    codes = (x / sv).round().clamp(-NIVEAU, NIVEAU).to(torch.int8)
    return emballer(codes.reshape(*v.shape[:-1], d)), sv.squeeze(-1).to(torch.float16)


def emballer(codes: torch.Tensor) -> torch.Tensor:
    """int8 [..., D] (valeurs dans [−7, 7]) → uint8 [..., D/2], pair en bas."""
    q = codes.to(torch.int16) & 0xF
    return (q[..., 0::2] | (q[..., 1::2] << 4)).to(torch.uint8)


def deballer(emballe: torch.Tensor) -> torch.Tensor:
    """uint8 [..., D/2] → int8 [..., D], extension de signe du quartet."""
    e = emballe.to(torch.int16)
    bas, haut = e & 0xF, (e >> 4) & 0xF
    codes = torch.stack((bas, haut), dim=-1).reshape(*emballe.shape[:-1], emballe.shape[-1] * 2)
    return ((codes ^ 8) - 8).to(torch.int8)


def dequantifier_v(emballe: torch.Tensor, echelles: torch.Tensor,
                   dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """(codes uint8 [..., D/2], échelles half [..., D/GROUPE]) → ``v`` [..., D]."""
    codes = deballer(emballe).to(torch.float32)
    d = codes.shape[-1]
    g = groupes(d)
    x = codes.reshape(*codes.shape[:-1], g, GROUPE) * echelles.to(torch.float32).unsqueeze(-1)
    return x.reshape(*codes.shape[:-1], d).to(dtype)
