"""C5-b — clés du cache KV int8 à échelle PAR CANAL (et par tête) sur chaque
bloc de 16 jetons ; jumeau torch des noyaux `kv_canal_*` et
`paged_attn_partial_kernel<CANAL>` (acvram_kernels.cu).

Pourquoi (chantier-c5b-19-09, porte mesurée à sec ×0,245) : après RoPE les
canaux de K ont des amplitudes très inégales (amax/rms par canal 5-7 sur les
K réels de Coder-30B) ; une échelle par jeton, prise sur le canal le plus
grand, écrase les petits canaux. L'échelle par canal sur un bloc de 16 jetons
divise l'erreur de K par 4. V garde son échelle par jeton (lu après le
softmax, l'échelle par canal de V ne se replie pas dans q).

Forme (identique au bit entre ce module et les noyaux) :

* déquantification UNIFORME d'une cellule : ``k = code × ks[jeton, tête] ×
  sc[bloc, tête, canal]`` — ``ks`` half (le tenseur d'aujourd'hui), ``sc``
  E4M3 [NB, HKV, D] (128 o par (bloc, tête) pour D = 128 : +3,0 % du KV) ;
* bloc FERMÉ par canal : ``sc = E4M3↑(amax_canal × 16 / 127)`` (arrondi vers
  le HAUT : aucun code ne sature), ``ks = 2⁻⁴`` (exact en half) — le facteur
  16 place les échelles usuelles (amax 0,12…3 500) dans la plage normale de
  l'E4M3 (2⁻⁶…448), sinon la moitié des canaux tombait dans les sous-normaux
  à 3 bits ; codes ``rn(k × 16 / sc)`` en division IEEE ;
* bloc PAR JETON (repli : réserve de tampons épuisée) : ``sc = 1``,
  ``ks = amax_jeton / 127`` — exactement le chemin d'aujourd'hui ;
* bloc COURANT (ouvert, < 16 jetons au décodage) : K gardé en bf16 dans une
  ligne de la réserve ``tampon`` [R, 16, HKV, D] (32 Kio par ligne pour
  Coder, K seul : 16 Kio), ``tampon_de[bloc]`` = ligne ou −1 (fermé) ou −2
  (par jeton) ; quantifié par canal à la FERMETURE (16ᵉ jeton) ; le lecteur
  lit ce bloc en bf16. Au préfill un bloc reçu entier est quantifié
  directement, sans passer par la réserve.

Les trois passes de l'écriture (`ecrire`) sont celles des noyaux
`kv_canal_plan_kernel` (rôles), `kv_canal_write_kernel` (V par jeton, K
selon le rôle), `kv_canal_close_kernel` (fermeture, libération) : tout
s'exécute sur l'appareil, sans scalaire hôte — capturable dans un graphe.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

__all__ = ["ACTIF", "quantifier_k_par_canal", "dequantifier_k_par_canal",
           "e4m3_haut", "attention_reference", "ecrire", "dequantifier_cache"]

# Régime : ACVRAM_KV_INT8_CANAL=1 → clés par canal ; 0 (défaut jusqu'au
# scellé C5-b) → le chemin par jeton, témoin. Lu à l'import (regime.VARIABLES).
ACTIF = os.environ.get("ACVRAM_KV_INT8_CANAL", "0") == "1"
# Lignes bf16 de la réserve des blocs courants, PAR COUCHE : au moins le
# nombre de séquences ouvertes en même temps (un bloc courant chacune), plus
# les blocs abandonnés à moitié pleins tant qu'ils ne sont pas réemployés.
# Épuisée → le bloc repart PAR JETON (qualité d'aujourd'hui, jamais faux).
RANGS = int(os.environ.get("ACVRAM_KV_CANAL_RANGS", "64"))

KS_CANAL = 0.0625          # 2^-4 : ks des blocs fermés par canal, exact en half
SC_PAR_JETON = 0x38        # E4M3 de 1,0 : sc des blocs par jeton
E4M3_MAX = 0x7E            # 448


def e4m3_haut(x: torch.Tensor) -> torch.Tensor:
    """Plus petit E4M3 ≥ x (x ≥ 0 fini, fp32) — octets uint8. x ≤ 0 → 2⁻⁹
    (jamais d'échelle nulle) ; x ≥ 448 → 448 (saturation : amax ≥ 3 556).
    Même arithmétique entière que `e4m3_haut` du .cu : les deux côtés rendent
    le même octet, la comparaison au bit du test carte en dépend."""
    x = x.to(torch.float32).contiguous()
    u = x.view(torch.int32)
    fe = ((u >> 23) & 0xFF) - 127
    # normal : exposant fe+7, mantisse = 3 bits de poids fort ; +1 si un reste
    b_norm = ((fe + 7) << 3) | ((u >> 20) & 7)
    b_norm = b_norm + ((u & 0xFFFFF) != 0).to(torch.int32)
    # sous-normal (x < 2⁻⁶) : multiples de 2⁻⁹ ; 8 = 0x08 = 2⁻⁶, le report tombe juste
    m = torch.floor(x * 512.0)
    b_sub = m.to(torch.int32) + (m * 0.001953125 < x).to(torch.int32)
    b = torch.where(fe < -6, b_sub, b_norm)
    b = torch.where(x >= 448.0, torch.full_like(b, E4M3_MAX), b)
    b = torch.where(x > 0, b, torch.ones_like(b))
    return b.clamp(max=E4M3_MAX).to(torch.uint8)


def e4m3_vers_float(octets: torch.Tensor) -> torch.Tensor:
    return octets.view(torch.float8_e4m3fn).to(torch.float32)


def quantifier_k_par_canal(k_bloc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``k_bloc`` [16, HKV, D] (bf16 ou fp32) → codes int8 [16, HKV, D] et
    échelles ``sc`` uint8 E4M3 [HKV, D] ; ``ks`` vaut 2⁻⁴ (`KS_CANAL`).

    amax par canal sur les 16 jetons ; ``sc = E4M3↑(amax × 16 / 127)`` ;
    ``code = rn(k × 16 / sc)`` (arrondi au plus proche, pair en cas d'égalité,
    comme `__float2int_rn` et `torch.round`), borné à ±127 — la borne ne mord
    jamais puisque sc est arrondi vers le haut."""
    assert k_bloc.dim() == 3, k_bloc.shape
    x = k_bloc.to(torch.float32)
    amax = x.abs().amax(dim=0)                                  # [HKV, D]
    sc = e4m3_haut(amax * 16.0 / 127.0)
    s = e4m3_vers_float(sc)
    codes = torch.round((x * 16.0) / s.unsqueeze(0)).clamp(-127, 127).to(torch.int8)
    return codes, sc


def dequantifier_k_par_canal(codes: torch.Tensor, sc: torch.Tensor,
                             ks: Optional[torch.Tensor] = None,
                             dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """codes [T, HKV, D] × ks [T, HKV] (2⁻⁴ par défaut) × sc [HKV, D] ou
    [T, HKV, D] E4M3 (uint8 ou float8) → [T, HKV, D]."""
    s = sc.view(torch.float8_e4m3fn).to(torch.float32) if sc.dtype == torch.uint8 \
        else sc.to(torch.float32)
    if s.dim() == 2:
        s = s.unsqueeze(0)
    if ks is None:
        ks = torch.full(codes.shape[:2], KS_CANAL, dtype=torch.float32, device=codes.device)
    return (codes.to(torch.float32) * ks.to(torch.float32).unsqueeze(-1) * s).to(dtype)


def attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        scale: float, window: int = 0) -> torch.Tensor:
    """Attention de décodage de référence en fp32 : ``q`` [HQ, D], ``k``/``v``
    [T, HKV, D] déjà déquantifiés (K par canal, V par jeton, ou bf16 exacts),
    GQA par répétition, fenêtre glissante optionnelle → [HQ, D] fp32."""
    HQ, D = q.shape
    T, HKV, _ = k.shape
    n_rep = HQ // HKV
    lo = max(0, T - window) if window else 0
    kk = k[lo:].to(torch.float32).repeat_interleave(n_rep, dim=1)   # [t, HQ, D]
    vv = v[lo:].to(torch.float32).repeat_interleave(n_rep, dim=1)
    s = torch.einsum("hd,thd->ht", q.to(torch.float32), kk) * scale
    p = torch.softmax(s, dim=-1)
    return torch.einsum("ht,thd->hd", p, vv)


# ---------------------------------------------------------------------------
# Écriture : jumeau des trois noyaux (rôles, écriture, fermeture)
# ---------------------------------------------------------------------------
ROLE_RIEN, ROLE_DIRECT, ROLE_TAMPON, ROLE_PAR_JETON = 0, 1, 2, 3


def _plan(cache, slots: torch.Tensor) -> tuple[list[int], list[int]]:
    """Rôle de chaque jeton de l'appel (kv_canal_plan_kernel) :
    0 rien (rembourrage, ou couvert par un meneur direct) · 1 meneur DIRECT
    (les 16 jetons du bloc sont dans cet appel, à partir du décalage 0 :
    quantification par canal immédiate ; ``rang`` = ligne périmée à rendre)
    · 2 TAMPON (bloc partiel : K en bf16 dans sa ligne) · 3 PAR JETON (réserve
    épuisée, ou bloc jamais ouvert par ce chemin).
    Alloue les lignes des blocs qui s'ouvrent ; le reste lit ``tampon_de``."""
    bs = cache.cfg.block_size
    sl = slots.tolist()
    T = len(sl)
    role, rang = [ROLE_RIEN] * T, [-1] * T
    td = cache.tampon_de
    for t, slot in enumerate(sl):
        if slot < 0:
            continue
        blk, off = slot // bs, slot % bs
        base = slot - off
        t0 = t - off
        meneur_ici = t0 >= 0 and sl[t0] == base
        if meneur_ici:
            plein = t0 + bs - 1 < T and sl[t0 + bs - 1] == base + bs - 1
            if plein:
                if t == t0:
                    role[t], rang[t] = ROLE_DIRECT, int(td[blk])
                    td[blk] = -1
                continue
            if t == t0:
                r = int(td[blk])
                if r < 0:
                    r = _prendre_ligne(cache)
                    if r < 0:
                        r = -2
                        _sc_par_jeton(cache, blk)
                    td[blk] = r
            role[t] = ROLE_TAMPON
            continue
        r = int(td[blk])
        if r >= 0:
            role[t] = ROLE_TAMPON
            continue
        if r == -1:                        # anomalie : bloc jamais ouvert par ce chemin
            td[blk] = -2
            _sc_par_jeton(cache, blk)
        role[t] = ROLE_PAR_JETON
    return role, rang


def _sc_par_jeton(cache, blk: int) -> None:
    """Octets E4M3 de 16,0 sur la ligne sc du bloc (bloc par jeton)."""
    cache.k_scale_canal.view(torch.uint8)[blk] = SC_PAR_JETON


def _prendre_ligne(cache) -> int:
    n = int(cache.tampon_sommet[0])
    if n <= 0:
        return -1
    cache.tampon_sommet[0] = n - 1
    return int(cache.tampon_libres[n - 1])


def _rendre_ligne(cache, r: int) -> None:
    n = int(cache.tampon_sommet[0])
    cache.tampon_libres[n] = r
    cache.tampon_sommet[0] = n + 1


def _par_jeton(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Une ligne [HKV, D] → codes int8 + échelle half par tête (le chemin
    d'aujourd'hui, kvcache._quantize)."""
    amax = x.abs().amax(dim=-1, keepdim=True).to(torch.float32)
    scale = (amax / 127.0).clamp(min=1e-8)
    q = (x.to(torch.float32) / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale.squeeze(-1).to(torch.float16)


def ecrire(cache, slots: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    """Écriture d'un appel (jumeau torch ; V par jeton toujours)."""
    bs = cache.cfg.block_size
    H = cache.cfg.num_kv_heads
    role, rang = _plan(cache, slots)
    sl = slots.tolist()
    kc, vc = cache.k, cache.v
    for t, slot in enumerate(sl):
        if slot < 0:
            continue
        blk, off = slot // bs, slot % bs
        vq, vs = _par_jeton(v[t])
        vc[blk, off] = vq
        cache.v_scale[blk, off] = vs
        ro = role[t]
        if ro == ROLE_RIEN:                        # couvert par son meneur direct
            continue
        if ro == ROLE_DIRECT:
            codes, sc = quantifier_k_par_canal(k[t:t + bs])
            kc[blk] = codes
            cache.k_scale_canal[blk] = sc.view(torch.float8_e4m3fn)
            cache.k_scale[blk] = KS_CANAL
            if rang[t] >= 0:
                _rendre_ligne(cache, rang[t])
        elif ro == ROLE_TAMPON and int(cache.tampon_de[blk]) >= 0:
            cache.tampon[int(cache.tampon_de[blk]), off] = k[t].to(torch.bfloat16)
        else:                                      # par jeton : sc = 1 déjà posé
            kq, ks = _par_jeton(k[t])
            kc[blk, off] = kq
            cache.k_scale[blk, off] = ks
    # fermeture (kv_canal_close_kernel) : après TOUTES les écritures de l'appel
    for t, slot in enumerate(sl):
        if slot < 0 or role[t] != ROLE_TAMPON:
            continue
        blk, off = slot // bs, slot % bs
        r = int(cache.tampon_de[blk])
        if off != bs - 1 or r < 0:
            continue
        codes, sc = quantifier_k_par_canal(cache.tampon[r])
        kc[blk] = codes
        cache.k_scale_canal[blk] = sc.view(torch.float8_e4m3fn)
        cache.k_scale[blk] = KS_CANAL
        cache.tampon_de[blk] = -1
        _rendre_ligne(cache, r)


def dequantifier_cache(cache, blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """K déquantifié des blocs ``blocks`` (forme quelconque […]) → [..., 16,
    HKV, D] : code × ks × sc, puis les blocs COURANTS remplacés par leur ligne
    bf16 — en `torch.where` à forme fixe (capturable, `gather_fixed`)."""
    codes = cache.k[blocks]                                   # [..., 16, HKV, D]
    ks = cache.k_scale[blocks].to(torch.float32).unsqueeze(-1)
    sc = cache.k_scale_canal[blocks].to(torch.float32).unsqueeze(-3)
    deq = codes.to(torch.float32) * ks * sc
    rangs = cache.tampon_de[blocks].to(torch.long)            # [...]
    brut = cache.tampon[rangs.clamp(min=0)].to(torch.float32)
    ouvert = (rangs >= 0).view(*rangs.shape, 1, 1, 1)
    return torch.where(ouvert, brut, deq).to(dtype)
