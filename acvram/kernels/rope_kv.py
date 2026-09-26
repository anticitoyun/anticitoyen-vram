"""RÉFUTÉ, TÉMOIN NOMMÉ (F3a, 17/09, verdict-f3a-finale-17-09 a3f1b7e) : ce noyau
n'est jamais défaut. Deux faits incompatibles, non résolus : (1) la seule
version saine en situ sous rejeu de graphe (celle-ci, d462c24 : PPL décodage
5,59 pour 5,54, cache Triton chaud ou vidé) ne rend pas les codes int8 de
kv_write_int8 aux demi-entiers exacts (tests carte 3/6 : contraction fma du
produit x·(1/sc), l'arrondi maison lit le produit non arrondi) ; (2) les quatre
variantes qui les rendent (sqrt_rn/fdiv ieee, inverses fp64, libdevice.rint,
produit fp64→fp32 : df4db39…ece3bed) corrompent le cache pas à pas SOUS GRAPHE
SEULEMENT (PPL 26 à 249 701, non reproductible ; saines en eager ; PTX sm_120
sans scratch, mémoire locale ni appel — cause non identifiée). Gain visé :
−48 lancements, −0,10 ms par pas à b=1 (rope_inplace + kv_write_int8). Reste
`ACVRAM_ROPE_KV=1`, opt-in, pour qui reprend la cause avec la sonde
outils/sonde-rope-kv-situ-17-09.py (à corriger : A et B dans deux processus,
sinon B charge DÉGRADÉ sans graphe et la sonde ne juge pas le régime qui casse).

Norme par tête + RoPE + écriture int8 du cache KV en UN lancement —
poste F, fusion (3a) (verdict-lancements-b1-17-09 : `rope_inplace` × 48
0,100 ms et `kv_write_int8` × 48 0,080 ms par pas à b = 1).

Un programme par (jeton, tête) : têtes de requête (norme q, rotation, écrit
q en place) puis têtes clé/valeur (norme k, rotation, k en place, puis k et
v quantifiés int8 — échelle max(amax/127, 1e-8) par (jeton, tête), codes
arrondis au plus proche — et dispersés au créneau du jeton ; créneau < 0 =
ligne de rembourrage, rien n'est écrit dans le cache).

Arithmétique de `rope_inplace_kernel` et `kv_write_int8_kernel`
(acvram_kernels.cu) : inv = rsqrt(moyenne(x²) + eps) sur la tête entière ;
pour i < d/2 : a = x[i]·inv·w[i], b = x[i+d/2]·inv·w[i+d/2],
out[i] = a·cos[i] − b·sin[i], out[i+d/2] = b·cos[i+d/2] + a·sin[i+d/2] ;
au-delà de d : x·inv·w ; une seule conversion bf16 en sortie. Juge :
tests/test_rope_kv.py (q, k à 1 ulp bf16 du calcul fp32 de référence, codes
int8 et échelles fp16 identiques, créneaux négatifs intacts). Décodage
``q_len = 1`` ; la version torch/CUDA reste le repli.

HISTORIQUE (17/09) — ce fichier est la version d462c24, la seule saine EN SITU
sous rejeu de graphe (poste3 : PPL décodage 5,59 pour 5,54, cache Triton chaud
ou vidé). Quatre variantes « plus exactes » (tl.sqrt_rn / tl.fdiv ieee, inverses
en fp64, libdevice.rint, produit fp64→fp32 : df4db39…ece3bed) rendent les
mêmes codes que kv_write_int8 aux demi-entiers exacts dans les tests unitaires,
mais corrompent le cache pas à pas SOUS GRAPHE SEULEMENT (PPL 26 à 249 701,
non reproductible ; saines en eager). Cause non identifiée : compilées hors
carte pour sm_120, elles n'ont ni scratch global, ni mémoire locale, ni appel
(outils/sonde-rope-kv-situ-17-09.py cherche où). Conséquence assumée : aux
demi-entiers exacts (x = amax/2), le code int8 peut différer de kv_write_int8
d'une unité — l'erreur de quantification y est la même (½ pas), le critère de
E (distance fp64 ≤ 1,1 ×) ne le voit pas ; tests/test_rope_kv.py le tolère.
"""
from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:                                        # noqa: BLE001
    triton = None
    tl = None


def disponible() -> bool:
    return triton is not None and (torch.cuda.is_available()
                                   or os.environ.get("TRITON_INTERPRET") == "1")


if triton is not None:

    @triton.jit
    def _rope_kv_kernel(q_ptr, k_ptr, v_ptr, cos_ptr, sin_ptr, pos_ptr, slots_ptr,
                        wq_ptr, wk_ptr, kc_ptr, ks_ptr, vc_ptr, vs_ptr,
                        HQ, HKV, eps,
                        stride_qt, stride_qh, stride_kt, stride_kh, stride_vt, stride_vh,
                        D: tl.constexpr, DR: tl.constexpr, NORME_Q: tl.constexpr, NORME_K: tl.constexpr):
        t = tl.program_id(0)
        h = tl.program_id(1)
        est_q = h < HQ
        i = tl.arange(0, D)
        tous = i >= 0
        base = tl.where(est_q, 0, 1)                     # 0 : tête q, 1 : tête k
        adr_q = q_ptr + t * stride_qt + h * stride_qh
        adr_k = k_ptr + t * stride_kt + (h - HQ) * stride_kh
        x = tl.load(adr_q + i, mask=tous & est_q, other=0.0).to(tl.float32) \
            + tl.load(adr_k + i, mask=tous & (base == 1), other=0.0).to(tl.float32)
        # norme par tête (RMSNorm sur la tête entière, eps partagé) ; poids 1 sans norme
        if NORME_Q:
            wq = tl.load(wq_ptr + i).to(tl.float32)
        else:
            wq = tl.full((D,), 1.0, tl.float32)
        if NORME_K:
            wk = tl.load(wk_ptr + i).to(tl.float32)
        else:
            wk = tl.full((D,), 1.0, tl.float32)
        w = tl.where(est_q, wq, wk)
        norme = tl.where(est_q, NORME_Q, NORME_K)        # constexpr 1/0 : normer cette tête ?
        inv = 1.0 / tl.sqrt(tl.sum(x * x, 0) / D + eps)
        inv = tl.where(norme == 1, inv, 1.0)
        y = x * inv * w
        # rotation des DR premières coordonnées (rotate_half) : partenaire i ± DR/2
        demi = DR // 2
        j = tl.where(i < demi, i + demi, i - demi)
        xp = tl.load(adr_q + j, mask=tous & est_q, other=0.0).to(tl.float32) \
            + tl.load(adr_k + j, mask=tous & (base == 1), other=0.0).to(tl.float32)
        if NORME_Q:
            wqp = tl.load(wq_ptr + j).to(tl.float32)
        else:
            wqp = tl.full((D,), 1.0, tl.float32)
        if NORME_K:
            wkp = tl.load(wk_ptr + j).to(tl.float32)
        else:
            wkp = tl.full((D,), 1.0, tl.float32)
        wp = tl.where(est_q, wqp, wkp)
        yp = xp * inv * wp
        pos = tl.load(pos_ptr + t)
        c = tl.load(cos_ptr + pos * DR + i, mask=i < DR, other=1.0)
        s = tl.load(sin_ptr + pos * DR + i, mask=i < DR, other=0.0)
        rot = tl.where(i < demi, y * c - yp * s, yp * s + y * c)
        rot = tl.where(i < DR, rot, y)
        out = rot.to(q_ptr.dtype.element_ty)
        tl.store(adr_q + i, out, mask=tous & est_q)
        tl.store(adr_k + i, out, mask=tous & (base == 1))
        # écriture int8 de k (tel qu'écrit, bf16) et de v au créneau du jeton
        slot = tl.load(slots_ptr + t)
        ecrit = (base == 1) & (slot >= 0)
        hk = tl.maximum(h - HQ, 0)
        kf = out.to(tl.float32)
        kmax = tl.maximum(tl.max(tl.abs(kf), 0) / 127.0, 1e-8)
        kcode = tl.minimum(tl.maximum(_rint(kf * (1.0 / kmax)), -127.0), 127.0)
        cell = tl.maximum(slot, 0) * HKV + hk
        tl.store(kc_ptr + cell * D + i, kcode.to(tl.int8), mask=tous & ecrit)
        tl.store(ks_ptr + cell + i * 0, kmax.to(tl.float16), mask=(i == 0) & ecrit)
        vv = tl.load(v_ptr + t * stride_vt + hk * stride_vh + i, mask=tous & (base == 1), other=0.0).to(tl.float32)
        vmax = tl.maximum(tl.max(tl.abs(vv), 0) / 127.0, 1e-8)
        vcode = tl.minimum(tl.maximum(_rint(vv * (1.0 / vmax)), -127.0), 127.0)
        tl.store(vc_ptr + cell * D + i, vcode.to(tl.int8), mask=tous & ecrit)
        tl.store(vs_ptr + cell + i * 0, vmax.to(tl.float16), mask=(i == 0) & ecrit)

    @triton.jit
    def _rint(x):
        """Arrondi au plus proche, égalités vers le pair (rint) — sans libdevice,
        pour que l'interpréteur et le noyau fassent la même chose."""
        f = tl.floor(x)
        r = x - f
        pair = (f % 2.0) == 0.0
        haut = (r > 0.5) | ((r == 0.5) & (pair == 0))
        return tl.where(haut, f + 1.0, f)


def rope_kv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cos32: torch.Tensor,
            sin32: torch.Tensor, positions: torch.Tensor, slots: torch.Tensor,
            wq, wk, eps: float, cache) -> None:
    """En place : ``q`` [T, HQ, D], ``k``/``v`` [T, HKV, D] bf16 (tranches
    d'une projection empilée acceptées : foulées passées), ``cos32/sin32``
    [max_pos, DR] fp32, ``positions`` [T] int64, ``slots`` [T] int64 (< 0 :
    rembourrage), ``wq``/``wk`` poids RMSNorm [D] ou None, ``cache``
    PagedKVCache int8 (k, v, k_scale, v_scale)."""
    T, HQ, D = q.shape
    HKV = k.shape[1]
    DR = cos32.shape[-1]
    assert D & (D - 1) == 0 and DR % 2 == 0 and DR <= D and k.shape[-1] == D and v.shape[-1] == D
    assert q.stride(2) == 1 and k.stride(2) == 1 and v.stride(2) == 1
    if T == 0:
        return
    pos = positions if positions.dtype == torch.int64 else positions.to(torch.int64)
    sl = slots if slots.dtype == torch.int64 else slots.to(torch.int64)
    _rope_kv_kernel[(T, HQ + HKV)](
        q, k, v, cos32, sin32, pos, sl,
        wq if wq is not None else q, wk if wk is not None else k,
        cache.k, cache.k_scale, cache.v, cache.v_scale,
        HQ, HKV, float(eps),
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        D=D, DR=DR, NORME_Q=1 if wq is not None else 0, NORME_K=1 if wk is not None else 0)
