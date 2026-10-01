"""État GDN en INT8 par fenêtre (LeapQuant, arXiv 2609.38166) — opt-in ``ACVRAM_ETAT_GDN=int8`` : la SORTIE CHANGE.

Pourquoi : au décodage, la récurrence fla lit et réécrit l'état fp32 entier à chaque pas (8 o/élément), et elle est
déjà au plancher lecture + écriture (revue/poste5-i5-mesure-verdict-01-10.md : 1,53 To/s au banc). Le seul levier
restant est de déplacer moins d'octets. Ici l'état d'une tête est gelé en INT8 au bord d'une fenêtre de ``P`` jetons ;
dans la fenêtre on ne relit que lui (1 o/élément) et l'on TAMPONNE les mises à jour de rang 1 de chaque jeton ; on ne
requantifie qu'au bord. L'erreur de quantification n'entre donc qu'une fois par fenêtre au lieu d'une fois par pas.

Récurrence (celle de fla, fused_recurrent.py:117-150, état [K, V]) : avec α_t = exp(g_t),
    S_t = α_t S_{t-1} + k_t w_tᵀ,   w_t = β_t (v_t − α_t S_{t-1}ᵀ k_t),   o_t = S_tᵀ q_t = α_t S_{t-1}ᵀ q_t + (k_t·q_t) w_t
(q, k normés L2 dans le noyau, q × K^-1/2). Dans une fenêtre commencée à l'état gelé S0, avec n enregistrements
(k_j, w_j, g_j) déjà tamponnés et P_j = Σ_{i≤j} g_i :
    S_nᵀ x = e^{P_n} S0ᵀ x + Σ_{j≤n} e^{P_n − P_j} (k_j·x) w_j .
S0 = C·(Z/127)·B + Σ_r k̃_r ũ_rᵀ : Z int8 [K, V] symétrique par colonne de valeur (échelle B [V]), lissage des lignes
de clé C [K] (racine de la moyenne des |valeurs| de la ligne, plancher), r = ``R`` « Compensator Tokens » fp16 ajustés
par itération de puissance (``ITERATIONS`` itérations, départ déterministe), soustraits AVANT la quantification.

Ce module porte la RÉFÉRENCE torch (processeur ou carte : la sémantique, que les tests jugent) et les noyaux Triton
(`pas`, `bord`) du décodage. Formes : ``N`` créneaux, ``HV`` têtes de valeur, ``H`` têtes de clé (GQA), K = V = 128.
"""

from __future__ import annotations

import os

import torch

P = int(os.environ.get("ACVRAM_ETAT_GDN_FENETRE", "16"))    # jetons par fenêtre (article : 16)
if P not in (1, 2, 4, 8, 16, 32, 64):
    raise ValueError(f"ACVRAM_ETAT_GDN_FENETRE={P} : puissance de 2 entre 1 et 64 (tl.arange du noyau)")
R = 4                                                        # compensateurs (article : 4 à 8 bits)
ITERATIONS = 2                                               # itérations de puissance par compensateur (article : non dit)
# Départ À CHAUD : chaque compensateur repart du ũ_r de la fenêtre précédente (l'état bouge peu en P jetons) ; à froid
# (premier gel) des normes L1 des colonnes. À froid il en faudrait ~8 : 68 produits matrice-vecteur par tête et par
# gel au lieu de 20, à un programme par SM (255 registres) — ~3 fois le temps d'un pas amorti sur la fenêtre.
PLANCHER_LISSAGE = 1e-6                                      # « small positive floor » de l'article
PLANCHER_ECHELLE = 1e-30


def nouvel_etat(N: int, HV: int, H: int, K: int = 128, V: int = 128, device="cpu",
                tampon: torch.dtype = torch.float16) -> dict:
    """Tampons d'un lot de créneaux ; un créneau neuf vaut l'état nul (Z = 0, n = 0). ``tampon`` : dtype des
    enregistrements (fp16 comme l'article ; fp32 sert au test de l'algèbre seule)."""
    f32, f16 = torch.float32, tampon
    return {
        "Z": torch.zeros(N, HV, K, V, dtype=torch.int8, device=device),
        "B": torch.zeros(N, HV, V, dtype=f32, device=device),       # échelle par colonne de valeur
        "C": torch.ones(N, HV, K, dtype=f32, device=device),        # lissage par ligne de clé
        "kc": torch.zeros(N, HV, R, K, dtype=torch.float16, device=device),   # compensateurs, côté clé
        "uc": torch.zeros(N, HV, R, V, dtype=torch.float16, device=device),   # compensateurs, côté valeur
        "bk": torch.zeros(N, H, P, K, dtype=f16, device=device),    # tampon : clés normées (par tête de clé)
        "bw": torch.zeros(N, HV, P, V, dtype=f16, device=device),   # tampon : w_j
        "bg": torch.zeros(N, HV, P, dtype=f32, device=device),      # tampon : g_j (log-décroissance)
        "n": torch.zeros(N, HV, dtype=torch.int32, device=device),  # enregistrements dans la fenêtre
    }


def octets_par_tete(K: int = 128, V: int = 128, H_sur_HV: float = 1 / 3) -> dict:
    """Octets stockés par tête de valeur (pour la note) : Z, échelles, compensateurs, tampon plein."""
    fixe = K * V + 4 * V + 4 * K + R * (K + V) * 2
    tampon = P * (2 * K * H_sur_HV + 2 * V + 4)
    return {"bord": fixe, "par_element_bord": fixe / (K * V), "tampon": tampon}


def _s0_t(et: dict, x: torch.Tensor) -> torch.Tensor:
    """S0ᵀ x pour x [N, HV, K] → [N, HV, V] (fp32)."""
    z = et["Z"].to(torch.float32)
    r = torch.einsum("nhkv,nhk->nhv", z, et["C"] * x) * et["B"] / 127.0
    kc, uc = et["kc"].to(torch.float32), et["uc"].to(torch.float32)
    return r + torch.einsum("nhrv,nhr->nhv", uc, torch.einsum("nhrk,nhk->nhr", kc, x))


def _s_t(et: dict, x: torch.Tensor, gqa: int) -> torch.Tensor:
    """S_nᵀ x (état courant de la fenêtre) pour x [N, HV, K]."""
    N, HV, _ = x.shape
    n = et["n"].to(torch.long)                                     # [N, HV]
    j = torch.arange(P, device=x.device)
    actif = (j[None, None, :] < n[..., None])                      # [N, HV, P]
    bg = torch.where(actif, et["bg"], torch.zeros_like(et["bg"]))
    pref = torch.cumsum(bg, dim=-1)                                # P_j
    Pn = pref.gather(-1, (n - 1).clamp(min=0)[..., None])[..., 0] * (n > 0)
    bk = et["bk"].to(torch.float32).repeat_interleave(gqa, dim=1)  # [N, HV, P, K]
    d = torch.einsum("nhpk,nhk->nhp", bk, x) * torch.exp(Pn[..., None] - pref) * actif
    return torch.exp(Pn)[..., None] * _s0_t(et, x) + torch.einsum("nhp,nhpv->nhv", d, et["bw"].to(torch.float32))


def etat_fp32(et: dict) -> torch.Tensor:
    """État courant reconstruit en fp32 [N, HV, K, V] (instantanés, export, bord)."""
    N, HV, K, V = et["Z"].shape
    gqa = HV // et["bk"].shape[1]
    S0 = et["C"][..., :, None] * et["Z"].to(torch.float32) / 127.0 * et["B"][..., None, :]
    S0 = S0 + torch.einsum("nhrk,nhrv->nhkv", et["kc"].to(torch.float32), et["uc"].to(torch.float32))
    n = et["n"].to(torch.long)
    j = torch.arange(P, device=S0.device)
    actif = (j[None, None, :] < n[..., None])
    bg = torch.where(actif, et["bg"], torch.zeros_like(et["bg"]))
    pref = torch.cumsum(bg, dim=-1)
    Pn = pref.gather(-1, (n - 1).clamp(min=0)[..., None])[..., 0] * (n > 0)
    poids = torch.exp(Pn[..., None] - pref) * actif                # [N, HV, P]
    bk = et["bk"].to(torch.float32).repeat_interleave(gqa, dim=1)
    return torch.exp(Pn)[..., None, None] * S0 + torch.einsum("nhp,nhpk,nhpv->nhkv", poids, bk,
                                                              et["bw"].to(torch.float32))


def geler(et: dict, S: torch.Tensor, masque: torch.Tensor | None = None) -> None:
    """Bord de fenêtre : S [N, HV, K, V] fp32 → compensateurs, lissage, INT8 ; tampon vidé. ``masque`` [N, HV] :
    seuls ces créneaux sont gelés (les autres gardent leur fenêtre en cours)."""
    N, HV, K, V = S.shape
    reste = S.to(torch.float32).clone()
    kc = torch.zeros(N, HV, R, K, dtype=torch.float16, device=S.device)
    uc = torch.zeros(N, HV, R, V, dtype=torch.float16, device=S.device)
    for r in range(R):
        chaud = et["uc"][:, :, r].to(torch.float32)
        froid = reste.abs().sum(dim=-2)                            # premier gel : normes L1 des colonnes
        v = torch.where((chaud.abs().sum(-1, keepdim=True) > 0), chaud, froid)
        v = v / v.norm(dim=-1, keepdim=True).clamp(min=PLANCHER_ECHELLE)
        for _ in range(ITERATIONS):
            u = torch.einsum("nhkv,nhv->nhk", reste, v)
            v = torch.einsum("nhkv,nhk->nhv", reste, u)
            v = v / v.norm(dim=-1, keepdim=True).clamp(min=PLANCHER_ECHELLE)
        u = torch.einsum("nhkv,nhv->nhk", reste, v)
        kc[:, :, r], uc[:, :, r] = u.to(torch.float16), v.to(torch.float16)
        reste = reste - kc[:, :, r].to(torch.float32)[..., :, None] * uc[:, :, r].to(torch.float32)[..., None, :]
    C = reste.abs().mean(dim=-1).sqrt().clamp(min=PLANCHER_LISSAGE)          # [N, HV, K]
    Rc = reste / C[..., :, None]
    B = Rc.abs().amax(dim=-2).clamp(min=PLANCHER_ECHELLE)                    # [N, HV, V]
    Z = torch.round(Rc / B[..., None, :] * 127.0).clamp(-127, 127).to(torch.int8)
    m = torch.ones(N, HV, dtype=torch.bool, device=S.device) if masque is None else masque
    et["Z"][m], et["B"][m], et["C"][m] = Z[m], B[m], C[m]
    et["kc"][m], et["uc"][m] = kc[m], uc[m]
    et["n"][m] = 0


def pas_reference(et: dict, q, k, v, g, beta, scale: float | None = None) -> torch.Tensor:
    """Un jeton de décodage par créneau, référence torch. q, k [N, H, K] (non normés), v [N, HV, V], g [N, HV]
    (log-décroissance), beta [N, HV] (après sigmoïde) — les formes de fla sans l'axe T. Rend o [N, HV, V] fp32 ;
    gèle les créneaux dont la fenêtre se remplit."""
    N, HV, V = v.shape
    H, K = q.shape[1], q.shape[2]
    gqa = HV // H
    scale = K ** -0.5 if scale is None else scale
    qn = q.float() / torch.sqrt((q.float() ** 2).sum(-1, keepdim=True) + 1e-6) * scale
    kn = k.float() / torch.sqrt((k.float() ** 2).sum(-1, keepdim=True) + 1e-6)
    kn16 = kn.to(et["bk"].dtype)                                   # la clé tamponnée (fp16) : on lit la même
    qh, kh = qn.repeat_interleave(gqa, dim=1), kn16.float().repeat_interleave(gqa, dim=1)
    a = torch.exp(g.float())
    sq, sk = _s_t(et, qh, gqa), _s_t(et, kh, gqa)
    w = beta.float()[..., None] * (v.float() - a[..., None] * sk)
    o = a[..., None] * sq + (kh * qh).sum(-1, keepdim=True) * w
    n = et["n"].to(torch.long)
    idx = n[..., None, None]
    et["bw"].scatter_(2, idx.expand(N, HV, 1, V), w.to(et["bw"].dtype)[:, :, None])
    et["bg"].scatter_(2, n[..., None], g.float()[..., None])
    nk = n.view(N, H, gqa)[..., 0]                                 # même n pour les têtes d'un groupe GQA
    et["bk"].scatter_(2, nk[..., None, None].expand(N, H, 1, K), kn16[:, :, None])
    et["n"] += 1
    plein = et["n"] >= P
    if bool(plein.any()):
        geler(et, etat_fp32(et), plein)
    return o


def pas_fp32(S: torch.Tensor, q, k, v, g, beta, scale: float | None = None) -> torch.Tensor:
    """La récurrence exacte (même algèbre que fla, en torch fp32) : témoin des tests ; S mis à jour en place."""
    N, HV, V = v.shape
    H, K = q.shape[1], q.shape[2]
    gqa = HV // H
    scale = K ** -0.5 if scale is None else scale
    qn = (q.float() / torch.sqrt((q.float() ** 2).sum(-1, keepdim=True) + 1e-6) * scale).repeat_interleave(gqa, 1)
    kn = (k.float() / torch.sqrt((k.float() ** 2).sum(-1, keepdim=True) + 1e-6)).repeat_interleave(gqa, 1)
    S.mul_(torch.exp(g.float())[..., None, None])
    w = beta.float()[..., None] * (v.float() - torch.einsum("nhkv,nhk->nhv", S, kn))
    S.add_(kn[..., :, None] * w[..., None, :])
    return torch.einsum("nhkv,nhk->nhv", S, qn)


# ---------------------------------------------------------------------------------------------------------------
# Noyaux Triton du décodage. Deux lancements par couche et par pas :
#   `_pas_kernel`  grille (N·HV, V/BVB) : lit Z (1 o/élément), échelles, compensateurs et le tampon, rend o, écrit
#                  l'enregistrement (w, g, k) à l'indice n — sans toucher n : les blocs de colonnes d'une même tête
#                  le lisent tous, l'incrémenter ici serait une course ;
#   `_bord_kernel` grille (N·HV) : n += 1 ; si la fenêtre est pleine, reconstruit S (récurrence sur le tampon, dans
#                  l'ordre), ajuste les compensateurs, lisse, quantifie, n = 0. Rare (1 pas sur P), un seul programme
#                  par tête car l'itération de puissance voit toute la matrice.
# Ils suivent la référence torch à l'ordre de sommation près (tolérance des tests sur carte), pas au bit.
# ---------------------------------------------------------------------------------------------------------------
import triton
import triton.language as tl


@triton.jit
def _portes(g_ptr, b_ptr, A_log, dt_bias, i_nh, i_hv, GATE_IN_KERNEL: tl.constexpr):
    g = tl.load(g_ptr + i_nh).to(tl.float32)
    b = tl.load(b_ptr + i_nh).to(tl.float32)
    if GATE_IN_KERNEL:                                     # F1 : portes brutes, comme fla (fused_recurrent.py:124-135)
        g = g + tl.load(dt_bias + i_hv).to(tl.float32)
        sp = tl.where(g > 20.0, g, tl.log(1.0 + tl.exp(g)))
        g = -tl.exp(tl.load(A_log + i_hv).to(tl.float32)) * sp
        b = tl.sigmoid(b)
    return g, b


@triton.jit
def _pas_kernel(q, k, v, g_ptr, b_ptr, A_log, dt_bias, o, Z, Bs, Cs, KC, UC, BKb, BWb, BGb, Nb, scale,
                H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BVB: tl.constexpr,
                P: tl.constexpr, R: tl.constexpr, GATE_IN_KERNEL: tl.constexpr):
    i_nh, i_vb = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    gqa = HV // H
    i_h = i_hv // gqa
    o_k = tl.arange(0, K)
    o_v = i_vb * BVB + tl.arange(0, BVB)
    o_j = tl.arange(0, P)
    o_r = tl.arange(0, R)
    n = tl.load(Nb + i_nh)

    b_q = tl.load(q + (i_n * H + i_h) * K + o_k).to(tl.float32)
    b_k = tl.load(k + (i_n * H + i_h) * K + o_k).to(tl.float32)
    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6) * scale
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_k16 = b_k.to(tl.float16)
    b_k = b_k16.to(tl.float32)                             # la clé tamponnée : on lit la même
    g, beta = _portes(g_ptr, b_ptr, A_log, dt_bias, i_nh, i_hv, GATE_IN_KERNEL)

    actif = o_j < n
    bg = tl.load(BGb + i_nh * P + o_j, mask=actif, other=0.0)
    pref = tl.cumsum(bg, 0)
    Pn = tl.sum(bg, 0)
    poids = tl.where(actif, tl.exp(Pn - pref), 0.0)                                         # [P]
    bk = tl.load(BKb + ((i_n * H + i_h) * P + o_j[:, None]) * K + o_k[None, :],
                 mask=actif[:, None], other=0.0).to(tl.float32)                             # [P, K]
    dq = tl.sum(bk * b_q[None, :], 1) * poids
    dk = tl.sum(bk * b_k[None, :], 1) * poids
    bw = tl.load(BWb + (i_nh * P + o_j[:, None]) * V + o_v[None, :], mask=actif[:, None], other=0.0).to(tl.float32)
    tq = tl.sum(dq[:, None] * bw, 0)
    tk = tl.sum(dk[:, None] * bw, 0)

    c = tl.load(Cs + i_nh * K + o_k)
    z = tl.load(Z + (i_nh * K + o_k[:, None]) * V + o_v[None, :]).to(tl.float32)            # [K, BVB]
    bsc = tl.load(Bs + i_nh * V + o_v) / 127.0
    s0q = tl.sum(z * (c * b_q)[:, None], 0) * bsc
    s0k = tl.sum(z * (c * b_k)[:, None], 0) * bsc
    kc = tl.load(KC + (i_nh * R + o_r[:, None]) * K + o_k[None, :]).to(tl.float32)        # [R, K]
    uc = tl.load(UC + (i_nh * R + o_r[:, None]) * V + o_v[None, :]).to(tl.float32)        # [R, BVB]
    s0q += tl.sum(tl.sum(kc * b_q[None, :], 1)[:, None] * uc, 0)
    s0k += tl.sum(tl.sum(kc * b_k[None, :], 1)[:, None] * uc, 0)

    eP = tl.exp(Pn)
    sq = eP * s0q + tq
    sk = eP * s0k + tk
    a = tl.exp(g)
    b_v = tl.load(v + i_nh * V + o_v).to(tl.float32)
    w = beta * (b_v - a * sk)
    b_o = a * sq + tl.sum(b_k * b_q) * w
    tl.store(o + i_nh * V + o_v, b_o.to(o.dtype.element_ty))
    tl.store(BWb + (i_nh * P + n) * V + o_v, w.to(tl.float16))
    if i_vb == 0:
        tl.store(BGb + i_nh * P + n, g)
        if i_hv % gqa == 0:
            tl.store(BKb + ((i_n * H + i_h) * P + n) * K + o_k, b_k16)


@triton.jit
def _bord_kernel(Z, Bs, Cs, KC, UC, BKb, BWb, BGb, Nb, total,
                 H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                 P: tl.constexpr, R: tl.constexpr, ITERATIONS: tl.constexpr,
                 PLANCHER_LISSAGE: tl.constexpr, PLANCHER_ECHELLE: tl.constexpr):
    # Persistant (grille ≤ nombre de SM) : 255 registres par fil, donc 1 programme par SM ; une grille d'un
    # programme par tête paierait ~3 vagues de sorties immédiates à chaque pas pour un gel sur P.
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    for i_nh in range(tl.program_id(0), total, tl.num_programs(0)):
      i_n, i_hv = i_nh // HV, i_nh % HV
      i_h = i_hv // (HV // H)
      n1 = tl.load(Nb + i_nh) + 1
      if n1 < P:
        tl.store(Nb + i_nh, n1)
      else:
        c = tl.load(Cs + i_nh * K + o_k)
        bsc = tl.load(Bs + i_nh * V + o_v) / 127.0
        S = c[:, None] * tl.load(Z + (i_nh * K + o_k[:, None]) * V + o_v[None, :]).to(tl.float32) * bsc[None, :]
        for r in tl.static_range(R):
            kc = tl.load(KC + (i_nh * R + r) * K + o_k).to(tl.float32)
            uc = tl.load(UC + (i_nh * R + r) * V + o_v).to(tl.float32)
            S += kc[:, None] * uc[None, :]
        for j in tl.static_range(P):                       # la récurrence elle-même, dans l'ordre du tampon
            gj = tl.load(BGb + i_nh * P + j)
            kj = tl.load(BKb + ((i_n * H + i_h) * P + j) * K + o_k).to(tl.float32)
            wj = tl.load(BWb + (i_nh * P + j) * V + o_v).to(tl.float32)
            S = S * tl.exp(gj) + kj[:, None] * wj[None, :]
        for r in tl.static_range(R):
            chaud = tl.load(UC + (i_nh * R + r) * V + o_v).to(tl.float32)        # départ à chaud (référence)
            vv = tl.where(tl.sum(tl.abs(chaud)) > 0, chaud, tl.sum(tl.abs(S), 0))
            vv = vv / tl.maximum(tl.sqrt(tl.sum(vv * vv)), PLANCHER_ECHELLE)
            for _ in tl.static_range(ITERATIONS):
                u = tl.sum(S * vv[None, :], 1)
                vv = tl.sum(S * u[:, None], 0)
                vv = vv / tl.maximum(tl.sqrt(tl.sum(vv * vv)), PLANCHER_ECHELLE)
            u16 = tl.sum(S * vv[None, :], 1).to(tl.float16)
            v16 = vv.to(tl.float16)
            tl.store(KC + (i_nh * R + r) * K + o_k, u16)
            tl.store(UC + (i_nh * R + r) * V + o_v, v16)
            S -= u16.to(tl.float32)[:, None] * v16.to(tl.float32)[None, :]
        c = tl.maximum(tl.sqrt(tl.sum(tl.abs(S), 1) / V), PLANCHER_LISSAGE)
        Rc = S / c[:, None]
        bmax = tl.maximum(tl.max(tl.abs(Rc), 0), PLANCHER_ECHELLE)
        x = Rc / bmax[None, :] * 127.0
        # arrondi au pair le plus proche (= torch.round, référence) écrit à la main : le rint des bibliothèques CUDA
        # n'existe pas sous TRITON_INTERPRET (CI sans carte). x − floor(x) est exact en fp32 pour |x| ≤ 127.
        f = tl.floor(x)
        d = x - f
        pair = (f - 2.0 * tl.floor(f * 0.5)) == 0.0
        zq = tl.where(d > 0.5, f + 1.0, tl.where(d < 0.5, f, tl.where(pair, f, f + 1.0)))
        zq = tl.minimum(tl.maximum(zq, -127.0), 127.0)
        tl.store(Z + (i_nh * K + o_k[:, None]) * V + o_v[None, :], zq.to(tl.int8))
        tl.store(Bs + i_nh * V + o_v, bmax)
        tl.store(Cs + i_nh * K + o_k, c)
        tl.store(Nb + i_nh, 0)


_SM: dict = {}


def _nb_sm(dev) -> int:
    if dev.type != "cuda":                                 # TRITON_INTERPRET=1 (CI sans carte) : grille arbitraire
        return 4
    if dev not in _SM:
        _SM[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    return _SM[dev]


def pas(et: dict, q, k, v, g, beta, A_log=None, dt_bias=None, BVB: int = 32) -> torch.Tensor:
    """Un jeton par créneau, sur carte. Formes de `GatedDeltaNet._lot_projete` : q, k [N, 1, H, K], v [N, 1, HV, V],
    g, beta [N, 1, HV] (brutes si ``A_log`` est donné : F1). Rend o [N, 1, HV, V] dans le dtype de v."""
    Nn, _, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    BVB = min(BVB, V)
    o = torch.empty_like(v)
    import contextlib
    with (torch.cuda.device(q.device.index) if q.is_cuda else contextlib.nullcontext()):
        _pas_kernel[(Nn * HV, V // BVB)](
            q, k, v, g, beta, A_log, dt_bias, o, et["Z"], et["B"], et["C"], et["kc"], et["uc"], et["bk"], et["bw"],
            et["bg"], et["n"], K ** -0.5, H=H, HV=HV, K=K, V=V, BVB=BVB, P=P, R=R,
            GATE_IN_KERNEL=A_log is not None, num_warps=4)
        _bord_kernel[(min(Nn * HV, _nb_sm(q.device)),)](
            et["Z"], et["B"], et["C"], et["kc"], et["uc"], et["bk"], et["bw"], et["bg"], et["n"], Nn * HV,
            H=H, HV=HV, K=K, V=V, P=P, R=R, ITERATIONS=ITERATIONS, PLANCHER_LISSAGE=PLANCHER_LISSAGE,
            PLANCHER_ECHELLE=PLANCHER_ECHELLE, num_warps=8)
    return o
