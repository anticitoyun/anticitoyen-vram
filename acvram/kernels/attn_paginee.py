"""Attention de décodage paginée INT8 en Triton — poste E (poste7-poste-d-verdict,
poste7-b0-et-cause-lm4-17-09) : `paged_attn_partial_kernel` (acvram_kernels.cu:891)
lit 4 octets par voie, un jeton par warp, et relit chaque page K/V une fois
PAR TÊTE DE REQUÊTE (grille (B·QL, HQ, C)) — ~1 Go par pas à 13 % de la bande
passante (poste3 cead3cb).

Ici un programme sert UN GROUPE GQA (b, tête KV, tranche de contexte) : les
``n_rep`` têtes de requête du groupe forment la tuile Q [16, D] et chaque
page de K/V n'est lue qu'une fois pour toutes (n_rep = 8 sur Coder-30B :
huit fois moins d'octets) ; les 16 jetons d'une page sont chargés d'un bloc
(128 octets contigus par jeton), les scores et la sortie passent par
``tl.dot`` en bf16 (int8 × échelle fp16 par jeton, exact en bf16), softmax en
ligne fp32 ; les tranches de contexte sont réduites par un second noyau.
Godet sans fantômes : une séquence de longueur 0 sort au premier test.

Périmètre : décodage ``q_len = 1`` (la vérification spéculative reste au
noyau CUDA), D ∈ {64, 128}. Sortie = noyau CUDA ± 2⁻⁸ (juge :
tests/test_attn_paginee.py, référence float64 et bras qui doivent casser).
Sans carte : ``TRITON_INTERPRET=1`` en fp16.
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

PAGE = 16                  # jetons par bloc du cache (kvcache.BLOCK_SIZE)
NREP_TUILE = 16            # lignes de la tuile Q : n_rep rembourré (tl.dot ≥ 16)
PAGES_PAR_TUILE = 4        # 64 jetons par itération
TRANCHES_MAX = 32          # au-delà, la réduction des tranches coûte plus que le parallélisme
                           # (b=1 ctx 2048 : 64 tranches de 32 jetons = 0,828 ms contre 0,704 CUDA)


def disponible() -> bool:
    return triton is not None and (torch.cuda.is_available()
                                   or os.environ.get("TRITON_INTERPRET") == "1")


if triton is not None:

    @triton.jit
    def _tranche(q, kc_ptr, ks_ptr, vc_ptr, vs_ptr, tables_ptr, b, hkv, N, debut, fin, debut_page, scale,
                 stride_page, stride_tok, stride_kvh, stride_sp, stride_st,
                 D: tl.constexpr, BN: tl.constexpr, PAGE_C: tl.constexpr, NREP_T: tl.constexpr):
        """Softmax en ligne d'UNE tranche de contexte pour la tuile q [NREP_T, D] :
        rend (m, l, acc) — le corps commun de `_partiel_kernel` (deux noyaux,
        témoin) et de `_partiel_reduit_kernel` (C15 niveau 3, un noyau)."""
        d = tl.arange(0, D)
        m = tl.full((NREP_T,), float("-inf"), tl.float32)
        l = tl.zeros((NREP_T,), tl.float32)
        acc = tl.zeros((NREP_T, D), tl.float32)
        tt = tl.arange(0, BN)
        for t0 in range(debut_page, fin, BN):
            t = t0 + tt
            valide = (t < fin) & (t >= debut)
            page = tl.load(tables_ptr + b * N + t // PAGE_C, mask=valide, other=0)
            cell = page * stride_page + (t % PAGE_C) * stride_tok + hkv * stride_kvh
            k = tl.load(kc_ptr + cell[:, None] * D + d[None, :], mask=valide[:, None], other=0)
            sk = tl.load(ks_ptr + page * stride_sp + (t % PAGE_C) * stride_st + hkv,
                         mask=valide, other=0.0)
            s = tl.dot(q, tl.trans(k.to(q.dtype)))                       # [NREP_T, BN] fp32
            s = s * (sk.to(tl.float32) * scale)[None, :]
            s = tl.where(valide[None, :], s, float("-inf"))
            m_new = tl.maximum(m, tl.max(s, 1))
            # tuile sans jeton valide (bord de fenêtre) : m reste -inf et
            # -inf - -inf = NaN ; on neutralise au lieu de propager
            m_sur = tl.where(m_new == float("-inf"), 0.0, m_new)
            corr = tl.exp(m - m_sur)
            p = tl.exp(s - m_sur[:, None])
            l = l * corr + tl.sum(p, 1)
            v = tl.load(vc_ptr + cell[:, None] * D + d[None, :], mask=valide[:, None], other=0)
            sv = tl.load(vs_ptr + page * stride_sp + (t % PAGE_C) * stride_st + hkv,
                         mask=valide, other=0.0)
            # p·v en fp16 : p ∈ [0, 1] × échelle et v int8 y sont exacts à 2⁻¹²
            # (en bf16, chaque probabilité perdait 2⁻⁹ : 264/384 sorties hors
            # 2⁻⁸ du noyau CUDA, poste3) ; la somme reste fp32
            pv = (p * sv.to(tl.float32)[None, :]).to(tl.float16)
            acc = acc * corr[:, None] + tl.dot(pv, v.to(tl.float16))
            m = m_new
        return m, l, acc

    @triton.jit
    def _partiel_kernel(q_ptr, kc_ptr, ks_ptr, vc_ptr, vs_ptr, tables_ptr, lens_ptr,
                        part_ptr, pm_ptr, pl_ptr,
                        HQ, HKV, N, C, chunk, scale, window,
                        stride_qb, stride_qh, stride_page, stride_tok, stride_kvh,
                        stride_sp, stride_st,
                        NREP: tl.constexpr, D: tl.constexpr, BN: tl.constexpr,
                        PAGE_C: tl.constexpr, NREP_T: tl.constexpr):
        # PAGE_C / NREP_T en arguments constexpr : Triton 3.8 refuse un global
        # du module dans un @jit (l'interpréteur l'acceptait — poste3, banc E)
        b = tl.program_id(0)
        hkv = tl.program_id(1)
        c = tl.program_id(2)
        slen = tl.load(lens_ptr + b)
        lo = tl.maximum(slen - window, 0)                 # window = « infini » si aucune fenêtre
        debut = tl.maximum(c * chunk, lo)                  # premier jeton VU par cette tranche
        fin = tl.minimum(slen, (c + 1) * chunk)
        debut_page = (debut // PAGE_C) * PAGE_C           # la boucle avance page par page
        rows = tl.arange(0, NREP_T)
        h = hkv * NREP + rows
        masque_h = rows < NREP
        d = tl.arange(0, D)
        base = ((b * HKV + hkv) * C + c)
        if debut >= fin:
            # tranche vide (contexte plus court, ou fantôme du godet)
            tl.store(pm_ptr + base * NREP_T + rows, tl.full((NREP_T,), float("-inf"), tl.float32))
            tl.store(pl_ptr + base * NREP_T + rows, tl.zeros((NREP_T,), tl.float32))
            tl.store(part_ptr + (base * NREP_T + rows[:, None]) * D + d[None, :],
                     tl.zeros((NREP_T, D), tl.float32))
            return
        # q lu tel quel (bf16 du modèle) : le facteur d'échelle s'applique aux
        # scores en fp32 — pré-multiplier q puis l'arrondir en 16 bits
        # déplaçait les logits de 20 × 2⁻¹¹ sur un puits, hors 2⁻⁸ en sortie
        q = tl.load(q_ptr + b * stride_qb + h[:, None] * stride_qh + d[None, :],
                    mask=masque_h[:, None], other=0.0)
        m, l, acc = _tranche(q, kc_ptr, ks_ptr, vc_ptr, vs_ptr, tables_ptr, b, hkv, N, debut, fin, debut_page,
                             scale, stride_page, stride_tok, stride_kvh, stride_sp, stride_st,
                             D, BN, PAGE_C, NREP_T)
        tl.store(pm_ptr + base * NREP_T + rows, m)
        tl.store(pl_ptr + base * NREP_T + rows, l)
        tl.store(part_ptr + (base * NREP_T + rows[:, None]) * D + d[None, :], acc)

    @triton.jit
    def _reduce_kernel(part_ptr, pm_ptr, pl_ptr, out_ptr, HKV, C,
                       stride_ob, stride_oh,
                       NREP: tl.constexpr, D: tl.constexpr, CT: tl.constexpr,
                       NREP_T: tl.constexpr, DB: tl.constexpr):
        b = tl.program_id(0)
        hkv = tl.program_id(1)
        db = tl.program_id(2)                             # tranche de D : la réduction
        rows = tl.arange(0, NREP_T)                       # de C tranches se parallélise
        masque_h = rows < NREP
        cs = tl.arange(0, CT)
        masque_c = cs < C
        base = (b * HKV + hkv) * C
        m = tl.load(pm_ptr + (base + cs[None, :]) * NREP_T + rows[:, None],
                    mask=masque_c[None, :], other=float("-inf"))          # [NREP_T, CT]
        mg = tl.max(m, 1)
        mg = tl.where(mg == float("-inf"), 0.0, mg)         # séquence vide (fantôme du godet)
        w = tl.where(masque_c[None, :], tl.exp(m - mg[:, None]), 0.0)
        l = tl.load(pl_ptr + (base + cs[None, :]) * NREP_T + rows[:, None],
                    mask=masque_c[None, :], other=0.0)
        lg = tl.sum(l * w, 1)
        d = db * DB + tl.arange(0, DB)
        acc = tl.zeros((NREP_T, DB), tl.float32)
        for c in range(0, C):
            wc = tl.exp(tl.load(pm_ptr + (base + c) * NREP_T + rows) - mg)
            a = tl.load(part_ptr + ((base + c) * NREP_T + rows[:, None]) * D + d[None, :])
            acc += a * wc[:, None]
        out = tl.where(lg[:, None] > 0, acc / lg[:, None], 0.0)   # fantôme : sortie nulle et FINIE
        tl.store(out_ptr + b * stride_ob + (hkv * NREP + rows[:, None]) * stride_oh + d[None, :],
                 out.to(out_ptr.dtype.element_ty), mask=masque_h[:, None])


    @triton.jit
    def _partiel_reduit_kernel(q_ptr, kc_ptr, ks_ptr, vc_ptr, vs_ptr, tables_ptr, lens_ptr,
                               part_ptr, pm_ptr, pl_ptr, cnt_ptr, out_ptr,
                               HQ, HKV, N, C, chunk, scale, window,
                               stride_qb, stride_qh, stride_page, stride_tok, stride_kvh,
                               stride_sp, stride_st, stride_ob, stride_oh,
                               NREP: tl.constexpr, D: tl.constexpr, BN: tl.constexpr,
                               PAGE_C: tl.constexpr, NREP_T: tl.constexpr, CT: tl.constexpr,
                               DEROULE: tl.constexpr = False):
        """C15 niveau 3 : `_partiel_kernel` + `_reduce_kernel` en UN lancement.
        Chaque programme écrit sa tranche (m, l, acc) puis incrémente le
        compteur de son groupe (b, tête KV) ; le DERNIER arrivé (n == C − 1)
        réduit les C tranches dans l'ordre 0..C−1 — le même ordre quel que soit
        l'arrivant, donc déterministe — avec l'arithmétique de `_reduce_kernel`,
        écrit la sortie et remet le compteur à zéro (aucun memset par pas, le
        graphe rejoue tel quel). Visibilité des tranches des autres programmes :
        barrière de bloc avant l'atomique acq_rel (portée gpu), barrière après,
        lectures .cg (L2). Une tranche vide (m = −inf, l = 0, acc = 0) tombe
        dans le même chemin que le témoin (pas de retour anticipé : la boucle
        ne tourne pas et rend ces valeurs)."""
        b = tl.program_id(0)
        hkv = tl.program_id(1)
        c = tl.program_id(2)
        slen = tl.load(lens_ptr + b)
        lo = tl.maximum(slen - window, 0)
        debut = tl.maximum(c * chunk, lo)
        fin = tl.minimum(slen, (c + 1) * chunk)
        debut_page = (debut // PAGE_C) * PAGE_C
        rows = tl.arange(0, NREP_T)
        h = hkv * NREP + rows
        masque_h = rows < NREP
        d = tl.arange(0, D)
        grp = b * HKV + hkv
        base = grp * C + c
        q = tl.load(q_ptr + b * stride_qb + h[:, None] * stride_qh + d[None, :],
                    mask=masque_h[:, None], other=0.0)
        m, l, acc = _tranche(q, kc_ptr, ks_ptr, vc_ptr, vs_ptr, tables_ptr, b, hkv, N, debut, fin, debut_page,
                             scale, stride_page, stride_tok, stride_kvh, stride_sp, stride_st,
                             D, BN, PAGE_C, NREP_T)
        tl.store(pm_ptr + base * NREP_T + rows, m)
        tl.store(pl_ptr + base * NREP_T + rows, l)
        tl.store(part_ptr + (base * NREP_T + rows[:, None]) * D + d[None, :], acc)
        tl.debug_barrier()
        n = tl.atomic_add(cnt_ptr + grp, 1, sem="acq_rel", scope="gpu")
        tl.debug_barrier()
        if n == C - 1:
            cs = tl.arange(0, CT)
            masque_c = cs < C
            base0 = grp * C
            mm = tl.load(pm_ptr + (base0 + cs[None, :]) * NREP_T + rows[:, None],
                         mask=masque_c[None, :], other=float("-inf"), cache_modifier=".cg")
            mg = tl.max(mm, 1)
            mg = tl.where(mg == float("-inf"), 0.0, mg)
            w = tl.where(masque_c[None, :], tl.exp(mm - mg[:, None]), 0.0)
            ll = tl.load(pl_ptr + (base0 + cs[None, :]) * NREP_T + rows[:, None],
                         mask=masque_c[None, :], other=0.0, cache_modifier=".cg")
            lg = tl.sum(ll * w, 1)
            somme = tl.zeros((NREP_T, D), tl.float32)
            if DEROULE:
                # Pièce 92 : la boucle série ci-dessous attend chaque tranche avant de lire la suivante — C
                # latences L2 sur le chemin critique du dernier programme. Déroulée (CT constexpr), toutes les
                # lectures partent ensemble ; les SOMMES restent dans l'ordre 0..C−1 et `where` laisse `somme`
                # intact au-delà de C : même arithmétique, sortie au bit.
                for cc in tl.static_range(CT):
                    ok = cc < C
                    wc = tl.exp(tl.load(pm_ptr + (base0 + cc) * NREP_T + rows, mask=ok & (rows >= 0),
                                        other=0.0, cache_modifier=".cg") - mg)
                    a = tl.load(part_ptr + ((base0 + cc) * NREP_T + rows[:, None]) * D + d[None, :],
                                mask=ok & (rows[:, None] >= 0), other=0.0, cache_modifier=".cg")
                    somme = tl.where(ok, somme + a * wc[:, None], somme)
            else:
                for cc in range(0, C):
                    wc = tl.exp(tl.load(pm_ptr + (base0 + cc) * NREP_T + rows, cache_modifier=".cg") - mg)
                    a = tl.load(part_ptr + ((base0 + cc) * NREP_T + rows[:, None]) * D + d[None, :],
                                cache_modifier=".cg")
                    somme += a * wc[:, None]
            out = tl.where(lg[:, None] > 0, somme / lg[:, None], 0.0)
            tl.store(out_ptr + b * stride_ob + (hkv * NREP + rows[:, None]) * stride_oh + d[None, :],
                     out.to(out_ptr.dtype.element_ty), mask=masque_h[:, None])
            tl.store(cnt_ptr + grp, 0)



def _tranches(n_pages: int, b: int, hkv: int, device) -> tuple[int, int]:
    """(C, chunk en jetons) : assez de programmes pour couvrir la carte à
    b = 1, une tranche par page au moins ; ne dépend que des formes (godet),
    donc stable sous un graphe CUDA."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count if device.type == "cuda" else 4
    voulu = max(1, min(TRANCHES_MAX, -(-2 * sms // max(1, b * hkv))))
    pages_par_tranche = max(PAGES_PAR_TUILE, -(-n_pages // voulu))
    pages_par_tranche = -(-pages_par_tranche // PAGES_PAR_TUILE) * PAGES_PAR_TUILE
    c = -(-n_pages // pages_par_tranche)
    return c, pages_par_tranche * PAGE


# C15-3c : le noyau fusionné à 8 warps (le témoin `_partiel_kernel` reste à 4) —
# les mêmes tuiles (k, v [64, 128] int8 → 16 bits, acc [16, 128] fp32, scores
# [16, 64]) réparties sur deux fois plus de fils : 178 registres/fil et 11 %
# d'occupation mesurés à 4 warps (verdict-c15-niveau3-coder-19-09 addendum
# 03 h 17) → prédit ≤ 110 registres, occupation ×2. Arithmétique inchangée
# (mêmes tuiles, mêmes tl.dot) ; seul l'ordre des réductions croisées (tl.sum
# de p, de l·w) peut suivre une autre disposition : au bit sous l'interpréteur,
# ≤ 1 ulp 16 bits sur carte (juge : tests/test_glue_compact.py).
# C15-3d : 8 warps = 103 registres/fil et occupation 24 % (poste2 04 h 40, contre 178 et 11 % à 4),
# durée inchangée 17,8 µs — et le bras B tire 369 W contre 349 (J +2,6 %, addendum 05 h 03) :
# ACVRAM_ATTN_WARPS_COMPACT=4 est le bras qui dit si ce sont ces warps (energie_par_poste).
WARPS_COMPACT = int(os.environ.get("ACVRAM_ATTN_WARPS_COMPACT", "8"))   # 8 : B8 +11,5 %, J 0,966 × ; B4 +9,1 %, 0,986 × et 369 W = 369 (poste2 05 h 15)
# Pièce 92 (23/09) : réduction des tranches déroulée (lectures en vol ensemble, mêmes sommes dans le même ordre) —
# au bit de la boucle série sur 11/11 cellules du banc L2 froid et au test (test_glue_compact : réduction déroulée),
# −1,9 % sur la moyenne du lot b=12 (15,46 → 15,17 µs/couche), −8,7 % à b=1 ctx 2 048. Témoin : =0.
REDUC_DEROULEE = os.environ.get("ACVRAM_ATTN_REDUC_DEROULEE", "1") == "1"
assert WARPS_COMPACT in (1, 2, 4, 8, 16), WARPS_COMPACT

_COMPTEURS: dict = {}


def _compteur(n: int, device) -> torch.Tensor:
    """Compteurs int32 [n] des groupes (b, tête KV), nuls entre deux lancements
    (le dernier programme remet le sien à zéro) : réservés une fois par forme
    et appareil, jamais pendant une capture de graphe (l'échauffement eager
    qui la précède les crée)."""
    cle = (n, str(device))
    c = _COMPTEURS.get(cle)
    if c is None:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("attn_paginee : compteurs du noyau fusionné réservés pendant une capture "
                               "de graphe — l'échauffement eager doit précéder (REGLES § 7)")
        c = _COMPTEURS[cle] = torch.zeros(n, dtype=torch.int32, device=device)
    return c


def paged_attention(q: torch.Tensor, kc: torch.Tensor, ks: torch.Tensor,
                    vc: torch.Tensor, vs: torch.Tensor, tables: torch.Tensor,
                    seq_lens: torch.Tensor, n_kv: int, scale: float,
                    window: int = 0, compact: bool = False) -> torch.Tensor:
    """``q`` [B, HQ, D] (un jeton par séquence), cache int8 ``kc/vc``
    [NB, 16, HKV, D] + échelles fp16 ``ks/vs`` [NB, 16, HKV], ``tables``
    [B, N] int64, ``seq_lens`` [B] int64 → [B, HQ, D] dans le dtype de q.
    ``compact`` (C15 niveau 3, ACVRAM_GLUE_COMPACT) : un seul lancement, la
    réduction des tranches faite par le dernier programme de chaque groupe."""
    B, HQ, D = q.shape
    n_rep = HQ // n_kv
    assert n_rep <= NREP_TUILE and D in (64, 128), (n_rep, D)
    N = tables.shape[1]
    C, chunk = _tranches(N, B, n_kv, q.device)
    f32 = dict(dtype=torch.float32, device=q.device)
    part = torch.empty(B * n_kv * C * NREP_TUILE, D, **f32)
    pm = torch.empty(B * n_kv * C * NREP_TUILE, **f32)
    pl = torch.empty(B * n_kv * C * NREP_TUILE, **f32)
    out = torch.empty_like(q)
    if compact:
        cnt = _compteur(B * n_kv, q.device)
        CT = 1
        while CT < C:
            CT *= 2
        _partiel_reduit_kernel[(B, n_kv, C)](
            q, kc, ks, vc, vs, tables, seq_lens, part, pm, pl, cnt, out,
            HQ, n_kv, N, C, chunk, float(scale), int(window) if window > 0 else 1 << 30,
            q.stride(0), q.stride(1), kc.stride(0) // D, kc.stride(1) // D, kc.stride(2) // D,
            ks.stride(0), ks.stride(1), out.stride(0), out.stride(1),
            NREP=n_rep, D=D, BN=PAGE * PAGES_PAR_TUILE, PAGE_C=PAGE, NREP_T=NREP_TUILE, CT=max(CT, 2),
            DEROULE=REDUC_DEROULEE, num_warps=WARPS_COMPACT, num_stages=2)
        return out
    _partiel_kernel[(B, n_kv, C)](
        q, kc, ks, vc, vs, tables, seq_lens, part, pm, pl,
        HQ, n_kv, N, C, chunk, float(scale), int(window) if window > 0 else 1 << 30,
        q.stride(0), q.stride(1), kc.stride(0) // D, kc.stride(1) // D, kc.stride(2) // D,
        ks.stride(0), ks.stride(1),
        NREP=n_rep, D=D, BN=PAGE * PAGES_PAR_TUILE, PAGE_C=PAGE, NREP_T=NREP_TUILE,
        num_warps=4, num_stages=2)
    CT = 1
    while CT < C:
        CT *= 2
    _reduce_kernel[(B, n_kv, D // 32)](part, pm, pl, out, n_kv, C, out.stride(0), out.stride(1),
                                       NREP=n_rep, D=D, CT=max(CT, 2), NREP_T=NREP_TUILE, DB=32,
                                       num_warps=4)
    return out
