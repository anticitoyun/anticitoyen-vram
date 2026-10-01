"""I5 (30/09) : récurrence GDN du décodage, AU BIT de `fused_recurrent_gated_delta_rule_fwd_kernel` (fla 0.5.2), avec
``J`` tuiles de valeurs par programme au lieu d'une.

Le noyau de fla lance un programme d'un warp par (séquence, tête de valeur, tuile de 8 colonnes) : à b=12 sur Qwen3.8,
9 216 programmes, dont chacun recharge q et k et refait leurs deux normes L2, ses portes et sa sigmoïde. Ici un programme
prend ``J`` tuiles CONSÉCUTIVES de la même tête et refait pour chacune EXACTEMENT le corps de fla, sur un bloc de même
forme [128, 8] et le même nombre de warps (1) — donc la même disposition des registres (`#blocked1` sizePerThread
[1, 1], threadsPerWarp [4, 8] : 32 lignes dans le fil, puis papillons 16, 8) et, vérifié à sec sur le PTX, la même
suite d'instructions flottantes (mêmes FMA contractées par LLVM). Le gain attendu vient de la grille (J fois moins de
programmes), pas d'un travail partagé : partager q, k et les portes change la contraction (voir le noyau).

L'égalité au bit n'est pas supposée : `tests/test_gdn_tuiles_au_bit.py` la vérifie sur carte (``torch.equal`` de la
sortie ET de l'état), et `scratchpad/poste5-i5-30-09/compare_ptx.py` compare à sec la suite des instructions flottantes
des deux PTX. ``J = 1`` redonne la grille de fla.
"""

from __future__ import annotations

import triton
import triton.language as tl

from fla.ops.utils.op import exp
from fla.ops.utils.softplus import softplus


@triton.jit(do_not_specialize=["T"])
def recurrence_tuiles_kernel(q, k, v, g, beta, A_log, dt_bias, o, S, scale, T,
                             H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                             BK: tl.constexpr, BV: tl.constexpr, J: tl.constexpr,
                             USE_GATE_IN_KERNEL: tl.constexpr, HAS_DT_BIAS: tl.constexpr,
                             APPLY_BETA_SIGMOID: tl.constexpr):
    # Le corps de fla (fused_recurrent.py:66-180, branches servies) recopié SANS le réordonner, répété pour J tuiles.
    # Partager q, k et les portes entre les tuiles change la contraction des FMA de LLVM (à sec, v4 : 12 voies FMA
    # par tuile au lieu de 64) : ce ne serait plus au bit. La boucle sur T (T = 1 à l'exécution) reste aussi : sans
    # elle, la contraction diffère déjà à J = 1 (à sec, K1 : 32 FMA contre 64).
    i_vj, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos = i_n * T
    for j in tl.static_range(J):
        o_k = tl.arange(0, BK)
        o_v = (i_vj * J + j) * BV + tl.arange(0, BV)
        p_q = q + (bos * H + i_h) * K + o_k
        p_k = k + (bos * H + i_h) * K + o_k
        p_v = v + (bos * HV + i_hv) * V + o_v
        p_g = g + bos * HV + i_hv
        p_beta = beta + bos * HV + i_hv
        p_o = o + (bos * HV + i_hv) * V + o_v
        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_k[:, None] & mask_v[None, :]
        b_h = tl.zeros([BK, BV], dtype=tl.float32)
        p_h0 = S + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
        for _ in tl.range(0, T):
            b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
            b_q = b_q * scale
            b_beta = tl.load(p_beta).to(tl.float32)
            if APPLY_BETA_SIGMOID:
                b_beta = tl.sigmoid(b_beta)
            b_g = tl.load(p_g).to(tl.float32)
            if USE_GATE_IN_KERNEL:
                b_A = tl.load(A_log + i_hv).to(tl.float32)
                if HAS_DT_BIAS:
                    b_g = b_g + tl.load(dt_bias + i_hv).to(tl.float32)
                b_g = -exp(b_A) * softplus(b_g)
            b_h *= exp(b_g)
            b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
            b_h += b_k[:, None] * b_v
            b_o = tl.sum(b_h * b_q[:, None], 0)
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
            p_q += H*K
            p_k += H*K
            p_v += HV*V
            p_g += HV
            p_beta += HV
            p_o += HV*V
        p_ht = S + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def recurrence_tuiles(q, k, v, g, beta, S, A_log=None, dt_bias=None, J: int = 2):
    """Même contrat que `gdn._recurrence_en_place` (T = 1, état ``S`` [N, HV, K, V] fp32 mis à jour en place, sortie
    ``o`` rendue) ; ``J`` tuiles de 8 colonnes par programme (diviseur de V / 8)."""
    import torch
    B, T, H, Kd = k.shape
    HV, Vd = v.shape[2], v.shape[-1]
    if T != 1:
        raise ValueError(f"recurrence_tuiles : décodage seul (T = 1), reçu T = {T}")
    BV = 8                                              # celui de fla : min(8, next_power_of_2(V))
    nv = triton.cdiv(Vd, BV)
    if nv % J:
        raise ValueError(f"recurrence_tuiles : J = {J} ne divise pas {nv} tuiles")
    o = torch.empty_like(v)
    with torch.cuda.device(q.device.index):
        recurrence_tuiles_kernel[(nv // J, B * HV)](
            q, k, v, g, beta, A_log, dt_bias, o, S, Kd ** -0.5, 1,
            H=H, HV=HV, K=Kd, V=Vd, BK=triton.next_power_of_2(Kd), BV=BV, J=J,
            USE_GATE_IN_KERNEL=A_log is not None, HAS_DT_BIAS=dt_bias is not None,
            APPLY_BETA_SIGMOID=A_log is not None, num_warps=1, num_stages=3)
    return o
