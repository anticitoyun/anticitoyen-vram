"""Sonde (β) du niveau 2 : `ACVRAM_MLA_QABS_DEUX_MOITIES=1` change l'ordre de somme de
q_abs sur le chemin =1 sans noyau — ≤ 1 ulp bf16 de l'einsum bf16, jamais plus (sinon la
sonde mesurerait autre chose qu'un ordre de somme)."""
from __future__ import annotations

import torch


def test_deux_moities_restent_a_un_ulp_de_l_einsum():
    torch.manual_seed(0)
    nh, rank, nope = 4, 64, 128
    k_b = (torch.randn(nh, rank, nope) * 0.1).to(torch.bfloat16); q = torch.randn(1, nh, nope).to(torch.bfloat16)
    ref = torch.einsum('hrn,thn->thr', k_b, q)
    m = nope // 2; kb = k_b.float(); qn = q.float()
    deux = (torch.einsum('hrn,thn->thr', kb[..., :m], qn[..., :m]) + torch.einsum('hrn,thn->thr', kb[..., m:], qn[..., m:])).to(torch.bfloat16)
    a = ref.float(); u = 2.0 ** (torch.floor(torch.log2(a.abs().clamp_min(1e-30))) - 7)
    ecart = ((a - deux.float()).abs() / u)
    assert float(ecart.max()) <= 1.0 + 1e-6, float(ecart.max())
    # à sec (CPU) l'einsum bf16 accumule en fp32 et arrondit une fois : les deux moitiés rendent
    # souvent les mêmes bits — le bras ne diffère que contre cuBLAS sur carte ; le juge de la
    # sonde est la PPL 8 192 + 512, pas ce test
