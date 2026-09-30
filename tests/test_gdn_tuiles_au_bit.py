"""I5 (30/09) : `gdn_tuiles.recurrence_tuiles` (J tuiles par programme) AU BIT du noyau fla en place (règle 9,
`torch.equal`, jamais une tolérance).

Deux étages :
* à sec (sans carte) : le PTX sm_120 de `recurrence_tuiles_kernel` a EXACTEMENT la suite d'instructions flottantes de
  `fused_recurrent_gated_delta_rule_fwd_kernel` répétée J fois (mêmes FMA contractées par LLVM). Casse si l'on partage
  q, k et les portes entre les tuiles (v4 du 30/09 : 12 voies FMA par tuile au lieu de 64) ou si l'on retire la boucle
  sur T (K1 : 32 FMA contre 64) — les deux fautes réellement commises en écrivant le noyau ;
* sur carte : sortie ET état égaux au bit à fla sur 3 pas chaînés, dimensions de Qwen3.8 (HV 48) et Qwen3.6 (HV 32),
  F1 (portes dans le noyau) et sans, b ∈ {1, 2, 8, 12, 16}, J ∈ {1, 2, 4}. Contrôle de sensibilité : fla lui-même en
  tuiles de 16 colonnes (autre disposition, autre ordre des sommes sur K) doit DIFFÉRER — sinon la comparaison ne voit
  pas l'ordre de sommation et ne prouve rien."""
import re

import pytest
import torch

fla = pytest.importorskip("fla")
triton = pytest.importorskip("triton")

from acvram.engine import gdn as G                                  # noqa: E402
from acvram.engine.gdn_tuiles import recurrence_tuiles, recurrence_tuiles_kernel  # noqa: E402

_OPS = re.compile(r"^\s*(?:@!?%p\d+\s+)?((?:add|sub|mul|fma|div|sqrt|rcp|ex2|lg2|max|min|neg|abs|cvt|shfl|setp)[.\w]*)")


def _flottants(ptx: str) -> list:
    seq = []
    for ligne in ptx.splitlines():
        m = _OPS.match(ligne)
        if not m or not re.search(r"f32|f16|bf16|shfl", m.group(1)):
            continue
        if m.group(1).startswith("shfl"):
            seq.append("shfl" + re.search(r",\s*(\d+),\s*31", ligne).group(1))
        else:
            seq.append(m.group(1))
    return seq


def _compiler(fn, sig: dict, const: dict) -> str:
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    sig = dict(sig, **{c: "constexpr" for c in const})
    src = ASTSource(fn=fn, signature=sig, constexprs={(fn.arg_names.index(a),): v for a, v in const.items()})
    return triton.compile(src, target=GPUTarget("cuda", 120, 32),
                          options=dict(num_warps=1, num_stages=3)).asm["ptx"]


@pytest.mark.parametrize("hv,porte", [(48, True), (48, False), (32, True)])
@pytest.mark.parametrize("J", [1, 2, 4])
def test_ptx_meme_suite_flottante_que_fla(hv, porte, J):
    from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel as F
    commun = dict(H=16, HV=hv, K=128, V=128, BK=128, BV=8, USE_GATE_IN_KERNEL=porte, HAS_DT_BIAS=porte,
                  APPLY_BETA_SIGMOID=porte)
    ptr = dict(q="*fp32", k="*fp32", v="*fp32", g="*fp32", beta="*fp32", A_log="*fp32", dt_bias="*fp32",
               o="*fp32", scale="fp32", T="i32")
    fla_ptx = _compiler(F.fn, dict(ptr, gk="*fp32", gv="*fp32", h0="*fp32", ht="*fp32", cu_seqlens="*i64"),
                        dict(commun, gk=None, gv=None, cu_seqlens=None, USE_G=True, USE_GK=False, USE_GV=False,
                             USE_QK_L2NORM_IN_KERNEL=True, IS_BETA_HEADWISE=True, USE_INITIAL_STATE=True,
                             STORE_FINAL_STATE=True, STATE_V_FIRST=False, IS_VARLEN=False, ALLOW_NEG_EIGVAL=False))
    nous = _compiler(recurrence_tuiles_kernel, dict(ptr, S="*fp32"), dict(commun, J=J))
    ref = _flottants(fla_ptx)
    assert len(ref) > 150, "extraction du PTX vide : l'instrument ne lit plus rien"
    assert _flottants(nous) == ref * J


carte = pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau Triton : carte requise")


def _entrees(b, hv, porte, graine):
    g_ = torch.Generator(device="cpu").manual_seed(graine)
    dev = torch.device("cuda:0")
    r = lambda *s, e=1.0: (torch.randn(*s, generator=g_) * e).to(dev)
    q, k, v = r(b, 1, 16, 128), r(b, 1, 16, 128), r(b, 1, hv, 128)
    if porte:
        return q, k, v, r(b, 1, hv), r(b, 1, hv), r(hv, e=0.5) - 1.0, r(hv, e=0.5)
    g = -torch.rand(b, 1, hv, generator=g_).to(dev)                 # log-décroissance ≤ 0, comme la voie non F1
    beta = torch.rand(b, 1, hv, generator=g_).to(dev)
    return q, k, v, g, beta, None, None


@carte
@pytest.mark.parametrize("hv,porte", [(48, True), (48, False), (32, True)])
@pytest.mark.parametrize("b", [1, 2, 8, 12, 16])
@pytest.mark.parametrize("J", [1, 2, 4])
def test_sortie_et_etat_au_bit_de_fla(hv, porte, b, J):
    S_ref = (torch.randn(b, hv, 128, 128, generator=torch.Generator().manual_seed(b)) * 0.1).to("cuda:0")
    S_ref[0, 0, 0, 0] = -0.0                                        # zeros + load : −0,0 devient +0,0 chez fla aussi
    S = S_ref.clone()
    for pas in range(3):
        q, k, v, g, beta, A, dt = _entrees(b, hv, porte, 1000 * b + pas)
        o_ref = G._recurrence_en_place(q, k, v, g, beta, S_ref, A, dt)
        o = recurrence_tuiles(q, k, v, g, beta, S, A, dt, J=J)
        assert torch.equal(o, o_ref), f"sortie ≠ fla au pas {pas}"
        assert torch.equal(S, S_ref), f"état ≠ fla au pas {pas}"
        assert torch.equal(torch.signbit(S), torch.signbit(S_ref))


@carte
def test_controle_sensible_a_l_ordre_des_sommes():
    """fla en tuiles de 16 colonnes : autre disposition des registres, autre ordre de la somme sur K. S'il rendait
    les mêmes bits que BV = 8 sur ces entrées, `torch.equal` ne verrait pas un ordre de sommation changé."""
    from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel as F
    b, hv = 12, 48
    q, k, v, g, beta, A, dt = _entrees(b, hv, True, 7)
    S0 = (torch.randn(b, hv, 128, 128, generator=torch.Generator().manual_seed(7)) * 0.1).to("cuda:0")
    sorties = []
    for BV in (8, 16):
        S, o = S0.clone(), torch.empty_like(v)
        F[(128 // BV, b * hv)](q=q, k=k, v=v, g=g, gk=None, gv=None, beta=beta, A_log=A, dt_bias=dt, o=o, h0=S, ht=S,
                               cu_seqlens=None, scale=128 ** -0.5, T=1, H=16, HV=hv, K=128, V=128, BK=128, BV=BV,
                               IS_BETA_HEADWISE=True, USE_QK_L2NORM_IN_KERNEL=True, APPLY_BETA_SIGMOID=True,
                               ALLOW_NEG_EIGVAL=False, STATE_V_FIRST=False, num_warps=1, num_stages=3)
        sorties.append((o, S))
    assert not (torch.equal(sorties[0][0], sorties[1][0]) and torch.equal(sorties[0][1], sorties[1][1]))
