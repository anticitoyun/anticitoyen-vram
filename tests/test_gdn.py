"""Gated DeltaNet contre l'implémentation de référence de transformers.

Mêmes poids des deux côtés ; le prefill (règle par blocs) et le décodage pas à
pas (règle récurrente) doivent reproduire la référence, y compris la
continuité de l'état entre prefill et décodage.
"""

import pytest
import torch

from acvram.engine.gdn import GatedDeltaNet, gdn_available

pytestmark = [
    pytest.mark.skipif(not gdn_available(),
                       reason="transformers/qwen3_next absent"),
    pytest.mark.skipif(not torch.cuda.is_available(),
                       reason="la référence est kernelisée Triton (GPU requis)"),
]
DEV = "cuda:0"

H, NK, NV, DK, DV, KER = 64, 2, 4, 16, 16, 4


def _paire():
    """(référence transformers, nôtre) avec des poids identiques."""
    from transformers.models.qwen3_next.configuration_qwen3_next import \
        Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import \
        Qwen3NextGatedDeltaNet

    torch.manual_seed(20260902)
    cfg = Qwen3NextConfig(
        hidden_size=H, linear_num_value_heads=NV, linear_num_key_heads=NK,
        linear_key_head_dim=DK, linear_value_head_dim=DV,
        linear_conv_kernel_dim=KER, hidden_act="silu", rms_norm_eps=1e-6,
        num_hidden_layers=1, layer_types=["linear_attention"])
    ref = Qwen3NextGatedDeltaNet(cfg, layer_idx=0).float().eval().to(DEV)
    for p in ref.parameters():
        p.data.uniform_(-0.4, 0.4)
    ref.A_log.data.uniform_(-2.0, 1.0)
    ref.dt_bias.data.uniform_(-0.5, 0.5)

    key_dim, value_dim = NK * DK, NV * DV

    # la référence entrelace q,k,v,z par groupe de têtes ; nos projections
    # sont séparées et plates — on désentrelace ses poids
    wqkvz = ref.in_proj_qkvz.weight.data          # [qkvz, H] entrelacé
    per_g = 2 * DK + 2 * DV * (NV // NK)
    wq, wk, wv, wz = [], [], [], []
    for gr in range(NK):
        bloc = wqkvz[gr * per_g:(gr + 1) * per_g]
        wq.append(bloc[:DK])
        wk.append(bloc[DK:2 * DK])
        wv.append(bloc[2 * DK:2 * DK + DV * (NV // NK)])
        wz.append(bloc[2 * DK + DV * (NV // NK):])
    wq, wk = torch.cat(wq), torch.cat(wk)
    wv, wz = torch.cat(wv), torch.cat(wz)
    wba = ref.in_proj_ba.weight.data              # [2·nv, H] entrelacé (b, a)
    wb, wa = [], []
    per_ba = 2 * (NV // NK)
    for gr in range(NK):
        bloc = wba[gr * per_ba:(gr + 1) * per_ba]
        wb.append(bloc[:NV // NK])
        wa.append(bloc[NV // NK:])
    wb, wa = torch.cat(wb), torch.cat(wa)

    def lin(w):
        m = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
        m.weight.data = w.clone()
        return m.float().to(DEV)

    notre = GatedDeltaNet(
        qkv=lin(torch.cat([wq, wk, wv])), gate=lin(wz),
        alpha=lin(wa), beta=lin(wb),
        out=lin(ref.out_proj.weight.data),
        conv_weight=ref.conv1d.weight.data.squeeze(1).clone(),
        dt_bias=ref.dt_bias.data.clone(), a_log=ref.A_log.data.clone(),
        norm_weight=ref.norm.weight.data.clone(),
        num_k_heads=NK, num_v_heads=NV, head_k_dim=DK, head_v_dim=DV).to(DEV)
    return ref, notre


def _ref_forward(ref, x):
    with torch.no_grad():
        return ref(x.unsqueeze(0))[0] if isinstance(ref(x.unsqueeze(0)), tuple) \
            else ref(x.unsqueeze(0))


def test_prefill_matches_reference():
    ref, notre = _paire()
    x = torch.randn(9, H, device=DEV) * 0.5
    with torch.no_grad():
        att = ref(x.unsqueeze(0))
        y_ref = att[0] if isinstance(att, tuple) else att
        y_nous, _ = notre(x)
    err = (y_nous - y_ref.squeeze(0)).abs().max().item()
    assert err < 1e-4, f"prefill : écart {err:.2e} avec la référence"


def test_decode_continuity():
    """Prefill puis décodage pas à pas == prefill de la séquence entière."""
    ref, notre = _paire()
    x = torch.randn(12, H, device=DEV) * 0.5
    with torch.no_grad():
        att = ref(x.unsqueeze(0))
        y_ref = (att[0] if isinstance(att, tuple) else att).squeeze(0)
        y_pre, etat = notre(x[:8])
        sorties = [y_pre]
        for i in range(8, 12):
            y_i, etat = notre(x[i:i + 1], etat)
            sorties.append(y_i)
        y_nous = torch.cat(sorties)
    err = (y_nous - y_ref).abs().max().item()
    assert err < 1e-3, f"décodage à état : écart {err:.2e}"


def test_detile_inverse_du_convertisseur():
    """Notre dé-tiling inverse exactement le réordonnancement du convertisseur
    llama.cpp (têtes V « tiled » pour le broadcast ggml)."""
    torch.manual_seed(0)
    nk, nvpk, hd = 4, 2, 8

    def reorder(t, dim):                  # la transformation du convertisseur
        forme = list(t.shape)
        neuf = forme[:dim] + [nk, nvpk, hd] + forme[dim + 1:]
        t = t.reshape(*neuf)
        perm = list(range(len(neuf)))
        perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
        return t.permute(*perm).contiguous().reshape(*forme)

    def detile(t, dim):                   # la nôtre (gguf.py)
        forme = list(t.shape)
        neuf = forme[:dim] + [nvpk, nk, hd] + forme[dim + 1:]
        t = t.reshape(*neuf)
        perm = list(range(len(neuf)))
        perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
        return t.permute(*perm).contiguous().reshape(*forme)

    w = torch.randn(nk * nvpk * hd, 13)
    assert torch.equal(detile(reorder(w, 0), 0), w)
    w2 = torch.randn(13, nk * nvpk * hd)
    assert torch.equal(detile(reorder(w2, 1), 1), w2)
