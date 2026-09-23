"""Chantier C10 (revue/chantier-c10-19-09) : le GEMV Marlin sert un MoE à
gate/up DISTINCTS (tables AWQ séparées : GLM k48-calibA, 13 couches sur 46) avec
expert partagé, sous ACVRAM_MARLIN_DISTINCT=1 — voie (b) : gate puis up par
`nvfp4_gemv_marlin` (NW=1), activation torch, down par le même noyau
(model.py, `_forward_grouped`, branche « distinct »).

À sec, sans carte ni nvcc : la disposition Marlin est construite par
`preparer_pile(repack=repack_torch)` (module marlin_port de C2) et le noyau
est remplacé par son jumeau torch `gemv_marlin_torch`, qui lit la disposition
aux mêmes places que lui. Le « chemin d'avant » (`_grouped` →
`nvfp4_gemv_grouped`, pile naturelle) est rejoué avec son noyau remplacé par
sa référence fp32 (celle que tests/test_gemv_marlin.py juge sur carte).
Juge : fp32 par ligne, |Δ| ≤ 2⁻⁷·max|y| — PAS le bit (l'ordre des FMA du
noyau n'est pas reproductible à sec) ; témoin cassant : échelles décalées.
"""
import json
import os
import pathlib
import sys

import pytest
import torch
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../outils'))
from racine_modeles import racine_modeles as _racine_modeles, alias_absent as _alias_absent  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from acvram import kernels as K                                                  # noqa: E402
from acvram.engine import model as MD                                            # noqa: E402
from acvram.engine import moe as MOE_D
from acvram.engine.layers import ChannelScaler, QuantLinear                       # noqa: E402
from acvram.engine.model import MLP, MoEBlock                                     # noqa: E402
from acvram.kernels import marlin_port as MP                                     # noqa: E402
from acvram.quant.formats import INT8Tensor, _quantize_int8                      # noqa: E402
from acvram.quant.nvfp4 import NVFP4Tensor, dequantize_nvfp4, quantize_nvfp4     # noqa: E402

torch.set_num_threads(min(8, torch.get_num_threads()))
TOL_HORS = 5e-4                       # part de valeurs hors 2⁻⁷ par ligne tolérée (P1, test_gemv_marlin.py)
GLM = _RACINE + "/GLM-4.7-Flash-srcbf16-nvfp4-k48-calibA"
COUCHE_GLM = 1                        # première couche distincte (13 : 1 2 4 6 9 11 12 13 16 17 19 20 23)


def _hors_par_ligne(y, ref):
    return int(((y.float() - ref.float()).abs() > 2 ** -7 * ref.float().abs().amax(1, keepdim=True)).sum())


class _ExtStub:
    """L'extension à sec : le seul GEMV Marlin, par son jumeau torch — ni
    `nvfp4_gemv_marlin_gateup` (inapplicable aux distincts) ni `moe_reduce`."""
    nvfp4_gemv_marlin = staticmethod(MP.gemv_marlin_torch)


def _reference_grouped(x32, qw, bs, gs, expert_ids, token_ids, k):
    """Le noyau v1 `nvfp4_gemv_grouped` remplacé par sa référence fp32 : poids
    déquantifiés (dequantize_nvfp4), x[t] @ Wᵀ par paire (test_gemv_marlin.py)."""
    assert qw is not None and bs is not None, "pile rendue passée au noyau"
    out = torch.zeros(expert_ids.numel(), qw.shape[1], dtype=torch.float32)
    for g, (e, t) in enumerate(zip(expert_ids.tolist(), token_ids.tolist())):
        if e < 0:
            continue
        w = dequantize_nvfp4(NVFP4Tensor(qw[e], bs[e].view(torch.float8_e4m3fn), gs[e], (qw.shape[1], k), k),
                             torch.float32)
        out[g] = x32[t, :k].float() @ w.T
    return out


def _disposition_a_sec(bloc):
    """`_construire_marlin` sans carte : même `preparer_pile`, repack torch (C2)."""
    out = {}
    for n in ("gate_proj", "up_proj", "down_proj"):
        _, qw, bs, gs, k, m = bloc._stacks[n]
        w, s, g = MP.preparer_pile(qw, bs.view(torch.float8_e4m3fn), gs.reshape(-1).float(), repack=MP.repack_torch)
        out[n] = (w, s, g, k, m)
    return out


def _routage(bloc, x):
    logits = bloc.router(x)
    topw, topi = torch.topk(torch.softmax(logits.float(), -1), bloc.top_k, dim=-1)
    return (topw / topw.sum(-1, keepdim=True)).float(), topi.to(torch.int32)


def _deux_chemins(bloc, x, monkeypatch):
    """Rend (y_avant, y_c10) de `_forward_grouped` sur le même routage : chemin
    d'avant (pile naturelle, `_grouped` avec sa référence fp32) puis C10
    (disposition Marlin à sec, pile rendue, jumeau torch) — chemin asserté."""
    topw, topi = _routage(bloc, x)
    monkeypatch.setattr(K, "nvfp4_gemv_grouped", _reference_grouped)
    monkeypatch.setattr(MD.kernels, "get_extension", lambda: None)
    assert bloc._stacks["gate_proj"][1] is not None
    y_avant = bloc._forward_grouped(x, topw, topi)
    assert "gemv_marlin" not in getattr(bloc, "chemins", {})
    bloc._stacks_marlin = _disposition_a_sec(bloc)
    bloc.__dict__["_piles_naturelles"] = {n: bloc._stacks[n] for n in ("gate_proj", "up_proj", "down_proj")}
    bloc._liberer_pile_naturelle()
    assert bloc._stacks["gate_proj"][1] is None and bloc.experts_layout in ("marlin", "marlin-w13")
    monkeypatch.setattr(MD.kernels, "get_extension", lambda: _ExtStub)
    y_c10 = bloc._forward_grouped(x, topw, topi)
    assert bloc.dernier_chemin == "gemv_marlin" and bloc.chemins["gemv_marlin"] == 1, bloc.chemins
    return y_avant, y_c10


# ---- bloc jouet : la forme de GLM (64 experts top-4, distincts, expert partagé int8) ----

def _bloc_glm_jouet(E=64, H=256, I=128, top_k=4):
    def lin(o, i, graine, sc=None):
        g = torch.Generator().manual_seed(graine)
        w = (torch.randn(o, i, generator=g) * 0.02).to(torch.bfloat16)
        return QuantLinear(quantize_nvfp4(w), out_features=o, in_features=i, scaler=sc).to_device("cpu")

    def scaler(i, graine):
        g = torch.Generator().manual_seed(graine)
        return ChannelScaler((0.5 + torch.rand(i, generator=g) * 1.5).to(torch.bfloat16), 0)

    def lin8(o, i, graine):
        g = torch.Generator().manual_seed(graine)
        w = (torch.randn(o, i, generator=g) * 0.02).to(torch.bfloat16)
        return QuantLinear(_quantize_int8(w, 128), out_features=o, in_features=i).to_device("cpu")
    experts = [MLP(lin(I, H, 10 * e + 1, scaler(H, 100 + e)), lin(I, H, 10 * e + 2, scaler(H, 300 + e)),
                   lin(H, I, 10 * e + 3, scaler(I, 200 + e))) for e in range(E)]
    partage = MLP(lin8(I, H, 7001), lin8(I, H, 7002), lin8(H, I, 7003))       # int8 comme le converti k48-calibA
    gen = torch.Generator().manual_seed(5)
    routeur = QuantLinear(_quantize_int8((torch.randn(E, H, generator=gen) * 0.02).to(torch.bfloat16), 128),
                          out_features=E, in_features=H).to_device("cpu")
    return MoEBlock(routeur, experts, top_k, shared=partage).to("cpu")


@pytest.mark.parametrize("t", [1, 12])
def test_c10_jouet_glm_prend_gemv_marlin_et_egale_le_chemin_d_avant(monkeypatch, t):
    monkeypatch.setattr(MOE_D, "_GEMV_LAYOUT", "marlin")
    monkeypatch.setattr(MOE_D, "_PREFILL_GROUPED", "marlin")
    monkeypatch.setattr(MOE_D, "_MARLIN_DISTINCT", "1")
    MoEBlock._marlin_refus_dit = False
    bloc = _bloc_glm_jouet()
    assert bloc._try_build_stacks(), bloc._raison_repli
    assert bloc._stacks_awq["up_distinct"] is True
    assert bloc._stacks_marlin is None                           # à sec : « hors CUDA », pas « distinctes »
    assert isinstance(bloc.shared.gate_proj.qweight, INT8Tensor)  # l'expert partagé reste dense int8
    x = (torch.randn(t, 256, generator=torch.Generator().manual_seed(t)) * 0.5).to(torch.bfloat16)
    y_avant, y_c10 = _deux_chemins(bloc, x, monkeypatch)
    assert y_avant.shape == y_c10.shape == (t, 256)
    assert _hors_par_ligne(y_c10, y_avant) == 0, _hors_par_ligne(y_c10, y_avant)
    # bloc entier : la contribution de l'expert partagé est la même des deux côtés
    partage = bloc._shared_out(x)
    assert _hors_par_ligne(y_c10.float() + partage.float(), y_avant.float() + partage.float()) == 0
    # témoin cassant : un octet d'échelle de décalage sur up → le juge dit faux
    w, s, g, k, m = bloc._stacks_marlin["up_proj"]
    bloc._stacks_marlin["up_proj"] = (w, torch.roll(s.view(torch.uint8), 1, dims=2).contiguous().view(s.dtype), g, k, m)
    topw, topi = _routage(bloc, x)
    y_faux = bloc._forward_grouped(x, topw, topi)
    assert _hors_par_ligne(y_faux, y_avant) > TOL_HORS * y_avant.numel()


def test_c10_sans_la_variable_le_refus_distinct_tient(monkeypatch):
    """ACVRAM_MARLIN_DISTINCT=0 (défaut) : le même bloc est refusé pour la forme,
    la pile naturelle reste, et le chemin d'avant seul répond."""
    monkeypatch.setattr(MOE_D, "_GEMV_LAYOUT", "marlin")
    monkeypatch.setattr(MOE_D, "_PREFILL_GROUPED", "marlin")
    monkeypatch.setattr(MOE_D, "_MARLIN_DISTINCT", "0")
    MoEBlock._marlin_refus_dit = False
    import contextlib
    import io
    tampon = io.StringIO()
    bloc = _bloc_glm_jouet()
    with contextlib.redirect_stdout(tampon):
        assert bloc._try_build_stacks(), bloc._raison_repli
    MoEBlock._marlin_refus_dit = False
    assert "distinctes" in tampon.getvalue() and bloc._stacks_marlin is None
    assert bloc._stacks["gate_proj"][1] is not None


# ---- tenseurs GLM réels : couche 1 (distincte), 64 experts + expert partagé ----

def _couche_glm():
    if not os.path.isdir(GLM):
        pytest.skip(f"converti absent : {GLM}")
    from acvram.engine.loader import _ShardReader, _linear
    manifest = json.load(open(os.path.join(GLM, "acvram_manifest.json")))
    reader = _ShardReader(GLM, manifest["weight_map"])
    p = f"model.layers.{COUCHE_GLM}.mlp."

    def lin(nom):
        m = _linear(p + nom, manifest, reader, 128)
        assert m is not None, p + nom
        return m.to_device("cpu")
    experts = [MLP(lin(f"experts.{e}.gate_proj.weight"), lin(f"experts.{e}.up_proj.weight"),
                   lin(f"experts.{e}.down_proj.weight")) for e in range(64)]
    partage = MLP(lin("shared_expert.gate_proj.weight"), lin("shared_expert.up_proj.weight"),
                  lin("shared_expert.down_proj.weight"))
    bloc = MoEBlock(lin("gate.weight"), experts, 4, shared=partage).to("cpu")
    reader.close()
    return bloc


@pytest.mark.skipif(bool(_alias_absent("GLM-4.7-Flash-srcbf16-nvfp4-k48-calibA")),
                    reason=_alias_absent("GLM-4.7-Flash-srcbf16-nvfp4-k48-calibA"))
@pytest.mark.parametrize("t", [1, 12])
def test_c10_couche_glm_reelle_egale_le_chemin_d_avant(monkeypatch, t):
    """Couche 1 de GLM-4.7-Flash k48-calibA (lue par le manifeste, `_linear`
    du chargeur) : tables AWQ gate ≠ up (`up_distinct`), expert partagé int8 ;
    b=1 (4 paires) et b=12 (48 paires, le godet certifié)."""
    monkeypatch.setattr(MOE_D, "_GEMV_LAYOUT", "marlin")
    monkeypatch.setattr(MOE_D, "_PREFILL_GROUPED", "marlin")
    monkeypatch.setattr(MOE_D, "_MARLIN_DISTINCT", "1")
    MoEBlock._marlin_refus_dit = False
    bloc = _couche_glm()
    assert bloc._try_build_stacks(), bloc._raison_repli
    assert bloc._stacks_awq["up_distinct"] is True                 # la prémisse de C10, sur les données
    assert bloc._stacks_awq.get("gate_proj") is not None and bloc._stacks_awq.get("up_proj") is not None
    assert isinstance(bloc.shared.gate_proj.qweight, INT8Tensor)
    _, qw, bs, gs, k, m = bloc._stacks["gate_proj"]
    assert (k, m) == (2048, 1536) and bloc._stacks["down_proj"][4:] == (1536, 2048)
    x = (torch.randn(t, 2048, generator=torch.Generator().manual_seed(40 + t)) * 0.5).to(torch.bfloat16)
    y_avant, y_c10 = _deux_chemins(bloc, x, monkeypatch)
    assert torch.isfinite(y_c10.float()).all()
    hors = _hors_par_ligne(y_c10, y_avant)
    assert hors == 0, (hors, float((y_c10.float() - y_avant.float()).abs().max()), float(y_avant.float().abs().max()))
    # disposition Marlin des tenseurs réels : dépaquetée, elle rend AU BIT les
    # poids déquantifiés de la pile naturelle (identité C2) pour les experts
    # routés — et aucune échelle de bloc n'a été annulée par le repack
    topw, topi = _routage(bloc, x)
    routes = sorted(set(topi.reshape(-1).tolist()))[:4]
    for n in ("gate_proj", "up_proj", "down_proj"):
        w, s, g, kk, mm = bloc._stacks_marlin[n]
        _, qw_n, bs_n, gs_n, _, _ = bloc._piles_naturelles[n]
        assert int((s.view(torch.uint8) == 0).sum()) == 0, n          # aucune échelle annulée par le repack
        for e in routes:
            assert getattr(bloc.experts[e], n).qweight.qweight.numel() == 0     # pile rendue : gabarit vide
            nat = dequantize_nvfp4(NVFP4Tensor(qw_n[e], bs_n[e].view(torch.float8_e4m3fn), gs_n[e], (mm, kk), kk),
                                   torch.bfloat16)
            assert torch.equal(MP.depaqueter_marlin(w[e], s[e], g[e], kk, mm, noyau="torch"), nat), (n, e)
    monkeypatch.setattr(MD.kernels, "get_extension", lambda: None)
    with pytest.raises(RuntimeError, match="pile NVFP4 naturelle rendue"):
        bloc._forward_grouped(x, topw, topi)                          # plus de repli vers la pile rendue
