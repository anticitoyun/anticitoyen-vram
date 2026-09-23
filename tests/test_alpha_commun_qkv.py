"""`--alpha-commun-qkv` (pièce 100 B, 23/09) : q_proj, k_proj et v_proj d'une
couche lisent la même entrée ; sans l'option chacun cherche son alpha AWQ et
`stack_nvfp4_linears` refuse la pile qkv (« scalers differents entre
projections », pièce 42 : trois GEMM, +0,9 ms/pas). Avec l'option les trois
échelles d'activation sont identiques par construction et le moteur empile.
Sans carte : MoE jouet, conversion et chargement sur processeur."""
from __future__ import annotations

import json

import pytest
import torch

from test_awq_seuil_echantillons import H, _act_scale, _magnitudes_variees, _stats_fabrique, _tiny_moe

from acvram.quant.calibrate import ActStats

QKV = [f"model.layers.0.self_attn.{p}_proj.weight" for p in "qkv"]


def _stats_attention():
    """Les mêmes statistiques d'entrée pour q, k et v (une seule activation),
    avec des canaux saillants pour que la recherche AWQ ne s'effondre pas à
    l'identité."""
    st = _stats_fabrique(n_riche=256, n_pauvre=256)
    for n in QKV:
        st[n] = ActStats(_magnitudes_variees(H), None, 256)
    return st


def _convertir(tmp_path, nom: str, **opts_kw):
    from acvram.engine.config import load_model_spec
    from acvram.hardware.profiles import load_profile
    from acvram.memory.tiering import PlannerOptions, auto_plan
    from acvram.quant.convert import ConversionOptions, convert_checkpoint
    ckpt = _tiny_moe(tmp_path / "hf")
    spec = load_model_spec(ckpt, "tiny-moe")
    plan, _ = auto_plan(spec, load_profile("rig-14900k-5090-3080ti"),
                        PlannerOptions(max_model_len=256, max_concurrent_seqs=2))
    out = tmp_path / nom
    convert_checkpoint(ckpt, plan, ConversionOptions(out_dir=str(out), **opts_kw),
                       spec=spec, stats=_stats_attention())
    return out


@pytest.fixture(scope="module")
def convertis(tmp_path_factory):
    d = tmp_path_factory.mktemp("alpha_qkv")
    return (_convertir(d / "a", "sans"), _convertir(d / "b", "avec", alpha_commun_qkv=True))


def _formats(out_dir):
    m = json.load(open(out_dir / "acvram_manifest.json"))
    return {n: m["tensors"][n]["format"] for n in QKV}


def test_avec_option_les_trois_echelles_sont_identiques_et_non_triviales(convertis):
    _, avec = convertis
    fmts = _formats(avec)
    if set(fmts.values()) != {"nvfp4"}:
        pytest.skip(f"le jouet ne sert pas q/k/v en nvfp4 sur ce plan : {fmts}")
    q, k, v = (_act_scale(avec, n) for n in QKV)
    assert torch.equal(q, k) and torch.equal(k, v), "alpha commun : trois échelles différentes"
    assert not torch.allclose(q, torch.ones_like(q)), "échelle à l'identité : rien n'a été partagé"


def test_le_manifeste_porte_l_option(convertis):
    sans, avec = convertis
    assert json.load(open(avec / "acvram_manifest.json"))["options"]["alpha_commun_qkv"] is True
    assert json.load(open(sans / "acvram_manifest.json"))["options"]["alpha_commun_qkv"] is False


def test_le_moteur_empile_qkv_avec_l_option(convertis):
    """Ce que l'option achète : la pile qkv (`attention.qkv_proj`) existe au
    chargement. Sans l'option elle peut exister aussi (alphas égaux par
    hasard) : on n'assert que le bras `avec`."""
    from acvram.engine import layers
    from acvram.engine.loader import load_model
    _, avec = convertis
    if set(_formats(avec).values()) != {"nvfp4"}:
        pytest.skip("q/k/v non nvfp4 sur ce plan")
    layers._REFUS_FUSION.clear()
    charge = load_model(str(avec), dtype=torch.float32, device_override="cpu")
    attn = [m for m in charge.model.modules() if hasattr(m, "qkv_proj")]
    assert attn, "aucun module d'attention avec qkv_proj"
    assert attn[0].qkv_proj is not None, f"pile qkv refusée : {dict(layers._REFUS_FUSION)}"
