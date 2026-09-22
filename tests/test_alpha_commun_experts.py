"""`--alpha-commun-experts` (22/09) : gate_proj et up_proj d'un expert MoE
lisent la MÊME entrée, et la pile groupée du moteur EXIGE qu'ils portent la
même échelle AWQ — sinon `_try_build_stacks` ne fusionne pas les deux tables
(`engine/moe.py:336-337`, `torch.equal` global sur [E, K]) et
`_construire_marlin` refuse la disposition Marlin (`moe.py:446`) : le
Qwen3-Coder-30B-A3B-nvfp4-qkv-22-09 charge en `experts_layout=naturel`,
8,62 ms/pas contre 6,7 (revue/poste1-disposition-naturel-qkv-22-09.md).

Le test mesure les deux bouts de la chaîne sur un MoE jouet :
1. sans l'option, la recherche par tenseur donne des alphas DIFFÉRENTS à gate
   et up (chacun minimise l'erreur de SON poids) — c'est la cause ;
2. avec l'option, les échelles sont identiques par construction, pour TOUS
   les experts ;
3. et le moteur le voit : `up_distinct` vrai sans, faux avec.
Ce qui le fait casser : retirer le partage, ou le poser sans que le moteur
fusionne (le point 3 est le seul qui juge le résultat, pas l'intention).
"""
from __future__ import annotations

import json

import pytest
import torch

from test_awq_seuil_echantillons import H, IK, _act_scale, _stats_fabrique, _tiny_moe


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
                       spec=spec, stats=_stats_fabrique(n_riche=256, n_pauvre=256))
    return out


def _paires(out_dir, experts=(0, 1)):
    return [(_act_scale(out_dir, f"model.layers.0.mlp.experts.{e}.gate_proj.weight"),
             _act_scale(out_dir, f"model.layers.0.mlp.experts.{e}.up_proj.weight")) for e in experts]


def _up_distinct(out_dir) -> bool:
    """Ce que le moteur en fait : la table [E, K] de gate est-elle LE MÊME
    objet que celle d'up après `_try_build_stacks` ?"""
    from acvram.engine.loader import load_model
    charge = load_model(str(out_dir), dtype=torch.float32, device_override="cpu")
    blocs = [m for m in charge.model.modules() if type(m).__name__.startswith("MoEBlock")]
    assert blocs, "le jouet doit porter au moins un MoEBlock"
    assert blocs[0]._try_build_stacks()
    return bool(blocs[0]._stacks_awq["up_distinct"])


@pytest.fixture(scope="module")
def convertis(tmp_path_factory):
    d = tmp_path_factory.mktemp("alpha")
    return (_convertir(d / "a", "sans"), _convertir(d / "b", "avec", alpha_commun_experts=True))


def test_sans_option_gate_et_up_divergent(convertis):
    sans, _ = convertis
    paires = _paires(sans)
    assert any(not torch.equal(g, u) for g, u in paires), (
        "le jouet doit reproduire la cause : au moins un expert dont gate ≠ up")


def test_avec_option_toutes_les_paires_sont_identiques(convertis):
    _, avec = convertis
    for e, (g, u) in enumerate(_paires(avec)):
        assert torch.equal(g, u), f"expert {e} : gate ≠ up malgré --alpha-commun-experts"


def test_le_moteur_fusionne_les_tables(convertis):
    sans, avec = convertis
    assert _up_distinct(sans) is True, "sans l'option, la disposition Marlin doit être refusée"
    assert _up_distinct(avec) is False, "avec l'option, les tables doivent fusionner"


def test_le_manifeste_porte_l_option(convertis):
    _, avec = convertis
    m = json.load(open(avec / "acvram_manifest.json"))
    assert m["options"].get("alpha_commun_experts") is True
