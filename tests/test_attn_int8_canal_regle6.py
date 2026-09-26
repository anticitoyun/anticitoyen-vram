"""Règle 6 (pièce 56, 23/09) : `--attn-qkvo-int8-canal` sans projection d'attention en int8
n'est plus un no-op silencieux — l'étiquette du manifeste suit les faits et un avertissement
nommé y est écrit. Ce test aurait cassé le 22/09 au soir (manifeste p55 : `attn_int8 = canal`,
0 tenseur int8)."""
import importlib.util
import os
from types import SimpleNamespace

from acvram.quant.convert import ConversionOptions, _bilan_attn_int8, convert_checkpoint


def _tensors(fmt_attn: str, n: int = 4) -> dict:
    d = {f"model.layers.{i}.self_attn.{p}.weight": {"format": fmt_attn} for i in range(n) for p in ("q_proj", "k_proj", "v_proj", "o_proj")}
    d.update({f"model.layers.{i}.mlp.{p}.weight": {"format": "nvfp4"} for i in range(n) for p in ("gate_proj", "up_proj", "down_proj")})
    d["model.layers.0.self_attn.q_norm.weight"] = {"format": "bf16"}
    return d


def test_drapeau_sans_projection_int8_donne_groupe_et_avertit():
    r = _bilan_attn_int8(SimpleNamespace(attn_qkvo_int8_canal=True), _tensors("nvfp4"))
    assert r["attn_int8"] == "groupe" and r["attn_int8_canal_demande"] is True
    assert r["avertissements"] and "SANS EFFET" in r["avertissements"][0] and "--snr-floor" in r["avertissements"][0]


def test_drapeau_avec_projections_int8_donne_canal_et_compte():
    r = _bilan_attn_int8(SimpleNamespace(attn_qkvo_int8_canal=True), _tensors("int8"))
    assert r["attn_int8"] == "canal" and r["attn_int8_canal_tenseurs"] == 16 and "avertissements" not in r


def test_sans_drapeau_rien_ne_change():
    assert _bilan_attn_int8(SimpleNamespace(attn_qkvo_int8_canal=False), _tensors("nvfp4")) == {"attn_int8": "groupe"}


def test_conversion_reelle_le_manifeste_dit_la_verite(tmp_path):
    """Bout en bout sur le petit MoE des tests : drapeau posé, aucune promotion (snr_floor 0) →
    le manifeste porte `groupe` + avertissement, pas `canal`. Sur le code d'hier : `canal`."""
    import json
    from acvram.engine.config import load_model_spec
    from acvram.hardware.profiles import load_profile
    from acvram.memory.tiering import PlannerOptions, auto_plan
    chemin = os.path.join(os.path.dirname(__file__), "test_awq_garde_repli.py")
    spec_mod = importlib.util.spec_from_file_location("garde", chemin); g = importlib.util.module_from_spec(spec_mod); spec_mod.loader.exec_module(g)
    ckpt = g._tiny_moe(tmp_path / "hf")
    spec = load_model_spec(ckpt, "tiny-moe")
    plan, _ = auto_plan(spec, load_profile("rig-14900k-5090-3080ti"), PlannerOptions(max_model_len=256, max_concurrent_seqs=2))
    out = tmp_path / "canal"
    convert_checkpoint(ckpt, plan, ConversionOptions(out_dir=str(out), attn_qkvo_int8_canal=True), spec=spec, stats=g._stats_toutes_saines())
    m = json.load(open(out / "acvram_manifest.json"))
    n_int8 = sum(1 for k, v in m["tensors"].items() if ".self_attn." in k and v.get("format") == "int8")
    if n_int8 == 0:
        assert m["attn_int8"] == "groupe" and m.get("avertissements"), m.get("attn_int8")
    else:
        assert m["attn_int8"] == "canal" and m["attn_int8_canal_tenseurs"] == n_int8
