"""Pièce 25 : (a) `outils/awq-stabilite-experts.py` (découpe du corpus,
courbe, seuil de stabilité, verdicts) ; (b) repli `mediane_couche` du convert
(statistique = médiane par canal des experts calibrés de la couche, ≥ 8
voisins ; liste des experts sans stats au manifeste) ; (c)
`invites-experts-sans-stats.py` (cibles depuis le manifeste, score d une
trace, verdict par ratio de KL avec alarme). À sec, sans carte."""
import importlib.util
import json
import os

import pytest
import torch

ICI = os.path.dirname(os.path.abspath(__file__))


def _module(rel):
    p = os.path.join(ICI, "..", rel)
    spec = importlib.util.spec_from_file_location(os.path.basename(rel).replace("-", "_").replace(".py", ""), p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------- (b) repli médiane de couche ----------

def _stats(couche, experts, proj, n=100, K=8):
    from acvram.quant.calibrate import ActStats
    out = {}
    for e in experts:
        out[f"model.layers.{couche}.mlp.experts.{e}.{proj}.weight"] = ActStats(
            mean_abs=torch.full((K,), float(e + 1)), max_abs=None, n_samples=n)
    return out


def test_repli_mediane_couche_unitaire():
    from acvram.quant.convert import _experts_par_couche, _stats_repli_mediane
    stats = {**_stats(3, range(9), "gate_proj"), **_stats(3, range(2), "down_proj"), **_stats(4, range(9), "gate_proj", n=3)}
    cache = {}
    st = _stats_repli_mediane(stats, "model.layers.3.mlp.experts.42.gate_proj.weight", 8, cache)
    assert st is not None and st.n_samples == 8 and torch.equal(st.mean_abs, torch.full((8,), 5.0))   # médiane de 1..9
    assert _stats_repli_mediane(stats, "model.layers.3.mlp.experts.42.down_proj.weight", 8, cache) is None   # 2 voisins < 8
    assert _stats_repli_mediane(stats, "model.layers.4.mlp.experts.42.gate_proj.weight", 8, cache) is None   # voisins sous le seuil
    assert _stats_repli_mediane(stats, "model.layers.3.mlp.gate_proj.weight", 8, cache) is None              # pas un expert
    assert cache[("model.layers.3", "gate_proj")] is st                                                      # mis en cache
    assert _experts_par_couche(["model.layers.3.mlp.experts.42.gate_proj.weight", "model.layers.3.mlp.experts.7.gate_proj.weight",
                                "model.layers.0.mlp.experts.1.down_proj.weight", "lm_head.weight"]) == \
        {"3": {"gate_proj": [42, 7]}, "0": {"down_proj": [1]}}


@pytest.fixture(scope="module")
def tiny_moe_10(tmp_path_factory, target_rig):
    """Un MoE minuscule à 10 experts (≥ 8 voisins calibrés) et son plan."""
    from safetensors.torch import save_file
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import PlannerOptions, auto_plan
    torch.manual_seed(3)
    H, I, NH, NKV, V, E, IK = 32, 48, 2, 1, 64, 10, 16
    d = tmp_path_factory.mktemp("moe10")
    json.dump({"architectures": ["Qwen3MoeForCausalLM"], "hidden_size": H, "intermediate_size": I, "num_hidden_layers": 1,
               "num_attention_heads": NH, "num_key_value_heads": NKV, "vocab_size": V, "max_position_embeddings": 64,
               "rms_norm_eps": 1e-6, "rope_theta": 1e5, "torch_dtype": "bfloat16", "model_type": "qwen3_moe",
               "num_experts": E, "num_experts_per_tok": 2, "moe_intermediate_size": IK}, open(d / "config.json", "w"))
    hd = H // NH
    sd = {"model.embed_tokens.weight": torch.randn(V, H, dtype=torch.bfloat16) * .2, "model.norm.weight": torch.ones(H, dtype=torch.bfloat16),
          "lm_head.weight": torch.randn(V, H, dtype=torch.bfloat16) * .2}
    p = "model.layers.0."
    for n, sh in [("self_attn.q_proj", (NH * hd, H)), ("self_attn.k_proj", (NKV * hd, H)), ("self_attn.v_proj", (NKV * hd, H)), ("self_attn.o_proj", (H, NH * hd))]:
        sd[p + n + ".weight"] = torch.randn(*sh, dtype=torch.bfloat16) * .2
    sd[p + "mlp.gate.weight"] = torch.randn(E, H, dtype=torch.bfloat16) * .2
    for e in range(E):
        for n, sh in [("gate_proj", (IK, H)), ("up_proj", (IK, H)), ("down_proj", (H, IK))]:
            sd[p + f"mlp.experts.{e}.{n}.weight"] = torch.randn(*sh, dtype=torch.bfloat16) * .2
    sd[p + "input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd[p + "post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    save_file(sd, str(d / "model.safetensors"))
    spec = load_model_spec(str(d), "moe10")
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=64, max_concurrent_seqs=1))
    return str(d), spec, plan


@pytest.mark.parametrize("repli", ["identite", "mediane_couche"])
def test_convert_repli_experts_et_manifeste(tiny_moe_10, tmp_path, repli):
    from acvram.quant.calibrate import ActStats
    from acvram.quant.convert import ConversionOptions, convert_checkpoint
    d, spec, plan = tiny_moe_10
    stats = {}
    for e in range(9):                                        # l expert 9 n a pas de stats, les 9 autres si
        for n, K in (("gate_proj", 32), ("up_proj", 32), ("down_proj", 16)):
            stats[f"model.layers.0.mlp.experts.{e}.{n}.weight"] = ActStats(
                mean_abs=torch.rand(K) + 0.5, max_abs=None, n_samples=100)
    out = str(tmp_path / f"out-{repli}")
    rapport = convert_checkpoint(d, plan, ConversionOptions(out_dir=out, quant_device="cpu", repli_experts=repli), spec=spec, stats=stats)
    m = json.load(open(os.path.join(out, "acvram_manifest.json")))
    assert rapport.experts_sans_stats == 3 and m["experts_sans_stats"] == 3
    assert m["experts_sans_stats_liste"] == {"0": {"gate_proj": [9], "up_proj": [9], "down_proj": [9]}}
    assert m["experts_repli"] == repli
    if repli == "identite":
        assert rapport.experts_repli_mediane == 0 and "experts_repli_mediane" not in m
    else:
        assert rapport.experts_repli_mediane == 3 and m["experts_repli_mediane"] == 3


# ---------- (c) invites ciblées ----------

def test_invites_cibles_score_et_verdict(tmp_path):
    m = _module("outils/gpu/mesure/invites-experts-sans-stats.py")
    manifest = {"experts_sans_stats_liste": {"3": {"gate_proj": [7, 12], "down_proj": [7]}, "5": {"up_proj": [0]}},
                "tensors": {f"model.layers.0.mlp.experts.{e}.gate_proj.weight": {} for e in range(128)}}
    cibles, E = m.sans_stats(manifest)
    assert cibles == {(3, 7), (3, 12), (5, 0)} and E == 128
    trace = tmp_path / "t.txt"
    trace.write_text("0 3 7,12,1,2\n0 5 0,4,5,6\n1 3 1,2,3,4\n")            # 3 touches sur 12 triplets
    sc, n = m.score_trace(str(trace), cibles)
    assert n == 12 and abs(sc - 0.25) < 1e-9
    # verdicts par ratio de KL
    d = tmp_path / "s"
    d.mkdir()
    for i, v in enumerate([0.30, 0.35, 0.40]):
        json.dump({"kl_max": v}, open(d / f"kl-ciblees-{i:02d}.json", "w"))
    for i, v in enumerate([0.10, 0.12, 0.11]):
        json.dump({"kl_max": v}, open(d / f"kl-temoins-{i:02d}.json", "w"))
    json.dump({"alarme": False}, open(d / "selection.json", "w"))
    r = m.juger(str(d))
    assert abs(r["ratio"] - 0.35 / 0.11) < 1e-9 and r["verdict"].startswith("RÉFUTÉ")
    for i, v in enumerate([0.11, 0.13, 0.12]):
        json.dump({"kl_max": v}, open(d / f"kl-ciblees-{i:02d}.json", "w"))
    assert m.juger(str(d))["verdict"].startswith("TENU")
    json.dump({"alarme": True}, open(d / "selection.json", "w"))
    assert m.juger(str(d))["verdict"].startswith("NON JUGÉ")


# ---------- (a) stabilité ----------

def test_stabilite_decoupe_courbe_et_seuil():
    m = _module("outils/awq-stabilite-experts.py")
    calib = [list(range(512)), list(range(512)), list(range(512))]
    assert sum(len(s) for s in m.decouper(calib, 100)) == 100 and len(m.decouper(calib, 100)) == 1
    assert sum(len(s) for s in m.decouper(calib, 1000)) == 1000 and len(m.decouper(calib, 1000)) == 2
    assert m.classe_de(1) == "1-7" and m.classe_de(8) == "8-31" and m.classe_de(600) == ">=512" and m.classe_de(0) == "0"
    lignes = [{"par_taille": {100: {"n": 3, "classe": "1-7", "ecart": 0.5}, 1000: {"n": 40, "classe": "32-127", "ecart": 0.05}}},
              {"par_taille": {100: {"n": 5, "classe": "1-7", "ecart": 0.4}, 1000: {"n": 20, "classe": "8-31", "ecart": 0.2}}},
              {"par_taille": {100: {"n": 0, "ecart": None}, 1000: {"n": 600, "classe": ">=512", "ecart": 0.01}}}]
    r = m.courbe(lignes, 0.10)
    assert r["courbe"]["1-7"]["ecart_mediane"] == 0.45 and r["seuil_stabilite"] == "32-127" and r["verdict"].startswith("TENU")
    assert m.courbe([{"par_taille": {100: {"n": 2, "classe": "1-7", "ecart": 0.02}}}], 0.10)["verdict"].startswith("RÉFUTÉ : stable dès")
    assert m.courbe([{"par_taille": {100: {"n": 900, "classe": ">=512", "ecart": 0.3}}}], 0.10)["verdict"].startswith("RÉFUTÉ : l échelle")


def test_stabilite_lecteur_de_poids_trois_dispositions(tmp_path):
    """verdict-piece25a-stabilite-awq-22-09 : 0 expert lu sur 17 235 — la source
    déquantifiée porte la disposition hub ≥ 5 (`experts.gate_up_proj` [E, H, 2I],
    `down_proj` [E, I, H]) et l instrument ne cherchait que les clés par expert.
    Le lecteur lit les trois dispositions et rend le MÊME poids ; un dossier
    sans aucune des trois rend None (et `mesurer` refuse à zéro au lieu de
    rendre « courbe: {} »)."""
    from safetensors.torch import save_file
    m = _module("outils/awq-stabilite-experts.py")
    E, H, I = 3, 8, 4
    torch.manual_seed(2)
    par_expert = {}
    for e in range(E):
        par_expert[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = torch.randn(I, H, dtype=torch.bfloat16)
        par_expert[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = torch.randn(I, H, dtype=torch.bfloat16)
        par_expert[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = torch.randn(H, I, dtype=torch.bfloat16)
    p = "model.layers.0.mlp.experts."
    hub = {"model.language_model.layers.0.mlp.experts.gate_up_proj": torch.stack([
               torch.cat([par_expert[p + f"{e}.gate_proj.weight"], par_expert[p + f"{e}.up_proj.weight"]], 0).t()
               for e in range(E)]).contiguous(),
           "model.language_model.layers.0.mlp.experts.down_proj": torch.stack(
               [par_expert[p + f"{e}.down_proj.weight"].t() for e in range(E)]).contiguous()}
    mm = {"model.language_model." + k[len("model."):]: v for k, v in par_expert.items()}
    dossiers = {}
    for nom, sd in (("par_expert", par_expert), ("hub", hub), ("multimodal", mm), ("vide", {"lm_head.weight": torch.zeros(2, 2)})):
        d = tmp_path / nom
        d.mkdir()
        save_file(sd, str(d / "model.safetensors"))
        dossiers[nom] = m.LecteurPoids(str(d))
    for k, w in par_expert.items():
        for nom in ("par_expert", "hub", "multimodal"):
            lu = dossiers[nom].charger(k)
            assert lu is not None and torch.equal(lu, w), (nom, k)
        assert dossiers["vide"].charger(k) is None
    assert dossiers["hub"].charger("lm_head.weight") is None
