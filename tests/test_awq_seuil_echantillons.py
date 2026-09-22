"""sage-hybrides-etape1-close-gemm-dense-17-09 : Nemotron calibA (PPL 1,4301
contre 1,0304 sans calibration, dégradation uniforme sur les 3 tranches).
Diff tenseur par tenseur (à sec, disque) : les tenseurs bf16 exclus
(mamba.in_proj/out_proj, self_attn) sont byte-identiques entre calibA et
precision-officielle -- pas la cause. L'expert partagé est sain (ratio de
norme 0,994 ± 0,015). Mais 6/346 `mlp.experts.*.down_proj` échantillonnés
ont un ratio de norme calibA/officielle de 0,29 à 0,69 : leur `act_scale`
s'étend sur 1,68e6×/2,4e5× (0,0001 à 177 ; 0,008 à 19) contre ~630× pour un
tenseur sain (0,013 à 8,1). Cause : `search_channel_scales` clampe CHAQUE
valeur à [1e-4, 1e4] mais jamais l'ÉTENDUE entre canaux -- sur un expert peu
routé (bras A, prose anglaise, quelques jetons captés), `mean_abs` par canal
n'estime plus rien, et la recherche tire un motif de salience extrême d'une
statistique bruitée. Le message existant (« jamais routés OU TROP PEU sur le
corpus ») nommait déjà ce régime sans jamais le TESTER : `experts_sans_stats`
ne comptait que `n_samples == 0`.
"""
import json

import torch
from safetensors.torch import save_file

from acvram.engine.config import load_model_spec
from acvram.hardware.profiles import load_profile
from acvram.memory.tiering import PlannerOptions, auto_plan
from acvram.quant.calibrate import ActStats
from acvram.quant.convert import ConversionOptions, convert_checkpoint

H, I, L, NH, NKV, V, E, IK = 64, 128, 1, 4, 2, 256, 4, 64


def _tiny_moe(tmp_path):
    torch.manual_seed(20260917)
    d = tmp_path
    d.mkdir(parents=True, exist_ok=True)
    json.dump({
        "architectures": ["Qwen3MoeForCausalLM"], "hidden_size": H,
        "intermediate_size": I, "num_hidden_layers": L,
        "num_attention_heads": NH, "num_key_value_heads": NKV,
        "vocab_size": V, "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "torch_dtype": "bfloat16", "model_type": "qwen3_moe",
        "num_experts": E, "num_experts_per_tok": 2,
        "moe_intermediate_size": IK,
    }, open(d / "config.json", "w"))
    hd = H // NH
    sd = {"model.embed_tokens.weight": torch.randn(V, H, dtype=torch.bfloat16) * .02}
    p = "model.layers.0."
    for n, sh in [("self_attn.q_proj", (NH * hd, H)), ("self_attn.k_proj", (NKV * hd, H)),
                  ("self_attn.v_proj", (NKV * hd, H)), ("self_attn.o_proj", (H, NH * hd))]:
        sd[p + n + ".weight"] = torch.randn(*sh, dtype=torch.bfloat16) * .02
    sd[p + "mlp.gate.weight"] = torch.randn(E, H, dtype=torch.bfloat16) * .02
    for e in range(E):
        for n, sh in [("gate_proj", (IK, H)), ("up_proj", (IK, H)), ("down_proj", (H, IK))]:
            sd[p + f"mlp.experts.{e}.{n}.weight"] = torch.randn(*sh, dtype=torch.bfloat16) * .02
    sd[p + "input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd[p + "post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["model.norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["lm_head.weight"] = torch.randn(V, H, dtype=torch.bfloat16) * .02
    save_file(sd, str(d / "model.safetensors"))
    return str(d)


def _magnitudes_variees(n):
    """Magnitude VARIEE par canal (colonnes x8 tous les 8, motif deja valide
    par test_quant.py::test_hadamard_helps_int4_on_outlier_channels) -- une
    statistique uniforme ne donne rien a optimiser a la recherche AWQ, qui
    s'effondrerait a l'identite meme avec beaucoup d'echantillons."""
    m = torch.full((n,), 0.02)
    m[::8] *= 8.0
    return m


def _stats_fabrique(n_riche, n_pauvre):
    """Expert 0 : statistique riche (256 jetons). Expert 1 : statistique
    pauvre (2 jetons, comme un expert presque jamais routé sur bras A) --
    un canal jamais vu (magnitude quasi nulle) cote a cote d'un canal
    heurte une fois par un jeton extreme (motif qui a produit l'étendue
    1,68e6x mesurée sur le vrai modèle)."""
    riche_h = ActStats(_magnitudes_variees(H), None, n_riche)
    riche_ik = ActStats(_magnitudes_variees(IK), None, n_riche)
    magnitudes = torch.full((IK,), 1e-6)
    magnitudes[0] = 50.0     # un seul jeton extrême sur un canal
    pauvre_down = ActStats(magnitudes, magnitudes, n_pauvre)
    pauvre_up = ActStats(_magnitudes_variees(H), None, n_pauvre)
    return {
        "model.layers.0.mlp.experts.0.gate_proj.weight": riche_h,
        "model.layers.0.mlp.experts.0.up_proj.weight": riche_h,
        "model.layers.0.mlp.experts.0.down_proj.weight": riche_ik,
        "model.layers.0.mlp.experts.1.gate_proj.weight": pauvre_up,
        "model.layers.0.mlp.experts.1.up_proj.weight": pauvre_up,
        "model.layers.0.mlp.experts.1.down_proj.weight": pauvre_down,
    }


def _convertir(tmp_path, n_pauvre):
    ckpt = _tiny_moe(tmp_path / "hf")
    spec = load_model_spec(ckpt, "tiny-moe")
    plan, _ = auto_plan(spec, load_profile("rig-14900k-5090-3080ti"),
                        PlannerOptions(max_model_len=256, max_concurrent_seqs=2))
    stats = _stats_fabrique(n_riche=256, n_pauvre=n_pauvre)
    out = tmp_path / f"out_{n_pauvre}"
    convert_checkpoint(ckpt, plan, ConversionOptions(out_dir=str(out)),
                       spec=spec, stats=stats)
    return out


def _act_scale(out_dir, name):
    """Un expert écrit TOUJOURS un `act_scale` explicite, même quand la
    recherche s'effondre à l'identité (`convert.py` : « le manifeste ne
    doit jamais mélanger échelle absente et échelle présente par
    ambiguïté d'absence ») -- `has_act_scale` seul ne distingue donc pas
    une VRAIE échelle AWQ d'un repli à `torch.ones`. Il faut lire la
    valeur."""
    from safetensors import safe_open
    for shard in sorted(out_dir.glob("acvram-*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as fh:
            if f"{name}.act_scale" in fh.keys():
                return fh.get_tensor(f"{name}.act_scale")
    raise KeyError(name)


def test_expert_a_statistique_pauvre_ne_recoit_pas_dawq(tmp_path):
    """Un changement qui doit casser : baisser `MIN_ECHANTILLONS_AWQ` sous
    2 (ou le retirer) fait sortir une échelle non triviale pour l'expert 1
    -- le motif exact du 17/09 (Nemotron, PPL 1,4301, act_scale étalée sur
    1,68e6x)."""
    out = _convertir(tmp_path, n_pauvre=2)
    riche = _act_scale(out, "model.layers.0.mlp.experts.0.down_proj.weight")
    pauvre = _act_scale(out, "model.layers.0.mlp.experts.1.down_proj.weight")
    assert not torch.allclose(riche, torch.ones_like(riche)), (
        "l'expert a statistique riche doit garder une vraie echelle AWQ")
    assert torch.allclose(pauvre, torch.ones_like(pauvre)), (
        "l'expert a 2 echantillons doit recevoir l'identite explicite, "
        "pas une echelle AWQ tiree d'une statistique bruitee")


def test_expert_a_statistique_suffisante_et_variee_recoit_awq(tmp_path):
    """Au-dessus du seuil (8), un expert échantillonné sur un motif de
    salience RAISONNABLE (variée mais pas dégénérée) garde son AWQ, sans
    déclencher la garde de norme -- le seuil protège le régime pauvre,
    pas tout ce qui n'est pas 256."""
    ckpt = _tiny_moe(tmp_path / "hf")
    spec = load_model_spec(ckpt, "tiny-moe")
    plan, _ = auto_plan(spec, load_profile("rig-14900k-5090-3080ti"),
                        PlannerOptions(max_model_len=256, max_concurrent_seqs=2))
    stats = _stats_fabrique(n_riche=256, n_pauvre=2)   # expert 1 ignoré ci-dessous
    stats["model.layers.0.mlp.experts.1.down_proj.weight"] = ActStats(
        _magnitudes_variees(IK), None, 32)
    stats["model.layers.0.mlp.experts.1.up_proj.weight"] = ActStats(
        _magnitudes_variees(H), None, 32)
    stats["model.layers.0.mlp.experts.1.gate_proj.weight"] = ActStats(
        _magnitudes_variees(H), None, 32)
    out = tmp_path / "out_varie"
    convert_checkpoint(ckpt, plan, ConversionOptions(out_dir=str(out)),
                       spec=spec, stats=stats)
    scale = _act_scale(out, "model.layers.0.mlp.experts.1.down_proj.weight")
    assert not torch.allclose(scale, torch.ones_like(scale))


def test_canal_extreme_avec_beaucoup_dechantillons_ne_declenche_plus_le_repli(tmp_path):
    """sage-awq-experts-peu-routes-portee-17-09 §2 : à l'origine, ce motif
    (un canal extrême à 50, les autres écrasés à 1e-6, n_samples=100 --
    au-dessus du seuil de 8) déclenchait la garde de norme et forçait un
    repli à l'identité (`sage-awq-relu2-garde-repli-17-09`, geste b) :
    le plancher ABSOLU (1e-6) laissait l'étendue exploser (50/1e-6 = 5e7).
    Depuis le correctif `sage-awq-plancher-median-faute-18-09` (plancher
    borné sur `max(mean_abs)/4096`, pas sur une statistique de position),
    ce même motif reste dans les bornes SANS repli : l'étendue est bornée
    à 4096 par construction quelle que soit l'amplitude du canal extrême.
    Un changement qui doit casser : revenir au plancher absolu ou au
    plancher par médiane fait réapparaître le repli ici."""
    ckpt = _tiny_moe(tmp_path / "hf")
    spec = load_model_spec(ckpt, "tiny-moe")
    plan, _ = auto_plan(spec, load_profile("rig-14900k-5090-3080ti"),
                        PlannerOptions(max_model_len=256, max_concurrent_seqs=2))
    magnitudes = torch.full((IK,), 1e-6)
    magnitudes[0] = 50.0
    stats = _stats_fabrique(n_riche=256, n_pauvre=2)
    stats["model.layers.0.mlp.experts.1.down_proj.weight"] = ActStats(
        magnitudes, magnitudes, 100)     # meme motif degenere, n_samples eleve
    out = tmp_path / "out_extreme"
    report = convert_checkpoint(ckpt, plan, ConversionOptions(out_dir=str(out)),
                                spec=spec, stats=stats)
    assert report.tenseurs_replies == []
