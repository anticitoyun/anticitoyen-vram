"""Les décisions du planificateur de placement, sur la vraie machine cible."""

import json
import os

import pytest

from acvram.engine.config import ModelSpec, load_model_spec
from acvram.hardware.detect import capabilities_for_sm
from acvram.memory.tiering import PlannerOptions, auto_plan, plan_placement


def _spec(tmp_path, name, **cfg):
    base = dict(architectures=["LlamaForCausalLM"], hidden_size=4096,
                intermediate_size=11008, num_hidden_layers=32,
                num_attention_heads=32, num_key_value_heads=8,
                vocab_size=128256, max_position_embeddings=8192)
    base.update(cfg)
    d = tmp_path / name
    d.mkdir()
    json.dump(base, open(d / "config.json", "w"))
    return load_model_spec(str(d), name)


def test_blackwell_a_le_fp4_ampere_non():
    assert capabilities_for_sm(120).weight_format == "nvfp4"
    assert capabilities_for_sm(120).fp4_tensor_core
    assert capabilities_for_sm(86).weight_format == "int4_awq"
    assert not capabilities_for_sm(86).fp4_tensor_core
    assert not capabilities_for_sm(86).fp8_tensor_core


def test_second_gpu_is_left_idle_when_the_model_fits_on_the_first(tmp_path, target_rig):
    """Étendre le pipeline à une carte plus lente coûte du débit mono-flux."""
    spec = _spec(tmp_path, "small", num_hidden_layers=24)
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=4096))
    assert {lp.exec_device for lp in plan.layers} == {"cuda:0"}
    assert any("laisse oisif a dessein" in w for w in plan.warnings)


def test_big_model_uses_both_gpus(tmp_path, target_rig):
    spec = _spec(tmp_path, "big", hidden_size=8192, intermediate_size=28672,
                 num_hidden_layers=80)
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=4096))
    assert {lp.exec_device for lp in plan.layers} == {"cuda:0", "cuda:1"}


def test_pipeline_crosses_gpus_exactly_once(tmp_path, target_rig):
    spec = _spec(tmp_path, "cross", hidden_size=8192, intermediate_size=28672,
                 num_hidden_layers=80)
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=4096))
    crossings = sum(1 for a, b in zip(plan.layers, plan.layers[1:])
                    if a.exec_device != b.exec_device)
    assert crossings <= 1


def test_attention_is_pinned_before_mlp_spills(tmp_path, target_rig):
    """L'attention est petite et critique en latence ; c'est le MLP qui part en RAM."""
    # 128 experts de 1536 sur 60 couches font environ 145 milliards de
    # paramètres : bien au-delà des 44 Go de VRAM cumulée, donc quelque chose
    # doit déborder.
    spec = _spec(tmp_path, "moe", hidden_size=4096, num_hidden_layers=60,
                 architectures=["Qwen3MoeForCausalLM"], num_experts=128,
                 num_experts_per_tok=8, moe_intermediate_size=1536)
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=4096))
    host_mlp = [lp for lp in plan.layers if lp.mlp_storage == "cpu"]
    host_attn = [lp for lp in plan.layers if lp.attn_storage == "cpu"]
    assert host_mlp, "ce modèle ne devrait pas tenir entièrement en VRAM"
    assert len(host_attn) < len(host_mlp)


def test_moe_reads_far_less_than_it_stores(tmp_path, target_rig):
    spec = _spec(tmp_path, "moe2", hidden_size=4096, num_hidden_layers=48,
                 architectures=["Qwen3MoeForCausalLM"], num_experts=128,
                 num_experts_per_tok=8, moe_intermediate_size=1536)
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=4096))
    assert plan.est_bytes_per_token < plan.total_weight_bytes / 4


def test_impossible_model_is_reported_not_silently_truncated(tmp_path, target_rig):
    spec = _spec(tmp_path, "huge", hidden_size=16384, intermediate_size=53248,
                 num_hidden_layers=126)
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=4096))
    assert any("ne tient pas sur cette machine" in w for w in plan.warnings)
    assert any("bits par poids" in w for w in plan.warnings)


def test_kv_budget_shrinks_to_keep_weights_in_vram(tmp_path, target_rig):
    spec = _spec(tmp_path, "seventy", hidden_size=8192, intermediate_size=28672,
                 num_hidden_layers=80)
    greedy = plan_placement(spec, target_rig,
                            PlannerOptions(max_model_len=32768,
                                           max_concurrent_seqs=4,
                                           kv_vram_fraction=0.55))
    tuned, _ = auto_plan(spec, target_rig,
                         PlannerOptions(max_model_len=32768, max_concurrent_seqs=4))
    assert tuned.est_decode_tok_s > greedy.est_decode_tok_s


def test_plan_survives_a_json_round_trip(tmp_path, target_rig):
    spec = _spec(tmp_path, "rt")
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=4096))
    assert json.loads(json.dumps(plan.to_dict()))["model"] == plan.model


# --- coût de l'exil : bead anticitoyen-vram-jt5 (geste 1 de ggrun) -----------
#
# On construit un plan synthétique plutôt que de passer par auto_plan : le but
# est de tester la fonction de coût sur des bandes et des octets fixés, pas la
# planification. Sans carte, sans modèle chargé (le SSD des modèles est
# injoignable).

from acvram.memory.tiering import (Plan, Tier, LayerPlacement,  # noqa: E402
                                   estimer_cout_exil)
from acvram.engine.loader import _signaler_cout_exil  # noqa: E402

_GIB = 2 ** 30


def _tier_5090(link_gbps: float = 26.8) -> Tier:
    # 5090 : VRAM en lecture ~1050 Go/s, MESURÉ le 9/09 (trois dispositifs
    # indépendants convergent, docs/PLAFOND-MEMOIRE-5090.md) — pas le pic de
    # plaque 1792 (trafic mixte, jamais ce que lit un GEMV). Corrigé le
    # 13/09 (poste7) : `detect.py:vram_bandwidth_gbps` prenait le pic, ce qui
    # sous-estimait le ratio de `estimer_cout_exil` d'un facteur 1,7.
    # Lien gen4x16 estimé ~26,8 Go/s (detect.py:200).
    return Tier(name="cuda:0", kind="gpu", device_index=0, capacity=32 * _GIB,
                weight_format="nvfp4", kv_format="fp8",
                read_bandwidth=1050.0, link_bandwidth=link_gbps)


def _plan_nemotron(n_exiles: int, link_gbps: float = 26.8) -> Plan:
    """Approche nemotron-lightning-exl3 : 32 couches, ~0,56 Gio de MLP actif
    par couche (17,8 Gio / 32), attention menue. `n_exiles` MLP en RAM hôte."""
    mlp = int(0.556 * _GIB)
    attn = int(0.05 * _GIB)
    layers = []
    for i in range(32):
        exile = i >= 32 - n_exiles
        layers.append(LayerPlacement(
            index=i, exec_device="cuda:0", attn_storage="cuda:0",
            mlp_storage="cpu" if exile else "cuda:0", fmt="nvfp4",
            attn_bytes=attn, mlp_bytes=mlp, mlp_active_bytes=mlp))
    return Plan(model="nemotron-lightning-exl3", tiers=[_tier_5090(link_gbps)],
                layers=layers)


def test_exil_franchit_le_seuil_falaise():
    # PRÉDICTION SCELLÉE (règle 4), lien gen4x16 estimé 26,8 Go/s, VRAM en
    # lecture 1050 Go/s (mesuré, pas le pic — voir _tier_5090) :
    #   T_transfert(1 couche) = 0,556 Gio / 26,8 Go/s ≈ 22,3 ms/jeton
    #   pas résident (32 couches, 0,606 Gio chacune) ≈ 19,8 ms
    # Donc UNE seule couche exilée coûte déjà plus que le pas entier (ratio
    # ≈ 1,12) : franchit le seuil de 20 %. Issue qui me gênerait : le modèle
    # over-signale si le lien réel est plus rapide — c'est pourquoi la bande
    # est un paramètre du plan (mesurable), pas une constante, et pourquoi le
    # test « pas de falaise » ci-dessous vérifie que le signal peut se taire.
    c = estimer_cout_exil(_plan_nemotron(8))
    assert c is not None
    assert c["n_couches_exilees"] == 8
    assert c["ratio"] > 1.0            # l'exil coûte plus qu'un pas entier
    assert c["franchit_seuil"]
    # une couche exilée seule franchit déjà le seuil
    assert estimer_cout_exil(_plan_nemotron(1))["franchit_seuil"]


def test_pas_de_falaise_quand_rien_n_est_exile():
    # Le contrôle DOIT pouvoir rendre « faux » (règle 5) : aucun MLP exilé →
    # rien à chiffrer → None, aucun avertissement.
    plan = _plan_nemotron(0)
    assert estimer_cout_exil(plan) is None
    _signaler_cout_exil(plan)
    assert not any("exil" in w for w in plan.warnings)


def test_pas_de_falaise_quand_le_lien_est_rapide():
    # Même exil, mais un lien aussi rapide que la VRAM en lecture (1050,
    # NVLink hypothétique) : le surcoût tombe sous le seuil. Deuxième preuve
    # que le signal se tait quand il le doit — sinon ce serait une alarme,
    # pas un contrôle.
    c = estimer_cout_exil(_plan_nemotron(1, link_gbps=1050.0))
    assert c is not None and not c["franchit_seuil"]
    assert c["ratio"] < 0.20


def test_bande_inconnue_ne_fabrique_pas_de_chiffre():
    # Règle 10 : un échec est un résultat. Lien à 0 (inconnu) → None, pas un
    # zéro trompeur qui dirait « exil gratuit ».
    plan = _plan_nemotron(4, link_gbps=0.0)
    assert estimer_cout_exil(plan) is None


def test_signal_exil_ecrit_un_avertissement_chiffre():
    plan = _plan_nemotron(8)
    _signaler_cout_exil(plan)
    assert any("exil" in w and "ms/jeton" in w for w in plan.warnings)


def test_le_plan_rend_le_maximum_de_contexte_qui_tient_pas_le_plus_petit(tmp_path, target_rig):
    """poste7 poste7-s2-k48-feu-vert-21-09 (addendum) : la colonne 262 144 ne demande pas « tient-il ? » mais
    « combien tiennent ? » — kv_max_tokens > 0, multiple de 1024, plus grand que le plan à la plus petite fraction
    (0,06, l'ancien repli), sans exiler un poids de plus. Cassant : l'ancien repli rendait le 0,06."""
    spec = _spec(tmp_path, "seventy", hidden_size=8192, intermediate_size=28672, num_hidden_layers=80)
    petit = plan_placement(spec, target_rig, PlannerOptions(max_model_len=262144, kv_vram_fraction=0.06))
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=262144))
    assert plan.kv_max_tokens > 0 and plan.kv_max_tokens % 1024 == 0 and not plan.overflowed
    assert plan.kv_max_tokens > petit.kv_max_tokens, (plan.kv_max_tokens, petit.kv_max_tokens)
    assert plan.bytes_per_tier.get("cpu", 0) <= petit.bytes_per_tier.get("cpu", 0)
    assert any("maximum apres les poids" in w for w in plan.warnings)


def test_zero_jeton_seulement_si_les_poids_ne_tiennent_pas_ou_sans_carte(tmp_path, target_rig):
    from acvram.hardware.detect import Rig
    huge = _spec(tmp_path, "huge", hidden_size=16384, intermediate_size=53248, num_hidden_layers=126)
    plan, _ = auto_plan(huge, target_rig, PlannerOptions(max_model_len=262144))
    assert plan.kv_max_tokens == 0 and any("ne tient pas sur cette machine" in w for w in plan.warnings)
    sans_carte = Rig(host=target_rig.host)
    plan2, _ = auto_plan(_spec(tmp_path, "petit"), sans_carte, PlannerOptions(max_model_len=262144))
    assert plan2.kv_max_tokens == 0 and any("aucune carte visible" in w for w in plan2.warnings)
