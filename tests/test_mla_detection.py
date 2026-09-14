"""Bead anticitoyen-vram-992 (14/09) : la détection MLA est portée par
`ModelSpec.est_mla` (kv_lora_rank > 0), pas par une liste de `model_type` —
GLM-4.7-Flash (`glm4_moe_lite`) en manquait aux trois sites (config.py,
convert.py, loader.py) et la conversion serait passée en silence par le
chemin non-MLA. `nope != v_head_dim` (192/256 sur ce modèle) n'a jamais été
posé chez nous avant : testé explicitement, pas supposé correct.
"""
import json

import torch

from acvram.engine.config import load_model_spec


def _config_glm4_moe_lite(tmp_path, kv_lora_rank=512):
    """Une configuration MINIMALE, avec un `model_type` volontairement
    absent de toute liste historique — le nom qui casse le test si une
    liste de noms est réintroduite à la place de `est_mla`."""
    cfg = {
        "model_type": "glm4_moe_lite",       # jamais dans aucune liste MLA
        "hidden_size": 128, "intermediate_size": 256,
        "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 4, "vocab_size": 1000,
        "max_position_embeddings": 512,
        "kv_lora_rank": kv_lora_rank, "q_lora_rank": 384,
        "qk_rope_head_dim": 32, "qk_nope_head_dim": 96, "v_head_dim": 128,
        "num_experts": 4, "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return str(p)


def test_glm4_moe_lite_est_reconnu_mla(tmp_path):
    """Un `model_type` absent de toute liste historique, avec
    `kv_lora_rank > 0` : `est_mla` et `mla_rope` doivent être vrais.
    Un changement qui doit casser : remettre `if mt in (...)` à la place
    de `if cfg.get("kv_lora_rank")` (config.py) romprait ce test."""
    spec = load_model_spec(_config_glm4_moe_lite(tmp_path))
    assert spec.est_mla
    assert spec.mla_rope
    assert spec.layer_types == ["full_attention", "full_attention"]


def test_sans_kv_lora_rank_pas_mla(tmp_path):
    """Le même `model_type`, sans compression KV : pas de MLA — le critère
    est bien `kv_lora_rank`, pas le nom."""
    spec = load_model_spec(_config_glm4_moe_lite(tmp_path, kv_lora_rank=0))
    assert not spec.est_mla
    assert not spec.mla_rope


def test_model_type_hors_liste_historique_avec_lora_est_mla():
    """N'importe quel `model_type`, même complètement inventé : `est_mla`
    ne regarde que `kv_lora_rank` (ModelSpec.est_mla, config.py:262-270) —
    une architecture nouvelle qui compresse son KV se reconnaît sans
    qu'on l'ajoute à une liste, par construction de la propriété elle-même."""
    from acvram.engine.config import ModelSpec
    spec = ModelSpec(name="x", architecture="TotalementInventeForCausalLM",
                     hidden_size=128, intermediate_size=256, num_layers=2,
                     num_attention_heads=4, num_key_value_heads=4,
                     vocab_size=1000, max_position_embeddings=512,
                     kv_lora_rank=512)
    assert spec.est_mla


class ModelSpecFactice:
    """Un objet minimal qui porte juste ce que `_adapt_hf` lit pour le
    bloc MLA -- eviter de construire un ModelSpec complet pour ce test."""
    def __init__(self, **kw):
        self.model_type = kw["model_type"]
        self.num_attention_heads = kw["num_attention_heads"]
        self.qk_nope_head_dim = kw["qk_nope_head_dim"]
        self.v_head_dim = kw["v_head_dim"]
        self.kv_lora_rank = kw["kv_lora_rank"]

    @property
    def est_mla(self) -> bool:
        return self.kv_lora_rank > 0


def test_scission_kv_b_proj_nope_different_de_v(tmp_path):
    """`convert.py` (bloc MLA, ~ligne 513) : la scission de `kv_b_proj`
    en `k_b_proj`/`v_b_proj` ne suppose PAS `nope == v_head_dim` — GLM-4.7-
    Flash a nope=192, v=256, jamais posé chez nous avant ce bead. Vérifie
    les FORMES et les VALEURS (pas seulement que ça ne lève pas)."""
    from acvram.quant.convert import _adapt_hf

    nh, nope, vd, rank = 4, 192, 256, 64
    spec = ModelSpecFactice(num_attention_heads=nh, qk_nope_head_dim=nope,
                            v_head_dim=vd, kv_lora_rank=rank,
                            model_type="glm4_moe_lite")

    kv_b = torch.arange(nh * (nope + vd) * rank, dtype=torch.float32) \
        .reshape(nh * (nope + vd), rank)
    source = [("model.layers.0.self_attn.kv_b_proj.weight", kv_b)]

    sortie = dict(_adapt_hf(iter(source), spec))
    k_b = sortie["model.layers.0.self_attn.k_b_proj.weight"]
    v_b = sortie["model.layers.0.self_attn.v_b_proj.weight"]

    assert k_b.shape == (nh, rank, nope)
    assert v_b.shape == (nh, vd, rank)

    # Valeurs : reconstruit la reference sans passer par _adapt_hf, pour
    # comparer un chemin de calcul independant du code teste.
    kv_ref = kv_b.reshape(nh, nope + vd, rank)
    k_ref = kv_ref[:, :nope, :].transpose(1, 2)
    v_ref = kv_ref[:, nope:, :]
    assert torch.equal(k_b, k_ref)
    assert torch.equal(v_b, v_ref)
