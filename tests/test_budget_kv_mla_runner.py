"""Le budget de jetons vivants d'un modèle MLA (sans cache paginé) venait
d'un défaut arbitraire, pas du plan réel — ordre poste7
(revue/poste7-glm-pile-correctif-16-09.md), découvert sur GLM le 16/09.

`Engine.__init__` (`runner.py:348`) calculait `n_blocks` par
`min((c.cfg.num_blocks for c in self.model.caches.values()), default=1024)`.
Sur un modèle MLA pur, `self.model.caches` est TOUJOURS vide — le cache
latent n'est jamais enregistré dans ce dict (`loader.py`, la branche MLA
ne fait jamais `caches[i] = ...`) — le générateur ne rend rien et le
défaut de 1024 blocs (`BLOCK_SIZE=16` → **16 384 jetons**) s'appliquait
silencieusement, quel que soit le plan réel (`auto_plan`, `kv_max_tokens`)
calculé pour ce modèle et ce rig.
"""
import json

import torch
from safetensors.torch import save_file

from acvram.engine.config import load_model_spec
from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams
from acvram.memory.kvcache import BLOCK_SIZE, BlockAllocator
from acvram.memory.tiering import PlannerOptions, auto_plan
from acvram.quant.convert import ConversionOptions, convert_checkpoint

H, INTER, L, NH, V = 128, 256, 2, 4, 1000
KV_LORA, Q_LORA, QK_ROPE, QK_NOPE, V_DIM = 512, 384, 32, 96, 128


def _checkpoint_mla(tmp_path):
    """Un point de contrôle MLA à 2 couches, forme DeepSeek-V2/GLM."""
    torch.manual_seed(20260916)
    d = tmp_path / "hf"
    d.mkdir()
    cfg = {
        "architectures": ["DeepseekV2ForCausalLM"], "model_type": "deepseek_v2",
        "hidden_size": H, "intermediate_size": INTER, "num_hidden_layers": L,
        "num_attention_heads": NH, "num_key_value_heads": NH, "vocab_size": V,
        "max_position_embeddings": 2048, "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0, "torch_dtype": "bfloat16",
        "kv_lora_rank": KV_LORA, "q_lora_rank": Q_LORA,
        "qk_rope_head_dim": QK_ROPE, "qk_nope_head_dim": QK_NOPE,
        "v_head_dim": V_DIM,
        "layer_types": ["full_attention"] * L,
    }
    json.dump(cfg, open(d / "config.json", "w"))

    def w(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16) * 0.02

    sd = {"model.embed_tokens.weight": w(V, H)}
    for i in range(L):
        p = f"model.layers.{i}."
        sd[p + "self_attn.q_a_proj.weight"] = w(Q_LORA, H)
        sd[p + "self_attn.q_a_layernorm.weight"] = torch.ones(Q_LORA, dtype=torch.bfloat16)
        sd[p + "self_attn.q_b_proj.weight"] = w(NH * (QK_NOPE + QK_ROPE), Q_LORA)
        sd[p + "self_attn.kv_a_proj_with_mqa.weight"] = w(KV_LORA + QK_ROPE, H)
        sd[p + "self_attn.kv_a_layernorm.weight"] = torch.ones(KV_LORA, dtype=torch.bfloat16)
        sd[p + "self_attn.k_b_proj.weight"] = w(NH, KV_LORA, QK_NOPE)
        sd[p + "self_attn.v_b_proj.weight"] = w(NH, V_DIM, KV_LORA)
        sd[p + "self_attn.o_proj.weight"] = w(H, NH * V_DIM)
        sd[p + "mlp.gate_proj.weight"] = w(INTER, H)
        sd[p + "mlp.up_proj.weight"] = w(INTER, H)
        sd[p + "mlp.down_proj.weight"] = w(H, INTER)
        sd[p + "input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
        sd[p + "post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["model.norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["lm_head.weight"] = w(V, H)
    save_file(sd, str(d / "model.safetensors"))
    return str(d)


def _converted_mla(tmp_path, target_rig, max_concurrent_seqs=2, max_model_len=512):
    src = _checkpoint_mla(tmp_path)
    spec = load_model_spec(src)
    assert spec.est_mla, "la config doit être détectée MLA pour que ce test soit probant"
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(
        max_model_len=max_model_len, max_concurrent_seqs=max_concurrent_seqs))
    out = str(tmp_path / "acvram")
    convert_checkpoint(src, plan, ConversionOptions(out_dir=out), spec=spec)
    return out, plan


def test_budget_mla_vient_du_plan_pas_du_defaut(tmp_path, target_rig):
    out, plan = _converted_mla(tmp_path, target_rig, max_concurrent_seqs=2, max_model_len=512)
    # Carte présente, le chargeur REPLANIFIE sur le rig réel (loader._replanifier,
    # plancher de contexte 2048 : 2 × 2048 = 4 096 jetons = 256 blocs, T4 20/09) ;
    # le test parle du plan de la conversion : il le redemande tel quel.
    loaded = load_model(out, dtype=torch.float32, device_override="cpu",
                        max_model_len=512, max_concurrent_seqs=2)
    assert not loaded.model.caches, \
        "un MLA pur ne doit rien enregistrer dans .caches (sinon ce test ne teste rien)"

    engine = Engine(loaded, None, max_batch_size=2, max_model_len=512,
                    enable_cuda_graphs=False)

    attendu = max(1, plan.kv_max_tokens // BLOCK_SIZE)
    assert engine.allocator.num_free == attendu, \
        f"budget attendu du plan ({attendu} blocs) pas {1024} (le défaut)"
    # Un changement qui doit casser : le plan réel d'un jouet, sur un vrai
    # rig, n'a aucune raison de retomber exactement sur l'ancien défaut.
    assert attendu != 1024, \
        "le plan et le défaut coïncident par accident, ce test ne serait pas probant"


def test_depassement_budget_journal_et_finish_reason(tmp_path, target_rig, capsys):
    out, _plan = _converted_mla(tmp_path, target_rig, max_concurrent_seqs=2, max_model_len=512)
    loaded = load_model(out, dtype=torch.float32, device_override="cpu")
    engine = Engine(loaded, None, max_batch_size=2, max_model_len=512,
                    enable_cuda_graphs=False)

    # Budget volontairement minuscule (3 blocs = 48 jetons) : un
    # dépassement doit survenir de façon déterministe pendant le
    # décodage, pas dépendre du budget réel — potentiellement généreux —
    # que le plan aurait calculé pour ce rig.
    engine.allocator = BlockAllocator(3, True)
    capsys.readouterr()

    produced = list(engine.generate([5, 42, 7],
                                    SamplingParams(temperature=0.0, max_tokens=64)))

    assert produced, "aucune sortie produite"
    assert produced[-1].finished
    assert produced[-1].finish_reason == "length", \
        "le contrat API (OpenAI/Anthropic) ne change pas : toujours \"length\""
    assert len(produced) < 64, \
        "doit s'arrêter avant max_tokens, faute de budget KV — sinon le budget de 3 " \
        "blocs n'a pas été le facteur limitant et ce test ne teste rien"

    journal = capsys.readouterr().out
    assert "budget KV épuisé" in journal, \
        "un dépassement de budget doit se journaliser, jamais silencieusement (REGLES)"
    assert "sortis=" in journal and "blocs_libres=" in journal
