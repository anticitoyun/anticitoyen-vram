"""poste7-hybrides-etape1-close-gemm-dense-17-09 : `_adapt_hf` (flux principal)
et `collect.py` (calibration) doivent renommer les tenseurs bruts nemotron_h
(`backbone.layers.N.mixer.*`) à l'identique -- `_nemotron_h_rename`/
`_nemotron_h_valeur` sont la table de vérité UNIQUE que les deux partagent
(refactor du 17/09, auparavant dupliqué en ligne dans `_adapt_hf` seul).
Ce test fige le comportement de `_adapt_hf` AVANT toute modification future
de cette table -- un changement qui doit casser : altérer le mapping
`_NEMOTRON_H_MAMBA_HEADS` ou l'ordre des conditions dans `_nemotron_h_rename`."""
import torch

from acvram.engine.config import ModelSpec
from acvram.quant.convert import _adapt_hf


def _spec_nemotron_h(num_layers=2):
    return ModelSpec(
        name="nemotron-h-test", model_type="nemotron_h", architecture="llama",
        hidden_size=64, intermediate_size=128, num_layers=num_layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        vocab_size=100, max_position_embeddings=64, rope_theta=10000.0,
        mamba_num_heads=4, mamba_head_dim=8, mamba_n_groups=1, mamba_state_size=16,
        moe_intermediate_size=32, shared_expert_intermediate_size=32,
    )


def _bruts():
    """Un jeu de tenseurs bruts nommés exactement comme le vrai point de
    contrôle bf16 (`backbone.embeddings.weight`, `backbone.layers.N.mixer.*`,
    sans suffixe `.weight` sur A_log/D/dt_bias -- vérifié le 17/09 par lecture
    directe du `model.safetensors.index.json` source)."""
    return [
        ("backbone.embeddings.weight", torch.randn(100, 64)),
        ("backbone.norm_f.weight", torch.randn(64)),
        ("backbone.layers.0.norm.weight", torch.randn(64)),
        ("backbone.layers.0.mixer.A_log", torch.zeros(4)),          # A = -exp(0) = -1
        ("backbone.layers.0.mixer.D", torch.arange(4, dtype=torch.float32)),
        ("backbone.layers.0.mixer.dt_bias", torch.arange(4, dtype=torch.float32)),
        ("backbone.layers.0.mixer.conv1d.weight", torch.randn(32, 1, 4)),
        ("backbone.layers.0.mixer.conv1d.bias", torch.randn(32)),
        ("backbone.layers.0.mixer.in_proj.weight", torch.randn(100, 64)),  # = n_in_proj (pas de rognage)
        ("backbone.layers.0.mixer.out_proj.weight", torch.randn(64, 32)),
        ("backbone.layers.0.mixer.norm.weight", torch.randn(32)),
        ("backbone.layers.1.norm.weight", torch.randn(64)),
        ("backbone.layers.1.mixer.experts.0.up_proj.weight", torch.randn(32, 64)),
        ("backbone.layers.1.mixer.experts.0.down_proj.weight", torch.randn(64, 32)),
        ("backbone.layers.1.mixer.shared_experts.up_proj.weight", torch.randn(32, 64)),
        ("backbone.layers.1.mixer.router.weight", torch.randn(8, 64)),
        ("backbone.layers.1.mixer.q_proj.weight", torch.randn(64, 64)),
        ("backbone.layers.1.mixer.o_proj.weight", torch.randn(64, 64)),
        ("backbone.layers.2.norm.weight", torch.randn(64)),         # hors plan (2 couches)
        ("mtp.embeddings.weight", torch.randn(100, 64)),
    ]


def test_renommage_et_valeurs_identiques_au_comportement_fige():
    out = dict(_adapt_hf(iter(_bruts()), _spec_nemotron_h(num_layers=2)))

    assert "model.embed_tokens.weight" in out
    assert "model.norm.weight" in out
    assert "model.layers.0.input_layernorm.weight" in out
    assert torch.allclose(out["model.layers.0.mamba.A.weight"], torch.full((4,), -1.0))
    assert torch.equal(out["model.layers.0.mamba.D.weight"], torch.arange(4, dtype=torch.float32))
    assert torch.equal(out["model.layers.0.mamba.dt_bias.weight"], torch.arange(4, dtype=torch.float32))
    assert out["model.layers.0.mamba.conv1d.weight"].shape == (32, 4)     # [d,1,L] -> [d,L]
    assert out["model.layers.0.mamba.conv1d.bias"].shape == (32,)         # inchangé (pas de reshape)
    assert out["model.layers.0.mamba.in_proj.weight"].shape == (100, 64)
    assert out["model.layers.0.mamba.out_proj.weight"].shape == (64, 32)
    assert out["model.layers.0.mamba.norm.weight"].shape == (32,)

    assert out["model.layers.1.mlp.experts.0.up_proj.weight"].shape == (32, 64)
    assert out["model.layers.1.mlp.experts.0.down_proj.weight"].shape == (64, 32)
    assert "model.layers.1.mlp.shared_expert.up_proj.weight" in out
    assert "model.layers.1.mlp.router.weight" in out
    assert "model.layers.1.self_attn.q_proj.weight" in out
    assert "model.layers.1.self_attn.o_proj.weight" in out

    # hors plan (couche 2 sur 2) et MTP : ignorés, pas dans la sortie
    assert not any(k.startswith("model.layers.2.") for k in out)
    assert not any(k.startswith("mtp.") or "embeddings" in k and k != "model.embed_tokens.weight"
                  for k in out)
    assert len(out) == 18   # 20 tenseurs bruts - 2 ignorés (couche 2 hors plan, mtp.)
