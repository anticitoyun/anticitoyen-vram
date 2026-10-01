"""evp (gpt-oss) : chaque écart d'architecture contre la couche de référence de transformers, au bit, sur processeur.

Scellé : scratchpad/poste1-evp-01-10/scelle.md. Les cliquets « au bit d'avant » protègent les alias déjà servis : une
correction propre à gpt-oss ne doit changer aucune autre sortie."""
import hashlib

import pytest
import torch

from acvram.engine.layers import RotaryEmbedding

tf = pytest.importorskip("transformers")


def _h(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


# Empreintes relevées sur bce7dc964, AVANT la pièce (inv_freq, cos, sin sur 300 positions, fp32).
_AVANT = {
    "deepseek_yarn": (dict(head_dim=64, base=10000.0, scaling={"rope_type": "yarn", "factor": 40,
                                                               "original_max_position_embeddings": 4096,
                                                               "beta_fast": 32, "beta_slow": 1,
                                                               "mscale": 1.0, "mscale_all_dim": 1.0}),
                      ("bb1f5b8e78d129b3", "55e6c49374975262", "0240483675a0f3b7")),
    "qwen_yarn": (dict(head_dim=128, base=1000000.0, scaling={"rope_type": "yarn", "factor": 4.0,
                                                              "original_max_position_embeddings": 32768}),
                  ("427ed49dc1d6e186", "b9ebf1eb803426d0", "bf4e51722423cdd0")),
}


@pytest.mark.parametrize("nom", sorted(_AVANT))
def test_yarn_sans_truncate_au_bit_d_avant(nom):
    c, attendu = _AVANT[nom]
    r = RotaryEmbedding(c["head_dim"], 4096, base=c["base"], scaling=c["scaling"])
    cos, sin = r(torch.arange(300), torch.device("cpu"), torch.float32, max_pos=300)
    assert (_h(r.inv_freq), _h(cos), _h(sin)) == attendu, "un alias YaRN déjà servi a changé de tables RoPE"


def _config_gpt_oss(**k):
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    base = dict(hidden_size=128, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=64, num_local_experts=4, num_experts_per_tok=2, vocab_size=97,
                sliding_window=8, rope_theta=150000.0,
                rope_scaling={"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0,
                              "original_max_position_embeddings": 4096, "truncate": False})
    base.update(k)
    return GptOssConfig(**base)


def test_yarn_gpt_oss_au_bit_de_transformers():
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding
    cfg = _config_gpt_oss()
    ref = GptOssRotaryEmbedding(cfg)
    pos = torch.arange(300)[None]
    cos_h, sin_h = ref(torch.zeros(1, 300, 64, dtype=torch.float32), pos)
    r = RotaryEmbedding(64, 4096, base=150000.0, scaling=dict(cfg.rope_scaling if getattr(cfg, "rope_scaling", None)
                                                               else cfg.rope_parameters))
    cos, sin = r(torch.arange(300), torch.device("cpu"), torch.float32, max_pos=300)
    assert torch.equal(r.inv_freq, ref.inv_freq.to(torch.float32))
    assert abs(r.facteur_attention() - ref.attention_scaling) == 0
    # gpt-oss rend les demi-tables (rotation par moitiés, `torch.chunk`) : la table acvram est [demi, demi]
    assert torch.equal(cos, torch.cat((cos_h[0], cos_h[0]), -1)) and torch.equal(sin, torch.cat((sin_h[0], sin_h[0]), -1))


def test_temoin_le_facteur_d_attention_change_les_tables():
    """Contrôle qui peut rendre faux : sans `attention_factor` (truncate retiré), les tables gpt-oss DIFFÈRENT."""
    s = {"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0,
         "original_max_position_embeddings": 4096, "truncate": False}
    a = RotaryEmbedding(64, 4096, base=150000.0, scaling=s)
    b = RotaryEmbedding(64, 4096, base=150000.0, scaling={k: v for k, v in s.items() if k != "truncate"})
    assert a.facteur_attention() > 1.3 and b.facteur_attention() == 1.0
    ca, _ = a(torch.arange(50), torch.device("cpu"), torch.float32, max_pos=50)
    cb, _ = b(torch.arange(50), torch.device("cpu"), torch.float32, max_pos=50)
    assert not torch.equal(ca, cb)
