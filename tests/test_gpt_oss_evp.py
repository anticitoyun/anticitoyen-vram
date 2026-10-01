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


def _eager_hf(q, k, v, sinks, scale, window):
    """Référence : `eager_attention_forward` de transformers (gpt-oss), masque causal (+ fenêtre) additif construit
    avec la même règle que `decode_attention_fixed` (clé j visible si j ≤ i et j > i − window)."""
    from types import SimpleNamespace
    from transformers.models.gpt_oss.modeling_gpt_oss import eager_attention_forward
    t, hq, _ = q.shape
    L, hkv, _ = k.shape
    off = L - t
    i = off + torch.arange(t)[:, None]
    j = torch.arange(L)[None, :]
    ok = (j <= i) & ((j > i - window) if window > 0 else torch.ones_like(j, dtype=torch.bool))
    m = torch.zeros(t, L).masked_fill(~ok, float("-inf"))[None, None]
    mod = SimpleNamespace(sinks=sinks, num_key_value_groups=hq // hkv, training=False)
    out, _ = eager_attention_forward(mod, q.permute(1, 0, 2)[None], k.permute(1, 0, 2)[None],
                                     v.permute(1, 0, 2)[None], m, scaling=scale, dropout=0.0)
    return out[0]                                     # [t, hq, d] (transformers rend [b, t, h, d])


@pytest.mark.parametrize("window", [0, 8])
@pytest.mark.parametrize("t,L", [(13, 13), (5, 21), (1, 30)])
def test_puits_contre_transformers(window, t, L):
    from acvram.engine.layers import attention_puits
    torch.manual_seed(0)
    hq, hkv, d = 8, 2, 64
    q, k, v = torch.randn(t, hq, d), torch.randn(L, hkv, d), torch.randn(L, hkv, d)
    sinks = torch.randn(hq) * 2
    ref = _eager_hf(q, k, v, sinks, d ** -0.5, window)
    out = attention_puits(q, k, v, sinks, d ** -0.5, q_offset=L - t, window=window, n_rep=hq // hkv, lignes=4)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


def test_puits_decodage_fixe_egal_au_chemin_variable():
    """Le pas à formes fixes (graphe) et le chemin par séquence rendent la même ligne ; témoin : un puits très
    négatif rend l'attention ordinaire, un puits fini la change (le contrôle qui peut rendre faux)."""
    from acvram.engine.layers import attention_puits, decode_attention_fixed, decode_attention_puits_fixe
    torch.manual_seed(1)
    b, S, hq, hkv, d, w = 3, 24, 8, 2, 64, 8
    lens = torch.tensor([24, 10, 3])
    q, k, v = torch.randn(b, hq, d), torch.randn(b, S, hkv, d), torch.randn(b, S, hkv, d)
    sinks = torch.randn(hq)
    fixe = decode_attention_puits_fixe(q, k, v, lens, sinks, hq // hkv, d ** -0.5, window=w)
    for i, n in enumerate(lens.tolist()):
        var = attention_puits(q[i:i + 1], k[i, :n], v[i, :n], sinks, d ** -0.5, q_offset=n - 1, window=w,
                              n_rep=hq // hkv)
        torch.testing.assert_close(fixe[i:i + 1], var, rtol=1e-5, atol=1e-6)
    sans = decode_attention_fixed(q, k, v, lens, hq // hkv, d ** -0.5, window=w)
    torch.testing.assert_close(decode_attention_puits_fixe(q, k, v, lens, torch.full((hq,), -1e4), hq // hkv,
                                                           d ** -0.5, window=w), sans, rtol=1e-5, atol=1e-6)
    assert (fixe - sans).abs().max() > 1e-2
