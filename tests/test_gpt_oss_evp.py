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


def _bloc_moe_depuis_hf(mlp_hf, top_k):
    """MoEBlock acvram construit depuis un GptOssMLP de transformers : gate/up DÉSENTRELACÉS (colonnes paires/impaires
    de gate_up_proj [E, h, 2I]), biais compris — la même transformation que fera le convertisseur."""
    from acvram.engine.attention import MLP
    from acvram.engine.layers import QuantLinear
    from acvram.engine.moe import MoEBlock
    from acvram.quant.formats import PlainTensor

    def lin(w, b):
        w = w.contiguous()
        return QuantLinear(PlainTensor(w, tuple(w.shape), "bf16"), None if b is None else b.contiguous())
    ex = mlp_hf.experts
    experts = []
    for e in range(ex.gate_up_proj.shape[0]):
        w, b = ex.gate_up_proj[e].detach(), ex.gate_up_proj_bias[e].detach()
        experts.append(MLP(lin(w[:, ::2].T, b[::2]), lin(w[:, 1::2].T, b[1::2]),
                           lin(ex.down_proj[e].detach().T, ex.down_proj_bias[e].detach()), act="swiglu_oss"))
    r = mlp_hf.router
    return MoEBlock(lin(r.weight.detach(), r.bias.detach()), experts, top_k, norm_topk_prob=True)


def test_moe_gpt_oss_contre_transformers():
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP
    torch.manual_seed(2)
    cfg = _config_gpt_oss()
    hf = GptOssMLP(cfg).float()
    with torch.no_grad():
        for p in hf.parameters():
            p.normal_(0, 0.5)                      # biais et poids assez grands pour que la borne de 7 serve
    bloc = _bloc_moe_depuis_hf(hf, cfg.num_experts_per_tok)
    assert bloc.act == "swiglu_oss" and not bloc._try_build_stacks()
    for t in (1, 7):
        x = torch.randn(t, cfg.hidden_size)
        ref = hf(x[None])
        ref = (ref[0] if isinstance(ref, tuple) else ref)[0]
        # fp32 des deux côtés ; seul l'ordre des sommes diffère (boucle par expert accumulée en fp32 ici)
        torch.testing.assert_close(bloc(x), ref, rtol=1e-4, atol=1e-4)
    # témoin : le biais du routeur compte (sans lui, d'autres experts sont choisis et la sortie s'écarte)
    bloc.router.bias = torch.zeros_like(bloc.router.bias)
    bloc.__dict__.pop("_router_w", None)
    x = torch.randn(7, cfg.hidden_size)
    ref = hf(x[None])
    ref = (ref[0] if isinstance(ref, tuple) else ref)[0]
    assert (bloc(x) - ref).abs().max() > 1e-2


def test_swiglu_borne_contre_transformers_et_temoin():
    """Activation seule, au bit de `GptOssExperts._apply_gate` (même ordre d'opérations) ; témoin : sans la borne
    (limit infinie) la sortie diffère sur des portes > 7 — le test voit la borne."""
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts
    from acvram.engine import attention as A
    torch.manual_seed(3)
    ex = GptOssExperts(_config_gpt_oss())
    gu = torch.randn(5, 2 * 128) * 6
    ref = ex._apply_gate(gu)
    mlp = A.MLP(None, None, None, act="swiglu_oss")
    assert torch.equal(mlp._fusionner(gu[..., ::2], gu[..., 1::2]), ref)
    lim = A._OSS_LIMIT
    try:
        A._OSS_LIMIT = float("inf")
        assert not torch.equal(mlp._fusionner(gu[..., ::2], gu[..., 1::2]), ref)
    finally:
        A._OSS_LIMIT = lim


_SOURCE_20B = "/mnt/2TO_2023_980PRO/Modeles/sources_hf_temporaire/gpt-oss-20b-mxfp4"


def test_mxfp4_vers_nvfp4_exact_synthetique_et_refus():
    from acvram.quant.mxfp4 import mxfp4_dequant, mxfp4_vers_nvfp4
    from acvram.quant.nvfp4 import dequantize_nvfp4
    g = torch.Generator().manual_seed(4)
    blocs = torch.randint(0, 256, (6, 5, 16), generator=g, dtype=torch.uint8)
    ech = torch.randint(110, 128, (6, 5), generator=g, dtype=torch.uint8)
    ech[0, 0], ech[5, 4] = 110, 127                      # 17 octaves : la limite tenue
    t = mxfp4_vers_nvfp4(blocs, ech)
    assert torch.equal(dequantize_nvfp4(t, torch.float32), mxfp4_dequant(blocs, ech))
    ech[0, 0] = 109                                      # 18 octaves : refus nommé
    with pytest.raises(ValueError, match="exacte impossible"):
        mxfp4_vers_nvfp4(blocs, ech)


@pytest.mark.skipif(not __import__("os").path.isdir(_SOURCE_20B), reason="source gpt-oss-20b absente")
def test_mxfp4_vers_nvfp4_au_bit_de_transformers_sur_le_20b():
    """Couche 0 réelle, experts 0 et 31 : les poids NVFP4 écrits (gate, up désentrelacés ; down) déquantifiés == la
    déquantification de transformers (`convert_moe_packed_tensors`), au bit."""
    import json
    import os
    from safetensors import safe_open
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors
    from acvram.quant.mxfp4 import mxfp4_vers_nvfp4
    from acvram.quant.nvfp4 import dequantize_nvfp4
    ou = json.load(open(os.path.join(_SOURCE_20B, "model.safetensors.index.json")))["weight_map"]

    def lire(k):
        with safe_open(os.path.join(_SOURCE_20B, ou[k]), "pt") as f:
            return f.get_tensor(k)
    for proj in ("gate_up_proj", "down_proj"):
        b, s = lire(f"model.layers.0.mlp.experts.{proj}_blocks"), lire(f"model.layers.0.mlp.experts.{proj}_scales")
        for e in (0, 31):
            ref = convert_moe_packed_tensors(b[e:e + 1], s[e:e + 1], dtype=torch.float32)[0].T   # [out, in]
            lignes = ((slice(0, None, 2), slice(1, None, 2)) if proj == "gate_up_proj" else (slice(None),))
            for li in lignes:
                t = mxfp4_vers_nvfp4(b[e, li].contiguous(), s[e, li].contiguous())
                assert torch.equal(dequantize_nvfp4(t, torch.float32), ref[li]), (proj, e, li)


# ---------------------------------------------------------------------------------------------------------------------
# Bout en bout : point de contrôle gpt-oss JOUET en MXFP4 → convertisseur (passage direct) → chargeur → logits, contre
# transformers GptOssForCausalLM aux mêmes poids (MXFP4 déquantifié), fp32 des deux côtés, processeur.

def _jouet_mxfp4(d):
    import json
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(5)
    H, I, L, NH, NKV, HD, E, V = 128, 128, 2, 4, 2, 64, 4, 97
    cfg = {"architectures": ["GptOssForCausalLM"], "model_type": "gpt_oss", "hidden_size": H, "intermediate_size": I,
           "num_hidden_layers": L, "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": HD,
           "num_local_experts": E, "num_experts_per_tok": 2, "experts_per_token": 2, "vocab_size": V,
           "max_position_embeddings": 4096, "rms_norm_eps": 1e-5, "rope_theta": 150000.0, "attention_bias": True,
           "sliding_window": 8, "layer_types": ["sliding_attention", "full_attention"], "swiglu_limit": 7.0,
           "rope_scaling": {"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0,
                            "original_max_position_embeddings": 4096, "truncate": False},
           "tie_word_embeddings": False, "torch_dtype": "bfloat16", "eos_token_id": 96, "pad_token_id": 95,
           "quantization_config": {"quant_method": "mxfp4", "modules_to_not_convert": [
               "model.layers.*.self_attn", "model.layers.*.mlp.router", "model.embed_tokens", "lm_head"]}}
    json.dump(cfg, open(d / "config.json", "w"))

    def bf(*s, k=0.05):
        return (torch.randn(*s, generator=g) * k).to(torch.bfloat16)

    def mx(*s):
        return (torch.randint(0, 256, (*s, 16), generator=g, dtype=torch.uint8),
                torch.randint(118, 124, s, generator=g, dtype=torch.uint8))
    sd = {"model.embed_tokens.weight": bf(V, H, k=0.5), "model.norm.weight": bf(H, k=1.0) + 1,
          "lm_head.weight": bf(V, H, k=0.2)}
    for i in range(L):
        p = f"model.layers.{i}."
        for n, (o, i_) in {"q": (NH * HD, H), "k": (NKV * HD, H), "v": (NKV * HD, H), "o": (H, NH * HD)}.items():
            sd[p + f"self_attn.{n}_proj.weight"], sd[p + f"self_attn.{n}_proj.bias"] = bf(o, i_), bf(o)
        sd[p + "self_attn.sinks"] = bf(NH, k=1.0)
        sd[p + "mlp.router.weight"], sd[p + "mlp.router.bias"] = bf(E, H, k=0.3), bf(E, k=0.3)
        b, s = mx(E, 2 * I, H // 32)
        sd[p + "mlp.experts.gate_up_proj_blocks"], sd[p + "mlp.experts.gate_up_proj_scales"] = b, s
        sd[p + "mlp.experts.gate_up_proj_bias"] = bf(E, 2 * I, k=0.5)
        b, s = mx(E, H, I // 32)
        sd[p + "mlp.experts.down_proj_blocks"], sd[p + "mlp.experts.down_proj_scales"] = b, s
        sd[p + "mlp.experts.down_proj_bias"] = bf(E, H)
        sd[p + "input_layernorm.weight"] = bf(H, k=0.1) + 1
        sd[p + "post_attention_layernorm.weight"] = bf(H, k=0.1) + 1
    save_file(sd, str(d / "model.safetensors"))
    json.dump({"weight_map": {k: "model.safetensors" for k in sd}}, open(d / "model.safetensors.index.json", "w"))
    return cfg, sd


def _hf_depuis_jouet(cfg, sd):
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
    from acvram.quant.mxfp4 import mxfp4_dequant
    c = GptOssConfig(**{k: v for k, v in cfg.items() if k not in ("quantization_config", "architectures",
                                                                    "torch_dtype", "model_type")})
    c._attn_implementation = "eager"
    hf = GptOssForCausalLM(c).float().eval()
    etat = {}
    for k, v in sd.items():
        if k.endswith("_scales"):
            continue
        if k.endswith("_blocks"):
            base = k[: -len("_blocks")]
            E = v.shape[0]
            w = torch.stack([mxfp4_dequant(v[e], sd[base + "_scales"][e]) for e in range(E)])   # [E, out, in]
            etat[base] = w.transpose(1, 2).contiguous()                                        # x @ W
            continue
        etat[k] = v.float()
    manquants, inattendus = hf.load_state_dict(etat, strict=False)
    assert not inattendus and all("rotary" in k for k in manquants), (manquants, inattendus)
    return hf


@pytest.fixture(scope="module")
def jouet_oss(tmp_path_factory):
    from acvram.engine.config import load_model_spec
    from acvram.memory.tiering import LayerPlacement, Plan, Tier
    from acvram.quant.convert import ConversionOptions, convert_checkpoint
    d = tmp_path_factory.mktemp("gpt-oss-jouet")
    cfg, sd = _jouet_mxfp4(d)
    spec = load_model_spec(str(d), "gpt-oss-jouet")
    GIB = 2 ** 30
    tier = Tier(name="cpu-test", kind="host", device_index=-1, capacity=8 * GIB, weight_format="bf16",
                kv_format="bf16", read_bandwidth=50.0, link_bandwidth=20.0)
    couches = [LayerPlacement(index=i, exec_device="cpu-test", attn_storage="cpu-test", mlp_storage="cpu-test",
                              fmt="bf16", attn_bytes=1, mlp_bytes=1, mlp_active_bytes=0, is_moe=True)
               for i in range(spec.num_layers)]
    kv = 2 * spec.num_layers * spec.num_key_value_heads * spec.head_dim * 2
    plan = Plan(model="gpt-oss-jouet", tiers=[tier], layers=couches, embed_device="cpu-test",
                lm_head_device="cpu-test", kv_bytes_per_token=kv, kv_budget={"cpu-test": kv * 4096})
    out = str(tmp_path_factory.mktemp("acvram-oss"))
    # autoriser_grossissement : sur un jouet de 0,7 Mo, l'en-tête safetensors (77 tenseurs) fait à lui seul +2 %
    convert_checkpoint(str(d), plan, ConversionOptions(out_dir=out, passage_direct=True, autoriser_grossissement=True),
                       spec=spec)
    return cfg, sd, out


def _logits_oss(model, prompt):
    from acvram.engine.model import ForwardBatch
    from acvram.memory.kvcache import BLOCK_SIZE, BlockAllocator
    n = len(prompt)
    alloc = BlockAllocator(model.caches[0].cfg.num_blocks)
    blocks = alloc.allocate((n + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)
    slots = torch.tensor([blocks[i // BLOCK_SIZE] * BLOCK_SIZE + i % BLOCK_SIZE for i in range(n)])
    batch = ForwardBatch(torch.tensor(prompt), torch.arange(n), [n], [n], [torch.tensor(blocks)], slots, True)
    with torch.no_grad():
        return model(batch)[0].float()


def test_jouet_converti_formats_et_chargement(jouet_oss):
    import json
    import os
    from acvram.engine.attention import Attention
    from acvram.engine.loader import load_model
    _, _, out = jouet_oss
    m = json.load(open(os.path.join(out, "acvram_manifest.json")))["tensors"]
    assert m["model.layers.0.mlp.experts.3.gate_proj.weight"]["format"] == "nvfp4"
    assert m["model.layers.0.mlp.experts.3.gate_proj.weight"].get("passage_direct") is True
    assert "model.layers.1.mlp.gate.bias" in m and "model.layers.1.self_attn.sinks" in m
    assert m["model.layers.1.self_attn.sinks"]["format"] == "bf16"
    loaded = load_model(out, dtype=torch.float32, device_override="cpu")
    attn = [x for x in loaded.model.modules() if isinstance(x, Attention)]
    assert [a.window for a in attn] == [8, 0] and all(a.sinks is not None for a in attn)


@pytest.mark.parametrize("n", [5, 20])
def test_jouet_logits_contre_transformers(jouet_oss, n):
    """n = 20 > fenêtre 8 : la couche glissante et le puits comptent. Témoin : sans puits, les logits s'écartent."""
    from acvram.engine.attention import Attention
    from acvram.engine.loader import load_model
    cfg, sd, out = jouet_oss
    torch.manual_seed(n)
    prompt = torch.randint(0, 95, (n,)).tolist()
    hf = _hf_depuis_jouet(cfg, sd)
    with torch.no_grad():
        ref = hf(input_ids=torch.tensor([prompt]), use_cache=False).logits[0, -1].float()
    loaded = load_model(out, dtype=torch.float32, device_override="cpu")
    y = _logits_oss(loaded.model, prompt)
    ecart = float((y - ref).abs().max() / ref.abs().max())
    assert ecart < 1e-4, ecart
    for a in loaded.model.modules():
        if isinstance(a, Attention):
            a.sinks = None
    assert float((_logits_oss(loaded.model, prompt) - ref).abs().max() / ref.abs().max()) > 1e-3


def test_jouet_pas_de_decodage_contre_transformers(jouet_oss):
    """Préfill de 19 jetons puis UN pas de décodage par le cache paginé (chemin `_decode` → puits à formes fixes) :
    logits du 20e jeton == transformers sur les 20 jetons."""
    from acvram.engine.loader import load_model
    from acvram.engine.model import ForwardBatch
    from acvram.memory.kvcache import BLOCK_SIZE, BlockAllocator
    cfg, sd, out = jouet_oss
    torch.manual_seed(7)
    prompt = torch.randint(0, 95, (20,)).tolist()
    hf = _hf_depuis_jouet(cfg, sd)
    with torch.no_grad():
        ref = hf(input_ids=torch.tensor([prompt]), use_cache=False).logits[0, -1].float()
    model = load_model(out, dtype=torch.float32, device_override="cpu").model
    alloc = BlockAllocator(model.caches[0].cfg.num_blocks)
    blocks = alloc.allocate(3)
    slot = lambda i: blocks[i // BLOCK_SIZE] * BLOCK_SIZE + i % BLOCK_SIZE          # noqa: E731
    n = 19
    with torch.no_grad():
        model(ForwardBatch(torch.tensor(prompt[:n]), torch.arange(n), [n], [n], [torch.tensor(blocks)],
                           torch.tensor([slot(i) for i in range(n)]), True))
        y = model(ForwardBatch(torch.tensor(prompt[n:]), torch.tensor([n]), [n + 1], [1], [torch.tensor(blocks)],
                               torch.tensor([slot(n)]), False))[0].float()
    ecart = float((y.reshape(-1) - ref).abs().max() / ref.abs().max())
    # le cache du plan jouet est bf16 : le pas de décodage relit K/V arrondis (le préfill garde les siens en fp32) —
    # mesuré 8,8e-4 ; sans puits 0,129 (le témoin d'en dessous)
    assert model.caches[0].cfg.dtype == "bf16"
    from acvram.engine.attention import Attention
    for a in model.modules():
        if isinstance(a, Attention):
            a.sinks = None
    with torch.no_grad():
        y0 = model(ForwardBatch(torch.tensor(prompt[n:]), torch.tensor([n]), [n + 1], [1], [torch.tensor(blocks)],
                                torch.tensor([slot(n)]), False))[0].float()
    assert ecart < 2e-3, ecart
    assert float((y0.reshape(-1) - ref).abs().max() / ref.abs().max()) > 5e-2
