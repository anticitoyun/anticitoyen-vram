"""Scaling « llama 4 » de ministral3/Devstral porté sur q (model.Attention
._echelle_llama4, après le RoPE) : q ← q · (1 + β·ln(1 + ⌊pos/plafond⌋)),
plafond = original_max_position_embeddings (8 192 sur Devstral), β = 0,1.

Équivalence contre transformers `Ministral3ForCausalLM` (mêmes poids : le
point de contrôle jouet de forme llama, dtype fp32 des deux côtés, sans
quantification : plan bf16 → tenseurs plats) sur les logits du dernier jeton
d'une séquence de 48 jetons placée aux positions 9 000-9 047 (⌊pos/8192⌋ = 1,
échelle 1 + 0,1·ln 2) et 16 384-16 431 (⌊·⌋ = 2, échelle 1 + 0,1·ln 3), et un
témoin sous le plafond (positions 0-47, échelle 1). Bras cassant : β = 0 côté
acvram (Attention.llama4 = None) → les logits au-delà du plafond diffèrent de
transformers (rouge) et coïncident sous le plafond.

Le refus « llama_4_scaling_beta non servi » (67aa280) tombe : Engine à
max_model_len = 16 384 se construit et le régime dit « (servi) »."""
import json
import os
import shutil

import pytest
import torch

from acvram.engine.config import load_model_spec
from acvram.engine.loader import load_model
from acvram.engine.model import Attention, ForwardBatch
from acvram.memory.tiering import LayerPlacement, Plan, Tier
from acvram.quant.convert import ConversionOptions, convert_checkpoint

pytest.importorskip("transformers")
GIB = 2 ** 30

ROPE_PARAMETERS = {"type": "yarn", "rope_type": "yarn", "rope_theta": 10000.0, "factor": 48.0,
                   "original_max_position_embeddings": 8192, "beta_fast": 32.0, "beta_slow": 1.0,
                   "mscale": 1.0, "mscale_all_dim": 1.0, "llama_4_scaling_beta": 0.1}
N = 48
BASES = {"sous_plafond": 0, "9000": 9000, "16384": 16384}


def _checkpoint_ministral3(tiny_checkpoint, tmp_path):
    """Le point de contrôle jouet, config Mistral3/ministral3 (text_config,
    rope_parameters yarn + llama_4_scaling_beta), poids inchangés."""
    src = json.load(open(os.path.join(tiny_checkpoint, "config.json")))
    d = tmp_path / "devstral-jouet"
    d.mkdir()
    shutil.copy(os.path.join(tiny_checkpoint, "model.safetensors"), d / "model.safetensors")
    texte = {k: v for k, v in src.items() if k != "architectures"}
    texte.update({"model_type": "ministral3", "max_position_embeddings": 32768,
                  "rope_parameters": ROPE_PARAMETERS, "head_dim": src["hidden_size"] // src["num_attention_heads"]})
    texte.pop("rope_theta", None)
    json.dump({"architectures": ["Mistral3ForConditionalGeneration"], "model_type": "mistral3",
               "text_config": texte, "torch_dtype": "bfloat16"}, open(d / "config.json", "w"))
    return str(d)


def _convertir_bf16(chemin, tmp_path):
    spec = load_model_spec(chemin, "devstral-jouet")
    tier = Tier(name="cpu-test", kind="host", device_index=-1, capacity=8 * GIB, weight_format="bf16",
                kv_format="bf16", read_bandwidth=50.0, link_bandwidth=20.0)
    couches = [LayerPlacement(index=i, exec_device="cpu-test", attn_storage="cpu-test", mlp_storage="cpu-test",
                              fmt="bf16", attn_bytes=1, mlp_bytes=1, mlp_active_bytes=0, is_moe=False)
               for i in range(spec.num_layers)]
    kv_par_jeton = 2 * spec.num_layers * spec.num_key_value_heads * spec.head_dim * 2     # bf16 : K et V
    plan = Plan(model="devstral-jouet", tiers=[tier], layers=couches, embed_device="cpu-test",
                lm_head_device="cpu-test", kv_bytes_per_token=kv_par_jeton,
                kv_budget={"cpu-test": kv_par_jeton * 4096})
    out = str(tmp_path / "acvram-bf16")
    convert_checkpoint(chemin, plan, ConversionOptions(out_dir=out), spec=spec)
    return out


def _logits_acvram(model, prompt, base):
    from acvram.memory.kvcache import BLOCK_SIZE, BlockAllocator
    n = len(prompt)
    alloc = BlockAllocator(model.caches[0].cfg.num_blocks)
    blocks = alloc.allocate((n + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)
    slots = torch.tensor([blocks[i // BLOCK_SIZE] * BLOCK_SIZE + i % BLOCK_SIZE for i in range(n)])
    for m in model.modules():
        if isinstance(m, Attention) and m.rope is not None:
            m.rope._ensure(base + n + 1, torch.device("cpu"), torch.float32)   # tables RoPE jusqu'à la position
    batch = ForwardBatch(torch.tensor(prompt), torch.arange(base, base + n), [n], [n],
                         [torch.tensor(blocks)], slots, True)
    with torch.no_grad():
        return model(batch)[0].float()


def _logits_hf(chemin, prompt, base):
    from safetensors.torch import load_file
    from transformers import Ministral3Config, Ministral3ForCausalLM
    cfg = json.load(open(os.path.join(chemin, "config.json")))["text_config"]
    conf = Ministral3Config(**{k: v for k, v in cfg.items() if k not in ("torch_dtype",)})
    conf._attn_implementation = "eager"
    hf = Ministral3ForCausalLM(conf).float().eval()
    sd = {k: v.float() for k, v in load_file(os.path.join(chemin, "model.safetensors")).items()}
    manquants, inattendus = hf.load_state_dict(sd, strict=False)
    assert not inattendus and all("rotary" in k for k in manquants), (manquants, inattendus)
    ids = torch.tensor([prompt])
    pos = torch.arange(base, base + len(prompt)).unsqueeze(0)
    with torch.no_grad():
        return hf(input_ids=ids, position_ids=pos, use_cache=False).logits[0, -1].float()


@pytest.fixture(scope="module")
def jouet(tiny_checkpoint, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("llama4")
    chemin = _checkpoint_ministral3(tiny_checkpoint, tmp)
    return chemin, _convertir_bf16(chemin, tmp)


def _ecart(a, b):
    return float((a - b).abs().max() / b.abs().max())


def test_l_attention_porte_le_scaling(jouet):
    _, conv = jouet
    loaded = load_model(conv, dtype=torch.float32, device_override="cpu")
    attn = [m for m in loaded.model.modules() if isinstance(m, Attention)]
    assert attn and all(m.llama4 == (0.1, 8192) for m in attn)
    assert loaded.spec.rope_scaling["llama_4_scaling_beta"] == 0.1


@pytest.mark.parametrize("nom", list(BASES))
def test_logits_egaux_a_transformers_au_dela_du_plafond(jouet, nom):
    chemin, conv = jouet
    base = BASES[nom]
    torch.manual_seed(base + 1)
    prompt = torch.randint(0, 1024, (N,)).tolist()
    ref = _logits_hf(chemin, prompt, base)
    loaded = load_model(conv, dtype=torch.float32, device_override="cpu")
    y = _logits_acvram(loaded.model, prompt, base)
    assert _ecart(y, ref) < 1e-4, _ecart(y, ref)                   # mesuré 1e-6 (bruit fp32)
    # bras cassant : sans le scaling (β = 0), l'écart au-delà du plafond doit
    # apparaître — et disparaître sous le plafond, où l'échelle vaut 1
    for m in loaded.model.modules():
        if isinstance(m, Attention):
            m.llama4 = None
    y0 = _logits_acvram(loaded.model, prompt, base)
    if base >= 8192:
        assert _ecart(y0, ref) > 1e-3, _ecart(y0, ref)             # mesuré 3,6e-3 (9 000) et 4,9e-3 (16 384)
    else:
        assert torch.equal(y0, y)


def test_le_moteur_ne_refuse_plus_au_dela_du_plafond_et_le_dit(jouet, target_rig):
    from unittest.mock import patch
    from acvram.engine.runner import Engine
    _, conv = jouet
    loaded = load_model(conv, max_model_len=16384, device_override="cpu")
    with patch("acvram.hardware.detect.detect_rig", return_value=target_rig):
        engine = Engine(loaded, None, max_batch_size=1, max_model_len=16384, enable_cuda_graphs=False)
    assert "llama4_scaling_beta=0.1(servi)" in engine.regime_ligne()
