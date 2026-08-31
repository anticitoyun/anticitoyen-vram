"""La normalisation par tête de Q et K, telle que Qwen3 l'exige.

Sans elle le moteur charge le modèle sans se plaindre et rend du charabia : les
poids `q_norm` et `k_norm` sont simplement ignorés. Ces tests fixent leur
présence, leur place dans l'ordre des opérations, et le fait que les modèles qui
n'en ont pas restent inchangés.
"""

import math

import torch

from acvram.engine.layers import RMSNorm, RotaryEmbedding, apply_rope
from acvram.engine.loader import load_model


def _layers(model):
    return getattr(model, "layers", None) or model.model.layers


def test_qk_norms_are_loaded(converted_qknorm):
    loaded = load_model(converted_qknorm, dtype=torch.float32, device_override="cpu")
    for layer in _layers(loaded.model):
        assert layer.self_attn.q_norm is not None, "q_norm perdue au chargement"
        assert layer.self_attn.k_norm is not None, "k_norm perdue au chargement"
        assert layer.self_attn.q_norm.weight.shape[-1] == layer.self_attn.head_dim


def test_model_without_qk_norm_stays_none(converted):
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    for layer in _layers(loaded.model):
        assert layer.self_attn.q_norm is None
        assert layer.self_attn.k_norm is None


def test_qk_norm_changes_the_output(converted_qknorm):
    """Le débrancher doit changer la réponse — sinon le test ne prouve rien."""
    from acvram.engine.runner import Engine, SamplingParams

    prompt = [5, 9, 2, 7, 1, 8, 3, 6]

    def run(loaded):
        e = Engine(loaded, None, max_batch_size=1, max_model_len=128)
        return [t for o in e.generate(prompt, SamplingParams(temperature=0.0,
                                                             max_tokens=8))
                for t in o.token_ids]

    with_norm = load_model(converted_qknorm, dtype=torch.float32,
                           device_override="cpu")
    a = run(with_norm)
    for layer in _layers(with_norm.model):
        layer.self_attn.q_norm = layer.self_attn.k_norm = None
    b = run(with_norm)
    assert a != b, "la normalisation de Q et K n'a eu aucun effet"


def test_norm_is_applied_before_rope():
    """L'ordre importe : normaliser après la rotation écraserait la norme que
    la RoPE vient de faire tourner, et ce n'est pas ce que le modèle a appris.
    """
    torch.manual_seed(0)
    hd, n, t = 16, 2, 5
    q = torch.randn(t, n, hd)
    k = torch.randn(t, n, hd)
    norm = RMSNorm(1 + torch.randn(hd) * 0.2, 1e-6)
    rope = RotaryEmbedding(hd, 64, 10000.0)
    cos, sin = rope(torch.arange(t), torch.device("cpu"), torch.float32)

    avant = apply_rope(norm(q), norm(k), cos, sin)
    qr, kr = apply_rope(q, k, cos, sin)
    apres = (norm(qr), norm(kr))
    assert not torch.allclose(avant[0], apres[0], atol=1e-4)


def test_eos_ids_survive_the_conversion(tmp_path_factory, converted):
    """Un modèle converti doit encore savoir où s'arrêter.

    Le manifeste ne recopiait ni ``eos_token_id`` ni le
    ``generation_config.json`` : le moteur ne s'arrêtait jamais de lui-même et
    rendait toujours ``max_tokens`` jetons, en repartant en roue libre après la
    réponse. Le symptôme, sur Qwen2.5, était un « Human: » qui poursuivait le
    dialogue tout seul.
    """
    import json
    import os

    from acvram.engine.runner import Engine

    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    d = loaded.path
    with open(os.path.join(d, "generation_config.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"eos_token_id": [151645, 151643]}, fh)

    e = Engine(loaded, None, max_batch_size=1, max_model_len=128)
    assert {151645, 151643} <= e._eos, "jetons d'arret perdus a la conversion"


def test_eos_from_generation_config_reaches_the_spec(tmp_path_factory):
    """``generation_config.json`` fait autorité sur ``config.json``."""
    import json

    from acvram.engine.config import load_model_spec

    d = tmp_path_factory.mktemp("gen")
    json.dump({"architectures": ["Qwen2ForCausalLM"], "hidden_size": 64,
               "intermediate_size": 128, "num_hidden_layers": 2,
               "num_attention_heads": 4, "num_key_value_heads": 2,
               "vocab_size": 256, "eos_token_id": 151643},
              open(d / "config.json", "w"))
    json.dump({"eos_token_id": [151645, 151643]},
              open(d / "generation_config.json", "w"))
    spec = load_model_spec(str(d))
    assert spec.eos_token_id == [151643, 151645]


def test_int8_kernels_match_reference():
    """Le GEMV et la déquantification INT8 fusionnés reproduisent la référence.

    Comme pour INT4 et NVFP4 : ``quant/*.py`` est la spécification, le noyau
    doit s'y conformer. Sans matériel CUDA le test vérifie le repli, qui est la
    référence elle-même — il reste utile comme test d'interface.
    """
    import pytest

    from acvram import kernels
    from acvram.quant.formats import quantize, dequantize

    torch.manual_seed(7)
    w = torch.randn(384, 256) * 0.05
    x = torch.randn(3, 256)
    t = quantize(w, "int8", group_size=128)
    ref_w = dequantize(t, torch.float32)
    ref_y = x @ ref_w.t()

    y = kernels.int8_matmul(x, t)
    assert torch.allclose(y.to(torch.float32), ref_y, atol=2e-2, rtol=1e-2)

    if not torch.cuda.is_available():
        pytest.skip("pas de CUDA : chemins fusionnés non exerçables")
    for i in range(torch.cuda.device_count()):
        tc = t.to(f"cuda:{i}")
        xc = x.to(f"cuda:{i}")
        yd = kernels.int8_dequant(tc, torch.float32)
        assert torch.allclose(yd.cpu(), ref_w, atol=1e-3), f"dequant cuda:{i}"
        yg = kernels.int8_matmul(xc[:1], tc)      # n=1 : chemin GEMV fusionné
        assert torch.allclose(yg.cpu().to(torch.float32), ref_y[:1],
                              atol=2e-2, rtol=1e-2), f"gemv cuda:{i}"
