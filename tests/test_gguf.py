"""Le lecteur GGUF : en-tête, métadonnées, déquantification.

Les types simples (F32, F16, Q8_0) sont vérifiés au bit près contre un fichier
fabriqué ici même. Les dispositions K-quants, trop retorses pour être
réécrites deux fois sans faute, sont couvertes par la conversion réelle d'un
modèle Q4_K_M — un déquantiseur faux y produit du charabia immédiat.
"""

import json
import struct

import numpy as np
import pytest
import torch

from acvram.quant.gguf import GGUFFile, is_gguf


def _w(fh, fmt, *vals):
    fh.write(struct.pack("<" + fmt, *vals))


def _wstr(fh, s):
    b = s.encode()
    _w(fh, "Q", len(b))
    fh.write(b)


def _make_gguf(path, tensors, kv):
    """Écrit un GGUF v3 minimal : kv de types u32/f32/str, tenseurs F32/F16/Q8_0."""
    with open(path, "wb") as fh:
        fh.write(b"GGUF")
        _w(fh, "I", 3)
        _w(fh, "QQ", len(tensors), len(kv))
        for k, v in kv.items():
            _wstr(fh, k)
            if isinstance(v, str):
                _w(fh, "I", 8); _wstr(fh, v)
            elif isinstance(v, float):
                _w(fh, "I", 6); _w(fh, "f", v)
            else:
                _w(fh, "I", 4); _w(fh, "I", v)
        blobs, offset = [], 0
        for name, (arr, ttype) in tensors.items():
            if ttype == 0:
                raw = arr.astype(np.float32).tobytes()
            elif ttype == 1:
                raw = arr.astype(np.float16).tobytes()
            else:                                     # Q8_0
                a = arr.astype(np.float32).reshape(-1, 32)
                d = np.abs(a).max(1, keepdims=True) / 127.0
                d[d == 0] = 1e-8
                q = np.round(a / d).clip(-127, 127).astype(np.int8)
                raw = b"".join(np.float16(d[i, 0]).tobytes() + q[i].tobytes()
                               for i in range(a.shape[0]))
            _wstr(fh, name)
            _w(fh, "I", arr.ndim)
            for dim in reversed(arr.shape):
                _w(fh, "Q", dim)
            _w(fh, "IQ", ttype, offset)
            blobs.append(raw)
            offset += (len(raw) + 31) // 32 * 32
        pad = (fh.tell() + 31) // 32 * 32 - fh.tell()
        fh.write(b"\0" * pad)
        for raw in blobs:
            fh.write(raw)
            fh.write(b"\0" * ((len(raw) + 31) // 32 * 32 - len(raw)))


@pytest.fixture()
def small_gguf(tmp_path):
    rng = np.random.default_rng(7)
    tensors = {
        "token_embd.weight": (rng.standard_normal((64, 16)).astype(np.float32), 0),
        "blk.0.attn_q.weight": (rng.standard_normal((16, 16)).astype(np.float32), 1),
        "blk.0.ffn_up.weight": (rng.standard_normal((32, 16)).astype(np.float32), 8),
        "output_norm.weight": (np.ones(16, np.float32), 0),
    }
    p = tmp_path / "petit.gguf"
    _make_gguf(str(p), tensors, {
        "general.architecture": "qwen3",
        "qwen3.block_count": 1, "qwen3.embedding_length": 16,
        "qwen3.feed_forward_length": 32, "qwen3.attention.head_count": 4,
        "qwen3.attention.head_count_kv": 2, "qwen3.context_length": 128,
        "qwen3.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen3.rope.freq_base": 1000000.0, "qwen3.vocab_size": 64,
        "tokenizer.ggml.eos_token_id": 2,
    })
    return str(p), tensors


def test_header_and_config(small_gguf):
    path, _ = small_gguf
    assert is_gguf(path)
    g = GGUFFile(path)
    cfg = g.hf_config()
    assert cfg["architectures"] == ["Qwen3ForCausalLM"]
    assert cfg["hidden_size"] == 16 and cfg["num_hidden_layers"] == 1
    assert cfg["num_key_value_heads"] == 2
    assert cfg["eos_token_id"] == 2
    assert cfg["rope_theta"] == 1000000.0
    assert cfg["tie_word_embeddings"] is True      # pas d'output.weight


def test_tensors_roundtrip(small_gguf):
    path, tensors = small_gguf
    g = GGUFFile(path)
    got = dict(g.iter_tensors())
    ref = torch.from_numpy(tensors["token_embd.weight"][0])
    assert torch.equal(got["model.embed_tokens.weight"], ref)
    ref16 = torch.from_numpy(
        tensors["blk.0.attn_q.weight"][0].astype(np.float16).astype(np.float32))
    assert torch.equal(got["model.layers.0.self_attn.q_proj.weight"], ref16)
    # Q8_0 : la reconstruction doit rester dans le pas de quantification.
    q8 = got["model.layers.0.mlp.up_proj.weight"]
    ref8 = torch.from_numpy(tensors["blk.0.ffn_up.weight"][0])
    step = ref8.abs().amax(dim=-1, keepdim=True) / 127.0
    assert (q8 - ref8).abs().max() <= step.max() * 0.5001


def test_from_model_spec(small_gguf):
    path, _ = small_gguf
    from acvram.engine.config import load_model_spec
    spec = load_model_spec(path)
    assert spec.num_layers == 1
    assert spec.eos_token_id == [2]
