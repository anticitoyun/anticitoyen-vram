"""C5-b : le budget KV « par jeton » du planificateur porte le surcoût du format
par canal (sc E4M3 [HKV, D] + tampon_de par bloc), à l'unité près de
`KVCacheConfig.bytes_per_block` — sinon `_kv_blocks_per_device` rend 3,1 % de
blocs en moins que les jetons planifiés (Coder b=12 : 23 824 < 24 576, une
séquence tronquée, Manon 20/09)."""
from __future__ import annotations

import pytest

from acvram.engine.config import ModelSpec
from acvram.memory import kv_canal
from acvram.memory.kvcache import BLOCK_SIZE, KVCacheConfig


def _spec():
    return ModelSpec(name="t", architecture="Qwen3MoeForCausalLM", hidden_size=2048, intermediate_size=6144,
                     num_layers=48, num_attention_heads=32, num_key_value_heads=4, vocab_size=151936,
                     max_position_embeddings=262144, rms_norm_eps=1e-6, rope_theta=1e7, rope_scaling=None,
                     head_dim=128)


@pytest.mark.parametrize("canal", [False, True])
def test_octets_par_jeton_egalent_le_bloc_du_format(monkeypatch, canal):
    monkeypatch.setattr(kv_canal, "ACTIF", canal)
    spec = _spec()
    bloc = KVCacheConfig(num_layers=1, num_kv_heads=4, head_dim=128, num_blocks=1, dtype="int8", canal=canal).bytes_per_block()
    par_couche = spec.kv_bytes_per_token(8) / spec.couches_avec_kv
    assert par_couche * BLOCK_SIZE == pytest.approx(bloc, abs=1)
    assert bloc == (17156 if canal else 16640)


def test_le_surcout_canal_est_celui_du_bloc(monkeypatch):
    spec = _spec()
    monkeypatch.setattr(kv_canal, "ACTIF", False); sans = spec.kv_bytes_per_token(8)
    monkeypatch.setattr(kv_canal, "ACTIF", True); avec = spec.kv_bytes_per_token(8)
    assert avec > sans and (avec - sans) * BLOCK_SIZE // spec.couches_avec_kv == 4 * 128 + 4   # témoin : sans le terme, égalité
