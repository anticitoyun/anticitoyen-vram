"""Pièce 235 (poste2, à sec) : `RotaryEmbedding._ensure` (engine/layers.py:721) ne revérifie que
`seq_len` et `device`, jamais `dtype` -- le même piège que la 213 (poste5) pour les caches RoPE du
service. Ici côté calibration : `quant/collect.py:257` construit son `RotaryEmbedding` sans passer
`dtype` (défaut `torch.float32`, `layers.py:657`), et `tables32()` (utilisée par le noyau RoPE fusionné,
`rope_fusee`, appelé EN PREMIER dans `attention.py` avant le repli `forward(dtype=x.dtype)`) lit
`self._dtype` -- gelé au premier `_ensure()` qui passe, quel que soit le dtype demandé ensuite.

Falsificateur : appeler `forward()` en bf16 PUIS `tables32()` (fp32 demandé) sur le même objet, au même
seq_len (donc pas de rebuild par la voie normale) -- `tables32()` doit rendre des tables IDENTIQUES à un
calcul fp32 frais, pas un requantage lossy du cache bf16 déjà arrondi. Sur le code d'aujourd'hui
(seule condition de rebuild : seq_len/device), ce test est ROUGE."""
import torch

from acvram.engine.layers import RotaryEmbedding


def _rope(dtype=torch.float32):
    return RotaryEmbedding(64, 128, 10000.0, None, dtype=dtype)


def test_tables32_apres_forward_bf16_rend_un_calcul_fp32_frais_235():
    device = torch.device("cpu")
    positions = torch.arange(8, dtype=torch.long)

    rope = _rope()
    # 1er appelant : le repli PyTorch, en bf16 (dtype des activations de calibration/service)
    cos_bf16, sin_bf16 = rope(positions, device, torch.bfloat16, max_pos=64)
    assert cos_bf16.dtype == torch.bfloat16

    # 2e appelant, MÊME objet, seq_len ≤ cache : le noyau fusionné demande ses tables fp32
    cos32, sin32 = rope.tables32(64, device)
    assert cos32.dtype == torch.float32

    # référence : un objet neuf, jamais touché en bf16, calcul fp32 direct
    ref = _rope()
    ref._ensure(64, device, torch.float32)

    assert torch.equal(cos32, ref._cos), (
        "tables32() a rendu un requantage du cache bf16 déjà arrondi, pas un calcul fp32 frais -- "
        "le dtype du premier appelant a gelé la précision du second (motif de la 213)")
    assert torch.equal(sin32, ref._sin)


def test_forward_apres_tables32_rend_bien_le_dtype_demande_235():
    """Sens inverse : tables32 (fp32) en premier, puis forward(bf16) au même seq_len -- forward() ne
    doit pas rendre du fp32 sous couvert de bf16 (silencieux, aucune erreur de type levée par torch
    lors d'un matmul fp32×bf16, juste un upcast qui masque le défaut)."""
    device = torch.device("cpu")
    positions = torch.arange(8, dtype=torch.long)

    rope = _rope()
    rope.tables32(64, device)  # 1er appelant : self._dtype (fp32 par défaut, ici explicite)

    cos_bf16, sin_bf16 = rope(positions, device, torch.bfloat16, max_pos=64)
    assert cos_bf16.dtype == torch.bfloat16, (
        f"forward(dtype=bf16) a rendu {cos_bf16.dtype} -- le cache est resté gelé au premier "
        "appelant (tables32, fp32) au lieu d'honorer le dtype demandé")
