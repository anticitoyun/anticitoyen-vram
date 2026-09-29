"""cqy (29/09, 0.7.15) : masque dense découpé D'EMBLÉE au-delà de MASQUE_OCTETS_MAX, blocs de taille CHOISIE PAR LE CODE
(≥ PLANCHER_LIGNES = 1 024). Contrôles : au bit contre le seul tenant (témoin MASQUE_OCTETS_MAX=0) — processeur ici, carte
sous carte.sh (grille bf16/fp32, GQA, q_offset 0 et > 0, q de 1 à 34 k) ; plancher refusé en dessous ; découpage vraiment fait."""
import os
import subprocess
import sys

import pytest
import torch

from acvram.engine import layers


class _Espion:
    def __init__(self):
        self.q = []

    def __getattr__(self, nom):
        return getattr(torch.nn.functional, nom)

    def scaled_dot_product_attention(self, qh, *a, **kw):
        self.q.append(qh.shape[2])
        return torch.nn.functional.scaled_dot_product_attention(qh, *a, **kw)


def _compare(monkeypatch, q_offset, q_len, dt, dev, n_rep, window=0, heads=8, dim=64):
    kv_len = q_offset + q_len
    torch.manual_seed(q_len + q_offset)
    q = torch.randn(q_len, heads, dim, dtype=dt, device=dev)
    k = torch.randn(kv_len, heads // n_rep, dim, dtype=dt, device=dev)
    v = torch.randn(kv_len, heads // n_rep, dim, dtype=dt, device=dev)
    monkeypatch.setattr(layers, "_MASQUE_OCTETS_MAX", 0)
    seul = layers.attention(q, k, v, True, 0.125, q_offset, window, n_rep)
    monkeypatch.undo()
    espion = _Espion()
    monkeypatch.setattr(layers, "F", espion)
    blocs = layers.attention(q, k, v, True, 0.125, q_offset, window, n_rep)
    monkeypatch.undo()
    dense = window > 0 or (q_len > 1 and q_offset > 0)
    attendu = (dense and q_len * kv_len * q.element_size() > layers._MASQUE_OCTETS_MAX
               and q_len > layers._lignes_par_bloc(kv_len, q.element_size()))
    assert (len(espion.q) > 1) == attendu, (espion.q, attendu)          # découpé exactement quand le seuil l exige
    assert all(n >= layers.PLANCHER_LIGNES for n in espion.q[:-1]), espion.q
    return seul, blocs


@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("q_offset,q_len,window", [(16, 2500, 0), (1024, 3000, 0), (0, 2600, 64), (5, 33, 0)])
def test_au_bit_processeur_seuil_abaisse(monkeypatch, dt, q_offset, q_len, window):
    """Processeur : seuil abaissé à 1 Mio pour découper des formes de quelques milliers (blocs choisis par le code).
    bf16 au bit ; fp32 à ≤ 1e-6 (le noyau CPU tuile selon q). L équivalence qui décide est celle de la carte."""
    monkeypatch.setattr(layers, "_MASQUE_OCTETS_MAX", 1 << 20)
    seuil = layers._MASQUE_OCTETS_MAX
    kv_len = q_offset + q_len
    torch.manual_seed(q_len)
    q = torch.randn(q_len, 2, 16, dtype=dt); k = torch.randn(kv_len, 2, 16, dtype=dt); v = torch.randn(kv_len, 2, 16, dtype=dt)
    monkeypatch.setattr(layers, "_MASQUE_OCTETS_MAX", 0)
    seul = layers.attention(q, k, v, True, 0.125, q_offset, window)
    monkeypatch.setattr(layers, "_MASQUE_OCTETS_MAX", seuil)
    espion = _Espion(); monkeypatch.setattr(layers, "F", espion)
    blocs = layers.attention(q, k, v, True, 0.125, q_offset, window)
    if q_len > 1024:
        assert len(espion.q) > 1 and espion.q[0] == 1024, espion.q
    if dt == torch.bfloat16:
        assert torch.equal(seul, blocs), (seul.float() - blocs.float()).abs().max()
    else:                        # fp32 sur processeur : tuilage du noyau CPU selon q (3,7e-9 mesuré) ; la carte est au bit
        assert torch.allclose(seul, blocs, rtol=0, atol=1e-6), (seul - blocs).abs().max()


GRILLE = [(q_offset, q_len) for q_offset in (0, 16, 1024) for q_len in (1, 33, 1000, 4096, 12288, 34558)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grille au bit sur carte : sous carte.sh")
@pytest.mark.parametrize("n_rep", [1, 4])
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("q_offset,q_len", GRILLE)
def test_au_bit_carte_grille(monkeypatch, q_offset, q_len, dt, n_rep):
    window = 256 if q_offset == 0 else 0            # q_offset 0 : masque dense par la fenêtre glissante
    seul, blocs = _compare(monkeypatch, q_offset, q_len, dt, "cuda", n_rep, window)
    assert torch.equal(seul, blocs), (seul.float() - blocs.float()).abs().max()
    del seul, blocs
    torch.cuda.empty_cache()


def test_le_plancher_ne_baisse_pas():
    """7 lignes par bloc n étaient pas au bit sur carte (2/20) ; 1 024 l étaient (10/10). Baisser le plancher casse ici."""
    assert layers.PLANCHER_LIGNES >= 1024
    assert layers._lignes_par_bloc(10_000_000, 2) >= 1024          # kv énorme : jamais sous le plancher


def test_plancher_refuse_en_dessous(monkeypatch):
    monkeypatch.setattr(layers, "_MASQUE_LIGNES_MIN", 512)
    with pytest.raises(ValueError, match="plancher de 1024 lignes"):
        layers._lignes_par_bloc(4096, 2)
    env = {**os.environ, "ACVRAM_MASQUE_LIGNES_MIN": "512", "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, "-c", "import acvram.engine.layers"], env=env, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode != 0 and "ACVRAM_MASQUE_LIGNES_MIN=512 refusé" in r.stderr, r.stderr[-400:]
