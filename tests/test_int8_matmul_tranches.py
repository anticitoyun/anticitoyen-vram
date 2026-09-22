"""Repli GEMM d'`int8_matmul` (n > seuil GEMV) : la déquantification se fait
par tranches de lignes quand la matrice entière dépasserait
`_DEQUANT_TRANCHE_MAX` — Gemma-4-31B, tête 262 144 × 5 376 : 5,25 Gio d'un
coup au premier préfill, OOM après un chargement juste (Laure 0cf7fe6).
Même arithmétique, mêmes valeurs au bit ; le bras qui doit différer : une
tranche sautée."""
import pytest
import torch

from acvram import kernels
from acvram.quant.formats import _quantize_int8


def _cas():
    torch.manual_seed(7)
    w = torch.randn(1000, 256)                       # 1000 lignes : pas multiple des tranches
    t = _quantize_int8(w, 64)
    x = torch.randn(40, 256, dtype=torch.float16)    # n = 40 < 80 mais sans carte : repli GEMM
    return t, x


def test_les_tranches_rendent_la_meme_sortie_au_bit(monkeypatch):
    t, x = _cas()
    entier = kernels.int8_matmul(x, t)
    par_ligne = t.qweight.shape[1] * (4 + x.dtype.itemsize)
    monkeypatch.setattr(kernels, "_DEQUANT_TRANCHE_MAX", 128 * par_ligne)     # tranches de 128 lignes
    appels = []
    orig = kernels.int8_dequant
    monkeypatch.setattr(kernels, "int8_dequant", lambda tt, dt: appels.append(tt.qweight.shape[0]) or orig(tt, dt))
    tranche = kernels.int8_matmul(x, t)
    assert appels == [128] * 7 + [104], appels
    assert tranche.shape == (40, 1000) and torch.equal(tranche, entier)


def test_une_tranche_sautee_se_voit(monkeypatch):
    t, x = _cas()
    entier = kernels.int8_matmul(x, t)
    par_ligne = t.qweight.shape[1] * (4 + x.dtype.itemsize)
    monkeypatch.setattr(kernels, "_DEQUANT_TRANCHE_MAX", 128 * par_ligne)
    orig = kernels.int8_dequant

    def sabote(tt, dt):
        w = orig(tt, dt)
        return torch.zeros_like(w) if tt.qweight.shape[0] == 104 else w      # la dernière tranche perdue
    monkeypatch.setattr(kernels, "int8_dequant", sabote)
    assert not torch.equal(kernels.int8_matmul(x, t), entier)


class _FauxQweightCuda:
    """`.is_cuda=True` sans carte réelle : un `torch.Tensor` ne se
    monkeypatch pas sur cet attribut, il ne dit que ce que son `.device`
    porte."""
    def __init__(self, shape):
        self.shape = shape
        self.is_cuda = True


class _FauxTenseurInt8Cuda:
    def __init__(self, group_size, k=256):
        self.qweight = _FauxQweightCuda((10, k))
        self.group_size = group_size


def test_group_size_non_128_refuse_proprement_sur_le_chemin_cuda(monkeypatch):
    """Le bead de Jérôme : `ext.int8_dequant` (extension CUDA) est calibré
    pour group_size=128 -- un group_size différent retenait ~12 Gio au
    premier prefill au lieu d'échouer. Le test qui casse : refuser, pas
    tourner en silence."""
    monkeypatch.setattr(kernels, "get_extension", lambda: object())
    t = _FauxTenseurInt8Cuda(group_size=64)
    x = torch.randn(200, 256, dtype=torch.float16)   # n=200 > ACVRAM_INT8_GEMV_MAX (80) : repli GEMM
    with pytest.raises(NotImplementedError, match="group_size=64.*non servi"):
        kernels.int8_matmul(x, t)


def test_group_size_128_ne_refuse_pas_sur_le_chemin_cuda(monkeypatch):
    monkeypatch.setattr(kernels, "get_extension", lambda: object())
    t = _FauxTenseurInt8Cuda(group_size=128)
    x = torch.randn(200, 256, dtype=torch.float16)
    # La fausse implémentation s'arrête ensuite ailleurs (scales/zeros absents,
    # `.contiguous()` inexistant sur le faux qweight) : hors périmètre de ce
    # test, seul le refus prématuré (celui qu'on corrige) ne doit pas arriver.
    with pytest.raises(AttributeError):
        kernels.int8_matmul(x, t)


def test_group_size_non_128_hors_cuda_n_est_pas_concerne():
    """`_dequantize_int8` (repli sans extension) gère n'importe quel
    group_size -- vu ci-dessus par les deux premiers tests (group_size=64,
    CPU) : le refus ne vaut QUE pour le chemin CUDA."""
    t, x = _cas()   # group_size=64, CPU
    kernels.int8_matmul(x, t)   # ne lève pas
