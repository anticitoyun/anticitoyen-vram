"""Repli GEMM d'`int8_matmul` (n > seuil GEMV) : la déquantification se fait
par tranches de lignes quand la matrice entière dépasserait
`_DEQUANT_TRANCHE_MAX` — Gemma-4-31B, tête 262 144 × 5 376 : 5,25 Gio d'un
coup au premier préfill, OOM après un chargement juste (poste3 0cf7fe6).
Même arithmétique, mêmes valeurs au bit ; le bras qui doit différer : une
tranche sautée."""
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
