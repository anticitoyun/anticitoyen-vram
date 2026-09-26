"""Pièce 260x (AU BIT, défaut) : la copie signée q − 128 du chemin cublas se fait par UN xor (q ^ 0x80 relu en int8).
(1) Équivalence au bit contre l'ancien aller-retour int16 : sur les 256 valeurs et sur le GEMM cublas entier.
(2) Cassant : au défaut, `_i8c_poids` ne lance QU'UNE opération de calcul, `bitwise_xor` — réintroduire l'ancienne copie
(`to(int16)`, `sub`, `to(int8)`) rend ce test rouge, même à sortie égale."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from acvram import kernels
from acvram.quant.formats import INT8Tensor

_VUES = {"view", "_to_copy_view", "alias", "detach", "_unsafe_view", "t", "transpose", "expand", "empty", "clone"}


class _Ops(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(func.overloadpacket.__name__)
        return func(*args, **(kwargs or {}))


def _i8c(n=64, k=128, graine=0):
    g = torch.Generator().manual_seed(graine)
    q = torch.randint(0, 256, (n, k), dtype=torch.uint8, generator=g)
    s = (torch.rand(n, 1, generator=g) * 0.02 + 0.001).to(torch.float16)
    return INT8Tensor(q, s, torch.full((n, 1), 128, dtype=torch.uint8), k, (n, k))


@pytest.mark.parametrize("mode", ["xor", "int16"])
def test_copie_signee_sur_les_256_valeurs(monkeypatch, mode):
    monkeypatch.setattr(kernels, "_I8C_COPIE", mode)
    q = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
    w = kernels.copie_signee(q)
    assert w.dtype == torch.int8 and w.is_contiguous()
    assert torch.equal(w, (q.to(torch.int32) - 128).to(torch.int8))


def test_xor_egale_int16_au_bit_sur_le_gemm_cublas(monkeypatch):
    t, x, ys = _i8c(), torch.randn(24, 128).to(torch.bfloat16), {}
    for mode in ("int16", "xor"):
        monkeypatch.setattr(kernels, "_I8C_COPIE", mode)
        t.__dict__.pop("_i8c", None)
        try:
            ys[mode] = kernels.gemm_i8c_cublas(x, t)
        except RuntimeError as e:                                  # _int_mm absent sur ce processeur
            pytest.skip(f"_int_mm indisponible à sec : {e}")
    assert ys["xor"] is not None and torch.equal(ys["xor"], ys["int16"])


def test_au_defaut_une_seule_operation_xor():
    assert kernels._I8C_COPIE == "xor"
    t = _i8c()
    t.__dict__.pop("_i8c", None)
    assert kernels._i8c_eligible(t)                               # hors mode : le test des zéros n'est pas compté
    with _Ops() as m:
        w = kernels._i8c_poids(t)
    calcul = [o for o in m.ops if o not in _VUES]
    assert calcul == ["bitwise_xor"], m.ops
    assert torch.equal(w, (t.qweight.to(torch.int16) - 128).to(torch.int8))


def test_le_temoin_int16_compte_plus_d_operations(monkeypatch):
    """Le contrôle du test précédent peut rendre « faux » : l'ancienne copie y échoue."""
    monkeypatch.setattr(kernels, "_I8C_COPIE", "int16")
    t = _i8c()
    kernels._i8c_eligible(t)
    with _Ops() as m:
        kernels._i8c_poids(t)
    assert [o for o in m.ops if o not in _VUES] != ["bitwise_xor"]
