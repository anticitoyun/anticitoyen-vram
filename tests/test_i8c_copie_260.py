"""Pièce 260. (1) AU BIT : la copie signée par xor (q ^ 0x80 relu en int8) égale l'aller-retour int16 (q − 128) sur les
256 valeurs et sur le GEMM cublas entier ; cassant : un masque faux (0x7F) ou un décalage rend ce fichier rouge.
(2) Opt-in ACVRAM_I8C_FP8_PREFILL : bf16 (défaut) marque les int8 d'origine fp8 `prefill_bf16` (139, déquant) ;
cublas ne les marque pas → éligibles au W8A8 int8, et la ligne de régime le dit."""

import pytest
import torch

from acvram import kernels, regime
from acvram.engine import loader
from acvram.quant.formats import INT8Tensor


def _i8c(n=64, k=128, graine=0):
    g = torch.Generator().manual_seed(graine)
    q = torch.randint(0, 256, (n, k), dtype=torch.uint8, generator=g)
    s = (torch.rand(n, 1, generator=g) * 0.02 + 0.001).to(torch.float16)
    return INT8Tensor(q, s, torch.full((n, 1), 128, dtype=torch.uint8), k, (n, k))


@pytest.mark.parametrize("mode", ["xor", "int16"])
def test_copie_signee_sur_les_256_valeurs(monkeypatch, mode):
    monkeypatch.setattr(kernels, "_I8C_COPIE", mode)
    q = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
    attendu = (q.to(torch.int32) - 128).to(torch.int8)
    w = kernels.copie_signee(q)
    assert w.dtype == torch.int8 and w.is_contiguous()
    assert torch.equal(w, attendu)


def test_xor_egale_int16_au_bit_sur_le_gemm_cublas(monkeypatch):
    t = _i8c()
    x = torch.randn(24, 128).to(torch.bfloat16)
    ys = {}
    for mode in ("int16", "xor"):
        monkeypatch.setattr(kernels, "_I8C_COPIE", mode)
        t.__dict__.pop("_i8c", None)
        try:
            ys[mode] = kernels.gemm_i8c_cublas(x, t)
        except RuntimeError as e:                                  # _int_mm absent sur ce processeur
            pytest.skip(f"_int_mm indisponible à sec : {e}")
    assert ys["xor"] is not None and torch.equal(ys["xor"], ys["int16"])


class _Lecteur:
    def __init__(self, sd):
        self.sd = sd

    def get(self, k):
        return self.sd[k]


def _construit(monkeypatch, mode):
    monkeypatch.setattr(kernels, "_I8C_FP8_PREFILL", mode)
    t = _i8c()
    sd = {"w.qweight": t.qweight, "w.scales": t.scales, "w.zeros": t.zeros}
    e = {"format": "int8", "shape": [64, 128], "keys": list(sd), "group_size": 128, "origine": "fp8"}
    return loader._build_quant(e, "w", _Lecteur(sd), 128)


def test_defaut_bf16_marque_l_origine_fp8(monkeypatch):
    t = _construit(monkeypatch, "bf16")
    assert t.__dict__.get("prefill_bf16") is True and not kernels._i8c_eligible(t)


def test_opt_in_cublas_rend_l_origine_fp8_eligible(monkeypatch):
    t = _construit(monkeypatch, "cublas")
    assert "prefill_bf16" not in t.__dict__ and kernels._i8c_eligible(t)


def test_ligne_de_regime_nomme_le_mode(monkeypatch):
    monkeypatch.setattr(regime, "_I8C_PREFILL_BF16", 233)
    monkeypatch.setattr(kernels, "_I8C_FP8_PREFILL", "bf16")
    assert regime.prefill_i8c_texte() == "prefill_int8=bf16(origine fp8 ×233)"
    monkeypatch.setattr(kernels, "_I8C_FP8_PREFILL", "cublas")
    assert regime.prefill_i8c_texte() == "prefill_int8=cublas(origine fp8 ×233)"


def test_defauts_260():
    assert kernels._I8C_FP8_PREFILL == "bf16" and kernels._I8C_COPIE == "xor"
