"""Les codecs, vérifiés face à leurs propres définitions."""

import math

import pytest
import torch

from acvram.quant.calibrate import (ActStats, hadamard_transform,
                                    largest_pow2_divisor,
                                    quantize_with_calibration)
from acvram.quant.formats import bits_per_weight, dequantize, quantize
from acvram.quant.int4 import pack_uint4, quantize_int4, unpack_uint4
from acvram.quant.nvfp4 import (E2M1_LEVELS, dequantize_nvfp4, pack_e2m1,
                                quantize_nvfp4, round_to_e2m1, unpack_e2m1)


def test_e2m1_grid_is_exact():
    """Des valeurs déjà sur la grille E2M1 doivent traverser l'aller-retour intactes."""
    w = torch.tensor([E2M1_LEVELS], dtype=torch.float32).repeat(4, 2)
    t = quantize_nvfp4(w)
    assert torch.equal(dequantize_nvfp4(t, torch.float32), w)


def test_e2m1_rounds_ties_to_even():
    # Les milieux doivent tomber sur un code pair, pas simplement monter ou descendre.
    m = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    assert round_to_e2m1(m).tolist() == [0, 2, 2, 4, 4, 6, 6]


def test_e2m1_saturates():
    assert round_to_e2m1(torch.tensor([1e9])).item() == 7


@pytest.mark.parametrize("shape", [(64, 64), (128, 256), (33, 100)])
def test_nvfp4_roundtrip(shape):
    w = torch.randn(*shape) * 0.02
    t = quantize_nvfp4(w)
    d = dequantize_nvfp4(t, torch.float32)
    assert d.shape == w.shape
    assert (d - w).norm() / w.norm() < 0.2


def test_nvfp4_bits_per_weight():
    # 4 bits par élément, plus un octet E4M3 pour 16 d'entre eux. L'unique
    # échelle globale fp32 par tenseur place le résultat un cheveu au-dessus de
    # 4,5 : d'où une tolérance qui n'est pas exacte.
    t = quantize_nvfp4(torch.randn(512, 4096) * 0.02)
    assert 4.5 <= t.bits_per_weight < 4.501


def test_int4_bits_per_weight():
    t = quantize_int4(torch.randn(512, 4096) * 0.02)
    assert math.isclose(t.bits_per_weight, 4.15625, rel_tol=1e-6)


@pytest.mark.parametrize("packer,unpacker",
                         [(pack_e2m1, unpack_e2m1), (pack_uint4, unpack_uint4)])
def test_nibble_packing_roundtrips(packer, unpacker):
    codes = torch.randint(0, 16, (8, 64), dtype=torch.uint8)
    assert torch.equal(unpacker(packer(codes)), codes)


def test_all_zero_weight_stays_zero():
    for fmt in ("nvfp4", "int4_awq", "int8"):
        t = quantize(torch.zeros(32, 256), fmt)
        assert dequantize(t, torch.float32).abs().max().item() == 0.0


def test_format_ordering_by_size_and_error():
    """Plus de bits doit signifier plus d'octets et une meilleure fidélité."""
    w = torch.randn(256, 1024) * 0.02
    prev_bytes, prev_err = 0, 1.0
    for fmt in ("int4_awq", "nvfp4", "int8", "bf16"):
        t = quantize(w, fmt)
        err = ((dequantize(t, torch.float32) - w).norm() / w.norm()).item()
        assert t.nbytes > prev_bytes, fmt
        assert err < prev_err + 1e-6, fmt
        prev_bytes, prev_err = t.nbytes, err


def test_hadamard_is_orthonormal_and_involutive():
    x = torch.randn(3, 1024)
    h = hadamard_transform(x)
    assert torch.allclose(x.norm(dim=-1), h.norm(dim=-1), atol=1e-4)
    assert torch.allclose(hadamard_transform(h), x, atol=1e-3)


def test_hadamard_handles_non_power_of_two():
    # 11008 = 256 × 43 : bloc-diagonal sur le plus grand diviseur puissance de deux.
    assert largest_pow2_divisor(11008) == 256
    x = torch.randn(2, 11008)
    assert torch.allclose(x.norm(dim=-1), hadamard_transform(x).norm(dim=-1),
                          atol=1e-3)


def test_hadamard_helps_int4_on_outlier_channels():
    """La rotation existe pour étaler les valeurs aberrantes ; vérifions-le."""
    torch.manual_seed(0)
    w = torch.randn(256, 1024) * 0.02
    w[:, ::64] *= 8.0
    act = torch.rand(1024).pow(3) * 10 + 0.1
    stats = ActStats(act, None, 64)
    _, _, plain = quantize_with_calibration(w, "int4_awq", stats,
                                            use_hadamard=False, use_awq=False)
    _, _, rotated = quantize_with_calibration(w, "int4_awq", stats,
                                              use_hadamard=True, use_awq=False)
    assert rotated["w_snr_db"] > plain["w_snr_db"] + 2.0


def test_bits_per_weight_matches_actual_storage():
    for fmt in ("nvfp4", "int4_awq", "int8"):
        t = quantize(torch.randn(256, 4096) * 0.02, fmt)
        assert math.isclose(t.bits_per_weight, bits_per_weight(fmt), rel_tol=1e-3)
