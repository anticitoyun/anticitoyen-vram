"""Empilement des projections NVFP4 (v0.4.62).

q, k, v — comme gate et up — lisent la même activation : les empiler remplace
trois GEMV par une. Leurs échelles globales diffèrent, et les unifier passerait
par un réarrondi e4m3 de l'ordre de 6 % ; le noyau lit donc une échelle par
ligne de sortie. Ces tests fixent les deux propriétés qui rendent l'opération
sûre : le résultat est identique au bit près, et rien n'est dupliqué en mémoire.
"""
import pytest
import torch

from acvram.engine.layers import QuantLinear, ROWS_PAR_BLOC, stack_nvfp4_linears
from acvram.quant.nvfp4 import NVFP4Tensor, quantize_nvfp4


def _lin(sorties: int, entrees: int, amplitude: float, graine: int) -> QuantLinear:
    """Une projection dont la dynamique — donc l'échelle globale — lui est propre."""
    g = torch.Generator().manual_seed(graine)
    w = torch.randn(sorties, entrees, generator=g) * amplitude
    return QuantLinear(quantize_nvfp4(w))


def _dequant(t: NVFP4Tensor, gs: float) -> torch.Tensor:
    """Déquantification de référence, échelle globale imposée."""
    from acvram.quant.nvfp4 import unpack_e2m1
    niveaux = unpack_e2m1(t.qweight).to(torch.float32)
    bs = t.block_scale.to(torch.float32).repeat_interleave(16, dim=1)
    return niveaux * bs * gs


def test_les_echelles_globales_different():
    """Sans quoi le test suivant ne prouverait rien."""
    a, b = _lin(64, 128, 1.0, 1), _lin(32, 128, 40.0, 2)
    assert a.qweight.global_scale_float() != b.qweight.global_scale_float()


def test_empilement_exact_au_bit_pres():
    lins = [_lin(64, 128, 1.0, 1), _lin(32, 128, 40.0, 2), _lin(32, 128, 0.05, 3)]
    avant = [_dequant(l.qweight, l.qweight.global_scale_float()) for l in lins]
    fus = stack_nvfp4_linears(lins)
    assert fus is not None
    pile = fus.qweight
    assert pile.global_scale_rows is not None
    assert pile.qweight.shape[0] == 128
    d = 0
    for att in avant:
        n = att.shape[0]
        seg = NVFP4Tensor(pile.qweight[d:d + n], pile.block_scale[d:d + n],
                          pile.global_scale, (n, att.shape[1]), pile.padded_in)
        gs = pile.global_scale_rows[d:d + n]
        assert gs.min() == gs.max()          # une échelle par segment, pas par ligne
        obtenu = _dequant(seg, float(gs[0]))
        assert torch.equal(obtenu, att)      # au bit près, pas « proche »
        d += n


def test_les_originaux_deviennent_des_vues():
    """Fusionner ne doit pas coûter un octet : le prefill lit la même mémoire."""
    lins = [_lin(64, 128, 1.0, 1), _lin(32, 128, 40.0, 2)]
    fus = stack_nvfp4_linears(lins)
    base = fus.qweight.qweight.data_ptr()
    fin = base + fus.qweight.qweight.numel()
    for l in lins:
        assert base <= l.qweight.qweight.data_ptr() < fin
        assert l.qweight.global_scale_rows is None      # chacun garde la sienne


def test_segment_non_aligne_refuse():
    """Le noyau lit l'échelle de la première ligne du bloc : un segment qui ne
    commence pas sur un multiple de la hauteur de bloc lui ferait appliquer
    l'échelle du voisin."""
    assert ROWS_PAR_BLOC > 1
    lins = [_lin(ROWS_PAR_BLOC + 1, 128, 1.0, 1), _lin(32, 128, 40.0, 2)]
    assert stack_nvfp4_linears(lins) is None


def test_pas_de_second_empilement():
    lins = [_lin(64, 128, 1.0, 1), _lin(32, 128, 40.0, 2)]
    fus = stack_nvfp4_linears(lins)
    assert stack_nvfp4_linears([fus, _lin(32, 128, 1.0, 4)]) is None


def test_temoin_desactive(monkeypatch):
    monkeypatch.setenv("ACVRAM_FUSION_NVFP4", "0")
    assert stack_nvfp4_linears([_lin(64, 128, 1.0, 1), _lin(32, 128, 1.0, 2)]) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau CUDA requis")
def test_gemv_empilee_identique_sur_gpu():
    """La preuve qui compte : la GEMV fusionnée rend exactement les trois
    GEMV séparées mises bout à bout."""
    from acvram.kernels import get_extension
    ext = get_extension()
    if not hasattr(ext, "nvfp4_gemv"):
        pytest.skip("extension sans nvfp4_gemv")
    dev = torch.device("cuda")
    lins = [_lin(512, 1024, 1.0, 1), _lin(128, 1024, 40.0, 2), _lin(128, 1024, 0.05, 3)]
    for l in lins:
        l.qweight = l.qweight.to(dev)
    x = torch.randn(1, 1024, device=dev, dtype=torch.bfloat16)
    separe = torch.cat([l(x) for l in lins], dim=-1)
    fus = stack_nvfp4_linears(lins)
    assert fus is not None
    assert torch.equal(fus(x), separe)
