"""Pièce 110 : q/k/v de formats mixtes — la paire de même format s'empile d'office (fusion partielle par format),
au bit des projections non empilées ; l'ancien code (fusion partielle derrière ACVRAM_FUSION_PARTIELLE=0) ne
l'empilait pas. À sec."""
import pytest
import torch
from torch import nn

from acvram.engine.attention import Attention
from acvram.engine.layers import QuantLinear
from acvram.quant.formats import _quantize_int8
from acvram.quant.nvfp4 import quantize_nvfp4

K = 64


def _lin(fmt, n, graine):
    w = torch.randn(n, K, generator=torch.Generator().manual_seed(graine)) * 0.05
    q = quantize_nvfp4(w.to(torch.bfloat16)) if fmt == "nvfp4" else _quantize_int8(w, group_size=32)
    return QuantLinear(q, None, None, n, K)


def _attention(q, k, v):
    a = Attention.__new__(Attention)
    nn.Module.__init__(a)
    a.q_proj, a.k_proj, a.v_proj, a.k_eq_v = q, k, v, False
    a.qkv_proj, a.qkv_tailles = None, ()
    return a


def test_paire_de_meme_format_empilee_sans_variable(monkeypatch):
    """Sans ACVRAM_FUSION_PARTIELLE, la paire nvfp4 (q, k) s'empile quand v est int8 — l'ancien code ne l'empilait
    pas. Octets de la pile = octets des projections (au bit) ; l'égalité AU BIT des SORTIES se juge sur carte
    (tests/test_fusion_partielle_numerique.py::test_le_partiel_rend_les_memes_nombres_que_trois_gemv, torch.equal) :
    à sec, le chemin torch n'applique pas l'échelle globale par ligne de la pile comme l'échelle scalaire."""
    monkeypatch.delenv("ACVRAM_FUSION_PARTIELLE", raising=False)
    q, k, v = _lin("nvfp4", 128, 1), _lin("nvfp4", 64, 2), _lin("int8", 64, 3)
    avant = [(l.qweight.qweight.clone(), l.qweight.block_scale.clone(), l.qweight.global_scale_float()) for l in (q, k)]
    a = _attention(q, k, v)
    assert a.fuse() is True and a.qkv_proj is None
    pile, (i, j), reste, tailles = a.qkv_partiel
    assert (i, j, reste) == (0, 1, 2), (i, j, reste)     # q+k nvfp4 ensemble, v int8 seul
    p = pile.qweight
    d = 0
    for qw, bs, gs in avant:
        n = qw.shape[0]
        assert torch.equal(p.qweight[d:d + n], qw) and torch.equal(p.block_scale[d:d + n], bs)
        assert float(p.global_scale_rows[d]) == gs and float(p.global_scale_rows[d + n - 1]) == gs
        d += n


def test_aucune_paire_de_meme_format(monkeypatch):
    monkeypatch.delenv("ACVRAM_FUSION_PARTIELLE", raising=False)
    q, k, v = _lin("nvfp4", 128, 1), _lin("int8", 64, 2), _lin("nvfp4", 64, 3)
    # q et v partagent le nvfp4 : c'est la paire (0, 2) qui s'empile
    a = _attention(q, k, v)
    assert a.fuse() is True and a.qkv_partiel[1] == (0, 2)
