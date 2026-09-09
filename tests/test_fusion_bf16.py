"""L'empilement bf16 doit rendre EXACTEMENT la même sortie, sans dupliquer.

Deux dangers, éprouvés tous les deux ici :

* **la sortie change.** Une optimisation qui change ce que le modèle produit
  est un bogue, pas une optimisation. Le GEMV empilé calcule ligne par ligne
  comme les deux séparés : chaque ligne de sortie voit la même suite de
  produits dans le même ordre, donc l'égalité doit être *exacte*, pas
  approchée. On la teste comme telle : un `allclose` laisserait passer une
  vraie divergence d'ordre de sommation.
* **les poids sont dupliqués.** q, k, v, gate et up font ~70 % d'un modèle
  dense : les copier doublerait 13,6 Gio sur Qwen2.5-14B. Les originaux doivent
  devenir des vues du tenseur empilé, pas des copies — on le vérifie par le
  `data_ptr`, seul témoin qui ne ment pas sur le partage de mémoire.
"""
import torch
import pytest

from acvram.engine.layers import QuantLinear, stack_plain_linears
from acvram.quant.formats import PlainTensor


def plain(out_f, in_f, graine):
    g = torch.Generator().manual_seed(graine)
    w = torch.randn(out_f, in_f, generator=g).to(torch.bfloat16)
    return QuantLinear(PlainTensor(w, (out_f, in_f), "bf16"))


def test_sortie_identique_au_bit_pres():
    a, b = plain(16, 8, 1), plain(16, 8, 2)
    x = torch.randn(3, 8).to(torch.bfloat16)
    attendu = torch.cat([a(x), b(x)], dim=-1)
    empile = stack_plain_linears([a, b])
    assert empile is not None
    assert torch.equal(empile(x), attendu)


def test_les_originaux_deviennent_des_vues():
    a, b = plain(16, 8, 1), plain(16, 8, 2)
    empile = stack_plain_linears([a, b])
    base = empile.qweight.weight.data_ptr()
    fin = base + 16 * 8 * empile.qweight.weight.element_size()
    # chaque original doit pointer DANS l'empilement, pas ailleurs
    assert a.qweight.weight.data_ptr() == base
    assert b.qweight.weight.data_ptr() == fin


def test_les_originaux_calculent_encore_juste():
    """Le repli prefill passe toujours par gate_proj/up_proj : leurs vues
    doivent rendre la même chose qu'avant l'empilement."""
    a, b = plain(16, 8, 1), plain(16, 8, 2)
    x = torch.randn(3, 8).to(torch.bfloat16)
    av_a, av_b = a(x).clone(), b(x).clone()
    stack_plain_linears([a, b])
    assert torch.equal(a(x), av_a)
    assert torch.equal(b(x), av_b)


@pytest.mark.parametrize("casse", ["biais", "entrees_differentes", "non_bf16"])
def test_refuse_ce_qu_il_ne_sait_pas_faire(casse):
    """Le défaut par défaut est le refus : chaque cas non couvert rend None,
    et le moteur garde le chemin séparé plutôt qu'un résultat faux."""
    a, b = plain(16, 8, 1), plain(16, 8, 2)
    if casse == "biais":
        b.bias = torch.zeros(16, dtype=torch.bfloat16)
    elif casse == "entrees_differentes":
        b = plain(16, 12, 2)
    else:
        from acvram.quant.formats import INT8Tensor
        b.qweight = INT8Tensor(torch.zeros(16, 8, dtype=torch.int8),
                               torch.ones(16, 1), torch.zeros(16, 1), 8, (16, 8))
    assert stack_plain_linears([a, b]) is None
