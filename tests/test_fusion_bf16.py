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


def test_empile_aussi_les_biais():
    """Qwen2.5 porte un biais sur q, k et v : les refuser laissait ses 48
    attentions sur le chemin a trois GEMV, et la garde de la mesure l'a
    attrape -- 0 attention fusionnee sur 48."""
    a, b = plain(16, 8, 1), plain(16, 8, 2)
    a.bias = torch.randn(16).to(torch.bfloat16)
    b.bias = torch.randn(16).to(torch.bfloat16)
    x = torch.randn(3, 8).to(torch.bfloat16)
    attendu = torch.cat([a(x), b(x)], dim=-1)
    empile = stack_plain_linears([a, b])
    assert empile is not None
    assert torch.equal(empile(x), attendu)


@pytest.mark.parametrize("casse", ["biais_partiel", "entrees_differentes", "non_bf16"])
def test_refuse_ce_qu_il_ne_sait_pas_faire(casse):
    """Le défaut par défaut est le refus : chaque cas non couvert rend None,
    et le moteur garde le chemin séparé plutôt qu'un résultat faux."""
    a, b = plain(16, 8, 1), plain(16, 8, 2)
    if casse == "biais_partiel":
        b.bias = torch.zeros(16, dtype=torch.bfloat16)   # a n'en a pas
    elif casse == "entrees_differentes":
        b = plain(16, 12, 2)
    else:
        from acvram.quant.formats import INT8Tensor
        b.qweight = INT8Tensor(torch.zeros(16, 8, dtype=torch.int8),
                               torch.ones(16, 1), torch.zeros(16, 1), 8, (16, 8))
    assert stack_plain_linears([a, b]) is None


# --- le noyau fusionné ------------------------------------------------------

def _ext_swiglu():
    """L'extension, ou None. On vérifie qu'elle porte bien le symbole : une
    extension présente mais compilée avant ce noyau répondrait à `is not None`
    et échouerait à l'appel."""
    if not torch.cuda.is_available():
        return None
    from acvram import kernels
    ext = kernels.get_extension()
    return ext if ext is not None and hasattr(ext, "swiglu_bf16") else None


@pytest.mark.skipif(_ext_swiglu() is None,
                    reason="extension CUDA sans swiglu_bf16 (ou pas de GPU)")
def test_swiglu_identique_a_torch_au_bit_pres():
    """Le noyau doit rendre exactement ce que rendait le chemin en deux
    lancements — arrondi intermédiaire compris. Calculer d'un trait en float
    serait plus exact, donc faux : d'autres jetons sortiraient."""
    ext = _ext_swiglu()
    g = torch.randn(3, 64, device="cuda").to(torch.bfloat16)
    u = torch.randn(3, 64, device="cuda").to(torch.bfloat16)
    gu = torch.cat([g, u], dim=-1).contiguous()
    attendu = torch.nn.functional.silu(g) * u
    assert torch.equal(ext.swiglu_bf16(gu), attendu)


@pytest.mark.skipif(_ext_swiglu() is None,
                    reason="extension CUDA sans swiglu_bf16 (ou pas de GPU)")
def test_swiglu_sur_valeurs_extremes():
    """Les grandes amplitudes sont là où un exp() approché diverge : c'est le
    cas qui distingue une équivalence réelle d'une équivalence sur du bruit
    centré."""
    ext = _ext_swiglu()
    vals = torch.tensor([-60., -8., -1e-3, 0., 1e-3, 8., 60.], device="cuda")
    g = vals.repeat(2, 1).to(torch.bfloat16)
    u = vals.flip(0).repeat(2, 1).to(torch.bfloat16)
    gu = torch.cat([g, u], dim=-1).contiguous()
    attendu = torch.nn.functional.silu(g) * u
    assert torch.equal(ext.swiglu_bf16(gu), attendu)


@pytest.mark.skipif(_ext_swiglu() is None,
                    reason="extension CUDA sans swiglu_bf16 (ou pas de GPU)")
@pytest.mark.parametrize("lot", [1, 8, 88, 171, 512])
def test_swiglu_sur_lot_large(lot):
    """Le noyau a été écrit et éprouvé pour le décodage — une ligne à la fois.
    Avant de l'employer au préremplissage il faut le voir juste sur un lot, et
    pas seulement supposer qu'il travaille par élément."""
    ext = _ext_swiglu()
    g = torch.randn(lot, 256, device="cuda").to(torch.bfloat16)
    u = torch.randn(lot, 256, device="cuda").to(torch.bfloat16)
    gu = torch.cat([g, u], dim=-1).contiguous()
    attendu = torch.nn.functional.silu(g) * u
    assert torch.equal(ext.swiglu_bf16(gu), attendu)


# --- le SwiGLU à deux entrées, pour le chemin NON fusionné -------------------

def _ext_swiglu2():
    if not torch.cuda.is_available():
        return None
    from acvram import kernels
    ext = kernels.get_extension()
    return ext if ext is not None and hasattr(ext, "swiglu2_bf16") else None


@pytest.mark.skipif(_ext_swiglu2() is None,
                    reason="extension CUDA sans swiglu2_bf16 (ou pas de GPU)")
@pytest.mark.parametrize("lot", [1, 8, 88, 512])
def test_swiglu2_identique_a_torch(lot):
    """Le chemin non fusionné payait silu PUIS produit. Le noyau à deux entrées
    doit rendre exactement la même chose — arrondi intermédiaire compris."""
    ext = _ext_swiglu2()
    g = torch.randn(lot, 256, device="cuda").to(torch.bfloat16)
    u = torch.randn(lot, 256, device="cuda").to(torch.bfloat16)
    assert torch.equal(ext.swiglu2_bf16(g, u),
                       torch.nn.functional.silu(g) * u)


@pytest.mark.skipif(_ext_swiglu2() is None,
                    reason="extension CUDA sans swiglu2_bf16 (ou pas de GPU)")
def test_swiglu2_et_swiglu_empile_concordent():
    """Les deux chemins doivent rendre la même chose, sans quoi un modèle
    changerait de sortie selon qu'il fusionne ou non."""
    ext = _ext_swiglu2()
    g = torch.randn(4, 128, device="cuda").to(torch.bfloat16)
    u = torch.randn(4, 128, device="cuda").to(torch.bfloat16)
    gu = torch.cat([g, u], dim=-1).contiguous()
    assert torch.equal(ext.swiglu2_bf16(g, u), ext.swiglu_bf16(gu))


@pytest.mark.skipif(_ext_swiglu2() is None,
                    reason="extension CUDA sans swiglu2_bf16 (ou pas de GPU)")
def test_swiglu2_sur_valeurs_extremes():
    """Là où un exp() approché divergerait."""
    ext = _ext_swiglu2()
    v = torch.tensor([-60., -8., -1e-3, 0., 1e-3, 8., 60.], device="cuda")
    g = v.repeat(2, 1).to(torch.bfloat16)
    u = v.flip(0).repeat(2, 1).to(torch.bfloat16)
    assert torch.equal(ext.swiglu2_bf16(g, u),
                       torch.nn.functional.silu(g) * u)
