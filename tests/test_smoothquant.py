"""SmoothQuant : l'échelle et le repliement, vérifiés sans carte ni modèle
chargé — la campagne réelle (Llama-2-7B, alpha ∈ {0,5; 0,65; 0,8}) suit
dans outils/smoothquant-a4-sweep.py."""
import math

import pytest
import torch

from acvram.quant.nvfp4 import dequantize_nvfp4, quantize_nvfp4
from acvram.quant.smoothquant import echelle_smoothquant, replier_smoothquant


def test_alpha_zero_ignore_les_activations():
    """A alpha=0, s ne depend QUE du poids (x**0 = 1) : toute la difficulte
    reste sur le poids, rien n'est deplace vers l'activation."""
    w = torch.tensor([2.0, 8.0, 0.5])
    x = torch.tensor([100.0, 1.0, 50.0])           # sans effet a alpha=0
    s = echelle_smoothquant(w, x, alpha=0.0)
    s2 = echelle_smoothquant(w, x * 7.0, alpha=0.0)
    assert torch.allclose(s, s2), "alpha=0 doit ignorer l'activation"
    assert torch.allclose(s, 1.0 / w)


def test_alpha_un_ignore_les_poids():
    """A alpha=1, s ne depend QUE de l'activation (w**0 = 1)."""
    w = torch.tensor([2.0, 8.0, 0.5])
    x = torch.tensor([100.0, 1.0, 50.0])
    s = echelle_smoothquant(w, x, alpha=1.0)
    s2 = echelle_smoothquant(w * 3.0, x, alpha=1.0)
    assert torch.allclose(s, s2), "alpha=1 doit ignorer le poids"
    assert torch.allclose(s, x)


def test_alpha_hors_bornes_leve():
    w = torch.ones(4)
    x = torch.ones(4)
    for mauvais in (-0.1, 1.1):
        with pytest.raises(ValueError, match="alpha"):
            echelle_smoothquant(w, x, alpha=mauvais)


def test_formes_incompatibles_levent():
    with pytest.raises(ValueError, match="formes"):
        echelle_smoothquant(torch.ones(4), torch.ones(5), alpha=0.5)


def test_canal_jamais_active_ne_rend_pas_nan():
    """Un canal a activation nulle (porte non routee dans le lot de
    calibration) ne doit jamais produire nan/inf, quel que soit alpha —
    sinon UN SEUL canal silencieux corrompt tout le tenseur reconstruit
    apres repliement."""
    w = torch.tensor([2.0, 0.0, 0.5])
    x = torch.tensor([0.0, 1.0, 50.0])
    for alpha in (0.0, 0.5, 0.65, 0.8, 1.0):
        s = echelle_smoothquant(w, x, alpha=alpha)
        assert torch.isfinite(s).all(), (alpha, s)
        assert (s > 0).all(), (alpha, s)


def test_le_repliement_preserve_le_produit_matriciel_en_pleine_precision():
    """Identite algebrique du module : X' @ W'^T == X @ W^T EXACTEMENT
    avant toute quantification. Verifiee ici sur le poids DEQUANTIFIE
    (pas sur le NVFP4 requantifie, qui introduit sa propre perte —
    l'objet de ce test est l'algebre du lissage, pas le format)."""
    torch.manual_seed(0)
    w = torch.randn(64, 128) * 0.02
    x = torch.randn(8, 128)
    x_absmax = x.abs().amax(dim=0)

    for alpha in (0.0, 0.5, 0.65, 0.8, 1.0):
        s = echelle_smoothquant(w.abs().amax(dim=0), x_absmax, alpha)
        w_smooth = w * s.unsqueeze(0)
        x_smooth = x / s.unsqueeze(0)
        ref = x @ w.t()
        lisse = x_smooth @ w_smooth.t()
        assert torch.allclose(ref, lisse, atol=1e-3, rtol=1e-3), alpha


def test_replier_smoothquant_rend_un_nvfp4_valide_et_l_echelle():
    torch.manual_seed(1)
    w = torch.randn(64, 128) * 0.02
    qw = quantize_nvfp4(w)
    x_absmax = torch.rand(128) * 5 + 0.1

    for alpha in (0.5, 0.65, 0.8):
        qw2, s = replier_smoothquant(qw, x_absmax, alpha)
        assert s.shape == (128,)
        assert torch.isfinite(s).all()
        deq = dequantize_nvfp4(qw2, torch.float32)
        assert deq.shape == w.shape
        assert torch.isfinite(deq).all()


def test_le_repliement_reste_proche_apres_requantification():
    """Le repliement + requantification NVFP4 doit rester une
    approximation raisonnable du produit d'origine — un ecart de deux
    ordres de grandeur signalerait un bogue de repliement (colonne au lieu
    de ligne, echelle inversee), pas seulement le bruit attendu du
    format."""
    torch.manual_seed(2)
    w = torch.randn(64, 128) * 0.02
    x = torch.randn(8, 128)
    qw = quantize_nvfp4(w)
    x_absmax = x.abs().amax(dim=0)

    qw2, s = replier_smoothquant(qw, x_absmax, alpha=0.65)
    x_smooth = x / s.unsqueeze(0)
    w_deq = dequantize_nvfp4(qw2, torch.float32)
    ref = x @ w.t()
    lisse = x_smooth @ w_deq.t()
    err = (lisse - ref).norm() / ref.norm()
    assert err < 0.5, f"erreur relative {err.item():.3f} apres repliement+requant"
