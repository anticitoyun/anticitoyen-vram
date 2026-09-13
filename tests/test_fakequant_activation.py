"""Fake-quant des activations (bead anticitoyen-vram-brd, étape 1) : les
deux primitives réutilisées pour le protocole W4A4/W4A8, vérifiées sans
carte ni modèle chargé — juste des tenseurs synthétiques."""
import torch

from acvram.quant.fakequant_activation import (fake_quantize_e4m3_activation,
                                               fake_quantize_nvfp4_activation)


def test_preserve_shape_et_dtype():
    x = torch.randn(3, 5, 4096, dtype=torch.bfloat16) * 0.3
    for fq in (fake_quantize_nvfp4_activation, fake_quantize_e4m3_activation):
        y = fq(x)
        assert y.shape == x.shape
        assert y.dtype == x.dtype


def test_e4m3_est_plus_fin_que_e2m1():
    """Le format à 8 bits doit reconstruire mieux que le format à 4 bits —
    sinon les deux fonctions mesureraient la même chose."""
    torch.manual_seed(0)
    x = torch.randn(16, 4096) * 0.3
    err4 = (fake_quantize_nvfp4_activation(x) - x).norm() / x.norm()
    err8 = (fake_quantize_e4m3_activation(x) - x).norm() / x.norm()
    assert err8 < err4 / 2, (err4.item(), err8.item())


def test_gere_une_dimension_non_multiple_de_16():
    """down_proj de Llama-2-7B a 11008 entrées (11008 = 16 × 688, un
    multiple exact) mais rien ne garantit qu'une autre architecture le
    soit ; les deux fonctions doivent survivre a une dimension quelconque."""
    x = torch.randn(4, 4097)
    for fq in (fake_quantize_nvfp4_activation, fake_quantize_e4m3_activation):
        y = fq(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()


def test_un_outlier_reste_confine_a_son_bloc_de_16():
    """Meme granularite de bloc (16) dans les deux fonctions : un outlier
    dans les 16 premieres colonnes ne doit pas degrader le bloc voisin.
    Sans cette localite, le protocole ne comparerait pas des formats a
    granularite egale, contrairement a ce que documente le module."""
    torch.manual_seed(1)
    x = torch.randn(4, 32) * 0.1
    x[:, 0] = 100.0
    for fq in (fake_quantize_nvfp4_activation, fake_quantize_e4m3_activation):
        y = fq(x)
        voisin_ref, voisin_q = x[:, 16:], y[:, 16:]
        err_voisin = (voisin_q - voisin_ref).norm() / voisin_ref.norm()
        assert err_voisin < 0.15, (fq.__name__, err_voisin.item())


def test_zero_reste_zero():
    for fq in (fake_quantize_nvfp4_activation, fake_quantize_e4m3_activation):
        y = fq(torch.zeros(4, 64))
        assert y.abs().max().item() == 0.0


def test_e2m1_degrade_avec_l_amplitude_de_bruit():
    """Non-regression minimale : plus le signal est bruite au depart, plus
    l'erreur de reconstruction relative reste dans un ordre de grandeur
    stable (la fonction n'explose pas, ne devient pas meilleure a mesure
    que le signal se degrade — ce qui trahirait un bogue de mise a
    l'echelle)."""
    torch.manual_seed(2)
    erreurs = []
    for amp in (0.05, 0.2, 1.0, 5.0):
        x = torch.randn(8, 4096) * amp
        y = fake_quantize_nvfp4_activation(x)
        erreurs.append(((y - x).norm() / x.norm()).item())
    # L'echelle globale FP32 rend l'erreur relative INVARIANTE a l'amplitude
    # (seule la resolution E2M1/E4M3 compte, pas la magnitude absolue) :
    # les quatre valeurs doivent donc rester proches les unes des autres.
    assert max(erreurs) - min(erreurs) < 0.02, erreurs
