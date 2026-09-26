"""Une pile NVFP4 dequantifiee doit lire `global_scale_rows`.

Le 9/09/2026, poste4 a mesure sur CPU un ecart de 0,121 sur la projection k
d'un groupe fusionne — un facteur d'echelle CONSTANT de 1,1216. Cause : ni
`dequantize_nvfp4` ni le noyau CPU ne lisaient `global_scale_rows`, que la
fusion pose pour que chaque projection empilee garde SON echelle globale.

Ce n'est pas un defaut de vitesse mais de JUSTESSE : tout tenseur fusionne
sorti du chemin CUDA — exil en RAM hote, extension absente — rendait des
nombres faux, en silence.
"""
import torch

from acvram.quant.nvfp4 import dequantize_nvfp4, quantize_nvfp4


def _pile(a, b):
    """Empile deux poids comme le fait `stack_nvfp4_linears`."""
    from acvram.quant.nvfp4 import NVFP4Tensor
    qa, qb = quantize_nvfp4(a), quantize_nvfp4(b)
    lignes = torch.cat([
        torch.full((qa.qweight.shape[0],), qa.global_scale_float()),
        torch.full((qb.qweight.shape[0],), qb.global_scale_float())])
    return NVFP4Tensor(
        torch.cat([qa.qweight, qb.qweight]).contiguous(),
        torch.cat([qa.block_scale, qb.block_scale]).contiguous(),
        qa.global_scale,
        (a.shape[0] + b.shape[0], a.shape[1]), qa.padded_in,
        global_scale_rows=lignes.contiguous()), qa, qb


def test_la_dequantification_lit_l_echelle_par_ligne():
    """Deux poids d'amplitudes tres differentes : sans `global_scale_rows`,
    le second herite de l'echelle du premier et sort faux d'un facteur."""
    g = torch.Generator().manual_seed(5)
    a = (torch.randn(64, 128, generator=g) * 0.02).to(torch.bfloat16)
    b = (torch.randn(64, 128, generator=g) * 2.0).to(torch.bfloat16)   # x100
    pile, qa, qb = _pile(a, b)
    d = dequantize_nvfp4(pile, torch.float32)
    da = dequantize_nvfp4(qa, torch.float32)
    db = dequantize_nvfp4(qb, torch.float32)
    assert torch.allclose(d[:64], da, atol=1e-5), "la premiere moitie derive"
    assert torch.allclose(d[64:], db, atol=1e-4), \
        "la seconde moitie herite de la mauvaise echelle globale"


def test_le_temoin_montre_que_l_epreuve_peut_echouer():
    """Sans echelle par ligne, l'ecart DOIT etre visible — sinon l'epreuve
    ci-dessus ne prouverait rien."""
    g = torch.Generator().manual_seed(5)
    a = (torch.randn(64, 128, generator=g) * 0.02).to(torch.bfloat16)
    b = (torch.randn(64, 128, generator=g) * 2.0).to(torch.bfloat16)
    pile, _, qb = _pile(a, b)
    pile.global_scale_rows = None            # on retire l'echelle par ligne
    faux = dequantize_nvfp4(pile, torch.float32)
    vrai = dequantize_nvfp4(qb, torch.float32)
    assert not torch.allclose(faux[64:], vrai, atol=1e-2), \
        "le temoin ne diverge pas : l'epreuve principale ne demontre rien"


def test_le_produit_cpu_sur_une_pile_est_juste():
    """Epreuve de COMPORTEMENT, pas de texte.

    La premiere version de ce test cherchait la chaine `dequantize_nvfp4` a
    moins de 200 caracteres de `global_scale_rows` dans le source — il aurait
    passe a l'identique sur un code qui ecrit la bonne ligne et calcule autre
    chose. C'est le motif du dossier applique a son propre filet, releve par
    poste4 sur mes cinq epreuves precedentes.
    """
    from acvram.kernels.cpu import nvfp4_matmul_cpu

    g = torch.Generator().manual_seed(7)
    a = (torch.randn(64, 128, generator=g) * 0.02).to(torch.bfloat16)
    b = (torch.randn(64, 128, generator=g) * 2.0).to(torch.bfloat16)
    pile, _, _ = _pile(a, b)
    # float32 des DEUX cotes : comparer un produit bf16 a une reference fp32
    # mesurerait l'arrondi du format, pas l'echelle. Premiere version de ce
    # test : 9,2e-2 d'ecart, entierement imputable a ce melange.
    x = (torch.randn(1, 128, generator=g) * 0.5).to(torch.float32)

    obtenu = nvfp4_matmul_cpu(x, pile).to(torch.float32)
    attendu = torch.nn.functional.linear(
        x, dequantize_nvfp4(pile, torch.float32))
    ecart = (obtenu - attendu).abs().max().item()
    echelle = attendu.abs().max().item()
    assert ecart <= 1e-3 * max(echelle, 1e-6), (
        f"le produit CPU derive de {ecart:.3e} sur une pile (echelle "
        f"{echelle:.3e}) — le noyau a probablement utilise une echelle "
        f"globale unique")
