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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau CUDA requis")
@pytest.mark.parametrize("n", [1, 4, 8, 9, 16, 64])
def test_pile_juste_a_tout_nombre_de_jetons(n):
    """La pile doit rendre la même chose que les projections séparées, que le
    lot passe par la GEMV (n ≤ 8) ou par les chemins de prefill (n > 8).

    Le 6 septembre 2026, GLM-4.7 répondait « de de de de » à tout prompt de dix
    jetons ou plus : les chemins de prefill prenaient l'échelle du premier
    segment pour toute la pile q_a/kv_a."""
    from acvram.kernels import get_extension
    if not hasattr(get_extension(), "nvfp4_gemv"):
        pytest.skip("extension sans nvfp4_gemv")
    from acvram.kernels import _NVFP4_GEMV_MAX
    dev = torch.device("cuda")
    lins = [_lin(768, 2048, 1.0, 11), _lin(576, 2048, 40.0, 12)]   # q_a, kv_a de GLM
    for l in lins:
        l.qweight = l.qweight.to(dev)
    g = torch.Generator(device="cuda").manual_seed(n)
    x = torch.randn(n, 2048, device=dev, dtype=torch.bfloat16, generator=g)
    from acvram.quant.nvfp4 import dequantize_nvfp4
    wref = torch.cat([dequantize_nvfp4(l.qweight, torch.bfloat16) for l in lins])
    exact = torch.nn.functional.linear(x, wref).float()
    fus = stack_nvfp4_linears(lins)
    obtenu = fus(x)

    # DEUX criteres, et ce que chacun peut rendre faux est ecrit ici.
    #
    # 1. Plancher de SNR DECLARE contre la reference bf16, par bloc de sortie,
    #    a TOUT n. Rend faux si le chemin de la pile applique la mauvaise
    #    echelle a un bloc — le defaut de GLM-4.7 du 6/09. Verifie : en
    #    sabotant le chemin de la pile (echelle de q_a imposee au bloc kv_a),
    #    les six valeurs de n rendent 0,22 dB et echouent, 6/6.
    # 2. Egalite bit a bit avec les projections separees, dans le regime GEMV
    #    seulement. Le critere etait ecrit « n <= 8 » quand le seuil valait 8 ;
    #    il suit maintenant la constante, sans quoi il se decale a chaque
    #    mesure du seuil — c'est ce decalage qui a fait echouer ce test a n = 9
    #    et 16 quand le seuil est passe a 32 le 10/09 (824d2fd), sur le chemin
    #    DEVENU LE PLUS JUSTE des deux : la tolerance par element de 2**-6
    #    etait calibree pour la route de la pile au-dela du seuil, pas pour la
    #    GEMV, qui rend 50 dB la ou la tolerance exige 1,56 %.
    #
    # CE QU'AUCUN DES DEUX NE PEUT RENDRE FAUX, et il faut le savoir : une
    # echelle fausse dans la DONNEE. La reference `exact` est construite en
    # dequantifiant les memes objets ; si `global_scale` est corrompu, la
    # reference l'est identiquement et le SNR reste a 50 dB. Verifie aussi :
    # 0/6. Le critere 2 y est aveugle pour la meme raison — la pile et les
    # GEMV separees lisent la meme echelle fausse et s'accordent en etant
    # toutes deux fausses. Ces deux criteres controlent le CHEMIN, pas la
    # donnee ; ce qui garde la donnee est le SNR par tenseur ecrit au manifeste
    # a la conversion.
    #
    # Plancher pose sur la PILE, mesure du 10/09 sur cette configuration :
    #
    #     n      pile q_a   pile kv_a   noyaux separes   chemin sabote
    #     16       51,1        50,4          49,3            0,22
    #     32       51,1        50,4          49,3              —
    #     64      333,1       365,2          28,3            0,22
    #    128      333,1       365,1          28,2              —
    #
    # La pile deroule vers bf16 au-dela du seuil, d'ou ses 333 dB ; les noyaux
    # separes y paient 28 dB par les tensor cores FP4, soit ~4 % RMS — le prix
    # annonce du format, pas un defaut, et c'est pourquoi le plancher porte sur
    # la pile et non sur eux. 20 dB laisse 30 dB de marge au pire regime mesure.
    obtenu = obtenu.float()
    PLANCHER_DB = 20.0
    debut = 0
    for i, largeur in enumerate((768, 576)):
        a_ = obtenu[:, debut:debut + largeur]
        b_ = exact[:, debut:debut + largeur]
        bruit = (a_ - b_).pow(2).mean()
        snr = 10 * torch.log10(b_.pow(2).mean() / bruit.clamp(min=1e-30)).item()
        assert snr >= PLANCHER_DB, (
            f"n={n}, bloc {i} ({largeur} sorties) : SNR de la pile "
            f"{snr:.2f} dB sous le plancher declare de {PLANCHER_DB} dB. "
            f"Attendu ~50 dB sous le seuil GEMV, ~333 dB au-dela. Une chute "
            f"a ce point signe une echelle prise au mauvais bloc — le defaut "
            f"de GLM-4.7 du 6/09, qui rend 0,22 dB et que la comparaison "
            f"pile/separees ne voit PAS puisque les deux lisent la meme "
            f"echelle fausse.")
        debut += largeur

    if n <= _NVFP4_GEMV_MAX:
        # En plus du plancher : la fusion ne doit rien changer aux bits que
        # rendent les GEMV separees.
        assert torch.equal(fus(x), torch.cat([l(x) for l in lins], dim=-1))
