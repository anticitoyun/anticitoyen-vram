"""La recurrence de l'attention lineaire se deroule-t-elle pareil des deux facons ?

Le harnais de perplexite deroule une fenetre entiere d'un coup ; le service va
jeton par jeton en propageant l'etat. Si les deux ne rendent pas la meme suite
de sorties, ils ne mesurent pas le meme modele — et un ecart de perplexite
entre acvram et llama.cpp serait imputable a cela plutot qu'au format.

Ecrit le 8/09/2026 pendant la recherche d'un facteur quatre sur un modele
hybride, alors que phi-4 dense s'accordait a llama.cpp a 4,7 % pres. La couche
est innocentee par ce test : il reste comme garde, pour qu'une optimisation
future du chemin en bloc ne s'ecarte pas du chemin pas a pas sans le dire.
"""
import pytest
import torch
import torch.nn as nn

from acvram.engine.kda import KimiDeltaAttention

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="pas de GPU")

CACHE, TETES, DIM_TETE, NOYAU = 256, 4, 32, 4
INTERNE = TETES * DIM_TETE


def _couche(dev):
    torch.manual_seed(5)

    def lin(i, o):
        m = nn.Linear(i, o, bias=False, device=dev, dtype=torch.bfloat16)
        with torch.no_grad():
            m.weight.mul_(0.05)
        return m

    def conv():
        return torch.randn(INTERNE, NOYAU, device=dev, dtype=torch.bfloat16) * 0.2

    return KimiDeltaAttention(
        lin(CACHE, INTERNE), lin(CACHE, INTERNE), lin(CACHE, INTERNE),
        lin(INTERNE, CACHE),
        lin(CACHE, TETES), lin(TETES, INTERNE),
        lin(CACHE, TETES), lin(TETES, INTERNE), lin(CACHE, TETES),
        conv(), conv(), conv(),
        torch.zeros(INTERNE, device=dev),
        -torch.rand(TETES, device=dev) - 0.5,
        torch.ones(DIM_TETE, device=dev, dtype=torch.bfloat16),
        TETES, DIM_TETE).to(dev)


def _ecart(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))


@CUDA
def test_bloc_et_pas_a_pas_concordent():
    """Une fenetre deroulee d'un coup contre la meme, jeton par jeton."""
    dev = torch.device("cuda:0")
    la = _couche(dev)
    h = torch.randn(64, CACHE, device=dev, dtype=torch.bfloat16) * 0.5

    with torch.no_grad():
        bloc, _ = la(h, None)
        etat, morceaux = None, []
        for t in range(h.shape[0]):
            y, etat = la(h[t:t + 1], etat)
            morceaux.append(y)
        pas = torch.cat(morceaux)

    # bf16 porte huit bits de mantisse ; l'etat, lui, est tenu en float32
    # (kda.py:132). Quelques 1e-3 sont l'arrondi des projections, pas une
    # derive de la recurrence — une derive croitrait avec la longueur.
    assert _ecart(pas, bloc) < 1e-2


@CUDA
def test_la_reprise_d_etat_ne_perd_rien():
    """Couper la sequence en deux et reprendre doit valoir la traiter entiere.

    C'est la propriete dont depend le service : un jeton decode apres un
    prefill doit voir exactement l'etat qu'aurait vu un prefill continu.
    """
    dev = torch.device("cuda:0")
    la = _couche(dev)
    h = torch.randn(64, CACHE, device=dev, dtype=torch.bfloat16) * 0.5

    with torch.no_grad():
        entier, _ = la(h, None)
        moitie, etat = la(h[:32], None)
        suite, _ = la(h[32:], etat)

    assert _ecart(torch.cat([moitie, suite]), entier) < 5e-3
