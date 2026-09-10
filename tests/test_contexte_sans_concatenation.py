"""Deux travaux proportionnels au contexte, faits a chaque pas pour rien.

`Sequence.all_ids` CONCATENE `prompt_ids` et `output_ids` : une liste neuve de
la taille du contexte entier, a chaque appel. Elle etait appelee deux fois par
sequence et par pas de decodage :

* `_register_complete_blocks` la construisait pour en lire SEIZE elements ;
* `_emit` la passait a `sample`, qui rend son argmax AVANT de la lire quand
  tout le lot est glouton et sans penalite — le cas courant.

A 120 jetons de contexte ce travail ne se voit pas ; a 4000 il grandit a
chaque jeton produit. Ces epreuves verifient que le raccourci rend exactement
la meme chose que la concatenation qu il remplace, y compris sur la frontiere
entre invite et sortie — le seul endroit ou une tranche peut se tromper.
"""
import pytest

from acvram.engine.runner import Sequence, _tranche
from acvram.engine.sampler import SamplingParams, besoin_historique


def _seq(n_invite, n_sortie):
    s = Sequence(id=1, prompt_ids=list(range(100, 100 + n_invite)),
                 params=SamplingParams())
    s.output_ids = list(range(900, 900 + n_sortie))
    return s


@pytest.mark.parametrize("n_invite,n_sortie", [(40, 0), (40, 7), (0, 40),
                                               (1, 1), (16, 16), (37, 45)])
def test_la_tranche_rend_ce_que_rendait_la_concatenation(n_invite, n_sortie):
    """Toutes les tranches possibles, comparees a `all_ids[a:b]`."""
    s = _seq(n_invite, n_sortie)
    total = n_invite + n_sortie
    for a in range(total + 1):
        for b in range(a, total + 2):
            assert _tranche(s, a, b) == tuple(s.all_ids[a:b]), (a, b)


def test_la_tranche_traverse_la_frontiere():
    """Le cas que la concatenation masquait : a cheval sur les deux listes.

    Sans ce test, une implementation qui ne lirait QUE l invite passerait les
    cas ou la tranche tombe entierement d un cote.
    """
    s = _seq(20, 20)
    t = _tranche(s, 16, 32)
    assert t == (116, 117, 118, 119, 900, 901, 902, 903,
                 904, 905, 906, 907, 908, 909, 910, 911)
    assert len(t) == 16


def test_l_historique_n_est_pas_lu_en_glouton():
    """La condition qui decide de CONSTRUIRE doit etre celle qui LIT.

    Elle vit dans `sampler` pour cette raison : deux copies de la meme regle
    finissent par diverger, et l historique serait alors soit construit pour
    rien, soit absent quand une penalite le reclame.
    """
    glouton = SamplingParams(temperature=0.0)
    assert besoin_historique([glouton, glouton]) is False
    # `SamplingParams()` par DEFAUT n'est pas glouton : sa temperature vaut
    # 1,0. Le raccourci ne joue donc que pour un client qui demande
    # explicitement une temperature nulle — a temperature non nulle
    # l'historique est lu, et le construire est justifie.
    assert besoin_historique([glouton, SamplingParams()]) is True


def test_une_seule_sequence_non_gloutonne_suffit_a_le_reclamer():
    """Le lot est traite d un bloc : une seule sequence a penalite oblige a
    construire l historique de TOUTES. Le contraire laisserait `sample` lire
    une case vide."""
    lot = [SamplingParams(temperature=0.0) for _ in range(8)]
    assert besoin_historique(lot) is False
    lot[5] = SamplingParams(temperature=1.0)
    assert besoin_historique(lot) is True
