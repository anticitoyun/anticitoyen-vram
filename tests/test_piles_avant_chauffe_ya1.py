"""ya1 (29/09) : sans graphes, les piles d experts (et le repack Marlin) se construisaient dans la 1re passe de chauffe,
leur transitoire s ajoutant au préfill plein (Coder-30B qkvo-i8c : 3 072 tenus sur 4 096, 0.7.12 comme 0.7.13).
`demarrer_service` les construit désormais AVANT la chauffe, comme `GraphRunner._eligible` avec graphes. À sec."""
import torch
import torch.nn as nn

from acvram.engine import contexte
from acvram.engine.moe import MoEBlock


class _FauxMoE(MoEBlock):
    """Comme les vrais : AUCUN paramètre ni buffer (les poids de QuantLinear sont des attributs ordinaires)."""
    def __init__(self, journal):
        nn.Module.__init__(self)
        self._stack_state = "?"
        self._journal = journal

    def _try_build_stacks(self):
        self._journal.append("piles")
        return True


class _Couche(nn.Module):
    def __init__(self, bloc, device):
        super().__init__()
        self.mlp = bloc
        self.device = torch.device(device)


class _FauxModele(nn.Module):
    def __init__(self, blocs, device):
        super().__init__()
        self.layers = nn.ModuleList([_Couche(b, device) for b in blocs])


class _FauxMoteur:
    def __init__(self, journal, n=3, device="cuda:0"):
        self.blocs = [_FauxMoE(journal) for _ in range(n)]
        self.model = _FauxModele(self.blocs, device)
        self.graphs = None
        self._journal = journal

    def chauffer_contexte(self, pas, strict):
        self._journal.append("chauffe")
        return 4096

    _construire_piles_avant_chauffe = contexte.ChauffeContexte._construire_piles_avant_chauffe


def test_piles_construites_avant_la_chauffe_sans_graphes():
    # Casse si la construction redevient paresseuse (dans le 1er forward de la chauffe), ou si la garde regarde les
    # paramètres du bloc (aucun sur un vrai MoE : 1re version, bras B de la prise ya1 identique au témoin)
    journal = []
    m = _FauxMoteur(journal)
    assert contexte.ChauffeContexte.demarrer_service(m) == (4096, 0)
    assert journal == ["piles"] * 3 + ["chauffe"]
    assert {b._stack_state for b in m.blocs} == {"oui"}


def test_hors_carte_rien_n_est_construit():
    journal = []
    m = _FauxMoteur(journal, device="cpu")
    contexte.ChauffeContexte.demarrer_service(m)
    assert journal == ["chauffe"] and {b._stack_state for b in m.blocs} == {"?"}


def test_une_couche_refusee_ne_touche_pas_les_autres():
    journal = []
    m = _FauxMoteur(journal)
    m.blocs[1]._try_build_stacks = lambda: False
    contexte.ChauffeContexte.demarrer_service(m)
    assert [b._stack_state for b in m.blocs] == ["oui", "non", "oui"]
