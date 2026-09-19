"""`exiger_regime_nominal` (outils/regime.py) doit refuser une cellule de
mesure sur un moteur dégradé — bead runner 14/09, précisé le 15/09 (poste7,
revue/poste7-glm-formats-mixtes-15-09.md §2) : `energie.py`/le duel
refusent une cellule dégradée plutôt que de la publier comme si de rien
n'était."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "outils"))
from regime import exiger_regime_nominal  # noqa: E402


class _MoteurFactice:
    def __init__(self, **override):
        self._r = {
            "graphes": True, "graphes_demandes": True, "graphes_raison": "",
            "couches_exilees": 0, "couches_total": 4,
            "experts_exiles": 0, "experts_total": 64,
            "piles_ok": True, "piles_raison": [],
            "cartes": ["cuda:0"], "chemin_moe": "mma-a4",
        }
        self._r.update(override)

    def regime(self):
        return self._r

    def regime_ligne(self):
        return "régime factice pour le test"


def test_nominal_ne_leve_rien():
    exiger_regime_nominal(_MoteurFactice())


def test_piles_ok_false_est_refuse():
    with pytest.raises(RuntimeError, match="repli eager"):
        exiger_regime_nominal(_MoteurFactice(piles_ok=False))


def test_piles_ok_false_nomme_la_cause_dans_le_message():
    """poste7 15/09 : pas "hétérogène" affirmé sans preuve — la cause
    relevée par Engine.regime() (piles_raison) doit apparaître, pas un
    texte générique."""
    with pytest.raises(RuntimeError, match="format PlainTensor uniforme"):
        exiger_regime_nominal(_MoteurFactice(
            piles_ok=False,
            piles_raison=["gate_proj : format PlainTensor uniforme mais non pris en charge"]))


def test_graphes_coupes_alors_que_demandes_est_refuse():
    with pytest.raises(RuntimeError, match="graphes désactivés"):
        exiger_regime_nominal(_MoteurFactice(graphes=False, graphes_raison="OOM"))


def test_couches_exilees_est_refuse():
    with pytest.raises(RuntimeError, match="exilées"):
        exiger_regime_nominal(_MoteurFactice(couches_exilees=1))


def test_plus_d_une_carte_est_refuse():
    with pytest.raises(RuntimeError, match="2 cartes"):
        exiger_regime_nominal(_MoteurFactice(cartes=["cuda:0", "cuda:1"]))


def test_piles_inconnues_autorisees_par_defaut():
    exiger_regime_nominal(_MoteurFactice(piles_ok=None))


def test_piles_inconnues_refusees_si_exige():
    with pytest.raises(RuntimeError, match="non vérifié"):
        exiger_regime_nominal(_MoteurFactice(piles_ok=None),
                              autoriser_piles_inconnues=False)
