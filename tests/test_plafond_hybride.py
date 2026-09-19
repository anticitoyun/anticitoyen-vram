"""Plafond des créneaux hybrides d'un graphe (GDN/KDA/Mamba2/MLA) : suit le
--max-batch du moteur quand ACVRAM_HYBRID_SLOTS n'est pas posé (G1 19/09 :
GLM tombait en eager dès b=5 sous `acvram serve` avec le défaut fixe 4 —
155 t/s au lieu de 568 —, `certifie` posant 12 avant l'import ne le voyait
pas) ; la variable posée garde la main ; le régime du moteur porte le plafond."""
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from acvram.engine.graphs import GraphRunner                                    # noqa: E402


def test_plafond_suit_le_lot_servi_sauf_variable_posee():
    assert GraphRunner.plafond_hybride(12, env="") == 12
    assert GraphRunner.plafond_hybride(1, env="") == 4                  # jamais sous 4 (l'ancien défaut)
    assert GraphRunner.plafond_hybride(None, env="") == 4
    assert GraphRunner.plafond_hybride(12, env="4") == 4                # la variable posée garde la main
    assert GraphRunner.plafond_hybride(2, env="16") == 16


def test_le_regime_porte_le_plafond():
    src = (pathlib.Path(__file__).resolve().parent.parent / "acvram" / "engine" / "runner.py").read_text()
    assert "max_batch_size=self.max_batch_size" in src                  # le GraphRunner reçoit le lot servi
    assert "hybrides≤" in src                                            # visible dans regime_ligne()
