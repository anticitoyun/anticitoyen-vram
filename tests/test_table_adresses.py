"""`verifier_table` : le contrat de la table d'adresses par expert (bead
anticitoyen-vram-pds, point 2) ne tolère jamais une entrée nulle — voir la
docstring de `memory/table_adresses.py` pour le contrat complet."""
from __future__ import annotations

import pytest
import torch

from acvram.memory.table_adresses import verifier_table


def test_table_sans_zero_ne_leve_pas():
    verifier_table(torch.tensor([1, 2, 3], dtype=torch.int64))


def test_table_avec_un_zero_leve():
    with pytest.raises(ValueError, match="jamais 0"):
        verifier_table(torch.tensor([1, 0, 3], dtype=torch.int64))


def test_table_vide_ne_leve_pas():
    verifier_table(torch.zeros(0, dtype=torch.int64))
