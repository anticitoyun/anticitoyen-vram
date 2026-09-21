"""`verifier_table`/`construire_table` : le contrat de la table d'adresses
par expert (bead anticitoyen-vram-pds, point 2) — voir la docstring de
`memory/table_adresses.py` pour le contrat complet. Aucune carte : `data_ptr()`
existe pour un tenseur CPU comme pour un tenseur CUDA."""
from __future__ import annotations

import pytest
import torch

from acvram.engine.layers import ExpertPool, QuantLinear
from acvram.engine.model import MLP
from acvram.memory.table_adresses import (adresse_expert, construire_table,
                                          verifier_table)
from acvram.quant.nvfp4 import quantize_nvfp4


def test_table_sans_zero_ne_leve_pas():
    verifier_table(torch.tensor([1, 2, 3], dtype=torch.int64))


def test_table_avec_un_zero_leve():
    with pytest.raises(ValueError, match="jamais 0"):
        verifier_table(torch.tensor([1, 0, 3], dtype=torch.int64))


def test_table_vide_ne_leve_pas():
    verifier_table(torch.zeros(0, dtype=torch.int64))


def _mlp_nvfp4(graine: int, sortie: int = 32, entree: int = 16) -> MLP:
    g = torch.Generator().manual_seed(graine)

    def lin(o, i, k):
        w = torch.randn(o, i, generator=g) * 0.02
        return QuantLinear(quantize_nvfp4(w), out_features=o, in_features=i)
    return MLP(lin(sortie, entree, 1), lin(sortie, entree, 2),
              lin(entree, sortie, 3))


def test_construire_table_une_entree_par_expert():
    experts = [_mlp_nvfp4(g) for g in range(4)]
    qw, bs = construire_table(experts, "gate_proj")
    assert qw.shape == (4,) and bs.shape == (4,)
    assert qw.dtype == torch.int64 and bs.dtype == torch.int64


def test_construire_table_adresses_distinctes_par_expert():
    # Quatre experts, quatre allocations distinctes : la table ne doit pas
    # aliaser deux experts sur la même adresse (un bogue de vue partagée les
    # ferait tous pointer vers le tenseur du dernier construit).
    experts = [_mlp_nvfp4(g) for g in range(4)]
    qw, _ = construire_table(experts, "down_proj")
    assert len(set(qw.tolist())) == 4


def test_construire_table_respecte_l_ordre_des_experts():
    experts = [_mlp_nvfp4(g) for g in range(3)]
    qw, _ = construire_table(experts, "up_proj")
    attendu = [e.up_proj.qweight.qweight.data_ptr() for e in experts]
    assert qw.tolist() == attendu


def test_adresse_expert_froid_lit_le_tampon_epingle_pas_l_original():
    """Régression : `to_device(streamed=True)` construit `.streamed.host` à
    partir d'une COPIE et ne touche jamais `.qweight` — lire `.qweight.qweight`
    pour un expert froid donnerait l'adresse de l'original ORPHELIN, pas
    celle que le noyau doit réellement lire."""
    mlp = _mlp_nvfp4(0)
    original_ptr = mlp.gate_proj.qweight.qweight.data_ptr()
    mlp.gate_proj.to_device(torch.device("cpu"), streamed=True)

    lue = adresse_expert(mlp.gate_proj, "qweight")
    reelle = mlp.gate_proj.streamed.host["qweight"].data_ptr()

    assert lue == reelle
    assert lue != original_ptr


def test_construire_table_mixte_resident_et_froid():
    """Un expert résident et un froid dans la MÊME table : chacun donne son
    adresse réelle, pas celle de l'autre ni celle de l'original du froid."""
    residents = _mlp_nvfp4(1)
    froid = _mlp_nvfp4(2)
    pool = ExpertPool(torch.device("cpu"), 4)
    froid.gate_proj.to_device(torch.device("cpu"), streamed=True, pool=pool)

    qw, _ = construire_table([residents, froid], "gate_proj")

    assert qw[0].item() == residents.gate_proj.qweight.qweight.data_ptr()
    assert qw[1].item() == froid.gate_proj.streamed.host["qweight"].data_ptr()
    assert qw[1].item() != froid.gate_proj.qweight.qweight.data_ptr()
