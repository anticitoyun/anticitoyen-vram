"""Pièce 201 (décision chef) : la tranche des replis nvfp4 (pièce 153) ne s'active que pour une copie fp32 entière
au-delà de 1 Gio — la tête d'un vocabulaire étendu, en PPL. Au seuil de `_DEQUANT_TRANCHE_MAX` seul, les projections
servies étaient tranchées au préfill et la sortie changeait (diag201, mixte 8 × 512 : logits ≠ main). Preuve sur le
modèle : scratchpad/poste5-p201-25-09 (sha256 des logits, bras ACVRAM_TRANCHE_COPIE_MIN=0)."""
import pytest

from acvram import kernels

SERVIES = {                                   # [N, K] des plus grandes projections nvfp4 servies
    "Qwen3.8 gate+up fusionné": (2 * 17408, 5120),
    "Llama-3.3-70B gate (non fusionné)": (28672, 8192),
    "gemma-4-31B gate+up fusionné": (2 * 21504, 5376),
}
TETES = {"Qwen3.8 (vocab étendu)": (248320, 5120), "gemma-4-31B": (262144, 5376)}


@pytest.mark.parametrize("nom", SERVIES)
def test_aucune_projection_servie_n_est_tranchee(nom):
    assert not kernels._tranche_copie(*SERVIES[nom]), f"{nom} tranché : la sortie du préfill servi changerait"


@pytest.mark.parametrize("nom", TETES)
def test_la_tete_d_un_vocabulaire_etendu_l_est(nom):
    assert kernels._tranche_copie(*TETES[nom]), f"{nom} non tranché : la PPL de la 153 retombe en OOM (4,74 Gio)"


def test_le_temoin_a_zero_tranche_tout(monkeypatch):
    monkeypatch.setattr(kernels, "_TRANCHE_COPIE_MIN", 0)
    assert all(kernels._tranche_copie(*nk) for nk in SERVIES.values())
