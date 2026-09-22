"""Opt-in ± 1 ulp des projections étroites int8 (REGLES § 1, 22/09) :
`ACVRAM_ETROITES_SPLITK` = facteur de programmes par SM du split-K de
`gemm_etroit`. Au bit du DÉFAUT : à F = 2 (vide), `decouper_k` rend
exactement les (tranches, groupes) de l arithmétique d avant, forme par
forme (qkv 80 × 4, o 32 × 11 sur 170 SM) ; un autre F change les tranches ;
la ligne de régime dit `serie` par défaut, `splitk±1ulp(F)` sinon ; le banc
mesure l écart en ulp bf16 contre la référence exacte."""
import importlib.util
import os

import pytest
import torch

from acvram.kernels import gemm_etroit as GE


def _ancienne(ng, tuiles_n, sm):
    voulu = -(-2 * sm // tuiles_n)
    tranches = max(1, min(ng, voulu))
    gpt = -(-ng // tranches)
    return -(-ng // gpt), gpt


@pytest.fixture(autouse=True)
def _defaut(monkeypatch):
    monkeypatch.delenv("ACVRAM_ETROITES_SPLITK", raising=False)
    monkeypatch.setattr(GE, "_FACTEUR", None)
    monkeypatch.setattr(GE, "_programmes", lambda device: 170)      # RTX 5090


def test_defaut_au_bit_de_l_arithmetique_d_avant():
    for N, K in ((5120, 2048), (2048, 4096), (4096, 2048), (1024, 2048), (151936, 2048)):
        ng, tuiles_n = K // 128, -(-N // GE.BN)
        assert GE.decouper_k(ng, tuiles_n, torch.device("cpu")) == _ancienne(ng, tuiles_n, 170), (N, K)
    assert GE.decouper_k(16, 80, torch.device("cpu")) == (4, 4)          # qkv : 80 tuiles × 4 tranches (verdict-detail-etroites)
    assert GE.decouper_k(32, 32, torch.device("cpu")) == (11, 3)         # o : 32 × 11
    assert GE.etroites_texte() == "serie"


def test_facteur_change_les_tranches_et_la_ligne(monkeypatch):
    monkeypatch.setenv("ACVRAM_ETROITES_SPLITK", "4")
    assert GE.facteur_splitk() == 4.0 and GE.etroites_texte() == "splitk±1ulp(F=4)"
    assert GE.decouper_k(16, 80, torch.device("cpu"))[0] > 4 and GE.decouper_k(32, 32, torch.device("cpu"))[0] > 11
    GE.regler_tranches(2.0)
    assert GE.decouper_k(16, 80, torch.device("cpu")) == (4, 4)          # le banc reprend la main
    GE.regler_tranches(None)
    assert GE.decouper_k(16, 80, torch.device("cpu"))[0] > 4


def test_ulp_bf16_du_banc():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outils", "gpu", "mesure", "banc-etroites-splitk.py")
    spec = importlib.util.spec_from_file_location("banc_etroites_splitk", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    y = torch.tensor([1.0, 2.0, 0.75, 1000.0], dtype=torch.bfloat16)
    assert m.ulp_bf16(y).tolist() == [2 ** -7, 2 ** -6, 2 ** -8, 2 ** 2]   # bf16 : 8 bits de mantisse
    ref = torch.tensor([1.0, 2.0, 0.75, 1000.0], dtype=torch.bfloat16)
    un_ulp = torch.tensor([1.0 + 2 ** -7, 2.0, 0.75, 1000.0], dtype=torch.bfloat16)
    e = m.ecart(un_ulp, ref)
    assert e["ulp_max"] == 1.0 and e["part_differents"] == 0.25
    assert m.ecart(ref, ref) == {"ulp_max": 0.0, "part_differents": 0.0, "rel_max": 0.0}


def test_poids_int8_du_banc_sur_l_appareil_demande():
    """poste2 8b68082b : les poids du banc restaient sur l hôte (repli silencieux) → Triton plantait."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outils", "gpu", "mesure", "banc-etroites-splitk.py")
    spec = importlib.util.spec_from_file_location("banc_etroites_splitk", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    from acvram.quant.formats import INT8Tensor
    t = m.poids_int8(256, 512, graine=1, device=torch.device("cpu"))
    assert isinstance(t, INT8Tensor) and t.qweight.shape == (256, 512) and t.group_size == 128
    assert t.qweight.device.type == "cpu" and t.scales.device.type == "cpu" and t.zeros.device.type == "cpu"
