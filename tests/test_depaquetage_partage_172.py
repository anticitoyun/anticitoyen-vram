"""Pièce 172 (B') : au préfill de plusieurs séquences, la boucle par séquence d'une couche à récurrence linéaire partage
le poids déquantifié de chaque linéaire NVFP4 (`kernels.depaquetage_partage`). AU BIT contre le chemin sans partage
(règle 9, `torch.equal`), sur des formes RÉELLES (Qwen3.8 : 6 144 × 5 120, 8 × 78 lignes et longueurs mêlées), là où
cuBLAS bf16 découpe sa réduction selon M (pièce 169).

Témoin de sensibilité, dans le même test : la GEMM GROUPÉE (le lot en un appel, ce que fait GDN_PREFILL_LOT=1) doit
DIFFÉRER du chemin par séquence sur ces formes — sinon le contrôle ne verrait pas qu'on a regroupé les GEMM, et le test
casse. Le test de couche casse aussi si la boucle de `DecoderLayerGDN` groupe les GEMM au lieu de partager le poids."""
import pytest
import torch
import torch.nn as nn

from acvram import kernels
from acvram.engine import couches
from acvram.engine.couches import DecoderLayerGDN
from acvram.engine.layers import QuantLinear
from acvram.engine.lot import ForwardBatch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuBLAS et déquantification sur carte")
DEV = torch.device("cuda:0")
N, K = 6144, 5120
COMPOSITIONS = {"C1": [78] * 8, "C2": [40, 78, 120, 200, 33, 90, 150, 64]}


@pytest.fixture(autouse=True)
def _sans_grad():
    with torch.no_grad():
        yield


def _lineaire(marlin: bool):
    from acvram.quant.nvfp4 import quantize_nvfp4
    torch.manual_seed(172)
    lin = QuantLinear(quantize_nvfp4(torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.02))
    if marlin:
        from acvram.kernels import marlin_port as MP
        if kernels.get_extension() is None or MP.charger(compiler=False) is None:
            pytest.skip("extension ou port Marlin absents")
        boite = nn.Module()
        boite.proj = lin
        if kernels.preparer_disposition_marlin(boite).get("seuls", 0) != 1:
            pytest.skip("disposition Marlin refusée pour ce poids")
    return lin


def _x(lens):
    g = torch.Generator(device="cpu").manual_seed(sum(lens))
    return (torch.randn(sum(lens), K, generator=g)).to(DEV, torch.bfloat16)


def _par_sequence(lin, x, lens):
    out, d = [], 0
    for n in lens:
        out.append(lin(x[d:d + n])); d += n
    return torch.cat(out)


@pytest.mark.parametrize("marlin", [False, True], ids=["naturel", "marlin"])
@pytest.mark.parametrize("comp", list(COMPOSITIONS))
def test_partage_au_bit_et_temoin_de_groupement(monkeypatch, marlin, comp):
    lens = COMPOSITIONS[comp]
    lin, x = _lineaire(marlin), _x(lens)
    monkeypatch.setattr(kernels, "_DEPAQ_PARTAGE", False)
    ref = _par_sequence(lin, x, lens)
    monkeypatch.setattr(kernels, "_DEPAQ_PARTAGE", True)
    c0 = dict(kernels.CHEMINS_NVFP4)
    with kernels.depaquetage_partage():
        bp = _par_sequence(lin, x, lens)
    fab = kernels.CHEMINS_NVFP4["depaquetage_partage_fabrique"] - c0.get("depaquetage_partage_fabrique", 0)
    reu = kernels.CHEMINS_NVFP4["depaquetage_partage_reutilise"] - c0.get("depaquetage_partage_reutilise", 0)
    assert (fab, reu) == (1, len(lens) - 1), (fab, reu)
    assert torch.equal(bp, ref)
    groupe = lin(x)
    assert not torch.equal(groupe, ref), "témoin aveugle : la GEMM groupée rend les mêmes bits sur ces formes"


class _AttnLineaire(nn.Module):
    """Couche à récurrence factice : une projection NVFP4 par séquence (ce que fait GatedDeltaNet.forward), et un
    `forward_lot` qui la groupe (ce que fait GDN_PREFILL_LOT=1)."""

    def __init__(self, lin):
        super().__init__()
        self.lin = lin

    def forward(self, x, etat):
        return self.lin(x)[:, :K], etat

    def forward_lot(self, x, etats, query_lens):
        return self.lin(x)[:, :K], etats


def _passe(couche, x, lens):
    b = ForwardBatch(tokens=torch.zeros(sum(lens), dtype=torch.long), positions=torch.zeros(sum(lens), dtype=torch.long),
                     seq_lens=lens, query_lens=lens, block_tables=[], slot_mapping=torch.zeros(sum(lens), dtype=torch.long),
                     is_prefill=True, seq_ids=list(range(len(lens))), gdn_store={})
    return couche(x, b)


def test_couche_gdn_partage_au_bit(monkeypatch):
    lens = COMPOSITIONS["C1"]
    lin, x = _lineaire(False), _x(lens)
    ident = nn.Identity()
    couche = DecoderLayerGDN(index=0, gdn=_AttnLineaire(lin), mlp=None, input_norm=ident, post_norm=ident, device=DEV)
    monkeypatch.setattr(couches, "_GDN_PREFILL_LOT", False)
    monkeypatch.setattr(kernels, "_DEPAQ_PARTAGE", False)
    ref = _passe(couche, x, lens)
    monkeypatch.setattr(kernels, "_DEPAQ_PARTAGE", True)
    c0 = kernels.CHEMINS_NVFP4["depaquetage_partage_reutilise"]
    assert torch.equal(_passe(couche, x, lens), ref)
    assert kernels.CHEMINS_NVFP4["depaquetage_partage_reutilise"] - c0 == len(lens) - 1, "partage non pris par la couche"
    monkeypatch.setattr(couches, "_GDN_PREFILL_LOT", True)            # GEMM groupée : doit différer
    assert not torch.equal(_passe(couche, x, lens), ref)


def test_une_seule_sequence_ne_garde_rien(monkeypatch):
    """Portée fermée pour UNE séquence (chef, 25/09) : aucun poids retenu, aucun pic en plus au long préfill."""
    lin, x = _lineaire(False), _x([300])
    ident = nn.Identity()
    couche = DecoderLayerGDN(index=0, gdn=_AttnLineaire(lin), mlp=None, input_norm=ident, post_norm=ident, device=DEV)
    monkeypatch.setattr(couches, "_GDN_PREFILL_LOT", False)
    monkeypatch.setattr(kernels, "_DEPAQ_PARTAGE", True)
    c0 = kernels.CHEMINS_NVFP4["depaquetage_partage_fabrique"]
    _passe(couche, x, [300])
    assert kernels.CHEMINS_NVFP4["depaquetage_partage_fabrique"] == c0
