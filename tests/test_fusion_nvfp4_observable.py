"""Un refus de fusion doit se voir.

`stack_nvfp4_linears` rendait `None` par cinq chemins differents, tous muets.
Le 9/09/2026, la question « les fusions s appliquent-elles au nvfp4 ? » a failli
partir en comptage de noyaux sous `ncu` alors que le code pouvait repondre.
Une absence n est un resultat que si l instrument pouvait rendre autre chose.
"""
import torch

from acvram.engine.layers import (QuantLinear, bilan_fusion_nvfp4,
                                  stack_nvfp4_linears)
from acvram.quant.nvfp4 import quantize_nvfp4


def _lin(sortie, entree, graine):
    g = torch.Generator().manual_seed(graine)
    w = (torch.randn(sortie, entree, generator=g) * 0.02).to(torch.bfloat16)
    return QuantLinear(quantize_nvfp4(w))


def test_les_formes_de_qwen25_14b_fusionnent():
    """q/k/v et gate/up du temoin : si l une refusait, le decodage paierait
    cinq lancements par couche au lieu de trois."""
    avant = dict(bilan_fusion_nvfp4())
    qkv = [_lin(5120, 5120, 1), _lin(1024, 5120, 2), _lin(1024, 5120, 3)]
    assert stack_nvfp4_linears(qkv) is not None, "q/k/v refuse"
    gate_up = [_lin(13824, 5120, 4), _lin(13824, 5120, 5)]
    assert stack_nvfp4_linears(gate_up) is not None, "gate/up refuse"
    assert bilan_fusion_nvfp4() == avant, \
        f"refus enregistre alors que les deux fusions ont reussi : {bilan_fusion_nvfp4()}"


def test_un_refus_est_enregistre_avec_sa_raison():
    """Le bilan doit nommer la cause, pas seulement compter."""
    avant = bilan_fusion_nvfp4().get("entrees de tailles differentes", 0)
    assert stack_nvfp4_linears([_lin(64, 128, 6), _lin(64, 256, 7)]) is None
    apres = bilan_fusion_nvfp4().get("entrees de tailles differentes", 0)
    assert apres == avant + 1, f"refus non compte : {bilan_fusion_nvfp4()}"


def test_le_temoin_est_lui_aussi_enregistre(monkeypatch):
    """Couper la fusion pour mesurer ne doit pas se confondre avec un echec."""
    monkeypatch.setenv("ACVRAM_FUSION_NVFP4", "0")
    avant = bilan_fusion_nvfp4().get("temoin ACVRAM_FUSION_NVFP4=0", 0)
    assert stack_nvfp4_linears([_lin(64, 128, 8), _lin(64, 128, 9)]) is None
    assert bilan_fusion_nvfp4()["temoin ACVRAM_FUSION_NVFP4=0"] == avant + 1
