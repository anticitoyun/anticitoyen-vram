"""L'empileur INT4-AWQ doit rendre EXACTEMENT ce que rendent n GEMV.

135 groupes du parc sont parfaitement homogènes en int4_awq et restaient
découpés faute d'empileur : `fuse()` n'essayait qu'int8, nvfp4 et bf16.

Le point qui décidait de la faisabilité : `INT4Tensor.scales` est
`[out, in//group]`, une échelle par ligne de sortie et par groupe. Contrairement
à NVFP4 — dont l'échelle globale scalaire a imposé `global_scale_rows` — il n'y
a rien à concilier entre segments. Ces épreuves le vérifient sur les nombres,
et la dernière retire l'hypothèse pour montrer qu'elle porte quelque chose.
"""
import pytest
import torch

from acvram.engine.layers import QuantLinear, stack_int4_awq_linears
from acvram.quant import formats

DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _lin(sortie, entree, graine, dev, biais=False):
    torch.manual_seed(graine)
    w = torch.randn(sortie, entree, dtype=torch.float32) * 0.02
    b = torch.randn(sortie, dtype=torch.bfloat16) * 0.01 if biais else None
    lin = QuantLinear(formats.quantize(w, "int4_awq", group_size=128), bias=b)
    lin.to_device(torch.device(dev))
    return lin


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("biais", [False, True])
def test_la_pile_int4_rend_les_memes_nombres(dev, biais):
    """Tailles distinctes : une permutation ou un mauvais découpage se voit."""
    lins = [_lin(256, 256, 1, dev, biais), _lin(64, 256, 2, dev, biais),
            _lin(64, 256, 3, dev, biais)]
    x = torch.randn(4, 256, dtype=torch.bfloat16, device=dev)
    ref = torch.cat([l(x) for l in lins], dim=-1)

    pile = stack_int4_awq_linears(lins)
    assert pile is not None, "un groupe int4_awq homogène doit s'empiler"
    obtenu = pile(x)

    assert obtenu.shape == ref.shape
    assert torch.equal(obtenu, ref), (
        f"la pile diffère des appels séparés (écart max "
        f"{(obtenu.float() - ref.float()).abs().max().item():.3e})")


@pytest.mark.parametrize("dev", DEVS)
def test_les_segments_gardent_chacun_leur_echelle(dev):
    """Témoin : si les échelles n'étaient pas portées par ligne, écraser
    celles d'un segment ne changerait rien. On les écrase, et on exige que
    l'écart apparaisse — sans quoi l'épreuve du dessus ne démontre rien."""
    lins = [_lin(256, 256, 1, dev), _lin(64, 256, 2, dev), _lin(64, 256, 3, dev)]
    x = torch.randn(4, 256, dtype=torch.bfloat16, device=dev)
    pile = stack_int4_awq_linears(lins)
    avant = pile(x).clone()

    pile.qweight.scales[256:] *= 2.0          # les segments k et v seulement
    apres = pile(x)

    assert torch.equal(avant[:, :256], apres[:, :256]), \
        "toucher k et v a changé q : les échelles ne sont pas par ligne"
    assert not torch.equal(avant[:, 256:], apres[:, 256:]), \
        "doubler les échelles de k et v n'a rien changé : elles ne sont pas lues"


def test_un_groupe_heterogene_est_refuse():
    """La pile ne doit pas se former sur un mélange — elle lirait des octets
    int4 comme s'ils étaient d'un autre format."""
    a = _lin(64, 256, 1, "cpu")
    b = QuantLinear(formats.quantize(
        torch.randn(64, 256) * 0.02, "int8", group_size=128))
    assert stack_int4_awq_linears([a, b]) is None
