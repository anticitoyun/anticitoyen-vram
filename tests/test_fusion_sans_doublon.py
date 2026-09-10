"""Les quatre empileurs doivent rendre les originaux en VUES, pas en copies.

Le 10/09/2026, `model.nbytes` sur Llama-2-7b-int8 rendait 7 223 386 112 octets
pour 6 761 930 752 sur disque : 461 455 360 de trop, exactement cinq gate_up
fusionnes a 92 291 072. `MLP.fuse()` construit la pile et NE LIBERE PAS
`gate_proj`/`up_proj` — a raison, puisque `forward` retombe sur eux au-dela de
SEUIL_FUSION. Les deux chemins sont necessaires ; les deux COPIES ne l'etaient
pas. `stack_plain_linears` et `stack_nvfp4_linears` repointaient deja les
originaux en vues, avec l'argument ecrit dans leur docstring ; `stack_int8` et
`stack_int4_awq` ne le faisaient que pour le BIAIS, et le commentaire de l'int8
affirmait pourtant « comme pour les poids ».

Cette epreuve tourne sur CPU : elle verifie le PARTAGE DE STOCKAGE, qui est une
propriete de disposition et non de materiel.
"""
import torch

from acvram.engine.layers import (QuantLinear, stack_int8_linears,
                                  stack_int4_awq_linears,
                                  stack_plain_linears)
from acvram.quant.formats import INT8Tensor, PlainTensor
from acvram.quant.int4 import INT4Tensor


def _int8(out, inn, groupe=128):
    ng = inn // groupe
    return INT8Tensor(torch.randint(0, 255, (out, inn), dtype=torch.uint8),
                      torch.rand(out, ng, dtype=torch.float16) + 0.01,
                      torch.randint(0, 255, (out, ng), dtype=torch.uint8),
                      groupe, (out, inn))


def _int4(out, inn, groupe=128):
    ng = inn // groupe
    return INT4Tensor(torch.randint(0, 255, (out, inn // 2), dtype=torch.uint8),
                      torch.rand(out, ng, dtype=torch.float16) + 0.01,
                      torch.randint(0, 255, (out, ng // 2), dtype=torch.uint8),
                      groupe, (out, inn), inn)


def _octets_int8(t):
    return t.qweight.numel() + t.scales.numel() * 2 + t.zeros.numel()


def test_int8_les_originaux_partagent_le_stockage_de_la_pile():
    a, b = _int8(256, 512), _int8(256, 512)
    la, lb = QuantLinear(a, out_features=256, in_features=512), \
        QuantLinear(b, out_features=256, in_features=512)
    avant = _octets_int8(a) + _octets_int8(b)
    pile = stack_int8_linears([la, lb])
    assert pile is not None, "l'empilement int8 a refuse un cas legitime"

    base = pile.qweight.qweight.untyped_storage().data_ptr()
    for l in (la, lb):
        assert l.qweight.qweight.untyped_storage().data_ptr() == base, \
            "l'original ne partage pas le stockage de la pile : il en est une COPIE"
        for champ in ("scales", "zeros"):
            assert (getattr(l.qweight, champ).untyped_storage().data_ptr()
                    == getattr(pile.qweight, champ).untyped_storage().data_ptr()), \
                f"{champ} duplique"
        # une tranche d'un `cat` sur l'axe 0 reste contigue : les noyaux
        # l'exigent, et c'est ce qui rend la vue utilisable
        assert l.qweight.qweight.is_contiguous(), "la vue n'est pas contigue"

    # Le compte qui a revele le defaut : sans les vues, le total valait le
    # double des originaux plus la pile.
    assert _octets_int8(pile.qweight) == avant, "la pile ne fait pas la somme"


def test_int8_les_vues_rendent_les_memes_valeurs():
    """Une vue doit etre exacte, pas seulement econome."""
    a, b = _int8(64, 256), _int8(32, 256)
    qa, qb = a.qweight.clone(), b.qweight.clone()
    la, lb = QuantLinear(a, out_features=64, in_features=256), \
        QuantLinear(b, out_features=32, in_features=256)
    assert stack_int8_linears([la, lb]) is not None
    assert torch.equal(la.qweight.qweight, qa)
    assert torch.equal(lb.qweight.qweight, qb)
    assert la.qweight.shape == (64, 256) and lb.qweight.shape == (32, 256)


def test_int4_awq_les_originaux_partagent_le_stockage():
    a, b = _int4(256, 512), _int4(256, 512)
    la, lb = QuantLinear(a, out_features=256, in_features=512), \
        QuantLinear(b, out_features=256, in_features=512)
    pile = stack_int4_awq_linears([la, lb])
    assert pile is not None, "l'empilement int4_awq a refuse un cas legitime"
    base = pile.qweight.qweight.untyped_storage().data_ptr()
    for l in (la, lb):
        assert l.qweight.qweight.untyped_storage().data_ptr() == base, \
            "int4_awq duplique encore les poids"
        assert l.qweight.qweight.is_contiguous()


def test_bf16_partageait_deja_le_stockage():
    """Temoin : ce chemin-la etait deja juste, et l'epreuve doit le confirmer —
    sinon elle ne mesurerait que ce que je viens d'ecrire."""
    a = PlainTensor(torch.randn(128, 256, dtype=torch.bfloat16), (128, 256), "bf16")
    b = PlainTensor(torch.randn(128, 256, dtype=torch.bfloat16), (128, 256), "bf16")
    la, lb = QuantLinear(a, out_features=128, in_features=256), \
        QuantLinear(b, out_features=128, in_features=256)
    pile = stack_plain_linears([la, lb])
    assert pile is not None
    base = pile.qweight.weight.untyped_storage().data_ptr()
    assert la.qweight.weight.untyped_storage().data_ptr() == base
    assert lb.qweight.weight.untyped_storage().data_ptr() == base
