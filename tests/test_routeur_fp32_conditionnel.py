"""Le fp32 du routeur MoE (revue/prediction-routeur-fp32-14-09.md) ne doit
coûter un cast + un F.linear fp32 supplémentaires que là où il sert
(sigmoid+biais, GLM-4.7-Flash) — pas sur un modèle softmax sans biais
(Coder-30B), régression trouvée par poste3 le 15/09 : +0,187 ms/pas à b=1."""

import torch

from acvram.engine.layers import QuantLinear
from acvram.engine.model import MLP, MoEBlock
from acvram.quant.formats import PlainTensor


def _plain(out_features, in_features, dtype=torch.bfloat16):
    t = torch.randn(out_features, in_features, dtype=dtype) * 0.02
    return QuantLinear(PlainTensor(t, tuple(t.shape), "bf16"),
                      out_features=out_features, in_features=in_features)


def _moe(scoring, score_bias, n_experts=4, hidden=16):
    router = _plain(n_experts, hidden)
    experts = [MLP(_plain(32, hidden), _plain(32, hidden), _plain(hidden, 32))
              for _ in range(n_experts)]
    return MoEBlock(router, experts, top_k=2, scoring=scoring,
                    score_bias=score_bias)


def test_softmax_sans_biais_reste_en_bf16():
    bloc = _moe("softmax", None)
    x = torch.randn(3, 16, dtype=torch.bfloat16)
    logits = bloc._router_logits(x)
    assert logits.dtype == torch.bfloat16


def test_sigmoid_avec_biais_passe_en_fp32():
    bloc = _moe("sigmoid", torch.zeros(4, dtype=torch.float32))
    x = torch.randn(3, 16, dtype=torch.bfloat16)
    logits = bloc._router_logits(x)
    assert logits.dtype == torch.float32


def test_c15_routeur_compact_suit_le_dtype_du_temoin():
    """C15 niveau 3 (route_logits_fusee) : l'arrondi bf16 des logits dans le
    noyau est rejoué exactement là où `_router_logits` les sort en bf16 —
    softmax, sigmoid sans biais — et pas pour sigmoid + biais (fp32)."""
    x = torch.randn(3, 16, dtype=torch.bfloat16)
    for scoring, biais, arrondi in (("softmax", None, True), ("sigmoid", None, True),
                                    ("sigmoid", torch.zeros(4, dtype=torch.float32), False)):
        bloc = _moe(scoring, biais)
        w, a = bloc._routeur_compact(x)
        assert a == arrondi and w.shape == (4, 16)
        # C15-3c : le MÊME tenseur que _router_logits (cache _router_w), au dtype du témoin
        lg = bloc._router_logits(x)
        assert w is bloc._router_w[lg.dtype] and w.dtype == lg.dtype and (lg.dtype == torch.float32) == (not a)
        assert torch.equal(torch.nn.functional.linear(x.to(w.dtype), w), lg)


def test_sigmoid_sans_biais_reste_en_bf16():
    """Le critère est sigmoid ET biais, pas sigmoid seul (lfm2/lfm2_moe
    routent en sigmoid sans nécessairement avoir de biais — pas de raison
    de leur imposer le coût fp32 sans le bénéfice qui le justifie)."""
    bloc = _moe("sigmoid", None)
    x = torch.randn(3, 16, dtype=torch.bfloat16)
    logits = bloc._router_logits(x)
    assert logits.dtype == torch.bfloat16
