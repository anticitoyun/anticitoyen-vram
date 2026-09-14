"""Critère d'équivalence par position (poste7, revue/poste7-glm-equivalence-
15-09.md § 2, précisé § 3 : cumulatif, échelle = max_j |logit_ref,j| de
la ligne). Les trois témoins négatifs sont les VRAIES fautes du 15/09 :
softmax au lieu de sigmoid, routeur bf16 sans le biais en fp32, repli
int4_awq du tiers hôte à sec. Chiffres pris sur les captures réelles de
l'équivalence GLM-4.7-Flash (positions dont l'écart top-1/top-2 de
référence est GRAND — pas un ex-aequo, un vrai désaccord)."""

import pytest

from acvram.quant.equivalence import (
    ulp_bf16,
    verdict_global,
    verdict_position,
)


def test_ulp_bf16_correspond_aux_valeurs_citees_par_poste7():
    # "sur des logits de 10-30, l'ulp bf16 vaut 0,0625-0,25"
    assert ulp_bf16(10.0) == pytest.approx(0.0625)
    assert ulp_bf16(28.0) == pytest.approx(0.125)


def test_position_normale_passe():
    v = verdict_position(delta=0.05, echelle_ref=15.0,
                         top1_ref=7, top1_nous=7, cos=0.99999,
                         ecart_top1_top2_ref=2.0)
    assert v.ok and not v.ex_aequo


def test_ex_aequo_prouve_passe_malgre_un_top1_different():
    v = verdict_position(delta=3.0, echelle_ref=10.0,
                         top1_ref=5, top1_nous=9, cos=0.9, ecart_top1_top2_ref=0.05)
    assert v.ok and v.ex_aequo


def test_top1_different_sans_ex_aequo_prouve_echoue():
    v = verdict_position(delta=3.0, echelle_ref=10.0,
                         top1_ref=5, top1_nous=9, cos=0.9, ecart_top1_top2_ref=5.0)
    assert not v.ok and not v.ex_aequo


def test_delta_seul_hors_seuil_echoue_meme_cos_bon():
    """Cumulatif (§ 3) : cos excellent ne suffit pas si delta dépasse."""
    v = verdict_position(delta=0.3, echelle_ref=10.0,   # seuil = 2*0.0625 = 0.125
                         top1_ref=3, top1_nous=3, cos=0.99999,
                         ecart_top1_top2_ref=None)
    assert not v.ok and not v.ex_aequo


def test_plus_de_deux_ex_aequo_refute_globalement():
    passe = verdict_position(delta=0.01, echelle_ref=1.0, top1_ref=0,
                             top1_nous=0, cos=0.9999, ecart_top1_top2_ref=None)
    ex_aequo = verdict_position(delta=3.0, echelle_ref=10.0, top1_ref=5,
                                top1_nous=9, cos=0.9, ecart_top1_top2_ref=0.01)
    ok, raison = verdict_global([passe] + [ex_aequo] * 3)
    assert not ok
    assert "3 ex-aequo" in raison


# -- témoins négatifs : les trois fautes réelles du 15/09 ------------------

def test_temoin_softmax_au_lieu_de_sigmoid():
    """poste2, capture initiale (routage en softmax, pas sigmoid) :
    position 1, top-1 réellement différent (acvram=1242, hf=77199),
    échelle de référence 16,625, écart top-1/top-2 de référence 0,375 —
    au-dessus de 2 ulp à cette échelle (2×0,125=0,25) : pas un ex-aequo,
    c'est le bogue."""
    v = verdict_position(delta=8.0, echelle_ref=16.625,
                         top1_ref=77199, top1_nous=1242, cos=0.9,
                         ecart_top1_top2_ref=0.375)
    assert not v.ok and not v.ex_aequo


def test_temoin_routeur_bf16_sans_biais_fp32():
    """Après le correctif routeur fp32 SEUL (biais encore en bf16, avant
    prediction-biais-fp32-14-09) : position 3, top-1 différent
    (acvram=3559, hf=16949), échelle 14,25, écart référence 0,625 —
    toujours pas un ex-aequo."""
    v = verdict_position(delta=2.31, echelle_ref=14.25,
                         top1_ref=16949, top1_nous=3559, cos=0.994,
                         ecart_top1_top2_ref=0.625)
    assert not v.ok and not v.ex_aequo


def test_temoin_repli_int4_awq_tiers_hote():
    """mini-acvram3 (tiering.py:372 avant correctif) : delta=6,64,
    cos=0,993 mesurés (revue/equivalence-glm-14-09.md) — même un top-1
    identique échoue largement sur delta ET sur cos, aucune ambiguïté."""
    v = verdict_position(delta=6.64, echelle_ref=16.625,
                         top1_ref=77199, top1_nous=77199, cos=0.993,
                         ecart_top1_top2_ref=0.375)
    assert not v.ok and not v.ex_aequo
