"""Critère d'équivalence par position (Sage, revue/sage-glm-equivalence-
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


def test_ulp_bf16_correspond_aux_valeurs_citees_par_sage():
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
    """Cumulatif (§ 3) : cos excellent ne suffit pas si delta dépasse
    largement le seuil recalé (5 ulp — voir MULTIPLICATEUR_ULP)."""
    v = verdict_position(delta=1.0, echelle_ref=10.0,   # seuil = 5*0.0625 = 0.3125
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
    """Manon, capture initiale (routage en softmax, pas sigmoid) :
    position 5, top-1 identique (42589) mais delta=5,01 et cos=0,954 —
    échoue largement sur delta ET cos, indépendamment du multiplicateur
    ulp (pas une histoire d'ex-aequo à la frontière)."""
    v = verdict_position(delta=5.0124, echelle_ref=21.125,
                         top1_ref=42589, top1_nous=42589, cos=0.954475,
                         ecart_top1_top2_ref=1.5)
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


def test_verdict_global_casse_sur_le_bogue_softmax():
    """Quatre positions RÉELLES de la capture avant le correctif routage
    sigmoid (Manon, logits-acvram.json vs logits-hf.json ; valeurs
    mesurées, pas reconstruites) : 3 ex-aequo (1, 10, 15 — gap réf ≤ 5
    ulp) et une franche (5, top-1 identique mais delta=5,01/cos=0,954).
    Même avec le seuil recalé et jusqu'à 2 ex-aequo tolérés, le verdict
    global reste RÉFUTÉ (soit par la position 5, soit par le compte
    d'ex-aequo) — REGLES §4 bis, témoin négatif."""
    lignes = [
        (77199, 1242, 2.0248, 0.997028, 25.2500, 0.3750),   # position 1
        (42589, 42589, 5.0124, 0.954475, 21.1250, 1.5000),  # position 5
        (96914, 90439, 0.8620, 0.999905, 24.6250, 0.0312),  # position 10
        (39649, 27608, 3.1620, 0.970630, 14.5625, 0.2500),  # position 15
    ]
    verdicts = [verdict_position(delta, echelle, t1r, t1n, cos, gap)
               for t1r, t1n, delta, cos, echelle, gap in lignes]
    ok, raison = verdict_global(verdicts)
    assert not ok, raison


def test_temoin_repli_int4_awq_tiers_hote():
    """mini-acvram3 (tiering.py:372 avant correctif) : delta=6,64,
    cos=0,993 mesurés (revue/equivalence-glm-14-09.md) — même un top-1
    identique échoue largement sur delta ET sur cos, aucune ambiguïté."""
    v = verdict_position(delta=6.64, echelle_ref=16.625,
                         top1_ref=77199, top1_nous=77199, cos=0.993,
                         ecart_top1_top2_ref=0.375)
    assert not v.ok and not v.ex_aequo
