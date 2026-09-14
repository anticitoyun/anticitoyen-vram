"""Critère de décodage (poste7 §7.2, revue/poste7-reprise-15-09-b.md) : au
décodage, un nombre d'ulp fixe ne juge pas (5 ulp échoue 739/768 sur le
témoin lui-même, poste3 f9fce44) — B/A doit dominer un témoin mesuré au
même lancement, pas un seuil absolu."""

import pytest

from acvram.quant.equivalence import verdict_decodage


def test_verdict_narrow_reel_de_poste3_passe():
    """Chiffres réels (revue/verdict-narrow-voyants-15-09.md) : narrow
    A/B (sous test) domine le témoin A-graphes/A-eager sur les trois
    statistiques (med 6,4 vs 7,0 ; p90 12,7 vs 13,4 ; max 48,8 vs 87,8) ;
    la divergence isolée s3@3 (14 ulp, hors ex-aequo) est sous le
    plancher du témoin — pas un bogue démontré. Onze valeurs choisies
    pour que médiane et p90 tombent exactement sur les chiffres publiés
    (indices 5 et 9 sur 11, sans interpolation), pas une reconstitution
    des 768 points bruts."""
    v = verdict_decodage(
        deltas_ulp_temoin=[7.0] * 9 + [13.4, 87.8],
        cos_temoin=[0.99909] * 11,
        deltas_ulp_ba=[6.4] * 9 + [12.7, 48.8],
        divergences_top1_hors_ex_aequo_ba=[14.0],
    )
    assert v.ok and not v.invalide, v.raison
    assert v.temoin["med"] == pytest.approx(7.0)
    assert v.temoin["p90"] == pytest.approx(13.4)
    assert v.ba["med"] == pytest.approx(6.4)
    assert v.ba["p90"] == pytest.approx(12.7)


def test_seuil_fixe_echoue_sur_son_propre_temoin():
    """Motif exact rapporté par poste7 : 5 ulp / cos>=0,9999 échoue
    739/768 sur le témoin — preuve que le seuil fixe ne peut pas être un
    contrôle valide (REGLES §4). Documenté, pas testé via
    verdict_position ici : ce test fixe juste le fait dans le dépôt."""
    temoin = [7.0] * 9 + [13.4, 87.8]
    echecs = sum(1 for d in temoin if d > 5.0)
    assert echecs == len(temoin), "le témoin devrait, comme mesuré, écraser un seuil à 5 ulp"


def test_temoin_hors_bornes_en_ulp_invalide_sans_juger():
    v = verdict_decodage(
        deltas_ulp_temoin=[7.0] * 10 + [151.0],
        cos_temoin=[0.999] * 11,
        deltas_ulp_ba=[1.0] * 11,
    )
    assert v.invalide and not v.ok
    assert "graphes a bougé" in v.raison


def test_temoin_cos_bas_invalide_sans_juger():
    v = verdict_decodage(
        deltas_ulp_temoin=[7.0] * 11,
        cos_temoin=[0.999] * 10 + [0.9975],
        deltas_ulp_ba=[1.0] * 11,
    )
    assert v.invalide and not v.ok


def test_ba_pire_que_le_temoin_est_refute():
    v = verdict_decodage(
        deltas_ulp_temoin=[7.0] * 100,
        cos_temoin=[0.999] * 100,
        deltas_ulp_ba=[20.0] * 100,   # bien au-dessus de 1,2x med/p90 et du max
    )
    assert not v.ok and not v.invalide
    assert "med(B/A)" in v.raison


def test_divergence_top1_au_dela_du_max_temoin_est_refutee():
    v = verdict_decodage(
        deltas_ulp_temoin=[7.0] * 100,
        cos_temoin=[0.999] * 100,
        deltas_ulp_ba=[6.0] * 100,
        divergences_top1_hors_ex_aequo_ba=[9.0],   # > max(temoin)=7.0
    )
    assert not v.ok and not v.invalide
    assert "hors ex-aequo" in v.raison


def test_divergence_top1_sous_le_max_temoin_passe():
    v = verdict_decodage(
        deltas_ulp_temoin=[7.0] * 99 + [50.0],
        cos_temoin=[0.999] * 100,
        deltas_ulp_ba=[6.0] * 100,
        divergences_top1_hors_ex_aequo_ba=[40.0],   # <= max(temoin)=50.0
    )
    assert v.ok and not v.invalide
