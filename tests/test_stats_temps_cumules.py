"""Les temps cumulés doivent être exposés, pas seulement les taux.

Un taux est une moyenne depuis le démarrage. Il ne permet pas de mesurer une
requête : deux relevés successifs de `prefill_tok_s` ne se soustraient pas.
Deux mesures s'y sont cassées le 9 septembre 2026 — l'une a obtenu un forward
supérieur au TTFT (impossible), l'autre a lu la clé absente comme un zéro et
conclu « 0,0 ms pour 39 410 jetons ».

L'épreuve porte sur ce qui rend la mesure possible : la SOUSTRACTION de deux
relevés doit rendre le temps écoulé entre eux.
"""
from acvram.engine.runner import EngineStats


def test_les_temps_cumules_sont_exposes():
    d = EngineStats().to_dict()
    assert "prefill_seconds" in d, "sans ce cumul, aucune requête n'est mesurable"
    assert "decode_seconds" in d


def test_deux_releves_se_soustraient():
    """Ce que le taux ne permet pas : isoler une requête entre deux relevés."""
    s = EngineStats()
    s.prefill_seconds, s.prefill_tokens = 1.0, 100
    avant = s.to_dict()
    s.prefill_seconds, s.prefill_tokens = 1.25, 150
    apres = s.to_dict()
    assert apres["prefill_seconds"] - avant["prefill_seconds"] == 0.25
    assert apres["prefill_tokens"] - avant["prefill_tokens"] == 50


def test_le_taux_seul_ne_suffit_pas():
    """Épreuve du piège lui-même : les taux peuvent être IDENTIQUES alors que
    du travail a eu lieu entre les deux relevés. Un contrôle qui s'appuierait
    sur eux ne verrait rien se passer."""
    s = EngineStats()
    s.prefill_seconds, s.prefill_tokens = 1.0, 100
    avant = s.to_dict()
    s.prefill_seconds, s.prefill_tokens = 2.0, 200      # même taux, 1 s de plus
    apres = s.to_dict()
    assert apres["prefill_tok_s"] == avant["prefill_tok_s"]
    assert apres["prefill_seconds"] > avant["prefill_seconds"]


def test_zero_reste_zero_sans_travail():
    """Et l'inverse : sans travail, le cumul ne bouge pas. Un compteur qui
    avance tout seul serait aussi trompeur qu'un compteur absent."""
    s = EngineStats()
    assert s.to_dict()["prefill_seconds"] == 0.0
    assert s.to_dict()["decode_seconds"] == 0.0
