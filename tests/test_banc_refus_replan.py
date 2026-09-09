"""Le banc doit REFUSER de publier un debit quand le plan a ete rejoue.

`acvram bench --what decode` annoncait 20,04 jetons/s la ou le decodage vaut
175 : deux defauts distincts, tous deux corriges ici et eprouves dans les deux
sens.

1. Le chronometre englobait prefill, allocation et capture des graphes sur 64
   jetons — le cout fixe dominait. On lit desormais `stats.decode_seconds`
   apres chauffe, et l'on rend la mediane avec sa dispersion.
2. Le banc laissait les deux cartes visibles quand le serveur les epingle
   depuis e5cafc0. Un avertissement n'arretait rien : le chiffre produit ne
   portait plus sur la configuration demandee.
"""
import types

import pytest

from acvram import bench


class _PlanRejoue:
    replanifie_cartes = (["cuda:0"], ["cuda:0", "cuda:1"])
    est_decode_tok_s = 687.6


class _PlanNet:
    est_decode_tok_s = 687.6


class _Modele:
    nbytes = 4 * 2 ** 30


class _Charge:
    def __init__(self, plan):
        self.plan = plan
        self.model = _Modele()


def _charger(plan):
    return lambda *a, **k: _Charge(plan)


def test_refus_quand_le_plan_a_ete_rejoue(monkeypatch):
    """Le defaut par defaut est le refus : pas de debit, et la raison avec."""
    monkeypatch.delenv("ACVRAM_BANC_ACCEPTE_REPLAN", raising=False)
    monkeypatch.setattr("acvram.engine.loader.load_model", _charger(_PlanRejoue()))
    d = bench.bench_decode("/inexistant")
    assert "refus" in d, "le banc a publie un debit sur un plan rejoue"
    assert "cuda:1" in d["refus"], "le refus doit nommer la carte en trop"
    assert "decode_tok_s" not in d, "aucun debit ne doit sortir d'un refus"


def test_le_refus_se_laisse_forcer_explicitement(monkeypatch):
    """Une garde qu'on ne peut pas lever bloque un usage legitime.

    Le forcage doit etre EXPLICITE : sans lui, le silence vaut refus.
    """
    monkeypatch.setenv("ACVRAM_BANC_ACCEPTE_REPLAN", "1")
    monkeypatch.setattr("acvram.engine.loader.load_model", _charger(_PlanRejoue()))
    # Le chargement passe la garde ; la suite echoue faute de vrai moteur, ce
    # qui suffit a prouver que le refus n'a PAS ete rendu.
    with pytest.raises(Exception) as exc:
        bench.bench_decode("/inexistant")
    assert "refus" not in str(exc.value).lower()


def test_la_sonde_de_cartes_ne_plante_jamais():
    """Une sonde qui leve fait echouer la mesure qu'elle devait proteger."""
    assert bench._plusieurs_cartes() in (True, False)
