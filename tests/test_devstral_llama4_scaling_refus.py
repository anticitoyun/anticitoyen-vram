"""sage-devstral-llama4-scaling-18-09 : llama_4_scaling_beta (yarn
ministral3/Devstral) est porte sur q par model.Attention quand le spec le
porte A LA CONSTRUCTION (tests/test_devstral_llama4_scaling.py) ; ici le
spec est modifie APRES load_model, aucune Attention ne le sert -- pas de
no-op silencieux, REFUS NOMME au-dela de original_max_position_embeddings,
valeur visible dans regime_ligne() meme sous le plafond (non refuse, mais
non servi non plus). Meme harnais que test_budget_kv_slots_chargeur.py (converted +
rig_fige + Engine direct), rope_scaling force par monkeypatch sur le spec
charge -- `converted` (conftest.py) n'a pas de yarn configure."""
from unittest.mock import patch

import pytest

from acvram.engine.loader import load_model
from acvram.engine.runner import Engine

YARN_LLAMA4 = {"type": "yarn", "original_max_position_embeddings": 8192,
              "llama_4_scaling_beta": 0.1, "factor": 48.0}


@pytest.fixture
def rig_fige(target_rig):
    with patch("acvram.hardware.detect.detect_rig", return_value=target_rig):
        yield target_rig


def _charger_avec_yarn(converted, rig_fige, max_model_len, rope_scaling):
    loaded = load_model(converted, max_model_len=max_model_len, device_override="cpu")
    loaded.spec.rope_scaling = rope_scaling
    return loaded


def test_refuse_au_dela_du_plafond(converted, rig_fige):
    loaded = _charger_avec_yarn(converted, rig_fige, 16384, YARN_LLAMA4)
    with pytest.raises(ValueError, match="llama_4_scaling_beta non servi"):
        Engine(loaded, None, max_batch_size=1, max_model_len=16384, enable_cuda_graphs=False)


def test_sous_le_plafond_ne_refuse_pas_mais_reste_visible_dans_regime(converted, rig_fige):
    loaded = _charger_avec_yarn(converted, rig_fige, 4096, YARN_LLAMA4)
    engine = Engine(loaded, None, max_batch_size=1, max_model_len=4096, enable_cuda_graphs=False)
    assert "llama4_scaling_beta=0.1(non_servi)" in engine.regime_ligne()


def test_temoin_cassant_beta_nul_ne_refuse_jamais(converted, rig_fige):
    """Bras casse (REGLES §5) : beta=0 (pas de llama_4_scaling du tout) ne
    doit jamais refuser, meme tres au-dela du plafond -- sinon le refus
    serait sur `original_max_position_embeddings` seul, pas sur le champ
    qui justifie vraiment le refus."""
    sans_beta = {**YARN_LLAMA4, "llama_4_scaling_beta": 0.0}
    loaded = _charger_avec_yarn(converted, rig_fige, 16384, sans_beta)
    engine = Engine(loaded, None, max_batch_size=1, max_model_len=16384, enable_cuda_graphs=False)
    assert "llama4_scaling_beta" not in engine.regime_ligne()


def test_temoin_cassant_rope_type_non_yarn_ne_refuse_jamais(converted, rig_fige):
    """Meme beta>0 et meme depassement, un rope non-yarn (llama3, linear...)
    n'a pas cette formule -- le refus ne doit pas se declencher au nom
    seul de llama_4_scaling_beta hors de son contexte yarn."""
    non_yarn = {**YARN_LLAMA4, "type": "llama3"}
    loaded = _charger_avec_yarn(converted, rig_fige, 16384, non_yarn)
    Engine(loaded, None, max_batch_size=1, max_model_len=16384, enable_cuda_graphs=False)
