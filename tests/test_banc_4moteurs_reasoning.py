"""poste2-banc (29/09) : `_texte_delta` (`outils/gpu/mesure/banc-4moteurs.py`) doit lire le
raisonnement d'un delta streaming vLLM quel que soit le champ où le moteur le rend — `content`,
`reasoning` (vLLM ≥ 0.13, protocol.py:71 en 0.29) ou l'ancien `reasoning_content` — sinon un
modèle à raisonnement servi par vLLM 0.29 passe pour une réponse vide, cause déjà vue et
corrigée côté syy/poste5-menus dans test-menus-reels.py."""
import importlib.util
import os

_CHEMIN = os.path.join(os.path.dirname(__file__), "..", "outils", "gpu", "mesure", "banc-4moteurs.py")
_spec = importlib.util.spec_from_file_location("banc_4moteurs_reasoning", _CHEMIN)
banc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(banc)


def test_texte_delta_lit_content():
    assert banc._texte_delta({"content": "Paris"}) == "Paris"


def test_texte_delta_lit_reasoning_vllm_029():
    """Le champ qui a cassé sans le correctif : vLLM 0.29 rend le raisonnement dans
    « reasoning », jamais dans « reasoning_content » — sans la lecture de ce champ, ce
    delta était compté vide."""
    assert banc._texte_delta({"content": None, "reasoning": "Je réfléchis..."}) == "Je réfléchis..."


def test_texte_delta_lit_ancien_reasoning_content():
    assert banc._texte_delta({"content": "", "reasoning_content": "hmm"}) == "hmm"


def test_texte_delta_vide_si_aucun_champ():
    assert banc._texte_delta({"role": "assistant", "content": ""}) == ""
