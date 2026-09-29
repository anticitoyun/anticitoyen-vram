"""Pièce claude (poste6, 29/09) : la campagne edz notait comme cause la DERNIÈRE ligne de stderr de claude —
« [claude-code:unrecognized_model] {…} », que claude imprime aussi sur 116 réussites de la liste-1. La cause est la
première ligne d'erreur nommée (API Error, REFUS…), la dernière ligne n'est qu'un repli."""
import importlib.util
import pathlib

RACINE = pathlib.Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("tmr", RACINE / "outils" / "test-menus-reels.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_l_erreur_api_prime_sur_la_ligne_unrecognized_model():
    m = _module()
    err = ("acvram sert déjà x\n\nAPI Error: 500 Jinja Exception: Only user, assistant and tool roles are supported\n"
           "⚠ claude.ai connectors are disabled\n[claude-code:unrecognized_model] {\"model\":\"x\"}\n")
    assert m.cause_de(err, "").startswith("API Error: 500")


def test_sans_ligne_d_erreur_la_derniere_ligne_reste_le_repli():
    m = _module()
    assert m.cause_de("a\nb\n", "") == "b"
    assert m.cause_de("", "sortie\n[claude-code:unrecognized_model] {}\n") == "[claude-code:unrecognized_model] {}"


def test_refus_du_lanceur_est_vu_dans_stdout():
    m = _module()
    assert m.cause_de("", "x\nREFUS : jeton absent\ny\n") == "REFUS : jeton absent"
