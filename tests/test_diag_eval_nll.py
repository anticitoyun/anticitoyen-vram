"""Pièce 37 (`diag-eval-nll.py`) à sec : PPL depuis un contexte minimal,
première divergence, verdicts P1/P2/P3 selon les trois bras. Le forward est carte."""
import importlib.util
import math
import os


def _m():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outils", "gpu", "mesure", "diag-eval-nll.py")
    spec = importlib.util.spec_from_file_location("diag_eval_nll", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_ppl_et_divergence():
    m = _m()
    nll = [5.0] * 32 + [math.log(10.0)] * 100
    assert abs(m.ppl(nll) - 10.0) < 1e-9 and abs(m.ppl(nll, depuis=0) - math.exp((5 * 32 + math.log(10) * 100) / 132)) < 1e-9
    assert m.premiere_divergence([1, 1, 1, 3], [1, 1, 1, 1]) == 3 and m.premiere_divergence([1, 1.5], [1, 1]) is None


def test_verdicts():
    m = _m()
    base = {"ppl_eval": 12.0, "ppl_serve": 12.0, "div_eval_serve": None, "champs_batch_serve": {}}
    assert m.verdict({**base, "ppl_hf": 11.5, "div_eval_hf": None, "div_serve_hf": None}).startswith("P1")
    assert m.verdict({**base, "ppl_hf": 11.5, "div_eval_hf": 2, "div_serve_hf": None, "div_eval_serve": 2}).startswith("P2")
    assert m.verdict({**base, "ppl_hf": 11.5, "div_eval_hf": 3, "div_serve_hf": 3}).startswith("P3")
    assert m.verdict({**base, "div_eval_serve": 1}).startswith("P2 (sans hf)")
    assert "forward plausible" in m.verdict(base)
    assert "bras hf requis" in m.verdict({**base, "ppl_eval": 27713.0})
