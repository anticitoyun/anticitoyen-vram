"""`banc-marlin-plancher.py` à sec : routage à D distincts exacts, octets par
couche (gate + up + down, E2M1 + E4M3), résumé To/s contre plancher et pic,
verdicts par seuils écrits avant. Le GEMV est carte (Manon)."""
import importlib.util
import os

import torch

ICI = os.path.dirname(os.path.abspath(__file__))


def _m():
    p = os.path.join(ICI, "..", "outils", "gpu", "mesure", "banc-marlin-plancher.py")
    spec = importlib.util.spec_from_file_location("banc_marlin_plancher", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_routage_a_D_distincts_et_octets():
    m = _m()
    g = torch.Generator().manual_seed(1)
    for D in (8, 42, 69, 96):
        topi = m.routage(12, D, g)
        assert topi.shape == (12, 8) and m.distincts(topi) == D
        assert all(len(set(r.tolist())) == 8 for r in topi)          # top-8 sans doublon par jeton
    assert m.OCTETS_EXPERT == 2 * (768 * 1024 + 768 * 128) + (2048 * 384 + 2048 * 48)   # 2,655 Mo
    assert abs(m.OCTETS_EXPERT / 1e6 - 2.655) < 0.01


def test_resume_et_verdicts():
    m = _m()
    octets = 42 * m.OCTETS_EXPERT
    r = m.resume([80.0, 82.0, 78.0, 81.0, 79.0], octets)
    assert r["us_mediane"] == 80.0 and abs(r["to_s"] - octets / 80e-6 / 1e12) < 1e-3
    assert abs(r["part_plancher"] - r["to_s"] / 1.55) < 1e-3
    base = {"couche": {"to_s": 1.40, "us_mediane": 80.0, "part_plancher": 0.9}, "amortissement_8_us": 0.5, "l2_us": 0.5, "rampe_dependance_us": 2.5}
    assert m.verdict(base)["verdict"].startswith("TENU :")
    assert m.verdict({**base, "couche": {**base["couche"], "to_s": 1.56}})["verdict"].startswith("RÉFUTÉ (le noyau")
    assert m.verdict({**base, "amortissement_8_us": 4.0})["verdict"].startswith("RÉFUTÉ (8 couches")
    assert m.verdict({**base, "l2_us": 3.0})["verdict"].startswith("TENU, L2")
