"""61w (poste1, 01/10) : `outils/gpu/mesure/kda-61w-abba.py comparer`, éprouvé à sec sur de faux bras. Le témoin (A1/A2, B1/B2)
doit pouvoir rendre « non déterministe » (rc 6, aucun verdict), une divergence d'ids doit être datée au jeton, et un gain b=12
sous 1,0 ms doit être dit FAUX (prédiction scellée, revue/poste1-p81-cake-kda-01-10.md)."""
import json
import subprocess
import sys
from pathlib import Path

OUTIL = Path(__file__).resolve().parent.parent / "outils" / "gpu" / "mesure" / "kda-61w-abba.py"
IDS1, IDS12 = [[1, 2, 3, 4]], [[5, 6, 7]] * 12


def _bras(tmp_path, nom, arbre, ids1, ids12, b1, b12):
    p = tmp_path / f"{nom}.json"
    p.write_text(json.dumps({"arbre": arbre, "regime": "", "precompile": "x",
                             "ids": {"b1": ids1, "b12": ids12}, "mur_ms": {"b1": b1, "b12": b12}}))
    return str(p)


def _comparer(tmp_path, a, b1, b2, a2):
    chemins = [_bras(tmp_path, n, **d) for n, d in (("A1", a), ("B1", b1), ("B2", b2), ("A2", a2))]
    r = subprocess.run([sys.executable, str(OUTIL), "comparer", *chemins], capture_output=True, text=True)
    return r.returncode, r.stdout


def _a(**k):
    return {**dict(arbre="/A", ids1=IDS1, ids12=IDS12, b1=[3.70, 3.71, 3.72], b12=[11.0, 11.1, 11.05]), **k}


def _b(**k):
    return {**dict(arbre="/B", ids1=IDS1, ids12=IDS12, b1=[3.33, 3.34, 3.33], b12=[9.45, 9.5, 9.48]), **k}


def test_identiques_et_gain_dans_la_bande():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        rc, out = _comparer(Path(d), _a(), _b(), _b(), _a())
    assert rc == 0, out
    assert "ids gloutons IDENTIQUES" in out and "dans la bande" in out and "FAUX" not in out


def test_le_temoin_peut_rendre_non_deterministe(tmp_path):
    rc, out = _comparer(tmp_path, _a(), _b(), _b(), _a(ids1=[[1, 2, 9, 4]]))
    assert rc == 6 and "pas déterministe" in out and "mur" not in out


def test_divergence_datee_et_gain_insuffisant_faux(tmp_path):
    b = _b(ids1=[[1, 2, 8, 4]], b1=[3.6] * 3, b12=[10.5] * 3)
    rc, out = _comparer(tmp_path, _a(), b, b, _a())
    assert rc == 0
    assert "PREMIÈRE DIVERGENCE au jeton 2" in out and "ids DIVERGENTS" in out
    assert "FAUX (gain < 1,0 ms)" in out


def test_arbres_incoherents_refuses(tmp_path):
    rc, _ = _comparer(tmp_path, _a(), _a(), _a(), _a())
    assert rc == 5
