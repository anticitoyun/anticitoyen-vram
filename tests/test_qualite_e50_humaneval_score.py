"""e50.3 § 7 (poste3, 01/10) : `outils/qualite-e50-humaneval-score.py` reconstruit
prompt+complétion+test pour chaque échantillon journalisé par lm-eval et le fait passer dans le
bac à sable (§ 2 bis) — jamais lm-eval ni le paquet qui exécuterait le code nu. Ce test vérifie
le compte (une complétion correcte, une fausse) à partir d'un jsonl construit à la main, sans
lm-eval ni serveur."""
import importlib.util
import json
import os
import shutil
import pytest

ICI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(not shutil.which("bwrap"), reason="bwrap (bubblewrap) absent")


def _module():
    spec = importlib.util.spec_from_file_location(
        "qualite_e50_humaneval_score", os.path.join(ICI, "outils", "qualite-e50-humaneval-score.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(autouse=True)
def _py_bac_a_sable(monkeypatch):
    monkeypatch.setenv("PY_BAC_A_SABLE", "/usr/bin/python3")


def test_une_bonne_une_fausse(tmp_path):
    jsonl = tmp_path / "samples.jsonl"
    jsonl.write_text("\n".join([
        json.dumps({"doc": {"prompt": "def inc(x):\n    \"\"\"increment\"\"\"\n",
                            "test": "def check(candidate):\n    assert candidate(1) == 2\n",
                            "entry_point": "inc"},
                   "filtered_resps": ["    return x + 1\n"]}),
        json.dumps({"doc": {"prompt": "def bad(x):\n    \"\"\"toujours faux\"\"\"\n",
                            "test": "def check(candidate):\n    assert candidate(1) == 2\n",
                            "entry_point": "bad"},
                   "filtered_resps": ["    return x + 999\n"]}),
    ]))
    r = _module().noter(str(jsonl))
    assert r == {"n": 2, "pass": 1, "pass_at_1": 0.5}, r


def test_toutes_bonnes(tmp_path):
    jsonl = tmp_path / "samples.jsonl"
    jsonl.write_text(json.dumps({
        "doc": {"prompt": "def inc(x):\n", "test": "def check(candidate):\n    assert candidate(1) == 2\n",
               "entry_point": "inc"},
        "filtered_resps": ["    return x + 1\n"]}))
    r = _module().noter(str(jsonl))
    assert r == {"n": 1, "pass": 1, "pass_at_1": 1.0}, r


def test_toutes_fausses(tmp_path):
    jsonl = tmp_path / "samples.jsonl"
    jsonl.write_text(json.dumps({
        "doc": {"prompt": "def inc(x):\n", "test": "def check(candidate):\n    assert candidate(1) == 2\n",
               "entry_point": "inc"},
        "filtered_resps": ["    return x + 0\n"]}))
    r = _module().noter(str(jsonl))
    assert r == {"n": 1, "pass": 0, "pass_at_1": 0.0}, r
