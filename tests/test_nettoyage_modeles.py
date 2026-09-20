"""`outils/nettoyage-modeles.sh` : déplacement en corbeille réversible, refus nommés, purge sous mot.

Joué sur des disques fabriqués (tmp_path) : les racines autorisées viennent de
ACVRAM_NETTOYAGE_RACINES, le verrou d'un chemin vide — jamais les vrais disques.
"""
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "outils" / "nettoyage-modeles.sh"


def _run(args, env, cwd):
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True,
                          env={**os.environ, **env}, cwd=cwd)


@pytest.fixture
def disques(tmp_path):
    a, b = tmp_path / "disqueA", tmp_path / "disqueB"
    (a / "models_gguf" / "M1").mkdir(parents=True); (a / "models_gguf" / "M1" / "w.gguf").write_bytes(b"x" * 4096)
    (b / "models" / "M2").mkdir(parents=True); (b / "models" / "M2" / "c.json").write_text("{}")
    ailleurs = tmp_path / "ailleurs" / "M3"; ailleurs.mkdir(parents=True); (ailleurs / "f").write_text("x")
    env = {"ACVRAM_NETTOYAGE_RACINES": f"{a}:{b}", "ACVRAM_VERROU": str(tmp_path / "verrou-vide")}
    return a, b, ailleurs, env


def test_deplace_en_corbeille_journalise_et_restaure(disques, tmp_path):
    a, b, ailleurs, env = disques
    liste = tmp_path / "liste.txt"
    liste.write_text(f"# commentaire\n{a}/models_gguf/M1\n{b}/models/M2   # avec commentaire\n")
    r = _run([str(liste)], env, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    corbs = sorted(a.glob("corbeille-*")) + sorted(b.glob("corbeille-*"))
    assert len(corbs) == 2
    assert not (a / "models_gguf" / "M1").exists() and (corbs[0] / "models_gguf" / "M1" / "w.gguf").exists()
    journal = (corbs[0] / "journal.tsv").read_text().splitlines()
    assert len(journal) == 1 and journal[0].split("\t")[1] == f"{a}/models_gguf/M1"
    assert "deplaces=2 refus=0" in r.stdout
    # restauration depuis le journal
    r2 = _run(["--restaurer", str(corbs[0])], env, tmp_path)
    assert r2.returncode == 0 and (a / "models_gguf" / "M1" / "w.gguf").exists() and "restaures=1" in r2.stdout


def test_refus_nommes_sans_arreter_la_liste(disques, tmp_path):
    a, b, ailleurs, env = disques
    liste = tmp_path / "liste.txt"
    liste.write_text(f"{ailleurs}\n{a}/models_gguf/ABSENT\n{a}/models_gguf/M1\n")
    r = _run([str(liste)], env, tmp_path)
    assert r.returncode == 1                       # des refus, mais la liste a été parcourue
    assert "REFUS hors des racines autorisees" in r.stdout and "REFUS absent" in r.stdout
    assert ailleurs.exists() and not (a / "models_gguf" / "M1").exists()
    assert "deplaces=1 refus=2" in r.stdout


def test_refuse_pendant_une_prise_de_carte(disques, tmp_path):
    a, b, ailleurs, env = disques
    Path(env["ACVRAM_VERROU"] + ".qui").write_text("poste2 1234")
    liste = tmp_path / "liste.txt"; liste.write_text(f"{a}/models_gguf/M1\n")
    r = _run([str(liste)], env, tmp_path)
    assert r.returncode == 75 and (a / "models_gguf" / "M1").exists()


def test_purge_seulement_avec_oui_et_sur_une_corbeille_datee(disques, tmp_path):
    a, b, ailleurs, env = disques
    liste = tmp_path / "liste.txt"; liste.write_text(f"{a}/models_gguf/M1\n")
    assert _run([str(liste)], env, tmp_path).returncode == 0
    corb = next(a.glob("corbeille-*"))
    r = _run(["--purger", str(corb)], env, tmp_path)
    assert r.returncode == 1 and corb.exists() and "purge non faite" in r.stdout
    r = _run(["--purger", str(a / "models_gguf")], env, tmp_path)
    assert r.returncode == 64 and (a / "models_gguf").exists()
    r = _run(["--purger", str(corb), "--oui"], env, tmp_path)
    assert r.returncode == 0 and not corb.exists()
