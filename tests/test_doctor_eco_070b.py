"""070 b : `acvram doctor` sortait 1 partout (hôte ET Flatpak v0.7.0) — `_doctor_eco` (poste7-eco 19/09) appelle
subprocess.run sans que cli.py importe subprocess : `NameError: name 'subprocess' is not defined` en toute dernière
ligne, après le rapport complet, donc invisible dans un terminal qui ne lit que les lignes « ok / ECHEC ». Vu par
verif-070 (doctor du bac à sable, code 1, 0 ECHEC sur l'hôte). Le test appelle _doctor_eco avec un sudo factice."""
import subprocess
import types

from acvram import cli


def test_070b_doctor_eco_ne_leve_pas(monkeypatch, capsys):
    monkeypatch.setattr(cli, "subprocess", types.SimpleNamespace(
        run=lambda *a, **k: subprocess.CompletedProcess(a, 1, "", ""), SubprocessError=subprocess.SubprocessError),
        raising=True)     # raising=True : si cli n'importe plus subprocess, le monkeypatch lui-même rend rouge
    cli._doctor_eco(cuda_ok=False)
    sortie = capsys.readouterr().out
    assert "mode demande" in sortie and "pas de droit sudo" in sortie
