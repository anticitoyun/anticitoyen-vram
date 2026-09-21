"""Crochet `ACVRAM_GUI_TEST` de parc/bin/claude-modeles : le gestionnaire d'un bouton est joué en processus, à sec
(rien lancé), son retour imprimé en une ligne `GUI_TEST {…}`. Sous Xvfb, parc par défaut (XDG_CONFIG_HOME vide) :
un bouton factice rend rc 2 ; ComfyUI sans extras.comfy_start refuse par un toast et ne lance rien ; un filtre sans
résultat rend 0 visible. Sauté sans xvfb-run ou sans GTK4/libadwaita."""
import json, os, shutil, subprocess, sys, pytest

ICI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUI = os.path.join(ICI, "parc", "bin", "claude-modeles")
PY = "/usr/bin/python3"                                   # le shebang de la GUI : gi n est pas dans le venv acvram


def _gi_ok():
    r = subprocess.run([PY, "-c", "import gi; gi.require_version('Gtk', '4.0'); gi.require_version('Adw', '1'); from gi.repository import Adw"],
                       capture_output=True)
    return r.returncode == 0


pytestmark = pytest.mark.skipif(not shutil.which("xvfb-run") or not _gi_ok(), reason="xvfb-run ou GTK4/libadwaita absent")


def jouer(test, tmp_path):
    env = {**os.environ, "ACVRAM_GUI_TEST": test, "XDG_CONFIG_HOME": str(tmp_path), "CUDA_VISIBLE_DEVICES": "",
           "PYTHONPATH": os.path.join(ICI, "parc", "lib")}
    env.pop("ACVRAM_PARC_CONFIG", None)
    r = subprocess.run(["xvfb-run", "-a", PY, GUI], capture_output=True, text=True, env=env, timeout=120)
    lignes = [l for l in r.stdout.splitlines() if l.startswith("GUI_TEST ")]
    assert len(lignes) == 1, r.stdout[-500:] + r.stderr[-500:]
    return r.returncode, json.loads(lignes[0][len("GUI_TEST "):])


def test_bouton_factice_rend_2(tmp_path):
    rc, r = jouer("clic:b_bidon", tmp_path)
    assert rc == 2 and r["erreur"] == "bouton inconnu : b_bidon" and r["spawns"] == []


def test_comfyui_non_configure_refuse_sans_rien_lancer(tmp_path):
    rc, r = jouer("clic:b_web", tmp_path)
    assert rc == 0 and r["spawns"] == [] and r["uris"] == [] and r["toasts"] == ["ComfyUI non configuré (extras.comfy_start de parc.toml)"]


def test_filtre_sans_resultat(tmp_path):
    rc, r = jouer("filtre:zzzz-rien", tmp_path)
    assert rc == 0 and r["visibles"] == 0
