"""Pièce 84 : liste vide après installation d'acvram-parc (aucun `~/.kimi-code/config.toml`, jamais de
parc-installer lancé) — `charger_parc()` rendait une RuntimeError affichée en toast, le store restait
vide sans que l'utilisateur sache pourquoi ni quoi faire. Vérifie ici : (1) config absente = parc vide,
pas une faute ; (2) le bouton « Rebalayer » relance l'outil EXISTANT `parc-installer`, restreint aux
dossiers déjà connus (`--sans-balayage`, aucun balayage de tous les disques depuis un clic de menu)."""
import json, os, shutil, subprocess, pytest

ICI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/usr/bin/python3"


def _gi_ok():
    r = subprocess.run([PY, "-c", "import gi; gi.require_version('Gtk', '4.0'); gi.require_version('Adw', '1'); from gi.repository import Adw"],
                       capture_output=True)
    return r.returncode == 0


pytestmark = pytest.mark.skipif(not shutil.which("xvfb-run") or not _gi_ok(), reason="xvfb-run ou GTK4/libadwaita absent")


def _jouer(gui, test, tmp_path, **env_sup):
    chemin = os.path.join(ICI, "parc", "bin", gui)
    env = {**os.environ, "ACVRAM_GUI_TEST": test, "XDG_CONFIG_HOME": str(tmp_path), "CUDA_VISIBLE_DEVICES": "",
           "PYTHONPATH": os.path.join(ICI, "parc", "lib"), **env_sup}
    env.pop("ACVRAM_PARC_CONFIG", None)
    r = subprocess.run(["xvfb-run", "-a", PY, chemin], capture_output=True, text=True, env=env, timeout=60)
    lignes = [l for l in r.stdout.splitlines() if l.startswith("GUI_TEST ")]
    assert len(lignes) == 1, r.stdout[-800:] + r.stderr[-800:]
    return r.returncode, json.loads(lignes[0][len("GUI_TEST "):])


@pytest.mark.parametrize("gui", ["claude-modeles", "kimi-modeles"])
def test_config_absente_liste_vide_pas_une_faute(tmp_path, gui):
    """Aucun ~/.kimi-code/config.toml (XDG_CONFIG_HOME neuf, HOME réel inchangé mais PARC_KIMI absent
    de tout parc.toml pointe vers ~/.kimi-code par défaut) : sur CE POSTE, si l'utilisateur a un vrai
    ~/.kimi-code/config.toml, ce test vise surtout charger_parc() en direct — voir test_charger_parc_config_absente."""
    rc, r = _jouer(gui, "filtre:", tmp_path)
    assert rc == 0
    assert "erreur" not in r, r


@pytest.mark.parametrize("gui", ["claude-modeles", "kimi-modeles"])
def test_clic_rebalayer_appelle_parc_installer_sans_balayage(tmp_path, gui):
    """Le bouton « Rebalayer » ne réécrit rien lui-même : il relance l'outil existant, restreint
    (--sans-balayage : pas de balayage de tous les disques depuis un clic)."""
    rc, r = _jouer(gui, "clic:b_rebalayer", tmp_path)
    assert rc == 0, r
    assert len(r["spawns"]) == 1, r
    argv = r["spawns"][0]
    assert argv[0].endswith("parc-installer"), argv
    assert "--auto" in argv and "--sans-balayage" in argv and "--sans-minuteur" in argv, argv
    assert "--racine" not in argv, "Rebalayer sans nouveau dossier ne doit rien ajouter : " + str(argv)


def test_charger_parc_config_absente(tmp_path):
    """Unit direct de la fonction (module `menu_modeles.parc`, qui a besoin de `gi` — donc du python3
    système, pas du venv acvram) : CONFIG absent → [], jamais une RuntimeError. Un config.toml PRÉSENT
    mais invalide reste une faute (contrôle négatif). kimi_dir pointé sous tmp_path : sinon le défaut
    « ~/.kimi-code » lirait le VRAI poste."""
    parc_toml = tmp_path / "parc.toml"
    kimi_dir = tmp_path / "kimi"
    parc_toml.write_text(f'[chemins]\nkimi_dir = "{kimi_dir}"\n')
    script = f'''
import sys
sys.path.insert(0, {os.path.join(ICI, "parc", "lib")!r})
from menu_modeles import config as cfg, parc as parcmod
assert not cfg.CONFIG.exists(), cfg.CONFIG
assert parcmod.charger_parc() == []
cfg.CONFIG.parent.mkdir(parents=True, exist_ok=True)
cfg.CONFIG.write_text("ceci n'est pas du toml valide [[[")
try:
    parcmod.charger_parc()
    print("PAS-DE-FAUTE")
except RuntimeError as e:
    assert "illisible" in str(e), e
    print("OK")
'''
    env = {**os.environ, "ACVRAM_PARC_CONFIG": str(parc_toml)}
    r = subprocess.run([PY, "-c", script], capture_output=True, text=True, env=env, timeout=30)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr
