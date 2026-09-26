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


def test_comfyui_verrou_refuse_spawn(tmp_path):
    import pathlib
    # Config : écrire COMFY_START. acvram_parc lit $XDG_CONFIG_HOME/acvram-parc/parc.toml
    # (pas .../acvram) — sinon comfy_start reste None et le toast « non configuré »
    # sort avant la garde de verrou.
    config_dir = tmp_path / "xdg" / "acvram-parc"
    config_dir.mkdir(parents=True)
    script = tmp_path / "comfyui.sh"
    script.write_text("#!/bin/bash\necho spawn\n")
    script.chmod(0o755)
    (config_dir / "parc.toml").write_text(f'[extras]\ncomfy_start = "{script}"\n')

    # Créer le verrou non-vide
    # JAMAIS le vrai /tmp/acvram-carte-0.lock.qui : l'écrire puis l'effacer détruisait l'étiquette d'une vraie
    # prise en cours (jxm, 24/09 — les .qui « disparus » pendant les CI)
    verrou_base = tmp_path / "acvram-carte-0.lock"
    verrou_path = pathlib.Path(str(verrou_base) + ".qui")
    verrou_path.write_text("mesure en cours")

    try:
        env = {**os.environ, "ACVRAM_VERROU": str(verrou_base), "ACVRAM_GUI_TEST": "clic:b_web", "XDG_CONFIG_HOME": str(config_dir.parent),
               "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": os.path.join(ICI, "parc", "lib")}
        env.pop("ACVRAM_PARC_CONFIG", None)
        r = subprocess.run(["xvfb-run", "-a", PY, GUI], capture_output=True, text=True, env=env, timeout=120)
        lignes = [l for l in r.stdout.splitlines() if l.startswith("GUI_TEST ")]
        assert len(lignes) == 1, r.stdout[-500:] + r.stderr[-500:]
        rc, result = r.returncode, json.loads(lignes[0][len("GUI_TEST "):])
        assert rc == 0 and result["spawns"] == [] and any("verrou acvram" in t for t in result.get("toasts", []))
    finally:
        verrou_path.unlink(missing_ok=True)


def _parc_fixture(tmp_path):
    """Parc minimal : moteur acvram present, 4 modeles dont 2 SERVENT des images
    (vision_status=vision) — l'un porte « Vision » dans son NOM — et 1 converti
    texte-seul dont l'ALIAS contient « vision » (le piege que le substring ramenait
    a tort). Rend (parc.toml, nb_vision_attendu)."""
    kimi = tmp_path / "kimi"; kimi.mkdir()
    tsv = tmp_path / "tsv"; tsv.mkdir()
    (kimi / "config.toml").write_text(
        '[models.acvram-alpha-vision]\nprovider="acvram"\nmodel="Alpha"\n'
        '[models.acvram-beta]\nprovider="acvram"\nmodel="Beta-Vision-8B"\n'
        '[models.acvram-gamma-vision]\nprovider="acvram"\nmodel="Gamma"\n'
        '[models.acvram-delta]\nprovider="acvram"\nmodel="Delta"\n')
    (tsv / "vision-modeles.tsv").write_text(
        "acvram-alpha-vision\tvision\n"
        "acvram-beta\tvision\n"
        "acvram-gamma-vision\ttexte-seul\n")
    parc = tmp_path / "parc.toml"
    parc.write_text(f'[chemins]\nkimi_dir="{kimi}"\ntsv_dir="{tsv}"\n[moteurs.acvram]\npresent=true\n')
    return parc, 2


@pytest.mark.parametrize("gui", ["claude-modeles", "kimi-modeles"])
def test_filtre_vision_egale_les_alias_vision_du_tsv(tmp_path, gui):
    """Decision chef (a), 21/09 : « vision » ne sort QUE les modeles qui
    servent des images (vision_status), pas ceux qui portent le mot dans leur
    alias ou leur nom. Cassant : avant, le substring ramenait aussi le converti
    texte-seul acvram-gamma-vision → 3 au lieu de 2. Verifie dans les DEUX menus
    (claude-modeles ET kimi-modeles), qui partagent le code de filtre."""
    parc, attendu = _parc_fixture(tmp_path)
    env = {**os.environ, "ACVRAM_GUI_TEST": "filtre:vision", "ACVRAM_PARC_CONFIG": str(parc),
           "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": os.path.join(ICI, "parc", "lib")}
    env.pop("XDG_CONFIG_HOME", None)
    chemin = os.path.join(ICI, "parc", "bin", gui)
    r = subprocess.run(["xvfb-run", "-a", PY, chemin], capture_output=True, text=True, env=env, timeout=120)
    lignes = [l for l in r.stdout.splitlines() if l.startswith("GUI_TEST ")]
    assert len(lignes) == 1, r.stdout[-500:] + r.stderr[-500:]
    result = json.loads(lignes[0][len("GUI_TEST "):])
    assert result["visibles"] == attendu == 2, (gui, result)
    assert result["total"] == 4, (gui, result)


@pytest.mark.parametrize("gui", ["claude-modeles", "kimi-modeles"])
def test_bouton_factice_rend_2_dans_les_deux_gui(tmp_path, gui):
    """Le crochet ACVRAM_GUI_TEST est présent et fonctionnel dans les DEUX menus :
    un bouton inconnu rend rc 2. (Il manquait à kimi-modeles — porté le 21/09,
    avec `import json`.)"""
    chemin = os.path.join(ICI, "parc", "bin", gui)
    env = {**os.environ, "ACVRAM_GUI_TEST": "clic:b_bidon", "XDG_CONFIG_HOME": str(tmp_path),
           "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": os.path.join(ICI, "parc", "lib")}
    env.pop("ACVRAM_PARC_CONFIG", None)
    r = subprocess.run(["xvfb-run", "-a", PY, chemin], capture_output=True, text=True, env=env, timeout=120)
    lignes = [l for l in r.stdout.splitlines() if l.startswith("GUI_TEST ")]
    assert len(lignes) == 1, r.stdout[-500:] + r.stderr[-500:]
    assert r.returncode == 2, (gui, r.returncode)
    assert json.loads(lignes[0][len("GUI_TEST "):])["erreur"] == "bouton inconnu : b_bidon"
