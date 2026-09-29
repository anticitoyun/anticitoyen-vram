"""GUI, pièce P1 (chef, 29/09) : un alias dont la fenêtre déclarée (config.toml,
max_context_size) est sous CTX_HORS_CHAMP (34 816 — poste1-edz-verdict-29-09.md § Classement,
décision chef) est marqué dans la colonne Contexte (tooltip + marqueur ⚠), et son
« Ouvrir dans … » dit pourquoi au lieu de tenter un lancement qui échouerait côté client
(claude-modele/kimi-modele). Aucun spawn ne doit partir pour un alias hors champ."""
import json
import os
import shutil
import subprocess

import pytest

ICI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/usr/bin/python3"


def _gi_ok():
    r = subprocess.run([PY, "-c", "import gi; gi.require_version('Gtk', '4.0'); gi.require_version('Adw', '1'); "
                                    "from gi.repository import Adw"], capture_output=True)
    return r.returncode == 0


pytestmark = pytest.mark.skipif(not shutil.which("xvfb-run") or not _gi_ok(),
                                reason="xvfb-run ou GTK4/libadwaita absent")


def _parc(tmp_path, ctx_hors_champ, ctx_dans_le_champ):
    kimi = tmp_path / "kimi"; kimi.mkdir()
    tsv = tmp_path / "tsv"; tsv.mkdir()
    (kimi / "config.toml").write_text(
        f'[models.acvram-etroit]\nprovider="acvram"\nmodel="Etroit"\nmax_context_size={ctx_hors_champ}\n'
        f'[models.acvram-large]\nprovider="acvram"\nmodel="Large"\nmax_context_size={ctx_dans_le_champ}\n')
    (tsv / "acvram-chemins.tsv").write_text(
        f"acvram-etroit\t{tmp_path}/etroit\t{ctx_hors_champ}\n"
        f"acvram-large\t{tmp_path}/large\t{ctx_dans_le_champ}\n")
    parc = tmp_path / "parc.toml"
    parc.write_text(f'[chemins]\nkimi_dir="{kimi}"\ntsv_dir="{tsv}"\n[moteurs.acvram]\npresent=true\n')
    return parc


def _jouer(parc, gui, clic):
    chemin = os.path.join(ICI, "parc", "bin", gui)
    env = {**os.environ, "ACVRAM_GUI_TEST": clic, "ACVRAM_PARC_CONFIG": str(parc),
           "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": os.path.join(ICI, "parc", "lib")}
    env.pop("XDG_CONFIG_HOME", None)
    r = subprocess.run(["xvfb-run", "-a", PY, chemin], capture_output=True, text=True, env=env, timeout=60)
    lignes = [l for l in r.stdout.splitlines() if l.startswith("GUI_TEST ")]
    assert len(lignes) == 1, r.stdout[-500:] + r.stderr[-500:]
    return r.returncode, json.loads(lignes[0][len("GUI_TEST "):])


@pytest.mark.parametrize("gui", ["claude-modeles", "kimi-modeles"])
def test_ouvrir_hors_champ_refuse_sans_spawn_et_dit_pourquoi(tmp_path, gui):
    parc = _parc(tmp_path, ctx_hors_champ=16384, ctx_dans_le_champ=65536)
    rc, r = _jouer(parc, gui, "clic:b_ouvrir@acvram-etroit")
    assert rc == 0
    assert r["spawns"] == []
    assert any("16 384" in t and "34 816" in t for t in r["toasts"]), r["toasts"]


@pytest.mark.parametrize("gui", ["claude-modeles", "kimi-modeles"])
def test_ouvrir_dans_le_champ_ne_porte_pas_le_refus_hors_champ(tmp_path, gui):
    parc = _parc(tmp_path, ctx_hors_champ=16384, ctx_dans_le_champ=65536)
    rc, r = _jouer(parc, gui, "clic:b_ouvrir@acvram-large")
    assert rc == 0
    assert not any("34 816" in t for t in r["toasts"]), r["toasts"]
