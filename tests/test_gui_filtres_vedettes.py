"""Filtres de lignée (« ≈ Opus/Fable », « code Android/Linux ») — pièce tri/filtres (30/09).

Repro réelle (chef, vrai écran, `~/.cache/acvram/gui-tri-repro-30-09/`) : sur le parc réel,
les trois boutons « Outils OK », « ≈ Opus/Fable », « code Android/Linux » rendaient 0/266.
Cause pour les deux derniers (confirmée par lecture de code + données réelles, voir
`acvram-memoire/revue/poste3-tri-filtres-vedettes-verdict-30-09.md`) : `outils/usages-modeles.py`
régénère la colonne `usage` depuis son vocabulaire fermé et perd le préfixe « ≈Opus »/« ≈Fable »/
« code android/linux » qu'écrit `scratchpad/notes-modeles-19-09/vedettes.py` — dès le premier
passage automatique (`modeles-a-jour`). Le correctif classe la lignée depuis l'ALIAS (fonction
pure `etiquette_vedette`, `parc.py`), jamais depuis `usage` : ce test vérifie que le filtre rend
> 0 quand des alias de la lignée existent, avec des alias SYNTHÉTIQUES calés sur les motifs
réels (aucune dépendance à `~/TSV` ni à une colonne usage)."""
import json, os, shutil, subprocess, pytest

ICI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/usr/bin/python3"


def _gi_ok():
    r = subprocess.run([PY, "-c", "import gi; gi.require_version('Gtk', '4.0'); gi.require_version('Adw', '1'); from gi.repository import Adw"],
                       capture_output=True)
    return r.returncode == 0


pytestmark = pytest.mark.skipif(not shutil.which("xvfb-run") or not _gi_ok(), reason="xvfb-run ou GTK4/libadwaita absent")

# un alias de chaque lignée (motifs de parc.py:_VEDETTE_*), un témoin qui NE doit
# jamais matcher (deepseek-coder, exclu explicitement par _VEDETTE_TEMOINS), et
# des alias neutres pour vérifier que le filtre EXCLUT bien le reste.
ALIAS_OPUS = "acvram-qwen3-6-35b-a3b-opus-nvfp4"
ALIAS_FABLE = "acvram-qwen38-fable-27b-nvfp4"
ALIAS_CODE = "acvram-devstral-small-nvfp4"
ALIAS_TEMOIN_CODE = "acvram-deepseek-coder-6-7b-nvfp4"    # jamais vedette (témoin)
ALIAS_NEUTRE = "acvram-mistral-nemo-12b-nvfp4"


def _parc_vedettes(tmp_path):
    kimi = tmp_path / "kimi"; kimi.mkdir()
    tsv = tmp_path / "tsv"; tsv.mkdir()
    noms = [ALIAS_OPUS, ALIAS_FABLE, ALIAS_CODE, ALIAS_TEMOIN_CODE, ALIAS_NEUTRE]
    (kimi / "config.toml").write_text(
        "".join(f'[models."{a}"]\nprovider="acvram"\nmodel="{a}"\n' for a in noms))
    parc = tmp_path / "parc.toml"
    parc.write_text(f'[chemins]\nkimi_dir="{kimi}"\ntsv_dir="{tsv}"\n[moteurs.acvram]\npresent=true\n')
    return parc, len(noms)


def _clic(tmp_path, gui, bouton):
    parc, _n = _parc_vedettes(tmp_path)
    chemin = os.path.join(ICI, "parc", "bin", gui)
    env = {**os.environ, "ACVRAM_GUI_TEST": f"clic:{bouton}", "ACVRAM_PARC_CONFIG": str(parc),
          "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": os.path.join(ICI, "parc", "lib")}
    env.pop("XDG_CONFIG_HOME", None)
    r = subprocess.run(["xvfb-run", "-a", "--server-args=-screen 0 1340x900x24", PY, chemin],
                       capture_output=True, text=True, env=env, timeout=60)
    lignes = [l for l in r.stdout.splitlines() if l.startswith("GUI_TEST ")]
    assert len(lignes) == 1, r.stdout[-800:] + r.stderr[-800:]
    d = json.loads(lignes[0][len("GUI_TEST "):])
    assert "erreur" not in d, d
    return d


@pytest.mark.parametrize("gui", ["claude-modeles", "kimi-modeles"])
def test_filtre_vedettes_trouve_opus_et_fable(tmp_path, gui):
    d = _clic(tmp_path, gui, "b_vedettes")
    assert d["actif"] is True
    assert d["visibles"] == 2, d          # ALIAS_OPUS + ALIAS_FABLE, ni le code ni le neutre
    assert d["total"] == 5, d


@pytest.mark.parametrize("gui", ["claude-modeles", "kimi-modeles"])
def test_filtre_code_os_trouve_le_code_exclut_le_temoin(tmp_path, gui):
    d = _clic(tmp_path, gui, "b_code_os")
    assert d["actif"] is True
    assert d["visibles"] == 1, d          # ALIAS_CODE seul ; ALIAS_TEMOIN_CODE est un témoin exclu
    assert d["total"] == 5, d
