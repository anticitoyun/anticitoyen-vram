"""Crochet `ACVRAM_GUI_TEST=trier:<titre>` : simule un clic sur l'en-tête d'une colonne EXACTEMENT comme le
gestionnaire interne du ColumnView le ferait (même appel : `Gtk.ColumnView.sort_by_column`), sur un parc de 60
alias sans fiche (le cas réel qui a fait remonter la pièce 83 : `sans fiche` en masse). Vérifie que le nombre de
libellés non vides sous la liste ne change pas avant/après le tri, pour CHAQUE colonne triable des deux menus.

Ne DÉTECTE PAS un défaut purement visuel (glyphe non repeint sans que la propriété `Gtk.Label.get_text()` change) :
seule une inspection de pixels le ferait. C'est un filet de régression sur les DONNÉES affichées (texte de cellule
resté vide/vidé), pas une preuve d'absence de défaut de peinture — voir revue/livraison-0.1.2-parc.md, pièce 83."""
import json, os, shutil, subprocess, tempfile, pytest

ICI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/usr/bin/python3"
COLONNES = ["Alias", "Moteur", "Qualité", "tok/s (* avant 20/09)", "Refus", "Usage", "Contexte"]


def _gi_ok():
    r = subprocess.run([PY, "-c", "import gi; gi.require_version('Gtk', '4.0'); gi.require_version('Adw', '1'); from gi.repository import Adw"],
                       capture_output=True)
    return r.returncode == 0


pytestmark = pytest.mark.skipif(not shutil.which("xvfb-run") or not _gi_ok(), reason="xvfb-run ou GTK4/libadwaita absent")


def _parc_60_sans_fiche(tmp_path):
    kimi = tmp_path / "kimi"; kimi.mkdir()
    tsv = tmp_path / "tsv"; tsv.mkdir()
    noms = [f"m{i:02d}-{c}" for i, c in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")]
    (kimi / "config.toml").write_text(
        "".join(f'[models.acvram-{a}]\nprovider="acvram"\nmodel="{a}"\n' for a in noms))
    parc = tmp_path / "parc.toml"
    parc.write_text(f'[chemins]\nkimi_dir="{kimi}"\ntsv_dir="{tsv}"\n[moteurs.acvram]\npresent=true\n')
    return parc, len(noms)


@pytest.mark.parametrize("gui", ["claude-modeles", "kimi-modeles"])
@pytest.mark.parametrize("colonne", COLONNES)
def test_tri_colonne_garde_le_texte(tmp_path, gui, colonne):
    parc, _n = _parc_60_sans_fiche(tmp_path)
    chemin = os.path.join(ICI, "parc", "bin", gui)
    env = {**os.environ, "ACVRAM_GUI_TEST": f"trier:{colonne}", "ACVRAM_PARC_CONFIG": str(parc),
           "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": os.path.join(ICI, "parc", "lib")}
    env.pop("XDG_CONFIG_HOME", None)
    r = subprocess.run(["xvfb-run", "-a", "--server-args=-screen 0 1340x900x24", PY, chemin],
                       capture_output=True, text=True, env=env, timeout=60)
    lignes = [l for l in r.stdout.splitlines() if l.startswith("GUI_TEST ")]
    assert len(lignes) == 1, r.stdout[-800:] + r.stderr[-800:]
    d = json.loads(lignes[0][len("GUI_TEST "):])
    assert "erreur" not in d, d
    avant = [t for t in d["avant"] if t.strip()]
    apres = [t for t in d["apres"] if t.strip()]
    assert len(avant) > 0, "rien de réalisé avant le tri : le crochet ne teste rien"
    assert len(apres) == len(avant), (colonne, len(avant), len(apres))
