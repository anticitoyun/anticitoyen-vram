"""t5e (poste1, 27/09, revue/poste1-t5e-verdict-27-09.md §11) : claude-modeles
et kimi-modeles affichaient deux lignes pour un seul modèle converti (deux
alias de config.toml pointant le même dossier — renommer-convertis.py ajoute
un alias neuf à chaque renommage et garde l'ancien, à dessein). Ordre de
chef (28/09) après une première version qui retirait l'alias de
config.toml : jamais ça — retirer un alias casse ce qui l'appelle par son
nom (scripts, campagnes en cours, habitudes). Le dédoublonnage se fait
UNIQUEMENT à l'affichage (`menu_modeles.parc._dedoublonner_par_dossier`) :
une ligne par dossier dans le parc rendu par `charger_parc()`, zéro octet
retiré de config.toml ni du TSV — l'alias masqué reste utilisable en ligne
de commande (`claude-modele <alias>`).

Module `gi`/GTK4, donc python3 système (voir test_gui_dossiers_modeles.py,
même convention) : sous-processus, ACVRAM_PARC_CONFIG pointé sous tmp_path
pour ne jamais lire le vrai poste."""
import os
import subprocess

ICI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/usr/bin/python3"


def _gi_ok():
    r = subprocess.run([PY, "-c", "import gi; gi.require_version('Gtk', '4.0'); gi.require_version('Adw', '1'); "
                                    "from gi.repository import Adw"], capture_output=True)
    return r.returncode == 0


import pytest  # noqa: E402

pytestmark = pytest.mark.skipif(not _gi_ok(), reason="GTK4/libadwaita absent")


def test_67_doublons_font_67_lignes_de_moins_zero_alias_retire(tmp_path):
    kimi_dir = tmp_path / "kimi"
    tsv_dir = tmp_path / "TSV"
    kimi_dir.mkdir()
    tsv_dir.mkdir()
    parc_toml = tmp_path / "parc.toml"
    parc_toml.write_text(
        f'[chemins]\nkimi_dir = "{kimi_dir}"\ntsv_dir = "{tsv_dir}"\n'
        '[moteurs.acvram]\npresent = true\n'
    )

    N = 67
    lignes_tsv = []
    entrees_toml = []
    for i in range(N):
        dossier = f"/mnt/x/modele-{i}-8B-nvfp4"
        canon = f"acvram-modele-{i}-8b-nvfp4"     # ce que produirait renommer-convertis.py aujourd'hui
        legacy = f"acvram-modele-{i}-srci1q4km"    # ancien alias, gardé par design
        lignes_tsv.append(f"{canon}\t{dossier}\t32768")
        lignes_tsv.append(f"{legacy}\t{dossier}\t32768")
        for alias in (canon, legacy):
            entrees_toml.append(
                f'[models.{alias}]\nprovider = "acvram"\nmodel = "modele-{i}"\nmax_context_size = 32768\n'
            )
    (tsv_dir / "acvram-chemins.tsv").write_text("\n".join(lignes_tsv) + "\n")
    (kimi_dir / "config.toml").write_text("\n".join(entrees_toml))
    toml_avant = (kimi_dir / "config.toml").read_text()

    script = f'''
import sys
sys.path.insert(0, {os.path.join(ICI, "parc", "lib")!r})
from menu_modeles import parc as parcmod
p = parcmod.charger_parc()
print("N_PARC", len(p))
print("ALIAS", ",".join(sorted(m.alias for m in p)))
'''
    env = {**os.environ, "ACVRAM_PARC_CONFIG": str(parc_toml)}
    r = subprocess.run([PY, "-c", script], capture_output=True, text=True, env=env, timeout=30)
    assert r.returncode == 0, r.stdout + r.stderr

    lignes = r.stdout.splitlines()
    n_parc = int([l for l in lignes if l.startswith("N_PARC")][0].split()[1])
    alias_affiches = [l for l in lignes if l.startswith("ALIAS")][0].split(" ", 1)[1].split(",")

    assert n_parc == N, f"attendu {N} lignes affichées (2N - N doublons), vu {n_parc}"
    for i in range(N):
        assert f"acvram-modele-{i}-8b-nvfp4" in alias_affiches      # le canonique, affiché
        assert f"acvram-modele-{i}-srci1q4km" not in alias_affiches  # le legacy, masqué au menu

    # config.toml : zéro octet retiré, y compris les alias masqués au menu.
    assert (kimi_dir / "config.toml").read_text() == toml_avant
    for i in range(N):
        assert f"[models.acvram-modele-{i}-srci1q4km]" in toml_avant
