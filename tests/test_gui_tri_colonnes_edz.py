"""anticitoyen-vram-edz (28/09, ordre chef/utilisateur) : chaque colonne
triable de claude-modeles/kimi-modeles triée COMME UN NOMBRE (134,9 > 95,
jamais l'ordre lexical des chaînes « 134,9* » < « 95 »), les étoiles de
qualité et le rang de refus comme des barèmes, « inconnu »/« non mesuré » en
fin de liste — DANS LES DEUX SENS (ascendant et descendant).

Complète `test_gui_tri_colonnes.py` (pièce 83, qui vérifie seulement que le
TEXTE ne se vide pas au tri, jamais l'ORDRE) sans le remplacer. Utilise
`self.selection` (le modèle RÉELLEMENT trié) plutôt que le texte des
`Gtk.Label` affichés : un `Gtk.ColumnView` virtualise les lignes hors écran,
`_textes_visibles()` n'en verrait qu'une partie — sans incidence ici (parc
de 6 lignes, toutes visibles), mais `ordre_avant`/`ordre_apres` (crochet
`trier:`, étendu par cette pièce) donne l'ordre vrai quel que soit l'écran.

Bogue trouvé et corrigé en l'écrivant : un clic d'en-tête simple
(`Gtk.ColumnView.sort_by_column`) INVERSE tout le comparateur pour
DESCENDING, en aveugle — un sentinel figé (« inconnu » = valeur la plus
haute ou la plus basse) ne peut donc rester « en fin de liste » que dans UN
SEUL sens. `Fenetre._comparer_inconnu_en_fin` (Qualité, tok/s, Refus)
interroge le sens réellement actif au moment de comparer et compense — les
témoins ci-dessous cassent sans ce correctif (voir le journal git)."""
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

# alias -> (refus, tps, qual, usage) ; m6 SANS fiche (charger_parc y met
# « non mesuré »/« non mesuré »/« inconnu » — le cas réel qui a fait remonter
# la pièce 83, ici en petit nombre pour un ordre entièrement vérifiable).
LIGNES = {
    "m1": ("aucun", "50", "★★", "code"),
    "m2": ("faible", "134,9*", "★★★★★", "chat"),
    "m3": ("moyen", "95", "★", "code"),
    "m4": ("élevé", "312,3", "★★★★", "agent"),
    "m5": ("aucun", "0", "★★★", "chat"),
}
SANS_FICHE = "m6"


def _parc_ordonne(tmp_path):
    kimi = tmp_path / "kimi"; kimi.mkdir()
    tsv = tmp_path / "tsv"; tsv.mkdir()
    tous = list(LIGNES) + [SANS_FICHE]
    (kimi / "config.toml").write_text(
        "".join(f'[models.acvram-{a}]\nprovider="acvram"\nmodel="{a}"\n' for a in tous))
    lignes_tsv = "\n".join(f"acvram-{a}\t{r}\t{t}\t{q}\t{u}" for a, (r, t, q, u) in LIGNES.items())
    (tsv / "notes-modeles.tsv").write_text(lignes_tsv + "\n")
    parc = tmp_path / "parc.toml"
    parc.write_text(f'[chemins]\nkimi_dir="{kimi}"\ntsv_dir="{tsv}"\n[moteurs.acvram]\npresent=true\n')
    return parc


def _trier(parc, gui, colonne, inverse=False):
    chemin = os.path.join(ICI, "parc", "bin", gui)
    arg = f"trier:{colonne}" + (":inverse" if inverse else "")
    env = {**os.environ, "ACVRAM_GUI_TEST": arg, "ACVRAM_PARC_CONFIG": str(parc),
           "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": os.path.join(ICI, "parc", "lib")}
    env.pop("XDG_CONFIG_HOME", None)
    r = subprocess.run(["xvfb-run", "-a", "--server-args=-screen 0 1340x900x24", PY, chemin],
                       capture_output=True, text=True, env=env, timeout=60)
    lignes = [l for l in r.stdout.splitlines() if l.startswith("GUI_TEST ")]
    assert len(lignes) == 1, r.stdout[-800:] + r.stderr[-800:]
    d = json.loads(lignes[0][len("GUI_TEST "):])
    assert "erreur" not in d, d
    ordre = d.get("ordre_apres")
    assert ordre, (colonne, d)
    return [m["alias"] for m in ordre]


@pytest.mark.parametrize("gui", ["claude-modeles", "kimi-modeles"])
def test_qualite_ordre_numerique_inconnu_en_fin_deux_sens(tmp_path, gui):
    parc = _parc_ordonne(tmp_path)
    asc = _trier(parc, gui, "Qualité")
    assert asc == ["acvram-m3", "acvram-m1", "acvram-m5", "acvram-m4", "acvram-m2", "acvram-m6"]
    desc = _trier(parc, gui, "Qualité", inverse=True)
    assert desc == ["acvram-m2", "acvram-m4", "acvram-m5", "acvram-m1", "acvram-m3", "acvram-m6"]


@pytest.mark.parametrize("gui", ["claude-modeles", "kimi-modeles"])
def test_debit_ordre_numerique_pas_lexical_inconnu_en_fin_deux_sens(tmp_path, gui):
    # lexicalement : "0" < "134,9*" < "312,3" < "50" < "95" — si l'ordre observé suivait
    # ça, le tri serait sur la CHAÎNE, pas la valeur. Numériquement : 0 < 50 < 95 < 134,9 < 312,3.
    parc = _parc_ordonne(tmp_path)
    asc = _trier(parc, gui, "tok/s (* avant 20/09)")
    assert asc == ["acvram-m5", "acvram-m1", "acvram-m3", "acvram-m2", "acvram-m4", "acvram-m6"]
    desc = _trier(parc, gui, "tok/s (* avant 20/09)", inverse=True)
    assert desc == ["acvram-m4", "acvram-m2", "acvram-m3", "acvram-m1", "acvram-m5", "acvram-m6"]


@pytest.mark.parametrize("gui", ["claude-modeles", "kimi-modeles"])
def test_refus_ordre_bareme_inconnu_en_fin_deux_sens(tmp_path, gui):
    # aucun(0) < faible(2) < moyen(3) < élevé(4) < inconnu(9) ; deux « aucun » (m1, m5)
    # ex æquo — stabilité non garantie entre eux, seul leur GROUPE compte ici.
    parc = _parc_ordonne(tmp_path)
    asc = _trier(parc, gui, "Refus")
    assert set(asc[:2]) == {"acvram-m1", "acvram-m5"}
    assert asc[2:] == ["acvram-m2", "acvram-m3", "acvram-m4", "acvram-m6"]
    desc = _trier(parc, gui, "Refus", inverse=True)
    assert desc[:3] == ["acvram-m4", "acvram-m3", "acvram-m2"]
    assert set(desc[3:5]) == {"acvram-m1", "acvram-m5"}
    assert desc[5] == "acvram-m6"


@pytest.mark.parametrize("gui", ["claude-modeles", "kimi-modeles"])
def test_alias_ordre_lexical(tmp_path, gui):
    parc = _parc_ordonne(tmp_path)
    asc = _trier(parc, gui, "Alias")
    assert asc == sorted(asc)


@pytest.mark.parametrize("gui", ["claude-modeles", "kimi-modeles"])
def test_usage_ordre_lexical(tmp_path, gui):
    parc = _parc_ordonne(tmp_path)
    asc = _trier(parc, gui, "Usage")
    usages = {"acvram-m1": "code", "acvram-m2": "chat", "acvram-m3": "code",
              "acvram-m4": "agent", "acvram-m5": "chat", "acvram-m6": "inconnu"}
    assert [usages[a] for a in asc] == sorted(usages.values())
