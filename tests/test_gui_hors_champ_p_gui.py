"""GUI, pièce P1 (chef, 29/09, corrigée après revue) : le seuil « mode réduit »/« refus » est
PAR CLIENT, tiré du code réel des lanceurs CLI, jamais une constante partagée entre claude-modeles
et kimi-modeles :
  - claude-modele : SEUIL_COMPLET=45000 (parc/bin/claude-modele:94, mode_outils()) — sous ce
    seuil, mode « essentiel » (Read/Edit/Bash/Grep, sans MCP), l'alias se lance quand même ;
    FENETRE_MIN=29120 (PROMPT_BASE_ESSENTIEL 11000 + compaction 13000 + conversation 4096 + sortie
    1024, tête du fichier ; ph1 30/09, 15096 avant menus) — sous CE seuil, claude-modele refuse
    (mesure de l'invite, err + exit 1), rien ne se lance. La GUI LIT ces seuils (`--seuils`),
    tests/test_seuils_lanceur_ph1.py ; aucune copie.
  - kimi-modele : KIMI_MCP_CTX_MIN=65536 (tête du fichier, lu par `--seuils` — 90q 30/09) — sous ce seuil, MCP
    coupés, l'alias se lance quand même ; aucun plancher dur générique (kimi-modele ne refuse
    jamais lui-même pour une fenêtre trop petite, hors le repli propre à l'engin acvram, géré
    par acvram-serveur).

La colonne Contexte marque ⛔ (refus) ou ◐ (réduit) avec la raison en infobulle ; « Ouvrir »
avertit avant de lancer en mode réduit (le client se lance normalement) et refuse SANS spawn en
mode refus (seul cas réel, propre à claude)."""
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


def _parc(tmp_path, ctx):
    kimi = tmp_path / "kimi"; kimi.mkdir()
    tsv = tmp_path / "tsv"; tsv.mkdir()
    (kimi / "config.toml").write_text(
        f'[models.acvram-m]\nprovider="acvram"\nmodel="M"\nmax_context_size={ctx}\n')
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


def test_claude_sous_seuil_minimum_refuse_sans_spawn(tmp_path):
    parc = _parc(tmp_path, 10000)
    rc, r = _jouer(parc, "claude-modeles", "clic:b_ouvrir@acvram-m")
    assert rc == 0 and r["spawns"] == []
    assert any("10 000" in t and "29 120" in t for t in r["toasts"]), r["toasts"]


def test_claude_entre_15096_et_29120_refuse_sans_spawn(tmp_path):
    """ph1 : le cas du bogue — 20 000 passait la copie 15 096 de la GUI, puis claude-modele refusait (29 120)."""
    parc = _parc(tmp_path, 20000)
    rc, r = _jouer(parc, "claude-modeles", "clic:b_ouvrir@acvram-m")
    assert rc == 0 and r["spawns"] == []
    assert any("20 000" in t and "29 120" in t for t in r["toasts"]), r["toasts"]


def test_claude_mode_reduit_entre_29120_et_45000_ne_bloque_pas(tmp_path):
    parc = _parc(tmp_path, 30000)
    rc, r = _jouer(parc, "claude-modeles", "clic:b_ouvrir@acvram-m")
    assert rc == 0
    assert any("30 000" in t and "45 000" in t and "réduit" in t for t in r["toasts"]), r["toasts"]


def test_claude_complet_au_dela_de_45000_pas_de_mention(tmp_path):
    parc = _parc(tmp_path, 50000)
    rc, r = _jouer(parc, "claude-modeles", "clic:b_ouvrir@acvram-m")
    assert rc == 0
    assert not any("45 000" in t or "29 120" in t for t in r["toasts"]), r["toasts"]


def test_kimi_ne_refuse_jamais_meme_tres_bas(tmp_path):
    """kimi-modele n'a pas de plancher dur générique (seuil_minimum=None côté profil) :
    même une fenêtre minuscule reste « mode réduit », jamais « refus »."""
    parc = _parc(tmp_path, 1000)
    rc, r = _jouer(parc, "kimi-modeles", "clic:b_ouvrir@acvram-m")
    assert rc == 0
    assert any("1 000" in t and "65 536" in t and "réduit" in t for t in r["toasts"]), r["toasts"]
    assert not any("refuse" in t for t in r["toasts"]), r["toasts"]


def test_kimi_complet_au_dela_de_65536_pas_de_mention(tmp_path):
    parc = _parc(tmp_path, 70000)
    rc, r = _jouer(parc, "kimi-modeles", "clic:b_ouvrir@acvram-m")
    assert rc == 0
    assert not any("65 536" in t for t in r["toasts"]), r["toasts"]
