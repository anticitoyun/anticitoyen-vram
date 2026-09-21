"""poste7 (verdict-vm-parc-21-09) : sur un parc VIDE (premier lancement, aucun moteur/modèle), les lanceurs
plantaient sous `set -u` (« id: unbound variable », « alias: unbound variable ») au lieu de dire « parc vide »
et sortir proprement. À sec, aucune carte, aucun serveur : parc.toml et config.toml réellement vides."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
PARC = RACINE / "parc"


def test_choisir_alias_sur_lister_alias_vide_rc0_message_nomme(tmp_path):
    """Unit direct : la fonction seule, sans passer par un lanceur entier."""
    script = tmp_path / "essai.sh"
    script.write_text(f'''#!/usr/bin/env bash
set -uo pipefail
err() {{ printf '%s\\n' "$*" >&2; }}
lister_alias() {{ :; }}  # rien : parc vide
CONFIG=/nulle/part
. "{PARC / "share" / "kimi-menu.lib.sh"}"
choisir_alias "titre" ""
echo "rc-interne=$?"
''')
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "parc vide" in r.stderr, r.stderr
    assert "unbound variable" not in r.stderr, r.stderr
    assert "rc-interne=0" in r.stdout, r.stdout


def test_kimi_modele_parc_vide_bout_en_bout_rc0(tmp_path):
    """De bout en bout : parc.toml sans moteur, config.toml sans [models.*] → rc 0, message nommé, rien lancé."""
    home = tmp_path / "home"; (home / ".config" / "acvram-parc").mkdir(parents=True)
    (home / ".kimi-code").mkdir()
    (home / ".kimi-code" / "config.toml").write_text('default_model = ""\n')  # aucun [models.*]
    (home / ".config" / "acvram-parc" / "parc.toml").write_text(
        f'[chemins]\nlib = "{PARC / "share" / "kimi-menu.lib.sh"}"\n[racines]\nmodeles = []\n[moteurs]\n[outils]\n[extras]\n')
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin", "ACVRAM_PARC_CONFIG": str(home / ".config" / "acvram-parc" / "parc.toml")}
    for lanceur in ("kimi-modele", "claude-modele"):
        r = subprocess.run(["bash", str(PARC / "bin" / lanceur)], capture_output=True, text=True, env=env, timeout=30)
        assert r.returncode == 0, f"{lanceur}: rc={r.returncode}\n{r.stdout}\n{r.stderr}"
        assert "parc vide" in r.stderr, f"{lanceur}: {r.stderr}"
        assert "unbound variable" not in r.stderr, f"{lanceur}: {r.stderr}"


def test_faute_construite_message_generique_ancien_ne_suffit_pas():
    """Faute construite : l'ancien message « Aucun modèle dans » n'est pas nommé « parc vide » — ce test
    est celui qui aurait dû détecter la régression ; il tombe si le message générique revient sans le mot-clé."""
    import re
    contenu = (PARC / "share" / "kimi-menu.lib.sh").read_text()
    ligne = re.search(r'^\s*\[ "\$\{#alias\[@\]\}" -gt 0 \].*$', contenu, re.M)
    assert ligne, "la garde de vide sur alias[] a disparu de choisir_alias"
    assert "parc vide" in ligne.group(0), "le message n'est plus « parc vide »"
    assert "return 0" in ligne.group(0), "le rc après « parc vide » n'est plus 0"
