"""anticitoyen-vram-edz (28/09, ordre chef/utilisateur) : les colonnes de
claude-modeles/kimi-modeles doivent afficher ce que porte notes-modeles.tsv
(tok/s de la campagne 290 de poste2, qualité, refus, usage), jamais une cellule
vide ni une valeur DÉCALÉE dans la mauvaise colonne — vérifié sur le vrai
`charger_parc()` avec des lignes fabriquées pour casser : une ligne à un champ
manquant (4 au lieu de 5, alias sans « usage »), une avec une TABULATION
NICHÉE dans le champ usage (copié-collé plausible, 6 colonnes au lieu de 5).

Vérification réelle du dépôt au moment d'écrire cette pièce (28/09) :
`charger_parc()` sur ~/TSV/notes-modeles.tsv réel → 281 modèles, 0 alias sans
fiche, 0 hors moteur, 0 cellule qual/tps/refus vide (28/09, campagne 290 en
cours) — rien à corriger côté données ; ce test protège la LOGIQUE de lecture
pour la prochaine fois qu'une ligne sera mal formée (les 66 alias nouveaux de
la campagne, ou n'importe quel alias futur)."""
import json
import os
import subprocess
from pathlib import Path

import pytest

ICI = Path(__file__).resolve().parent.parent
PY = "/usr/bin/python3"


def _gi_ok():
    r = subprocess.run([PY, "-c", "import gi; gi.require_version('Gtk', '4.0'); gi.require_version('Adw', '1'); "
                                    "from gi.repository import Adw"], capture_output=True)
    return r.returncode == 0


pytestmark = pytest.mark.skipif(not _gi_ok(), reason="GTK4/libadwaita absent")


def _parc(tmp_path, notes_lignes):
    kimi = tmp_path / "kimi"; kimi.mkdir()
    tsv = tmp_path / "tsv"; tsv.mkdir()
    noms = ["complet", "champ-manquant", "tab-niche-dans-usage", "sans-fiche-du-tout"]
    (kimi / "config.toml").write_text(
        "".join(f'[models.acvram-{a}]\nprovider="acvram"\nmodel="{a}"\n' for a in noms))
    (tsv / "notes-modeles.tsv").write_text(notes_lignes)
    parc = tmp_path / "parc.toml"
    parc.write_text(f'[chemins]\nkimi_dir="{kimi}"\ntsv_dir="{tsv}"\n[moteurs.acvram]\npresent=true\n')
    return parc


def _charger(parc):
    script = f'''
import sys, json
sys.path.insert(0, {str(ICI / "parc" / "lib")!r})
from menu_modeles import parc as parcmod
p = parcmod.charger_parc()
print(json.dumps([{{"alias": m.alias, "qual": m.qual, "tps": m.tps,
                    "refus": m.refus, "usage": m.usage}} for m in p], ensure_ascii=False))
'''
    r = subprocess.run([PY, "-c", script], capture_output=True, text=True, timeout=30,
                       env={**os.environ, "ACVRAM_PARC_CONFIG": str(parc)})
    assert r.returncode == 0, r.stdout + r.stderr
    lignes = [l for l in r.stdout.splitlines() if l.startswith("[")]
    assert lignes, r.stdout + r.stderr
    return {m["alias"]: m for m in json.loads(lignes[-1])}


def test_ligne_complete_affiche_ses_quatre_champs_sans_decalage(tmp_path):
    notes = "acvram-complet\taucun\t134,9*\t★★★★★\tcode\n"
    d = _charger(_parc(tmp_path, notes))
    m = d["acvram-complet"]
    assert (m["refus"], m["tps"], m["qual"], m["usage"]) == ("aucun", "134,9*", "★★★★★", "code")


def test_champ_manquant_tombe_en_sans_fiche_jamais_un_decalage(tmp_path):
    # 4 colonnes au lieu de 5 (usage manquant) : `lire_tsv(mini=5)` doit
    # IGNORER la ligne entière — jamais lire "★★★" comme usage ou "aucun"
    # comme qualité (un décalage d'une colonne vers la gauche).
    notes = "acvram-champ-manquant\taucun\t50\t★★★\n"
    d = _charger(_parc(tmp_path, notes))
    m = d["acvram-champ-manquant"]
    assert m["qual"] == "non mesuré", m       # jamais "★★★" pris pour un refus décalé
    assert m["refus"] == "inconnu", m
    assert m["tps"] == "non mesuré", m
    assert m["usage"] == "inconnu", m


def test_tabulation_nichee_dans_usage_ne_deborde_pas_sur_les_autres_alias(tmp_path):
    # 6 colonnes (une tabulation en trop dans "usage") : la ligne SUIVANTE
    # (alias distinct) ne doit jamais hériter d'un fragment de la précédente.
    notes = (
        "acvram-tab-niche-dans-usage\taucun\t95\t★★\tcode\tagent\n"
        "acvram-sans-fiche-du-tout\taucun\t20\t★\tchat\n"
    )
    d = _charger(_parc(tmp_path, notes))
    # la 5e valeur ("code") est prise comme usage, la 6e ("agent") ignorée —
    # jamais lue comme une nouvelle ligne ni fusionnée dans l'alias suivant.
    assert d["acvram-tab-niche-dans-usage"]["usage"] == "code"
    assert d["acvram-sans-fiche-du-tout"]["refus"] == "aucun"
    assert d["acvram-sans-fiche-du-tout"]["tps"] == "20"
    assert d["acvram-sans-fiche-du-tout"]["qual"] == "★"
    assert d["acvram-sans-fiche-du-tout"]["usage"] == "chat"


def test_alias_absent_du_tsv_reste_visible_avec_des_placeholders_jamais_vide(tmp_path):
    d = _charger(_parc(tmp_path, "acvram-complet\taucun\t100\t★★★★★\tcode\n"))
    m = d["acvram-sans-fiche-du-tout"]
    assert m["qual"] and m["tps"] and m["refus"] and m["usage"], m   # jamais ""
    assert m["qual"] == "non mesuré" and m["refus"] == "inconnu"
