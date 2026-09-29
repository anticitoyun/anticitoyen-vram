"""Commun claude-modeles / kimi-modeles — parc et constantes partagées.

Le seul endroit où le parc (acvram-parc) est chargé pour les deux menus : chemins
des TSV, coffre de secrets, moteurs déclarés présents, barèmes de tri des fiches.
Les constantes DIVERGENTES (icône, couleurs, bouton web, chemins d'outils propres
à chaque menu) restent dans chaque lanceur.
"""
import os
import re
import sys
from pathlib import Path

# ─── parc.toml (acvram-parc) : le seul endroit où vivent chemins, ports et moteurs ───
sys.path.insert(0, "/usr/share/acvram-parc/lib")
_LIB_LOCALE = Path(__file__).resolve().parent.parent
if (_LIB_LOCALE / "acvram_parc.py").exists():
    sys.path.insert(0, str(_LIB_LOCALE))
from acvram_parc import charger as _charger_parc, ORDRE_MOTEUR as _ORDRE_PARC  # noqa: E402

from .moteur import Moteur  # noqa: E402

PARC = _charger_parc()
MAISON = Path.home()
KIMI_DIR = PARC.kimi_dir
TSV_DIR = PARC.tsv_dir
CONFIG = PARC.config_kimi
NOTES = PARC.tsv("notes")
GGUF_TSV = PARC.tsv("gguf")
VLLM_TSV = PARC.tsv("vllm")
ACVRAM_TSV = PARC.tsv("acvram")
VISION_TSV = PARC.tsv("vision")   # alias → vision | texte-seul (modeles-a-jour)
SECRETS = PARC.secrets
BIN = PARC.bin
DOSSIERS_LANCEMENT = list(PARC.dossiers_lancement)
DOSSIER_LANCEMENT_DEFAUT = DOSSIERS_LANCEMENT[0]
# Test en processus, sans clic ni X pilotable (Broadway rend un canvas noir, xdotool exige sudo) :
# ACVRAM_GUI_TEST=clic:<attribut du bouton>[@<alias>] | filtre:<texte> | trier:<titre de colonne> — le gestionnaire est appelé comme par un clic,
# À SEC (aucun processus lancé, aucune URI ouverte : argv et URI journalisés), le retour est imprimé en une ligne
# `GUI_TEST {…}` puis la fenêtre quitte ; rc 2 si le bouton n'existe pas. poste3 le joue sous Xephyr/Xvfb.
GUI_TEST = os.environ.get("ACVRAM_GUI_TEST")

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Moteurs déclarés présents dans parc.toml, et eux seuls : un moteur absent n'a
# ni alias ni port sondé.
MOTEURS = {
    cle: Moteur(cle, m["nom"], int(m["port"]), m["route"], m["var_cle"], m["cle_defaut"],
                m.get("journal") or None, m.get("tokens") or "")
    for cle, m in PARC.moteurs.items()
}
ORDRE_MOTEUR = {cle: _ORDRE_PARC.get(cle, 9) for cle in MOTEURS}
# alias sans fiche dans notes-modeles.tsv : jamais « ? » (poste7-menus-cloture-19-09 § 1)
FICHE_ABSENTE = ["inconnu", "non mesuré", "non mesuré", "inconnu"]

# ─── données ──────────────────────────────────────────────────────────────────
NOTE_MOTS = {"★★★★★": 5, "★★★★": 4, "★★★": 3, "★★": 2, "★": 1,
             "excellent": 5, "très bon": 4, "tres bon": 4, "bon": 3, "moyen": 2}
REFUS_RANG = {"aucun": 0, "nul": 0, "rares": 1, "très faible": 1, "tres faible": 1,
              "faible": 2, "moyen": 3, "élevé": 4, "eleve": 4}
