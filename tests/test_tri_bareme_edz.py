"""anticitoyen-vram-edz (28/09, ordre chef/utilisateur) : les barèmes de tri
de claude-modeles/kimi-modeles (rang_qualite, rang_refus, Modele.debit) — la
part testable SANS GTK/xvfb affiché, en gardant `sorted(key=...)` comme modèle
exact de ce qu'un clic d'en-tête fait (`_colonne` compare avec
`(f(a)>f(b))-(f(a)<f(b))`, strictement équivalent à `sorted(key=f)`).

`menu_modeles.parc` importe `gi` (Gtk/Adw), absent du venv (torch) — comme
`test_gui_dossiers_modeles.py`, les valeurs sont calculées dans un sous-processus
python3 SYSTÈME (une seule fois, `_valeurs` module-scope), jamais importées ici.

Bogue trouvé en l'écrivant : `rang_qualite` rendait -1 pour une fiche absente
(« non mesuré », ni un mot de NOTE_MOTS ni une étoile) — la valeur la PLUS
BASSE, donc triée AVANT ★ en ascendant, l'inverse de « en fin de liste ».
Même défaut sur `Modele.debit` (« non mesuré » → -1). `rang_refus` avait
déjà la bonne convention (9, « en queue de tri ») — les deux autres alignés
dessus (voir parc.py)."""
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

_SCRIPT = '''
import sys, json
sys.path.insert(0, {lib!r})
from menu_modeles.parc import Modele, rang_qualite, rang_refus
from menu_modeles.config import ORDRE_MOTEUR

def m(qual="", tps="", refus=""):
    return Modele("a", "acvram", "nom", 0, refus, tps, qual, "")

out = {{
    "qual_etoiles": sorted(["\\u2605\\u2605\\u2605\\u2605\\u2605", "\\u2605\\u2605", "\\u2605\\u2605\\u2605\\u2605", "\\u2605"], key=rang_qualite),
    "qual_mots": sorted(["excellent", "moyen", "bon", "tr\\u00e8s bon"], key=rang_qualite),
    "rang_non_mesure": rang_qualite("non mesur\\u00e9"),
    "rang_etoile_5": rang_qualite("\\u2605\\u2605\\u2605\\u2605\\u2605"),
    "refus_mots": sorted(["\\u00e9lev\\u00e9", "aucun", "moyen", "faible"], key=rang_refus),
    "refus_fraction": sorted(["4/5 refus", "1/5 refus", "3/5 refus"], key=rang_refus),
    "rang_inconnu": rang_refus("inconnu"),
    "rang_eleve": rang_refus("\\u00e9lev\\u00e9"),
    "debit_134": m(tps="134,9*").debit,
    "debit_95": m(tps="95").debit,
    "debit_non_mesure": m(tps="non mesur\\u00e9").debit,
    "ordre_moteur": {{k: v for k, v in ORDRE_MOTEUR.items() if k in ("acvram", "vllm", "llamacpp")}},
}}
print(json.dumps(out, ensure_ascii=False))
'''.format(lib=str(ICI / "parc" / "lib"))


@pytest.fixture(scope="module")
def valeurs():
    r = subprocess.run([PY, "-c", _SCRIPT], capture_output=True, text=True, timeout=30,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout)


# ---- Qualité : étoiles numériques, pas lexicales ; mots ; inconnu toujours en fin -------------

def test_qualite_etoiles_triees_numeriquement_pas_lexicalement(valeurs):
    assert valeurs["qual_etoiles"] == ["★", "★★", "★★★★", "★★★★★"]


def test_qualite_mots_ordonnes_bon_avant_excellent(valeurs):
    assert valeurs["qual_mots"] == ["moyen", "bon", "très bon", "excellent"]


def test_qualite_non_mesure_toujours_apres_toute_qualite_connue(valeurs):
    # `rang_qualite` seul ne garantit l'ordre qu'en ASCENDANT (sentinel fixe, 9) ;
    # « en fin de liste DANS LES DEUX SENS » est la garantie du comparateur GTK
    # direction-aware (`Fenetre._comparer_inconnu_en_fin`), vérifié séparément
    # dans test_gui_tri_colonnes_edz.py — un `reverse=True` nu sur ce seul rang
    # renverrait l'inconnu en TÊTE, la faute inverse (piège trouvé en l'écrivant).
    assert valeurs["rang_non_mesure"] > valeurs["rang_etoile_5"]


# ---- Refus : mots, fractions « n/5 », inconnu toujours en fin ----------------------------------

def test_refus_mots_ordonnes_aucun_avant_eleve(valeurs):
    assert valeurs["refus_mots"] == ["aucun", "faible", "moyen", "élevé"]


def test_refus_fraction_triee_numeriquement(valeurs):
    assert valeurs["refus_fraction"] == ["1/5 refus", "3/5 refus", "4/5 refus"]


def test_refus_inconnu_toujours_en_fin(valeurs):
    # même remarque que Qualité : garantie ASCENDANTE seulement à ce niveau.
    assert valeurs["rang_inconnu"] > valeurs["rang_eleve"]


# ---- tok/s (débit) : « 134,9* » > 95, numérique, inconnu toujours en fin -----------------------

def test_debit_compare_les_valeurs_pas_les_chaines(valeurs):
    # lexicalement "134,9*" < "95" (le caractère '1' < '9') ; numériquement 134,9 > 95
    assert valeurs["debit_134"] > valeurs["debit_95"]


def test_debit_ignore_letoile(valeurs):
    assert valeurs["debit_134"] == 134.9


def test_debit_non_mesure_toujours_apres_tout_debit_connu(valeurs):
    # même remarque que Qualité : garantie ASCENDANTE seulement à ce niveau.
    assert valeurs["debit_non_mesure"] > valeurs["debit_134"]


# ---- Moteur : ordre déclaré, pas alphabétique ---------------------------------------------------

def test_moteur_ordre_declare_pas_alphabetique(valeurs):
    o = valeurs["ordre_moteur"]
    if {"acvram", "llamacpp", "vllm"} <= set(o):
        assert o["acvram"] < o["vllm"] < o["llamacpp"]
