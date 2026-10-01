"""bd e50.1 (P1, demandé par l'utilisatrice, relayé par chef 01/10) : les capacités affichées
au menu étaient sparses — `m.capacites` venait SEULEMENT de `config.toml` (« capabilities »,
posée ailleurs, jamais par ce dépôt) : 14 alias « thinking », 1 seul « video_in » pour une
trentaine de noms VL, aucune étiquette pour les 92 noms « heretic » ni les ~30 « abliterated ».

Chaque capacité doit venir d'une SOURCE nommée, jamais inventée (`parc.py:deriver_capacites`) :
alias/nom pour heretic/abliterated/sans-censure/nsfw/coder ; `config.json` (`vision_config`,
jeton vidéo) pour image_in/video_in ; `chat_template` de `tokenizer_config.json` (`<think>`,
appels d'outils) pour thinking/tools.

Ce test tourne sur le PARC RÉEL de ce poste (`charger_parc()`, aucun ACVRAM_PARC_CONFIG) — via
`/usr/bin/python3` en sous-processus (le gi/PyGObject de `menu_modeles.parc` n'est pas dans le
venv de pytest). Il saute si le parc réel est absent ou trop petit pour être significatif."""
import json
import os
import re
import shutil
import subprocess
import pytest

ICI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/usr/bin/python3"


def _gi_ok():
    r = subprocess.run([PY, "-c", "import gi; gi.require_version('Gtk', '4.0'); gi.require_version('Adw', '1'); from gi.repository import Adw"],
                       capture_output=True)
    return r.returncode == 0


def _assez_de_parc_reel():
    tsv = os.path.expanduser("~/TSV/acvram-chemins.tsv")
    try:
        return sum(1 for l in open(tsv) if l.strip() and not l.startswith("#")) >= 100
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not shutil.which(PY) or not _gi_ok() or not _assez_de_parc_reel(),
    reason="GTK4/libadwaita, /usr/bin/python3 ou parc réel (~/TSV, ≥100 alias) absent")

SCRIPT = r"""
import sys, json
sys.path.insert(0, "/usr/share/acvram-parc/lib")
sys.path.insert(0, %(parc_lib)r)
from menu_modeles.parc import charger_parc
parc = charger_parc()
out = [{"alias": m.alias, "nom": m.nom, "caps": sorted(m.capacites),
       "vision_status": m.vision_status, "dossier": m.dossier} for m in parc]
print(json.dumps(out))
"""


def _parc_reel():
    parc_lib = os.path.join(ICI, "parc", "lib")
    r = subprocess.run([PY, "-c", SCRIPT % {"parc_lib": parc_lib}],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-2000:]
    lignes = [l for l in r.stdout.splitlines() if l.startswith("[")]
    assert lignes, r.stdout[-2000:] + r.stderr[-2000:]
    return json.loads(lignes[-1])


def test_assez_dalias_pour_etre_significatif():
    parc = _parc_reel()
    assert len(parc) >= 100, f"parc réel trop petit pour juger la dérivation : {len(parc)} alias"


def test_heretic_et_abliterated_dans_le_nom_portent_leur_etiquette():
    parc = _parc_reel()
    heretic_sans_etiquette = [m["alias"] for m in parc
                              if "heretic" in m["alias"].lower() and "heretic" not in m["caps"]]
    abliterated_sans_etiquette = [m["alias"] for m in parc
                                  if "abliterated" in m["alias"].lower() and "abliterated" not in m["caps"]]
    assert not heretic_sans_etiquette, (
        f"alias « heretic » sans l'étiquette dédiée : {heretic_sans_etiquette[:5]} "
        f"(+{max(0, len(heretic_sans_etiquette) - 5)} autres)")
    assert not abliterated_sans_etiquette, (
        f"alias « abliterated » sans l'étiquette dédiée : {abliterated_sans_etiquette[:5]} "
        f"(+{max(0, len(abliterated_sans_etiquette) - 5)} autres)")
    # preuve que le contrôle compte quelque chose de réel, pas un ensemble vide par coïncidence
    assert sum(1 for m in parc if "heretic" in m["caps"]) >= 50, "trop peu d'alias heretic réels"
    assert sum(1 for m in parc if "abliterated" in m["caps"]) >= 10, "trop peu d'alias abliterated réels"


def test_vision_status_vision_porte_toujours_image_in():
    parc = _parc_reel()
    manquants = [m["alias"] for m in parc if m["vision_status"] == "vision" and "image_in" not in m["caps"]]
    assert not manquants, f"vision_status=vision sans image_in : {manquants}"
    assert sum(1 for m in parc if m["vision_status"] == "vision") >= 5, "trop peu d'alias vision réels"


def test_config_json_video_token_porte_video_in():
    """Repris directement de la source (bd e50.1) : un config.json avec une clé `video_*`
    (`video_token_id`, Qwen3-VL) doit donner `video_in`, QUE le dossier soit déclaré vision ou
    non — le token vidéo est son propre signal, indépendant de vision_status (mmproj llama.cpp)."""
    parc = _parc_reel()
    candidats = [m for m in parc if m["dossier"]]
    verifies = 0
    manquants = []
    for m in candidats:
        chemin = os.path.join(m["dossier"], "config.json")
        try:
            with open(chemin, encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            continue
        if any(k.startswith("video_") for k in cfg):
            verifies += 1
            if "video_in" not in m["caps"]:
                manquants.append(m["alias"])
    assert verifies >= 5, "trop peu de config.json avec un jeton vidéo pour juger"
    assert not manquants, f"config.json avec jeton vidéo mais sans l'étiquette video_in : {manquants}"


def test_le_controle_peut_rendre_faux():
    """Bras cassant direct : un alias dont le NOM ne contient ni heretic ni abliterated ne doit
    PAS porter l'étiquette — sinon le contrôle ci-dessus ne détecterait jamais une régression qui
    poserait l'étiquette partout."""
    parc = _parc_reel()
    # la source est « alias + nom » (le modèle sous-jacent, deriver_capacites) — un témoin
    # doit n'avoir ni mot dans NI L'UN NI L'AUTRE, sinon « abl » dans le nom réel d'un alias
    # au nom court (ex. llamacpp-agents-4b-kimi -> Agents-A1-4B-kimi-heretic-Q6_K) ressort à
    # tort comme faux positif alors que la source le porte légitimement.
    def _texte(m):
        return f"{m['alias']} {m.get('nom') or ''}".lower()
    _mot_abl = re.compile(r"abliterated|ablit[ée]rated|(?<![a-z])abl(?![a-z])", re.I)
    temoins = [m for m in parc if "heretic" not in _texte(m) and not _mot_abl.search(_texte(m))]
    assert temoins, "aucun témoin sans heretic/abliterated dans alias+nom — le contrôle est aveugle"
    faux_positifs = [m["alias"] for m in temoins if "heretic" in m["caps"] or "abliterated" in m["caps"]]
    assert not faux_positifs, (
        f"étiquette heretic/abliterated posée sans le mot dans le nom : {faux_positifs[:5]} — "
        "ce n'est plus une dérivation depuis la source, c'est inventé")
