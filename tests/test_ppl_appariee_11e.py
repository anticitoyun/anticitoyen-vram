"""outils/gpu/mesure/ppl-appariee-bootstrap.py (11e) : sur des fenêtres synthétiques, un modèle identique rend Δ ≈ 0 et
|z| < 2 (non conclusif) ; un modèle 3 % pire partout rend Δ ≈ +3 % et z > 2 ; des fenêtres non appariées → ÉCHEC (rc 2).
Cassure : remplacer le tirage commun par un tirage par modèle → le test « identique » garde Δ ≈ 0 mais z n'est plus ≈ 0
(σ gonflée) — le test « 3 % » avec z > 2 casse sur des fenêtres bruitées."""
import json
import os
import random
import subprocess
import sys

OUTIL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outils", "gpu", "mesure", "ppl-appariee-bootstrap.py")


def _json(tmp_path, nom, fenetres):
    p = tmp_path / nom
    p.write_text(json.dumps([{"par_fenetre": fenetres}]))
    return str(p)


def _lancer(*chemins):
    r = subprocess.run([sys.executable, OUTIL, *chemins, "--tirages", "2000"], capture_output=True, text=True)
    return r.returncode, r.stdout


def test_identique_non_conclusif_et_pire_conclusif(tmp_path):
    rng = random.Random(1)
    a = [(2048 * rng.uniform(1.5, 2.5), 2048) for _ in range(40)]      # nll moyen ~2 par position, très variable
    b = [(s * 1.03, n) for s, n in a]                                  # 3 % de nll en plus partout → PPL ×e^(0,06)
    rc, out = _lancer(_json(tmp_path, "a.json", a), _json(tmp_path, "a2.json", a), _json(tmp_path, "b.json", b))
    assert rc == 0, out
    lignes = [l for l in out.splitlines() if "− référence" in l]
    assert "Δ +0.000 %" in lignes[0] and "non conclusif" in lignes[0], lignes[0]
    assert "→ conclusif" in lignes[1] and "z +" in lignes[1], lignes[1]


def test_fenetres_non_appariees_echec(tmp_path):
    a = [(4096.0, 2048)] * 10
    rc, out = _lancer(_json(tmp_path, "a.json", a), _json(tmp_path, "c.json", a[:-1]))
    assert rc == 2 and "non appariées" in out
