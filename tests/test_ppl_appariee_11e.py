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


# 11e (01/10, prise réelle) : `acvram eval --json > fichier` écrit son en-tête AVANT le JSON, dont des lignes qui commencent par
# « [ » ([acvram], [régime]). json.load sur le fichier entier cassait l'analyse (données intactes). Copie de la vraie forme.
ENTETE_EVAL = ("  variables ignorees (le code ne les lit nulle part) : ACVRAM_DUREE_MAX, ACVRAM_POSTE\n"
               "  cadrage : min_context=0 window=2048 stride=2048 max_tokens=65536\n"
               "[acvram] réserve de préfill : formule seule — mesure de chauffe ignorée (aucun fichier)\n"
               "[régime] ACVRAM_CPUS=0-15 ACVRAM_GEMV_LAYOUT=marlin extension=oui\n")


def test_lit_la_sortie_reelle_d_acvram_eval_avec_son_entete(tmp_path):
    rng = random.Random(2)
    a = [(2048 * rng.uniform(1.5, 2.5), 2048) for _ in range(20)]
    chemins = []
    for nom in ("a.json", "a2.json"):
        p = tmp_path / nom
        p.write_text(ENTETE_EVAL + json.dumps([{"par_fenetre": a}], indent=2) + "\n")
        chemins.append(str(p))
    rc, out = _lancer(*chemins)
    assert rc == 0, out
    assert "Δ +0.000 %" in out


def test_sans_json_le_refus_est_nomme(tmp_path):
    p = tmp_path / "vide.json"
    p.write_text(ENTETE_EVAL)
    r = subprocess.run([sys.executable, OUTIL, str(p), str(p)], capture_output=True, text=True)
    assert r.returncode != 0 and "aucun JSON" in (r.stdout + r.stderr)
