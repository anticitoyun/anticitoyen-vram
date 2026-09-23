"""Pièce 37 : le bras « decode » de `diag-eval-nll.py` voit-il bien les mêmes
logits que le préfill ? Sur un modèle jouet (processeur, fp32), la NLL jeton
par jeton du chemin de GÉNÉRATION (cache KV, un pas par position) doit
coïncider avec celle du préfill complet : c est le teacher-forcing, même
modèle, même contexte. Un écart signe un défaut de montage du bras
(positions, créneaux du cache, croissance des blocs) — pas un défaut du
modèle mesuré. Ce test doit casser si l on y touche.
"""
from __future__ import annotations

import importlib.util
import os

import torch

CHEMIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outils", "gpu", "mesure", "diag-eval-nll.py")


def _module():
    spec = importlib.util.spec_from_file_location("diag_eval_nll", CHEMIN)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_bras_decode_egale_le_prefill(converted):
    from acvram.engine.loader import load_model
    m = _module()
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    ids = [5, 42, 7, 99, 13, 8, 21, 3, 64, 2, 17, 55]
    e = m.nll_eval(loaded, ids)
    d = m.nll_decode(loaded, None, ids)
    assert len(d) == len(e) == len(ids) - 1
    # Tolérance 0,05 nat, pas « au bit » : le cache KV est quantifié (int8) et
    # la réduction d un préfill batché n est pas celle d un GEMV par jeton.
    # Mesuré sur le jouet : 5,2e-3 nat. Le seuil de l instrument est 1 nat,
    # vingt fois plus haut — un vrai défaut de montage se voit bien avant.
    ecart = max(abs(x - y) for x, y in zip(e, d))
    assert ecart < 0.05, f"decode ≠ préfill de {ecart} nats — le bras est mal monté"
    # position 1 : contexte d un seul jeton, aucun cache lu — le contrôle
    # intégré de l instrument (`montage_decode_ok`) exige 1e-2 ici.
    assert abs(d[0] - e[0]) <= 1e-2, f"position 1 : Δ = {abs(d[0] - e[0])}"


def test_verdict_refuse_un_bras_mal_monte():
    m = _module()
    r = {"montage_decode_ok": False, "delta_position_1": 4.2, "ppl_decode": 9.0,
         "ppl_eval": 99.0, "nll_decode_mediane": 2.2, "div_eval_decode": 3}
    assert "INVALIDE" in m.verdict(r)
    r["montage_decode_ok"] = True
    assert m.verdict(r).startswith("P5")
    r["ppl_decode"] = 120.0
    assert m.verdict(r).startswith("P4")


def test_reference_hf_invalide_arrete_le_verdict():
    """Contrôle de la RÉFÉRENCE (22/09, Manon 9b757c1b : HF rend PPL 108 039
    sur les mêmes ids). Une référence qui échoue sur son PROPRE encodage, ou
    qui génère du bruit, ne juge rien — le verdict doit le dire et s arrêter
    là, au lieu d accuser le moteur."""
    m = _module()
    bonne = {"ppl_propre": 9.4, "suite_distincts": 17, "suite_texte": "the quick brown fox"}
    assert m.juger_reference(bonne)["reference_valide"] is True
    mauvaise = m.juger_reference({"ppl_propre": 108039.0, "suite_distincts": 2, "suite_texte": "!!!!"})
    assert mauvaise["reference_valide"] is False
    assert len(mauvaise["reference_motifs"]) == 2
    v = m.verdict({**mauvaise, "ppl_eval": 227232.0, "ppl_decode": 227000.0,
                   "montage_decode_ok": True, "nll_decode_mediane": 12.3, "div_eval_decode": 0})
    assert v.startswith("RÉFÉRENCE INVALIDE"), v
    assert "P4" not in v and "P5" not in v          # le moteur n est pas jugé


def test_reference_sans_bras_propre_est_invalide():
    m = _module()
    r = m.juger_reference({"suite_distincts": 12})
    assert r["reference_valide"] is False and "pas de bras propre" in r["reference_motifs"][0]
