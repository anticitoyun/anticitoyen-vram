"""Pièce 27(a) : trace de routage par modalité (v2) — masque aligné sur le lot,
lignes v2 relues par les deux lecteurs (v1 compatible), table Q conditionnelle,
verdict « voie texte fermée » sur les experts froids. Coût nul hors trace, et
la sortie du modèle ne change pas (la trace n écrit que)."""
import json

import pytest
import torch

from acvram.memory import trace_routage as TR


@pytest.fixture(autouse=True)
def _propre(monkeypatch):
    monkeypatch.setattr(TR, "_ACTIF", False)
    monkeypatch.setattr(TR, "_MODALITES", None)
    monkeypatch.setattr(TR, "_FICHIER", None)
    monkeypatch.setattr(TR, "_JETON", 0)
    monkeypatch.setattr(TR, "_BASE", None)


def test_masque_de_modalite_du_lot():
    ids = torch.tensor([5, 262144, 262144, 7, -1, 9])
    assert TR.modalites_du_lot(ids, image_token_id=262144) == ["t", "i", "i", "t", "s", "t"]
    assert TR.modalites_du_lot(ids, None, [(1, 3)]) == ["t", "i", "i", "t", "s", "t"]      # par plages
    assert TR.modalites_du_lot([1, 2, 3]) == ["t", "t", "t"]                                # sans image : tout texte
    assert TR.modalites_du_lot([], None) == []


def test_cout_nul_hors_trace(monkeypatch):
    TR.poser_modalites(["i", "t"])
    assert TR._MODALITES is None                       # hors trace, rien n est posé
    TR.noter(0, torch.tensor([[1, 2]]), torch.tensor([[0.5, 0.5]]))   # ne doit rien écrire ni lever


def test_ecriture_v2_et_relecture(tmp_path, monkeypatch):
    chemin = str(tmp_path / "t.txt")
    monkeypatch.setattr(TR, "_ACTIF", True)
    monkeypatch.setattr(TR, "_CHEMIN", chemin)
    TR.poser_modalites(["i", "t"])
    TR.noter(0, torch.tensor([[3, 7], [1, 2]]), torch.tensor([[0.75, 0.25], [0.6, 0.4]]))
    TR.noter(1, torch.tensor([[5, 9], [4, 8]]), torch.tensor([[0.5, 0.5], [0.9, 0.1]]))
    TR.fermer()
    lignes = [l for l in open(chemin).read().splitlines() if not l.startswith("#")]
    assert lignes[0] == "0 0 i 3:0.75,7:0.25" and lignes[1] == "1 0 t 1:0.6,2:0.4"
    # v1 : le lecteur historique rend toujours (jeton, couche, experts)
    assert list(TR.relire(chemin))[:2] == [(0, 0, [3, 7]), (1, 0, [1, 2])]
    # v2 : modalité et poids
    v2 = list(TR.relire_modalites(chemin))
    assert v2[0] == (0, 0, "i", [(3, 0.75), (7, 0.25)]) and v2[2][2] == "i"
    # une trace v1 (sans modalité) est refusée par le lecteur v2, jamais devinée
    v1 = str(tmp_path / "v1.txt")
    open(v1, "w").write("0 0 3,7\n1 0 1,2\n")
    assert list(TR.relire(v1)) == [(0, 0, [3, 7]), (1, 0, [1, 2])]
    with pytest.raises(ValueError, match="trace v1"):
        list(TR.relire_modalites(v1))


def _module():
    import importlib.util
    import os
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outils", "gpu", "mesure", "routage-par-modalite.py")
    spec = importlib.util.spec_from_file_location("routage_par_modalite", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_table_Q_et_verdict_froids(tmp_path):
    m = _module()
    t = str(tmp_path / "trace.txt")
    with open(t, "w") as f:
        # expert 3 : image seule ; expert 7 : texte seul ; expert 1 : les deux
        for j in range(10):
            f.write(f"{j} 0 i 3:1.0,1:0.5\n")
        for j in range(10, 20):
            f.write(f"{j} 0 t 7:1.0,1:0.5\n")
    tab = m.table(m.agreger(t), 128)
    l3, l7, l1 = tab["lignes"]["0/3"], tab["lignes"]["0/7"], tab["lignes"]["0/1"]
    assert l3["n"] == {"i": 10} and l7["n"] == {"t": 10} and l1["n"] == {"i": 10, "t": 10}
    assert l3["Q"]["i"] == 2.0 and l3["Q"]["t"] == 0.0        # 10/20 du trafic image, part attendue 10/40
    assert l1["Q"]["i"] == 1.0 and l1["Q"]["t"] == 1.0        # sans préférence
    assert abs(l3["masse"]["i"] - 10.0) < 1e-6
    manifeste = str(tmp_path / "m.json")
    json.dump({"experts_sans_stats_liste": {"0": {"gate_proj": [3, 7, 42]}}}, open(manifeste, "w"))
    v = m.verdict_froids(tab, m.experts_froids(manifeste))
    assert v["froids_declares"] == 3 and v["froids_vus"] == 2 and v["froids_jamais_vus"] == 1
    assert v["froids_image_seule"] == 1 and v["part_image_seule"] == 0.5 and v["verdict"].startswith("MARGINAL")
    json.dump({"experts_sans_stats_liste": {"0": {"gate_proj": [3]}}}, open(manifeste, "w"))
    assert m.verdict_froids(tab, m.experts_froids(manifeste))["verdict"].startswith("TENU")
    json.dump({"experts_sans_stats_liste": {"0": {"gate_proj": [7]}}}, open(manifeste, "w"))
    assert m.verdict_froids(tab, m.experts_froids(manifeste))["verdict"].startswith("RÉFUTÉ")
    json.dump({"experts_sans_stats_liste": {"9": {"gate_proj": [99]}}}, open(manifeste, "w"))
    assert m.verdict_froids(tab, m.experts_froids(manifeste))["verdict"].startswith("NON JUGÉ")
