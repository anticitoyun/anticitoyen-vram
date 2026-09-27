"""53j-b : la Pipe OWUI « ComfyUI médias » (parc/share/openwebui/comfyui_medias.py), à sec.

Casse si : un graphe référence un nœud absent ; le graphe Klein de la Pipe s'écarte du workflow du réglage Images
natif (flux2-klein-9b.api.json) ; la table de nœuds natif pointe hors du workflow ; une tâche d'OWUI (titre,
suggestions) déclenche une génération."""
import asyncio
import importlib.util
import json
import pathlib

import pytest

pytest.importorskip("pydantic")

RACINE = pathlib.Path(__file__).resolve().parent.parent / "parc" / "share" / "openwebui"
_spec = importlib.util.spec_from_file_location("comfyui_medias", RACINE / "comfyui_medias.py")
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)


def _liens_valides(g):
    for nid, n in g.items():
        for v in n["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in g, f"nœud {nid} ({n['class_type']}) lié à {v[0]} absent"


@pytest.mark.parametrize("cle", list(cm.MODELES))
def test_chaque_graphe_est_ferme_et_porte_l_invite(cle):
    g, desc = cm.construire(cle, "une pomme rouge 768x512 3 s", 42)
    _liens_valides(g)
    assert any(n["class_type"] == "CLIPTextEncode" and n["inputs"]["text"] == "une pomme rouge" for n in g.values())
    assert cm.MODELES[cle]["nom"] in desc and "768×512" in desc
    assert json.dumps(g)                                       # sérialisable pour /prompt


def test_klein_de_la_pipe_est_le_workflow_natif():
    natif = json.loads((RACINE / "flux2-klein-9b.api.json").read_text())
    g = cm.graphe_klein(cm.MODELES["klein9b"], "Prompt", 1024, 1024, 0)
    g["9"]["inputs"]["filename_prefix"] = natif["9"]["inputs"]["filename_prefix"]
    assert g == natif


def test_table_de_noeuds_native_pointe_dans_le_workflow():
    natif = json.loads((RACINE / "flux2-klein-9b.api.json").read_text())
    for n in json.loads((RACINE / "flux2-klein-9b.noeuds.json").read_text()):
        for nid in n["node_ids"]:
            assert n["key"] in natif[nid]["inputs"], (n, natif[nid]["class_type"])


def test_options_bornees():
    assert cm.lire_options("chat 5000x10", (512, 512)) == ("chat 5000x10", 512, 512, 2.0)   # 5000x10 : pas une taille
    assert cm.lire_options("chat 1000x700 12 s", (512, 512)) == ("chat", 992, 688, 10.0)
    assert cm.graphe_video(cm.MODELES["wan22"], "x", 832, 480, 1, 2.0)["6"]["inputs"]["length"] == 49


def test_premier_fichier_ignore_les_temporaires():
    sorties = {"9": {"images": [{"filename": "a.png", "type": "temp"}, {"filename": "b.png", "type": "output"}]}}
    assert cm.premier_fichier(sorties)["filename"] == "b.png"
    assert cm.premier_fichier({"9": {"images": [{"filename": "a.png", "type": "temp"}]}}) is None


def test_une_tache_owui_ne_genere_rien(monkeypatch):
    p = cm.Pipe()
    monkeypatch.setattr(p, "_requete", lambda *a, **k: pytest.fail("appel ComfyUI pendant une tâche"))
    body = {"model": "comfyui_medias.klein9b", "messages": [{"role": "user", "content": "une pomme"}]}
    assert asyncio.run(p.pipe(body, __task__="title_generation")) == "Génération ComfyUI"
    assert [m["id"] for m in p.pipes()] == list(cm.MODELES)
