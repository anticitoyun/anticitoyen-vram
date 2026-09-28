"""296 : modèles vidéo à gabarit de la Pipe « ComfyUI médias » (Wan 2.2 14B I2V/T2V, Wan VACE, LTX-2.3), à sec.

Workflow SIMULÉ (les vrais viennent d'poste6, parc/share/openwebui/videos/). Casse si : la pièce jointe n'arrive
pas au nœud d'entrée ; un modèle qui exige une image se lance sans ; la longueur n'est pas multiple·k + 1 ; un nœud
de la table absent du workflow est ignoré en silence ; le téléversement ComfyUI n'est pas un multipart lisible ;
l'installateur ne peut plus inliner GABARITS."""
import base64
import email.parser
import importlib.util
import pathlib

import pytest

pytest.importorskip("pydantic")

RACINE = pathlib.Path(__file__).resolve().parent.parent / "parc" / "share" / "openwebui"
_spec = importlib.util.spec_from_file_location("comfyui_medias_296", RACINE / "comfyui_medias.py")
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)

_GRAPHE = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "exemple.png"}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["9", 0]}},
    "3": {"class_type": "WanImageToVideo", "inputs": {"width": 0, "height": 0, "length": 0, "start_image": ["1", 0]}},
    "4": {"class_type": "KSampler", "inputs": {"seed": 0}},
    "5": {"class_type": "CreateVideo", "inputs": {"fps": 0}},
    "6": {"class_type": "LoadVideo", "inputs": {"file": "ref.mp4"}},
    "9": {"class_type": "CLIPLoader", "inputs": {}},
}
_NOEUDS = [{"type": "prompt", "node_ids": ["2"], "key": "text"}, {"type": "width", "node_ids": ["3"], "key": "width"},
           {"type": "height", "node_ids": ["3"], "key": "height"}, {"type": "length", "node_ids": ["3"], "key": "length"},
           {"type": "seed", "node_ids": ["4"], "key": "seed"}, {"type": "fps", "node_ids": ["5"], "key": "fps"}]


@pytest.fixture(autouse=True)
def gabarits(monkeypatch):
    img = {"graphe": _GRAPHE, "noeuds": _NOEUDS + [{"type": "image", "node_ids": ["1"], "key": "image"}]}
    vid = {"graphe": _GRAPHE, "noeuds": _NOEUDS + [{"type": "video", "node_ids": ["6"], "key": "file"},
                                                   {"type": "image", "node_ids": ["1"], "key": "image"}]}
    monkeypatch.setattr(cm, "GABARITS", {"wan22-14b-i2v": img, "wan22-14b-t2v": {"graphe": _GRAPHE, "noeuds": _NOEUDS},
                                         "wan-vace": vid, "ltx23": img})


def test_quatre_modeles_distincts_au_selecteur_avec_usage():
    ids = [p["id"] for p in cm.Pipe().pipes()]
    for k in ("wan22_i2v", "wan22_t2v", "wan_vace", "ltx23"):
        assert k in ids and cm.MODELES[k]["usage"] and cm.MODELES[k]["nom"].startswith("Vidéo · ")
    assert len({cm.MODELES[k]["nom"] for k in ids}) == len(ids)


def test_i2v_pose_l_image_jointe_la_duree_et_la_taille():
    g, desc = cm.construire("wan22_i2v", "un chat qui saute 1280x720 3 s", 7, ("image", "owui-ab.png"))
    assert g["1"]["inputs"]["image"] == "owui-ab.png" and g["2"]["inputs"]["text"] == "un chat qui saute"
    assert (g["3"]["inputs"]["width"], g["3"]["inputs"]["height"], g["3"]["inputs"]["length"]) == (1280, 720, 49)
    assert g["4"]["inputs"]["seed"] == 7 and g["5"]["inputs"]["fps"] == 16
    assert _GRAPHE["1"]["inputs"]["image"] == "exemple.png"               # gabarit jamais modifié en place
    assert "3 s" in desc and "1280×720" in desc


def test_longueur_multiple_plus_un():
    assert cm.longueur_images(5, 16, 4) == 81 and cm.longueur_images(5, 24, 8) == 121
    assert cm.longueur_images(0.1, 16, 4) == 5
    g, _ = cm.construire("ltx23", "mer", 1)
    assert (g["3"]["inputs"]["length"] - 1) % 8 == 0 and g["5"]["inputs"]["fps"] == 24


def test_entree_exigee_facultative_ou_absente():
    img = [{"mime": "image/png", "data": b"x"}]
    vid = [{"mime": "video/mp4", "id": "f1"}]
    with pytest.raises(ValueError, match="Joignez une image"):
        cm.choisir_entree(cm.MODELES["wan22_i2v"], vid)                  # une vidéo n'est pas l'image de départ
    with pytest.raises(ValueError):
        cm.choisir_entree(cm.MODELES["wan_vace"], [])
    assert cm.choisir_entree(cm.MODELES["wan_vace"], vid)[0] == "video"
    assert cm.choisir_entree(cm.MODELES["wan_vace"], img)[0] == "image"   # pose de référence
    assert cm.choisir_entree(cm.MODELES["ltx23"], []) is None
    assert cm.choisir_entree(cm.MODELES["ltx23"], img)[0] == "image"
    assert cm.choisir_entree(cm.MODELES["wan22_t2v"], img) is None


def test_entree_hors_gabarit_et_noeud_absent_sont_des_erreurs():
    with pytest.raises(KeyError, match="ne prend pas de video"):
        cm.construire("wan22_i2v", "x", 1, ("video", "v.mp4"))
    with pytest.raises(KeyError, match="nœud 8.text absent"):
        cm.appliquer({"graphe": _GRAPHE, "noeuds": [{"type": "prompt", "node_ids": ["8"], "key": "text"}]}, {"prompt": "x"})
    cm.GABARITS.pop("ltx23")
    with pytest.raises(KeyError, match="réinstaller"):
        cm.construire("ltx23", "x", 1)


def test_references_jointes_data_fichier_owui_et_video():
    png = base64.b64encode(b"\x89PNG").decode()
    messages = [{"role": "user", "content": "ancien"},
                {"role": "user", "content": [{"type": "text", "text": "anime"},
                                             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}},
                                             {"type": "image_url", "image_url": {"url": "/api/v1/files/0a1b2c3d-4e5f/content"}}]}]
    fichiers = [{"id": "0a1b2c3d-4e5f", "content_type": "image/png"},
                {"type": "file", "file": {"id": "9f8e7d6c", "meta": {"content_type": "video/mp4"}}},
                {"id": "txt1", "content_type": "text/plain"}]
    refs = cm.references_jointes(messages, fichiers)
    assert refs == [{"mime": "image/png", "data": b"\x89PNG"}, {"mime": "image/png", "id": "0a1b2c3d-4e5f"},
                    {"mime": "video/mp4", "id": "9f8e7d6c"}]
    assert cm.references_jointes([{"role": "user", "content": "texte"}], None) == []


def test_multipart_de_televersement_lisible():
    corps, ct = cm.corps_multipart("owui-1.mp4", b"\x00\x01video", "video/mp4")
    msg = email.parser.BytesParser().parsebytes(f"Content-Type: {ct}\r\n\r\n".encode() + corps)
    parts = {p.get_param("name", header="content-disposition"): p for p in msg.get_payload()}
    assert parts["image"].get_filename() == "owui-1.mp4" and parts["image"].get_payload(decode=True) == b"\x00\x01video"
    assert parts["type"].get_payload() == "input" and parts["overwrite"].get_payload() == "true"


def test_l_installateur_peut_inliner_gabarits():
    source = (RACINE / "comfyui_medias.py").read_text()
    assert source.count("\nGABARITS = {}\n") == 1                         # openwebui-medias l'exige
    gab = {"wan22-14b-i2v": {"graphe": {"1": {"inputs": {"b": True, "n": None}}}, "noeuds": []}}
    espace = {}
    exec(compile(source.replace("\nGABARITS = {}\n", "\nGABARITS = " + repr(gab) + "\n"), "pipe", "exec"), espace)
    assert espace["GABARITS"] == gab
