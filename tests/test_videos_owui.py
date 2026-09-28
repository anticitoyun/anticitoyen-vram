"""296 (poste6) : les graphes vidéo ComfyUI de parc/share/openwebui/videos/, à sec.

Casse si : un lien pointe vers un nœud absent ; une entrée de <nom>.noeuds.json vise un nœud ou une clé absents du graphe ;
un type de nœud sort de la liste validée par `validate_prompt` de ComfyUI 0.37.4 le 28/09 ; une longueur par défaut
n'est pas 4k+1 (Wan) ou 8k+1 (LTX) ; une dimension n'est pas multiple de 16 (Wan) ou 32 (LTX)."""
import json
import pathlib

import pytest

DOSSIER = pathlib.Path(__file__).resolve().parent.parent / "parc" / "share" / "openwebui" / "videos"
NOMS = ["wan22-14b-i2v", "wan22-14b-t2v", "wan-vace", "ltx23", "ltx23-t2v"]
TYPES_VALIDES = {
    "UNETLoader", "CLIPLoader", "VAELoader", "CLIPTextEncode", "ModelSamplingSD3", "LoadImage", "WanImageToVideo",
    "EmptyHunyuanLatentVideo", "KSamplerAdvanced", "VAEDecode", "VHS_VideoCombine", "VHS_LoadVideo", "ImageResizeKJv2",
    "WanVideoVACEModelSelect", "WanVideoModelLoader", "WanVideoTextEmbedBridge", "WanVideoVAELoader", "WanVideoVACEEncode",
    "WanVideoSampler", "WanVideoDecode", "CheckpointLoaderSimple", "LoraLoaderModelOnly", "LTXAVTextEncoderLoader",
    "LTXVConditioning", "LTXVAudioVAELoader", "EmptyLTXVLatentVideo", "LTXVEmptyLatentAudio", "RandomNoise",
    "KSamplerSelect", "ManualSigmas", "CFGGuider", "LTXVSeparateAVLatent", "VAEDecodeTiled", "LTXVAudioVAEDecode",
    "ImageScale", "LTXVPreprocess", "LTXVImgToVideoInplace", "LTXVConcatAVLatent", "SamplerCustomAdvanced",
}
TYPES_NOEUDS = {"prompt", "negatif", "width", "height", "length", "fps", "seed", "steps", "image", "video"}


def _charger(nom):
    return (json.loads((DOSSIER / f"{nom}.api.json").read_text()), json.loads((DOSSIER / f"{nom}.noeuds.json").read_text()))


@pytest.mark.parametrize("nom", NOMS)
def test_liens_et_types(nom):
    g, _ = _charger(nom)
    for nid, n in g.items():
        assert n["class_type"] in TYPES_VALIDES, f"{nom} : nœud {nid} de type {n['class_type']} jamais validé"
        for v in n["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in g, f"{nom} : nœud {nid} lié à {v[0]} absent"
    assert any(n["class_type"] == "VHS_VideoCombine" for n in g.values())


@pytest.mark.parametrize("nom", NOMS)
def test_table_de_noeuds(nom):
    g, noeuds = _charger(nom)
    types = set()
    for e in noeuds:
        assert e["type"] in TYPES_NOEUDS, f"{nom} : type {e['type']} inconnu de la Pipe"
        types.add(e["type"])
        for nid in e["node_ids"]:
            assert nid in g, f"{nom} : entrée {e['type']} vise le nœud {nid} absent"
            assert e["key"] in g[nid]["inputs"], f"{nom} : nœud {nid} sans clé {e['key']}"
            assert not isinstance(g[nid]["inputs"][e["key"]], list), f"{nom} : {nid}.{e['key']} est un lien, pas une valeur"
    assert {"prompt", "width", "height", "length", "fps", "seed"} <= types
    assert ("image" in types) == (nom in ("wan22-14b-i2v", "wan-vace", "ltx23"))
    assert ("video" in types) == (nom == "wan-vace")


@pytest.mark.parametrize("nom", NOMS)
def test_longueur_et_dimensions(nom):
    g, noeuds = _charger(nom)
    ltx = nom.startswith("ltx")
    for e in noeuds:
        for nid in e["node_ids"]:
            v = g[nid]["inputs"][e["key"]]
            if e["type"] == "length":
                assert (v - 1) % (8 if ltx else 4) == 0, f"{nom} : longueur {v} hors {'8k+1' if ltx else '4k+1'}"
            if e["type"] in ("width", "height"):
                assert v % (32 if ltx else 16) == 0, f"{nom} : {e['type']} {v} non multiple de {32 if ltx else 16}"
            if e["type"] == "fps":
                assert v == (24 if ltx else 16)
