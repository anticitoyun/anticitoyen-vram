"""Contrat multimodal, pièce (a) P0 conversion (revue/poste7-go-multimodal-
organisation-20-09 § 2) : les tenseurs de la tour visuelle (`model.vision_tower.*`,
`model.embed_vision.*` — Gemma 4 ; `model.visual.*` — Qwen3-VL) sont GARDÉS en
bf16 sous leur nom source, jamais quantifiés ; le manifeste dit `vision: oui`
et `vision_bytes` = Σ exact ; un alias sans tour reste identique au bit à ce
que la conversion produisait avant cette règle (témoin sérialisé ci-dessous,
sha256 par clé, produit par le code d'origine sur le même mini-modèle).

À sec, sans carte. Un mini-modèle VL factice : une config Qwen3-VL (text_config
+ vision_config), deux couches texte minuscules, et des tenseurs vision qui
couvrent les trois préfixes du contrat — dont un `…encoder.layers.3…` : indice
de couche hors plan (2 couches texte), le piège de `TensorRouter.layer_index`.
"""
import hashlib
import json
import os
import shutil

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from acvram.engine.config import load_model_spec
from acvram.memory.tiering import PlannerOptions, auto_plan
from acvram.quant.convert import ConversionOptions, convert_checkpoint

# Les préfixes du contrat, écrits ICI (le test est le contrat, pas le code) ;
# test_les_prefixes_sont_ceux_du_contrat les confronte à convert.VISION_PREFIXES.
VISION_PREFIXES = ("model.vision_tower.", "model.embed_vision.", "model.visual.")

H, INTER, L, NH, NKV, V = 64, 128, 2, 4, 2, 500
VH = 32                                                      # largeur de la tour
GRAINE = 20260920

# Témoin : sha256 (dtype, forme, octets) par clé écrite, conversion du MÊME
# mini-modèle (alias texte) par le code d'ORIGINE (a4427a56, avant la règle
# vision). Le code d'origine jetait la tour : sa sortie sur le modèle VL et sur
# l'alias texte était la même — c'est elle qui est figée ici.
# Régénération (code d'origine dans un worktree temporaire T, à sec) :
#   CUDA_VISIBLE_DEVICES= PYTHONPATH=T python tests/test_mm_conversion.py T/sortie
TEMOIN_TEXTE = {
    "lm_head.weight.block_scale":
        "be533df191777216e7cb1c00c8e590b986586de418be68d515fe1b4806676723",
    "lm_head.weight.global_scale":
        "fe86ebc1992196a2f0492301af3affb921c3c88378c8bb008af3e26a3b20159a",
    "lm_head.weight.qweight":
        "56ec31063608ee2875d5088ff1ac51fa16f1c2a48a5076e8e0dfcecd1bb436f1",
    "model.embed_tokens.weight":
        "a6ec388da76907789c7af9e75862605a2cb7d4b2e29f11923e42290b695988b9",
    "model.layers.0.input_layernorm.weight":
        "d16217b910541669d4ee5a39c34dcb8c6cc3bf1cd201fa011e145b69f7a21b8c",
    "model.layers.0.mlp.down_proj.weight.block_scale":
        "6cdf0188614af83862f60f2be62e4a8f495357922e4b639da9e5bf95a4ac089e",
    "model.layers.0.mlp.down_proj.weight.global_scale":
        "c4ce584365f30d25b35a92451642bf76e34812a8f1e52be23654c386edc087d4",
    "model.layers.0.mlp.down_proj.weight.qweight":
        "8bf3b0a7770c9aa9db62aa6c2f31ee36f09c043a80171b78c9c2b6ce3dd049cf",
    "model.layers.0.mlp.gate_proj.weight.block_scale":
        "8ede2de46bcc2c89490a3fe6f4f01b405191d4da021bbdbc43ba1dab46572285",
    "model.layers.0.mlp.gate_proj.weight.global_scale":
        "c844cbd4ea99e6dc4c0f2b48320ed5df765050a0019e6296615279f171ebde01",
    "model.layers.0.mlp.gate_proj.weight.qweight":
        "de7c524804ea14f5bf19a4c8cd2d528c0215d55a42aed22076dad46faa4396ef",
    "model.layers.0.mlp.up_proj.weight.block_scale":
        "893d8cf4603c18b9b7049258dad915e43f22b0be368156e1a0593299498423d1",
    "model.layers.0.mlp.up_proj.weight.global_scale":
        "c844cbd4ea99e6dc4c0f2b48320ed5df765050a0019e6296615279f171ebde01",
    "model.layers.0.mlp.up_proj.weight.qweight":
        "6927e9ffa099ae1fb7f702d610505861f5b3ff39b6d9ada76472912bd5abdeb9",
    "model.layers.0.post_attention_layernorm.weight":
        "d16217b910541669d4ee5a39c34dcb8c6cc3bf1cd201fa011e145b69f7a21b8c",
    "model.layers.0.self_attn.k_proj.weight.block_scale":
        "483a694cb23a30207423030786c81f9160956804302d9d5f8680348cfc131306",
    "model.layers.0.self_attn.k_proj.weight.global_scale":
        "57e5a74d82270e6f3a11d8fb129094fb54d06020f322913320fe968187be9521",
    "model.layers.0.self_attn.k_proj.weight.qweight":
        "65239af602de9986de8107a60a9cbb13c17c043d735203726d52db1813d0b175",
    "model.layers.0.self_attn.o_proj.weight.block_scale":
        "95af90c289ddbaabcdb3ba87d56396e3e6c7991340f9c92e2004985bd1465dec",
    "model.layers.0.self_attn.o_proj.weight.global_scale":
        "c844cbd4ea99e6dc4c0f2b48320ed5df765050a0019e6296615279f171ebde01",
    "model.layers.0.self_attn.o_proj.weight.qweight":
        "5206217fa01ad3e7b27de9652c52e3b2e0a88cd98e69e60ceb17f627d70ec841",
    "model.layers.0.self_attn.q_proj.weight.block_scale":
        "4782e46741d602c2ad048749d1f28b6c9800f348bbdf03ff567018c4e38bab2e",
    "model.layers.0.self_attn.q_proj.weight.global_scale":
        "b6e782b75abbb3fb1a02a430c0ffcc4d151b567de253d7a3700c8ea78752bc4d",
    "model.layers.0.self_attn.q_proj.weight.qweight":
        "fff1630db5ab46ee33e1bd0b00a70c99044f6978a0d7f3b4c48c98c9f856ea40",
    "model.layers.0.self_attn.v_proj.weight.block_scale":
        "7900d089be7daed0453d0eb01dc04823cc0bd3054ed45df4706cd8a1dbb732d7",
    "model.layers.0.self_attn.v_proj.weight.global_scale":
        "82f17e32bb1e83e2391713fd837b6273615b3c0cd440bce7d0c5708edb655294",
    "model.layers.0.self_attn.v_proj.weight.qweight":
        "3c1ecb2bd84f18681c8f5e3a7f2124719e8e7e4623e2283393de9f7031737f9e",
    "model.layers.1.input_layernorm.weight":
        "d16217b910541669d4ee5a39c34dcb8c6cc3bf1cd201fa011e145b69f7a21b8c",
    "model.layers.1.mlp.down_proj.weight.block_scale":
        "54367a517cc052b8fee558f73b68500135a5c3ed14249aace4f95eeafe06e834",
    "model.layers.1.mlp.down_proj.weight.global_scale":
        "df96ca3019353bad08323ffae0a3a1b0495aa63d5675a0597f87dba3d917d87d",
    "model.layers.1.mlp.down_proj.weight.qweight":
        "b3b74927f563250b9c037acabd16b2c230eb187bee5f9527e0fc83e7493f4dd0",
    "model.layers.1.mlp.gate_proj.weight.block_scale":
        "9eac675c68ab041605648b718a4b624e0c8160d36b7fee4f8be16fd9d0dc46f1",
    "model.layers.1.mlp.gate_proj.weight.global_scale":
        "5d088fb0194b0732a32d5334d5e5fc23f3ec169cf3bb9a1f2811895acf6df87f",
    "model.layers.1.mlp.gate_proj.weight.qweight":
        "2040f869a2acb8377b27c2ee46e8e7c312698f1b252002ea66c4ded4735ae92b",
    "model.layers.1.mlp.up_proj.weight.block_scale":
        "e20ea3c71c02fe3f2e101532f1b30543874b3ab5358d24503a1c1d710e21a6a3",
    "model.layers.1.mlp.up_proj.weight.global_scale":
        "ee5a498e8e0240430b693da0e6b4757b8dca1c0cf8dfb391d1feea86db9cd30f",
    "model.layers.1.mlp.up_proj.weight.qweight":
        "764f32e9373515853a11898f509f88fb423af7943b72c3bcd45b4fc495cc7ab0",
    "model.layers.1.post_attention_layernorm.weight":
        "d16217b910541669d4ee5a39c34dcb8c6cc3bf1cd201fa011e145b69f7a21b8c",
    "model.layers.1.self_attn.k_proj.weight.block_scale":
        "ec105d23226bd7fe50ea53aa8055ededff53f1f3070269b2d29c254f7cf32a41",
    "model.layers.1.self_attn.k_proj.weight.global_scale":
        "d50056fb7e0da6a5fc28d14328d6abab94f1019331c95506f11358eb172b0cd6",
    "model.layers.1.self_attn.k_proj.weight.qweight":
        "6d615053074bacaecb015fea40ac89ca77bd281dc819261e1fa5c4cbf2536ffa",
    "model.layers.1.self_attn.o_proj.weight.block_scale":
        "475bc4c4d77ca08deb0ec8d5f20c1b197015fae854386e2d16f75a5be83c983d",
    "model.layers.1.self_attn.o_proj.weight.global_scale":
        "fe86ebc1992196a2f0492301af3affb921c3c88378c8bb008af3e26a3b20159a",
    "model.layers.1.self_attn.o_proj.weight.qweight":
        "94f58aaafe480bbd3e1d74734116d494eac1c086737ce021301635c5acb5942d",
    "model.layers.1.self_attn.q_proj.weight.block_scale":
        "e19aa6f7da7dcbbd34a10a0e461e51de973df38939a71f45adc4654d818950b9",
    "model.layers.1.self_attn.q_proj.weight.global_scale":
        "1b29be94c5ae07526069a0be9d9566c1799a83b3c0e5f2d6ad499a619ca9b90c",
    "model.layers.1.self_attn.q_proj.weight.qweight":
        "881e36c41628ad4efaac5f6571cf9c7c20ed553aba9f80df5690377b287a0555",
    "model.layers.1.self_attn.v_proj.weight.block_scale":
        "e1523c5cf63b607e8e912b68de05ab713ae60a3a2ae2c5109e1b16c2e8199ef2",
    "model.layers.1.self_attn.v_proj.weight.global_scale":
        "277b9ac9fdef618e3eb6bdc0d6f7279f6334e13c490f8fec123b4a485b85cf9b",
    "model.layers.1.self_attn.v_proj.weight.qweight":
        "341f0eed19bd21ac1ed6f53f68f2385fa8d426d1cd0f30c186ae6068a2b1f209",
    "model.norm.weight":
        "d16217b910541669d4ee5a39c34dcb8c6cc3bf1cd201fa011e145b69f7a21b8c",
}


def _texte_config() -> dict:
    return {
        "model_type": "qwen3_vl_text", "hidden_size": H, "intermediate_size": INTER,
        "num_hidden_layers": L, "num_attention_heads": NH, "num_key_value_heads": NKV,
        "vocab_size": V, "max_position_embeddings": 2048, "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0, "hidden_act": "silu", "tie_word_embeddings": False,
    }


def _config(vision: bool) -> dict:
    cfg = {"architectures": ["Qwen3VLForConditionalGeneration"], "model_type": "qwen3_vl",
           "text_config": _texte_config(), "torch_dtype": "bfloat16"}
    if vision:
        cfg["vision_config"] = {"model_type": "qwen3_vl", "hidden_size": VH, "depth": 1,
                                "patch_size": 14, "in_channels": 3, "out_hidden_size": H}
    return cfg


def _tenseurs_texte() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(GRAINE)

    def w(*shape):
        return (torch.randn(*shape, generator=g) * 0.02).to(torch.bfloat16)

    hd = H // NH
    p = "model.language_model."
    sd = {p + "embed_tokens.weight": w(V, H)}
    for i in range(L):
        q = f"{p}layers.{i}."
        sd[q + "self_attn.q_proj.weight"] = w(NH * hd, H)
        sd[q + "self_attn.k_proj.weight"] = w(NKV * hd, H)
        sd[q + "self_attn.v_proj.weight"] = w(NKV * hd, H)
        sd[q + "self_attn.o_proj.weight"] = w(H, NH * hd)
        sd[q + "mlp.gate_proj.weight"] = w(INTER, H)
        sd[q + "mlp.up_proj.weight"] = w(INTER, H)
        sd[q + "mlp.down_proj.weight"] = w(H, INTER)
        sd[q + "input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
        sd[q + "post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd[p + "norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["lm_head.weight"] = w(V, H)
    return sd


def _tenseurs_vision() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(GRAINE + 1)

    def w(*shape):
        return (torch.randn(*shape, generator=g) * 0.05).to(torch.bfloat16)

    return {
        # Qwen3-VL
        "model.visual.patch_embed.proj.weight": w(VH, 3, 2, 14, 14),
        "model.visual.blocks.0.attn.qkv.weight": w(3 * VH, VH),
        "model.visual.blocks.0.attn.qkv.bias": w(3 * VH),
        "model.visual.blocks.0.mlp.linear_fc1.weight": w(4 * VH, VH),
        "model.visual.merger.linear_fc2.weight": w(H, 4 * VH),
        # Gemma 4 : encodeur (indice de couche 3 : hors plan texte) et projecteur
        "model.vision_tower.vision_model.embeddings.patch_embedding.weight": w(VH, 3, 14, 14),
        "model.vision_tower.vision_model.encoder.layers.3.self_attn.q_proj.weight": w(VH, VH),
        "model.vision_tower.vision_model.encoder.layers.3.mlp.fc1.weight": w(4 * VH, VH),
        "model.embed_vision.embedding_projection.weight": w(H, VH),
        "model.embed_vision.embedding_post_projection_norm.weight": torch.ones(H, dtype=torch.bfloat16),
    }


def _ecrire_source(d, vision: bool) -> str:
    d.mkdir(parents=True, exist_ok=True)
    json.dump(_config(vision), open(d / "config.json", "w"))
    sd = _tenseurs_texte()
    if vision:
        sd.update(_tenseurs_vision())
        (d / "processor_config.json").write_text('{"processor_class": "Qwen3VLProcessor"}')
        (d / "preprocessor_config.json").write_text('{"patch_size": 14}')
        (d / "chat_template.jinja").write_text("{{ messages }}")
    save_file(sd, str(d / "model.safetensors"))
    return str(d)


def _convertir(src: str, out: str, target_rig) -> str:
    spec = load_model_spec(src)
    plan, _ = auto_plan(spec, target_rig, PlannerOptions(max_model_len=512, max_concurrent_seqs=2))
    convert_checkpoint(src, plan, ConversionOptions(out_dir=out), spec=spec)
    return out


def _lire(dossier: str) -> dict[str, torch.Tensor]:
    out = {}
    for fn in sorted(os.listdir(dossier)):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(dossier, fn), framework="pt", device="cpu") as fh:
                for k in fh.keys():
                    out[k] = fh.get_tensor(k)
    return out


def _empreinte(t: torch.Tensor) -> str:
    h = hashlib.sha256(f"{t.dtype}|{tuple(t.shape)}|".encode())
    if t.numel():
        h.update(t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _empreintes(dossier: str) -> dict[str, str]:
    return {k: _empreinte(t) for k, t in _lire(dossier).items()}


def _controle_vision(src: str, out: str) -> list[str]:
    """Les défauts du converti `out` face à la source `src`, [] si le contrat
    est tenu. Doit pouvoir dire FAUX : voir test_une_faute_construite_est_vue."""
    defauts = []
    source = {k: t for k, t in _lire(src).items() if k.startswith(VISION_PREFIXES)}
    sortie = _lire(out)
    vus = {k for k in sortie if k.startswith(VISION_PREFIXES)}
    for k, t in source.items():
        s = sortie.get(k)
        if s is None:
            defauts.append(f"absent : {k}")
            continue
        if s.dtype != torch.bfloat16:
            defauts.append(f"dtype {s.dtype} : {k}")
        elif tuple(s.shape) != tuple(t.shape) or not torch.equal(s, t):
            defauts.append(f"valeurs ou forme différentes : {k}")
    for k in sorted(vus - set(source)):
        defauts.append(f"clé vision inconnue de la source : {k}")
    for k in sortie:                       # rien de quantifié sous un préfixe vision
        if k.rsplit(".", 1)[-1] in ("qweight", "scales", "block_scale") \
                and k.startswith(VISION_PREFIXES):
            defauts.append(f"tenseur vision quantifié : {k}")
    manifest = json.load(open(os.path.join(out, "acvram_manifest.json")))
    attendu = sum(t.numel() * t.element_size() for t in source.values())
    if manifest.get("vision_bytes") != attendu:
        defauts.append(f"vision_bytes {manifest.get('vision_bytes')} ≠ {attendu}")
    if manifest.get("vision") != ("oui" if source else "non"):
        defauts.append(f"vision={manifest.get('vision')!r}")
    for k in source:
        fmt = manifest["tensors"].get(k, {}).get("format")
        if fmt != "bf16":
            defauts.append(f"manifeste format={fmt} : {k}")
    return defauts


@pytest.fixture(scope="module")
def convertis(tmp_path_factory, target_rig):
    base = tmp_path_factory.mktemp("mm")
    src_vl = _ecrire_source(base / "src_vl", vision=True)
    src_texte = _ecrire_source(base / "src_texte", vision=False)
    out_vl = _convertir(src_vl, str(base / "out_vl"), target_rig)
    out_texte = _convertir(src_texte, str(base / "out_texte"), target_rig)
    return src_vl, out_vl, src_texte, out_texte


def test_les_prefixes_sont_ceux_du_contrat():
    from acvram.quant import convert
    assert tuple(convert.VISION_PREFIXES) == VISION_PREFIXES


def test_la_tour_est_gardee_en_bf16_sous_son_nom(convertis):
    src_vl, out_vl, _, _ = convertis
    assert _controle_vision(src_vl, out_vl) == []
    manifest = json.load(open(os.path.join(out_vl, "acvram_manifest.json")))
    assert manifest["vision"] == "oui" and manifest["vision_bytes"] > 0
    # config.json : vision_config et model_type source conservés ; processeur copié
    cfg = json.load(open(os.path.join(out_vl, "config.json")))
    assert cfg["model_type"] == "qwen3_vl" and "vision_config" in cfg
    for fn in ("processor_config.json", "preprocessor_config.json", "chat_template.jinja"):
        assert os.path.isfile(os.path.join(out_vl, fn)), fn
    # les couches texte, elles, sont bien quantifiées (le plan s'applique toujours)
    formats = {e["format"] for k, e in manifest["tensors"].items() if ".layers." in k
               and k.startswith("model.layers.") and k.endswith("proj.weight")}
    assert formats and formats.isdisjoint({"bf16", "fp16", "fp32"}), formats


def test_l_alias_texte_est_identique_au_bit_a_la_conversion_d_avant(convertis):
    _, out_vl, _, out_texte = convertis
    texte = _empreintes(out_texte)
    assert not [k for k in texte if k.startswith(VISION_PREFIXES)]
    assert json.load(open(os.path.join(out_texte, "acvram_manifest.json")))["vision"] == "non"
    assert texte == TEMOIN_TEXTE, "l'alias texte ne donne plus les octets du code d'origine"
    # et la tour n'a rien changé aux tenseurs texte du modèle VL
    vl_texte = {k: v for k, v in _empreintes(out_vl).items() if not k.startswith(VISION_PREFIXES)}
    assert vl_texte == texte


@pytest.mark.parametrize("faute", ["renommee", "quantifiee", "un_bit"])
def test_une_faute_construite_est_vue(convertis, tmp_path, faute):
    src_vl, out_vl, _, _ = convertis
    faux = str(tmp_path / faute)
    shutil.copytree(out_vl, faux)
    cible = "model.visual.blocks.0.attn.qkv.weight"
    manifest = json.load(open(os.path.join(faux, "acvram_manifest.json")))
    fragments = set(manifest["weight_map"].values())
    assert len(fragments) == 1, fragments               # mini-modèle : un seul fragment
    fn = fragments.pop()
    sd = _lire(faux)
    if faute == "renommee":
        sd["model.visuel.blocks.0.attn.qkv.weight"] = sd.pop(cible)
    elif faute == "quantifiee":
        t = sd.pop(cible).float()
        echelle = t.abs().amax(dim=1, keepdim=True) / 127
        sd[cible + ".qweight"] = (t / echelle).round().to(torch.int8)
        sd[cible + ".scales"] = echelle.to(torch.bfloat16)
        manifest["tensors"][cible]["format"] = "int8"
    else:
        t = sd[cible].view(torch.int16).clone()
        t.view(-1)[0] ^= 1
        sd[cible] = t.view(torch.bfloat16)
    save_file(sd, os.path.join(faux, fn))
    json.dump(manifest, open(os.path.join(faux, "acvram_manifest.json"), "w"))
    assert _controle_vision(src_vl, faux), faute


if __name__ == "__main__":                       # génère le témoin : voir TEMOIN_TEXTE
    import pathlib
    import sys
    from acvram.hardware.profiles import load_profile
    base = pathlib.Path(sys.argv[1])
    src = _ecrire_source(base / "src_texte", vision=False)
    out = _convertir(src, str(base / "out_texte"), load_profile("rig-14900k-5090-3080ti"))
    print(json.dumps(_empreintes(out), indent=4, sort_keys=True))
