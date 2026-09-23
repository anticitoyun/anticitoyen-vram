"""Contrat multimodal, pièce (a) P0 conversion (revue/sage-go-multimodal-
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
VISION_PREFIXES = ("model.vision_tower.", "model.embed_vision.", "model.visual.", "model.vision_embedder.")

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


# Témoin de l'alias texte du mini-modèle Qwen3-VL MoE srcAWQ (ci-dessous), produit par le
# code d'ORIGINE (b4ef64e3, base de la branche) : voie HFQuantCheckpoint, à sec.
TEMOIN_QVL_MOE_TEXTE = {
    "lm_head.weight.block_scale":
        "2406eb97bf6d794481618f7663527c9b8dea9847e738f31d0e9323b0bc58a3eb",
    "lm_head.weight.global_scale":
        "0f1be68273451dfde36cc6aede6ce97d97437396b67b4d0934be3f7cbb020c09",
    "lm_head.weight.qweight":
        "6090cf953d200da1882f6ac7c9b571fcec82649626ae92aa9932e8ee3dd44151",
    "model.embed_tokens.weight":
        "dd3adff573a6f52f4e50938037a84e72f4137238e28728368da457ad8b4d1a6d",
    "model.layers.0.input_layernorm.weight":
        "d16217b910541669d4ee5a39c34dcb8c6cc3bf1cd201fa011e145b69f7a21b8c",
    "model.layers.0.mlp.experts.0.down_proj.weight.act_scale":
        "971f236f54d23c5f57784f3654486f988ff3ba7562ce3557279a4b01d74d5e92",
    "model.layers.0.mlp.experts.0.down_proj.weight.block_scale":
        "0722d6b821faa088bf3e6c9558c4611d1db587074a63772ff6c43ff72a8b6356",
    "model.layers.0.mlp.experts.0.down_proj.weight.global_scale":
        "5d088fb0194b0732a32d5334d5e5fc23f3ec169cf3bb9a1f2811895acf6df87f",
    "model.layers.0.mlp.experts.0.down_proj.weight.qweight":
        "56fbf725886f2383ecb4794824990a64ee921b6b0052246ea5b3e9cf03596e16",
    "model.layers.0.mlp.experts.0.gate_proj.weight.act_scale":
        "298cc135f9c60098d5759a62b249037e398fa5faee86180b8bc9077897007278",
    "model.layers.0.mlp.experts.0.gate_proj.weight.block_scale":
        "5ba1388c5ec1fd432d0a6b68c35418700825ac95d81bdcab1aa0639754f5d2d2",
    "model.layers.0.mlp.experts.0.gate_proj.weight.global_scale":
        "ab33b53a0498952cf8c6890041884130439a72f359740d2697417594d1f8bb7b",
    "model.layers.0.mlp.experts.0.gate_proj.weight.qweight":
        "a59abb5627d2951710b0e155ab0b5ca9cd860617cb3516a52e9f36889b08b961",
    "model.layers.0.mlp.experts.0.up_proj.weight.act_scale":
        "298cc135f9c60098d5759a62b249037e398fa5faee86180b8bc9077897007278",
    "model.layers.0.mlp.experts.0.up_proj.weight.block_scale":
        "aac2b4847be7af197b69236e950c9d796b9761f2ce4e18f709ff8415b4bf8ab5",
    "model.layers.0.mlp.experts.0.up_proj.weight.global_scale":
        "4cdc2a6ffc257f46e9fe85321ef93b3c2f273f9346d51d6b070291f52d73ce97",
    "model.layers.0.mlp.experts.0.up_proj.weight.qweight":
        "2e391f0c90ff47d47573c18324f394844a4f2265118637202f99b4fefbfd3c22",
    "model.layers.0.mlp.experts.1.down_proj.weight.act_scale":
        "971f236f54d23c5f57784f3654486f988ff3ba7562ce3557279a4b01d74d5e92",
    "model.layers.0.mlp.experts.1.down_proj.weight.block_scale":
        "e86c13845aa1a97733c5dab9cf742bd68dbad0817fc15e35cde017c7698ea442",
    "model.layers.0.mlp.experts.1.down_proj.weight.global_scale":
        "861177c7b84cd836c8e86e9a81287d02fcdbe2220e683131abd88aabc1ce4356",
    "model.layers.0.mlp.experts.1.down_proj.weight.qweight":
        "87f77d4d28df61fd3f8da77d6735902b8ca725b1690bb3875fb1632264b4643c",
    "model.layers.0.mlp.experts.1.gate_proj.weight.act_scale":
        "298cc135f9c60098d5759a62b249037e398fa5faee86180b8bc9077897007278",
    "model.layers.0.mlp.experts.1.gate_proj.weight.block_scale":
        "c7eb45012f564754267143ee62dfd977e517d927cfa76b33561821eca4a6fbbf",
    "model.layers.0.mlp.experts.1.gate_proj.weight.global_scale":
        "c844cbd4ea99e6dc4c0f2b48320ed5df765050a0019e6296615279f171ebde01",
    "model.layers.0.mlp.experts.1.gate_proj.weight.qweight":
        "401d463589e60f6c2f66f18bbb84d5072fca9c22981b5de79f03d52a759aa9ff",
    "model.layers.0.mlp.experts.1.up_proj.weight.act_scale":
        "298cc135f9c60098d5759a62b249037e398fa5faee86180b8bc9077897007278",
    "model.layers.0.mlp.experts.1.up_proj.weight.block_scale":
        "bfaaaefc406b16c29da1df212c9c10cd949f65556f7dfebc43898401cd6a9d0e",
    "model.layers.0.mlp.experts.1.up_proj.weight.global_scale":
        "8167cf27e90ebb811abe992f74f1371ee40ca8f12f9b914cca7a83f0a33c0039",
    "model.layers.0.mlp.experts.1.up_proj.weight.qweight":
        "1dc456da6deeaa08458beee87f1c57cc2a1118b0b6cbcdbd5c0da672a320009b",
    "model.layers.0.mlp.experts.2.down_proj.weight.act_scale":
        "971f236f54d23c5f57784f3654486f988ff3ba7562ce3557279a4b01d74d5e92",
    "model.layers.0.mlp.experts.2.down_proj.weight.block_scale":
        "fb2283540cd7c230b34eb3041a71b54c0b7dd67b10d8469d82a4e1a0b0894cfe",
    "model.layers.0.mlp.experts.2.down_proj.weight.global_scale":
        "c4ce584365f30d25b35a92451642bf76e34812a8f1e52be23654c386edc087d4",
    "model.layers.0.mlp.experts.2.down_proj.weight.qweight":
        "7dea17aef1866eee56992e161fe1a983d7e9359c28277265c4eff40ab4ca06e2",
    "model.layers.0.mlp.experts.2.gate_proj.weight.act_scale":
        "298cc135f9c60098d5759a62b249037e398fa5faee86180b8bc9077897007278",
    "model.layers.0.mlp.experts.2.gate_proj.weight.block_scale":
        "693eb58c6dbfd1fdb9c3f4a4e2cb570efb451c5ede79a356df2a12ab3d30c1ff",
    "model.layers.0.mlp.experts.2.gate_proj.weight.global_scale":
        "df96ca3019353bad08323ffae0a3a1b0495aa63d5675a0597f87dba3d917d87d",
    "model.layers.0.mlp.experts.2.gate_proj.weight.qweight":
        "f4cc4fbe225f90368ebaf12f0ab670506a0323b6d5ab3c094fb735df0bba6b32",
    "model.layers.0.mlp.experts.2.up_proj.weight.act_scale":
        "298cc135f9c60098d5759a62b249037e398fa5faee86180b8bc9077897007278",
    "model.layers.0.mlp.experts.2.up_proj.weight.block_scale":
        "356aeb562f13191c1c7c7351f33eba961afaa852833411f53df7fdc907448475",
    "model.layers.0.mlp.experts.2.up_proj.weight.global_scale":
        "df96ca3019353bad08323ffae0a3a1b0495aa63d5675a0597f87dba3d917d87d",
    "model.layers.0.mlp.experts.2.up_proj.weight.qweight":
        "ab82594cd609f111776ae2075122372885f94312d9c43a48353eda68ac797230",
    "model.layers.0.mlp.experts.3.down_proj.weight.act_scale":
        "971f236f54d23c5f57784f3654486f988ff3ba7562ce3557279a4b01d74d5e92",
    "model.layers.0.mlp.experts.3.down_proj.weight.block_scale":
        "a777785e9b8772a01a0f8586957cd4e816faacd71f56f3e0a60fa1c59e62c9dc",
    "model.layers.0.mlp.experts.3.down_proj.weight.global_scale":
        "267a8166cff7bad7b6a5357f31a0e867b40fb87077806c1a81b65184d23ff65d",
    "model.layers.0.mlp.experts.3.down_proj.weight.qweight":
        "36e1cef279876c054f88d1b13595d683136d080b3b08dcd61cddf2984d562384",
    "model.layers.0.mlp.experts.3.gate_proj.weight.act_scale":
        "298cc135f9c60098d5759a62b249037e398fa5faee86180b8bc9077897007278",
    "model.layers.0.mlp.experts.3.gate_proj.weight.block_scale":
        "a58042c36c25d183f99b76f62640717040331b567048b7279f75c00a3419d587",
    "model.layers.0.mlp.experts.3.gate_proj.weight.global_scale":
        "373ead1549aea137a44d7aa51394cd41af6ce0708db03eab9dfb8d96e76eaf80",
    "model.layers.0.mlp.experts.3.gate_proj.weight.qweight":
        "d79bf9f1b6abbcb76af70e667fb2ce01048e5b04408b06310c44ed9b4f757fee",
    "model.layers.0.mlp.experts.3.up_proj.weight.act_scale":
        "298cc135f9c60098d5759a62b249037e398fa5faee86180b8bc9077897007278",
    "model.layers.0.mlp.experts.3.up_proj.weight.block_scale":
        "be099ee94ac5d39532b796fef845d40c1b5d4f9a0dcdd8f1b66f43a50a485c4b",
    "model.layers.0.mlp.experts.3.up_proj.weight.global_scale":
        "5d088fb0194b0732a32d5334d5e5fc23f3ec169cf3bb9a1f2811895acf6df87f",
    "model.layers.0.mlp.experts.3.up_proj.weight.qweight":
        "13a8a214591d95f5df8353f9aded585570676752e19e9916ad85b79032b7c8a6",
    "model.layers.0.mlp.gate.weight":
        "ce0e54d3e3dbb27b7e55163617c84e43c43ff29df9fff9a3df0e55710e099857",
    "model.layers.0.post_attention_layernorm.weight":
        "d16217b910541669d4ee5a39c34dcb8c6cc3bf1cd201fa011e145b69f7a21b8c",
    "model.layers.0.self_attn.k_norm.weight":
        "29d48a64f07cc074051a975431366f9ac17ee3dbc9f6f735628ee43d64cf25bf",
    "model.layers.0.self_attn.k_proj.weight.block_scale":
        "825183efa28fe75830e1ab78013b5905dbdf04682213e59e32d0915f2a77d338",
    "model.layers.0.self_attn.k_proj.weight.global_scale":
        "62fbee03a90cc93c9e3ca42f8503acc798e8ac4e240ae2becb9002ab3e3efe4c",
    "model.layers.0.self_attn.k_proj.weight.qweight":
        "018d63cca783bd169cb6162d4020c626fca261a37149b523c06eadb221b0c8c7",
    "model.layers.0.self_attn.o_proj.weight.block_scale":
        "9bdbd06593dffcdde250d5936c671d22de89937440b6493a0dad72482e560f7e",
    "model.layers.0.self_attn.o_proj.weight.global_scale":
        "636157ad34de53c3fb680f0e57bd8069d5ebe2e2b68ca0c97000a17c85f92fac",
    "model.layers.0.self_attn.o_proj.weight.qweight":
        "92d4f32f0b3b0e2f407a6973649072c15f3ef806f498f9fb46a3ffe65c9f601b",
    "model.layers.0.self_attn.q_norm.weight":
        "29d48a64f07cc074051a975431366f9ac17ee3dbc9f6f735628ee43d64cf25bf",
    "model.layers.0.self_attn.q_proj.weight.block_scale":
        "f0167da19c6fa4416b4075df07c61f035f7a521783dd0d37a1eae840ccd9ad39",
    "model.layers.0.self_attn.q_proj.weight.global_scale":
        "58cb3fe221867809216c7a1d9f7f10e3d56ecb3489a7e7dcb90a3365a353aeaa",
    "model.layers.0.self_attn.q_proj.weight.qweight":
        "b99238dffad6e9f85c0baf3fce4230f2a19d75ae155a2fb57c4aaf219b7737a8",
    "model.layers.0.self_attn.v_proj.weight.block_scale":
        "d0fced84fed4014ec53dc58599c3e29c943dd733a83d2c2bae2d7a0506db96ad",
    "model.layers.0.self_attn.v_proj.weight.global_scale":
        "57e5a74d82270e6f3a11d8fb129094fb54d06020f322913320fe968187be9521",
    "model.layers.0.self_attn.v_proj.weight.qweight":
        "4b4203cfa6cfcec778616133a3e9d8cc0125505e6b343588cebeaf8540d40b02",
    "model.layers.1.input_layernorm.weight":
        "d16217b910541669d4ee5a39c34dcb8c6cc3bf1cd201fa011e145b69f7a21b8c",
    "model.layers.1.mlp.experts.0.down_proj.weight.act_scale":
        "971f236f54d23c5f57784f3654486f988ff3ba7562ce3557279a4b01d74d5e92",
    "model.layers.1.mlp.experts.0.down_proj.weight.block_scale":
        "b6f7e674b724d43cf0c41302a41958d839f5659f803fda12851ed34787f82bc7",
    "model.layers.1.mlp.experts.0.down_proj.weight.global_scale":
        "79e7747c7871339edd8bfe662e91d7aa74e583cd21eff031ca57a79f147e2fab",
    "model.layers.1.mlp.experts.0.down_proj.weight.qweight":
        "c665433f52d7d4764f70560595b6cce06f327bf0cd12b2be2022defe11b98418",
    "model.layers.1.mlp.experts.0.gate_proj.weight.act_scale":
        "298cc135f9c60098d5759a62b249037e398fa5faee86180b8bc9077897007278",
    "model.layers.1.mlp.experts.0.gate_proj.weight.block_scale":
        "eab114f365920b325e4e1618ee3051392caa90f35882188f3f27f3b178984f0a",
    "model.layers.1.mlp.experts.0.gate_proj.weight.global_scale":
        "d50056fb7e0da6a5fc28d14328d6abab94f1019331c95506f11358eb172b0cd6",
    "model.layers.1.mlp.experts.0.gate_proj.weight.qweight":
        "cf1ba091f874f8a58368c63a08604451d0cd53785f552c0a77bff36fa04240de",
    "model.layers.1.mlp.experts.0.up_proj.weight.act_scale":
        "298cc135f9c60098d5759a62b249037e398fa5faee86180b8bc9077897007278",
    "model.layers.1.mlp.experts.0.up_proj.weight.block_scale":
        "6addf8e35ef53e53a7e3eb4300a40f1a96ff107f5344e952a2190d7ce78b6332",
    "model.layers.1.mlp.experts.0.up_proj.weight.global_scale":
        "7d2d472e95d2e7434fbcb9bf1da44cdbf65d18175e1d7ceca1fa938ef79a85ed",
    "model.layers.1.mlp.experts.0.up_proj.weight.qweight":
        "8ae447e504838378d0f4f668c124c4f1128bd22363d1e3d290786a24b88cf065",
    "model.layers.1.mlp.experts.1.down_proj.weight.act_scale":
        "971f236f54d23c5f57784f3654486f988ff3ba7562ce3557279a4b01d74d5e92",
    "model.layers.1.mlp.experts.1.down_proj.weight.block_scale":
        "da46d726d2be974271af7711421b2f57c86165b9b6661067b4741ba2c87bd4d1",
    "model.layers.1.mlp.experts.1.down_proj.weight.global_scale":
        "0d1ac0c23d750443dc7c184db7f2bc18a654f1f4c0d9fbb01732e16322af3294",
    "model.layers.1.mlp.experts.1.down_proj.weight.qweight":
        "8488a4529c2775bb2c37a2df200c1f3bf07bbe33c0c155c09fe4877151266b0a",
    "model.layers.1.mlp.experts.1.gate_proj.weight.act_scale":
        "298cc135f9c60098d5759a62b249037e398fa5faee86180b8bc9077897007278",
    "model.layers.1.mlp.experts.1.gate_proj.weight.block_scale":
        "fba4b0ddb84825f20734e9f04207f49899e76566d4dc5b50dfb6586f27c7d256",
    "model.layers.1.mlp.experts.1.gate_proj.weight.global_scale":
        "13bf8dee265d893ac63701d9b5f89c9a7c59f50744e7556f0bc8bbc5dd588138",
    "model.layers.1.mlp.experts.1.gate_proj.weight.qweight":
        "8cbf9a072ddbc296aabb69e744a7a5e6ec6238aa1667396e16bd2c11e8db9d5c",
    "model.layers.1.mlp.experts.1.up_proj.weight.act_scale":
        "298cc135f9c60098d5759a62b249037e398fa5faee86180b8bc9077897007278",
    "model.layers.1.mlp.experts.1.up_proj.weight.block_scale":
        "acc29a9722cbbf887c2ef7c288f415b13633901155838053fd61ab60e53059e3",
    "model.layers.1.mlp.experts.1.up_proj.weight.global_scale":
        "bab3e0e6b9291918fbaf7c1857d1cd077986eeb8fe5e378b71cb1e22a24d43d9",
    "model.layers.1.mlp.experts.1.up_proj.weight.qweight":
        "7447c31b3befb44fd03139ff80e6e11bd1b5ddade4edb5ff49d96dc2d77a43bf",
    "model.layers.1.mlp.experts.2.down_proj.weight.act_scale":
        "971f236f54d23c5f57784f3654486f988ff3ba7562ce3557279a4b01d74d5e92",
    "model.layers.1.mlp.experts.2.down_proj.weight.block_scale":
        "f8cecc73aecc7f111be7436fd2b8e8be53ae4f3cf194c28cd776354ce3aa1bb4",
    "model.layers.1.mlp.experts.2.down_proj.weight.global_scale":
        "a83e5b5614dc002e42ad3c0682c86998a5732863e775460f5ed5e6901abdcfd6",
    "model.layers.1.mlp.experts.2.down_proj.weight.qweight":
        "c828f9af3666db041b3b949104ff14a9beb47f7a5af7d50b01369a2d6f2e3b5e",
    "model.layers.1.mlp.experts.2.gate_proj.weight.act_scale":
        "298cc135f9c60098d5759a62b249037e398fa5faee86180b8bc9077897007278",
    "model.layers.1.mlp.experts.2.gate_proj.weight.block_scale":
        "9b875b42dcb92e8ec1e3a4421c2f02663ee209d28461370f0c9a49a5fa2ddc75",
    "model.layers.1.mlp.experts.2.gate_proj.weight.global_scale":
        "636157ad34de53c3fb680f0e57bd8069d5ebe2e2b68ca0c97000a17c85f92fac",
    "model.layers.1.mlp.experts.2.gate_proj.weight.qweight":
        "bf232c44bd1aac380200d2c6eeebd8c13dfb5b4e7058f4a0980515d930b56569",
    "model.layers.1.mlp.experts.2.up_proj.weight.act_scale":
        "298cc135f9c60098d5759a62b249037e398fa5faee86180b8bc9077897007278",
    "model.layers.1.mlp.experts.2.up_proj.weight.block_scale":
        "77844f3aa305496e9fd206073797a85edecc7fab0ec5d46ba8d0405d88c57bce",
    "model.layers.1.mlp.experts.2.up_proj.weight.global_scale":
        "13bf8dee265d893ac63701d9b5f89c9a7c59f50744e7556f0bc8bbc5dd588138",
    "model.layers.1.mlp.experts.2.up_proj.weight.qweight":
        "c4d64d2b7f46260602b4279994ad3c56f710ef83aff175a073f64019020d3256",
    "model.layers.1.mlp.experts.3.down_proj.weight.act_scale":
        "971f236f54d23c5f57784f3654486f988ff3ba7562ce3557279a4b01d74d5e92",
    "model.layers.1.mlp.experts.3.down_proj.weight.block_scale":
        "116eb21a634cf1839709ebfdbcb8bb1b00f52e4170f2259db4ad048f0e63b331",
    "model.layers.1.mlp.experts.3.down_proj.weight.global_scale":
        "8167cf27e90ebb811abe992f74f1371ee40ca8f12f9b914cca7a83f0a33c0039",
    "model.layers.1.mlp.experts.3.down_proj.weight.qweight":
        "51de2f13d71697eec284c75533b8a30df54cd1b1e0179cde25a430c88a059f44",
    "model.layers.1.mlp.experts.3.gate_proj.weight.act_scale":
        "298cc135f9c60098d5759a62b249037e398fa5faee86180b8bc9077897007278",
    "model.layers.1.mlp.experts.3.gate_proj.weight.block_scale":
        "580ecbf12680e82246d0d5223ea1513ec5d09cf71bd32602a1d69fbe3f7597c6",
    "model.layers.1.mlp.experts.3.gate_proj.weight.global_scale":
        "cdc0bdefab93352334c80a8f8e54353b6aaad42d8b2a7eb98fa5de114927ac41",
    "model.layers.1.mlp.experts.3.gate_proj.weight.qweight":
        "b841e11575be3bf58f65b73a140df146336aeac9f051b5175687d5563c3ec445",
    "model.layers.1.mlp.experts.3.up_proj.weight.act_scale":
        "298cc135f9c60098d5759a62b249037e398fa5faee86180b8bc9077897007278",
    "model.layers.1.mlp.experts.3.up_proj.weight.block_scale":
        "adafb0472999a2192d161944e8b35ac9ed8bb41ea20074555cf06f8f9a9823fb",
    "model.layers.1.mlp.experts.3.up_proj.weight.global_scale":
        "636157ad34de53c3fb680f0e57bd8069d5ebe2e2b68ca0c97000a17c85f92fac",
    "model.layers.1.mlp.experts.3.up_proj.weight.qweight":
        "5900741326fbb3d577b87eca2f7afa22a9575fc2f6cb6ffe2b48a9ec7e7394f9",
    "model.layers.1.mlp.gate.weight":
        "6574fa09a8307d552a30278153a610eb3839c55da4e3d129155319f5b4dd2893",
    "model.layers.1.post_attention_layernorm.weight":
        "d16217b910541669d4ee5a39c34dcb8c6cc3bf1cd201fa011e145b69f7a21b8c",
    "model.layers.1.self_attn.k_norm.weight":
        "29d48a64f07cc074051a975431366f9ac17ee3dbc9f6f735628ee43d64cf25bf",
    "model.layers.1.self_attn.k_proj.weight.block_scale":
        "908bad69197d9689acc60b6b0c35e4d39280c5c6c80ff4e496186ec8c5bbf143",
    "model.layers.1.self_attn.k_proj.weight.global_scale":
        "b6e782b75abbb3fb1a02a430c0ffcc4d151b567de253d7a3700c8ea78752bc4d",
    "model.layers.1.self_attn.k_proj.weight.qweight":
        "4cfcc96a56977da04b7f0ba9f8b6fa1e897a0a11338815b6a14865dbb11a7c2a",
    "model.layers.1.self_attn.o_proj.weight.block_scale":
        "31f3a70cd5e0b91dc3a4dbdb910d1d5b34ff4eb4fa88cc953daa09a03ec04661",
    "model.layers.1.self_attn.o_proj.weight.global_scale":
        "0d1ac0c23d750443dc7c184db7f2bc18a654f1f4c0d9fbb01732e16322af3294",
    "model.layers.1.self_attn.o_proj.weight.qweight":
        "e052dd5bf5222533fd1661807b9e50cf85c0a866941487b626472eb0a8d739ee",
    "model.layers.1.self_attn.q_norm.weight":
        "29d48a64f07cc074051a975431366f9ac17ee3dbc9f6f735628ee43d64cf25bf",
    "model.layers.1.self_attn.q_proj.weight.block_scale":
        "8d983c92dc98be368eb74b1b3d0ba5b78244efd3a03a4ae8544b0c760c72714c",
    "model.layers.1.self_attn.q_proj.weight.global_scale":
        "bab3e0e6b9291918fbaf7c1857d1cd077986eeb8fe5e378b71cb1e22a24d43d9",
    "model.layers.1.self_attn.q_proj.weight.qweight":
        "65affe60edc897e531332ff3d4918e75656fee05f9ab12f75f46312de2661e9c",
    "model.layers.1.self_attn.v_proj.weight.block_scale":
        "ab566748c11a7b64a469f1ecc78d7f575e947c41df7a58d4699c4654cfeab91e",
    "model.layers.1.self_attn.v_proj.weight.global_scale":
        "e745d07605139e8a807dacf4c781d796f97a95658dd2e1b2ab040f70647c5b4c",
    "model.layers.1.self_attn.v_proj.weight.qweight":
        "e4dc9c42677e3e9f8d574b16d8e46470841ff815a63aaf0ae89797f7ce3c60cd",
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


def test_la_ligne_de_regime_nomme_la_vision_seulement_modele_charge():
    from acvram import regime
    try:
        regime.declarer_modele_charge(None)
        assert "vision=" not in regime.regime_ligne()
        regime.declarer_modele_charge({"vision": "oui"})
        assert " vision=declaree(tour absente) " in regime.regime_ligne() + " "   # oui sans tour chargée (engine/vision) : nommé
        regime.declarer_modele_charge({"vision": "non"})
        assert " vision=off " in regime.regime_ligne() + " "
        regime.declarer_modele_charge({})
        assert " vision=off " in regime.regime_ligne() + " "
    finally:
        regime.declarer_modele_charge(None)
    assert "vision=" not in regime.regime_ligne()


# ---- gemma4_unified (12B, 20/09 : Manon) : vision = model.vision_embedder.* + model.embed_vision, PAS de SigLIP ;
# ---- audio (model.embed_audio.*) non servi : écarté ET nommé (manifeste audio: "non servi"), jamais en silence.

def _source_unified(d) -> str:
    d.mkdir(parents=True, exist_ok=True)
    cfg = _config(False)
    cfg.update({"architectures": ["Gemma4UnifiedForConditionalGeneration"], "model_type": "gemma4_unified",
                "vision_config": {"model_type": "gemma4_unified_vision", "hidden_size": VH},
                "audio_config": {"model_type": "gemma4_unified_audio", "hidden_size": 8}})
    json.dump(cfg, open(d / "config.json", "w"))
    g = torch.Generator().manual_seed(GRAINE + 7)
    w = lambda *sh: (torch.randn(*sh, generator=g) * 0.05).to(torch.bfloat16)
    sd = _tenseurs_texte()
    sd.update({"model.vision_embedder.patch_dense.weight": w(VH, 3 * 14 * 14), "model.vision_embedder.patch_dense.bias": w(VH),
               "model.vision_embedder.patch_ln1.weight": w(3 * 14 * 14), "model.vision_embedder.pos_embedding": w(64, VH),
               "model.embed_vision.embedding_projection.weight": w(H, VH),
               "model.embed_audio.embedding_projection.weight": w(H, 8)})
    (d / "processor_config.json").write_text('{"processor_class": "Gemma4Processor"}')
    save_file(sd, str(d / "model.safetensors"))
    return str(d)


def test_unified_garde_l_embedder_de_patches_et_nomme_l_audio_non_servi(tmp_path, target_rig, capsys):
    src = _source_unified(tmp_path / "src_u")
    out = _convertir(src, str(tmp_path / "out_u"), target_rig)
    man = json.load(open(os.path.join(out, "acvram_manifest.json")))
    sortie = _lire(out)
    vision = {k: t for k, t in _lire(src).items() if k.startswith(VISION_PREFIXES)}
    assert len(vision) == 5
    for k, t in vision.items():                         # tous gardés, au bit, bf16, nom source
        assert k in sortie and sortie[k].dtype is torch.bfloat16 and torch.equal(sortie[k], t), k
    assert man["vision"] == "oui" and man["vision_bytes"] == sum(t.numel() * 2 for t in vision.values())
    assert "model.embed_audio.embedding_projection.weight" not in sortie      # audio : pas gardé…
    assert man.get("audio") == "non servi"                                     # … mais nommé au manifeste
    assert "audio non servi" in capsys.readouterr().out                         # … et au journal


# ---- Qwen3-VL MoE (30B-A3B, contrat sage-go-qwen3vl-parallele-20-09 § 2, pièce (a)) : tour
# ---- model.visual.* + model.visual.merger.* + model.visual.deepstack_merger_list.{0,1,2}.* gardés
# ---- bf16 au bit sous leur nom source, depuis une source bf16 ET depuis la source du parc
# ---- (« srcAWQ » = compressed-tensors pack-quantized : weight_packed/weight_scale/weight_shape,
# ---- tour dans `ignore` donc bf16 en clair — voie HFQuantCheckpoint) ; manifeste deepstack +
# ---- deepstack_visual_indexes + mrope_section + mrope_interleaved lus de la config, jamais devinés.

E, EI, GS = 4, 32, 32                         # experts, largeur d'expert, groupe int4
DEEPSTACK_IDX = [8, 16, 24]
MROPE = [24, 20, 20]
DEEPSTACK_PREFIX = "model.visual.deepstack_merger_list."
QUANT_CT = {"quant_method": "compressed-tensors", "format": "pack-quantized",
            "quantization_status": "compressed", "version": "0.14.0",
            "config_groups": {"group_0": {"format": "pack-quantized", "targets": ["Linear"],
                                          "weights": {"num_bits": 4, "group_size": GS, "strategy": "group",
                                                      "symmetric": True, "type": "int"}}},
            "ignore": ["lm_head"]}


def _config_qvl_moe(vision: bool, awq: bool) -> dict:
    txt = {**_texte_config(), "model_type": "qwen3_vl_moe_text", "head_dim": H // NH,
           "num_experts": E, "num_experts_per_tok": 2, "moe_intermediate_size": EI,
           "decoder_sparse_step": 1, "mlp_only_layers": [], "norm_topk_prob": True,
           "rope_theta": 5000000, "rope_scaling": {"mrope_interleaved": True, "mrope_section": MROPE,
                                                   "rope_type": "default"}}
    cfg = {"architectures": ["Qwen3VLMoeForConditionalGeneration"], "model_type": "qwen3_vl_moe",
           "text_config": txt, "dtype": "bfloat16", "tie_word_embeddings": False}
    if vision:
        cfg["vision_config"] = {"model_type": "qwen3_vl_moe", "hidden_size": VH, "depth": 1, "num_heads": 2,
                                "patch_size": 16, "in_channels": 3, "out_hidden_size": H, "spatial_merge_size": 2,
                                "temporal_patch_size": 2, "deepstack_visual_indexes": DEEPSTACK_IDX}
    if awq:
        cfg["quantization_config"] = QUANT_CT
    return cfg


def _tenseurs_texte_moe() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(GRAINE + 3)

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
        sd[q + "self_attn.q_norm.weight"] = torch.ones(hd, dtype=torch.bfloat16)
        sd[q + "self_attn.k_norm.weight"] = torch.ones(hd, dtype=torch.bfloat16)
        sd[q + "mlp.gate.weight"] = w(E, H)
        for e in range(E):
            sd[q + f"mlp.experts.{e}.gate_proj.weight"] = w(EI, H)
            sd[q + f"mlp.experts.{e}.up_proj.weight"] = w(EI, H)
            sd[q + f"mlp.experts.{e}.down_proj.weight"] = w(H, EI)
        sd[q + "input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
        sd[q + "post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd[p + "norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["lm_head.weight"] = w(V, H)
    return sd


def _tenseurs_vision_qvl() -> dict[str, torch.Tensor]:
    """Les familles de clés de la source réelle (index.json de
    Qwen3-VL-30B-A3B-abliterated-AWQ, lu le 20/09) : patch_embed, pos_embed,
    blocks.N.{attn.qkv, attn.proj, mlp.linear_fc1, mlp.linear_fc2, norm1, norm2}
    weight+bias, merger.{norm, linear_fc1, linear_fc2}, deepstack_merger_list.N idem."""
    g = torch.Generator().manual_seed(GRAINE + 4)

    def w(*shape):
        return (torch.randn(*shape, generator=g) * 0.05).to(torch.bfloat16)

    sd = {"model.visual.patch_embed.proj.weight": w(VH, 3, 2, 16, 16),
          "model.visual.patch_embed.proj.bias": w(VH),
          "model.visual.pos_embed.weight": w(9, VH)}
    b = "model.visual.blocks.0."
    for nom, forme in (("attn.qkv", (3 * VH, VH)), ("attn.proj", (VH, VH)),
                       ("mlp.linear_fc1", (4 * VH, VH)), ("mlp.linear_fc2", (VH, 4 * VH))):
        sd[b + nom + ".weight"] = w(*forme)
        sd[b + nom + ".bias"] = w(forme[0])
    for nom in ("norm1", "norm2"):
        sd[b + nom + ".weight"] = torch.ones(VH, dtype=torch.bfloat16)
        sd[b + nom + ".bias"] = w(VH)
    for m in ["model.visual.merger."] + [f"{DEEPSTACK_PREFIX}{i}." for i in range(len(DEEPSTACK_IDX))]:
        sd[m + "norm.weight"] = torch.ones(4 * VH, dtype=torch.bfloat16)
        sd[m + "norm.bias"] = w(4 * VH)
        sd[m + "linear_fc1.weight"] = w(4 * VH, 4 * VH)
        sd[m + "linear_fc1.bias"] = w(4 * VH)
        sd[m + "linear_fc2.weight"] = w(H, 4 * VH)
        sd[m + "linear_fc2.bias"] = w(H)
    return sd


def _empaqueter_ct(wt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """bf16 [out, in] → (weight_packed int32 [out, in/8], weight_scale bf16
    [out, in/GS], weight_shape int64 [2]), convention compressed-tensors
    pack_to_int32 : entiers signés décalés de +8, nibble faible d'abord."""
    w = wt.float()
    out, inn = w.shape
    grp = w.reshape(out, inn // GS, GS)
    scale = (grp.abs().amax(-1) / 7.0).clamp(min=1e-8).to(torch.bfloat16)
    q = (grp / scale.float()[..., None]).round().clamp(-8, 7).to(torch.int32).reshape(out, inn) + 8
    packed = torch.zeros(out, inn // 8, dtype=torch.int32)
    for j in range(8):
        packed |= q[:, j::8] << (4 * j)
    return packed, scale, torch.tensor([out, inn], dtype=torch.int64)


def _dequantifier_ct(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Référence indépendante de hfquant._ct_int4 (même arithmétique, écrite ici)."""
    n = packed.shape[1] * 8
    q = torch.stack([(packed >> (4 * j)) & 0xF for j in range(8)], -1).reshape(packed.shape[0], n)
    return ((q.float() - 8.0) * scale.float().repeat_interleave(GS, 1)).to(torch.bfloat16)


def _est_lineaire_texte(k: str) -> bool:
    return k.startswith("model.language_model.layers.") and k.endswith(".weight") \
        and ("_proj." in k)                      # q/k/v/o et experts ; mlp.gate, normes, lm_head en clair (ignore)


def _ecrire_source_qvl(d, vision: bool, awq: bool, dequantifie: bool = False) -> str:
    """`awq` : texte empaqueté compressed-tensors (la source du parc) ;
    `dequantifie` : source bf16 dont les projections valent EXACTEMENT le
    déquantifié de la source awq (le clair que la voie HFQuant doit rendre)."""
    d.mkdir(parents=True, exist_ok=True)
    json.dump(_config_qvl_moe(vision, awq), open(d / "config.json", "w"))
    sd = {}
    for k, t in _tenseurs_texte_moe().items():
        if _est_lineaire_texte(k) and (awq or dequantifie):
            packed, scale, shape = _empaqueter_ct(t)
            if awq:
                base = k[:-len(".weight")]
                sd[base + ".weight_packed"], sd[base + ".weight_scale"], sd[base + ".weight_shape"] = packed, scale, shape
            else:
                sd[k] = _dequantifier_ct(packed, scale)
        else:
            sd[k] = t
    if vision:
        sd.update(_tenseurs_vision_qvl())
        (d / "preprocessor_config.json").write_text('{"patch_size": 16, "merge_size": 2}')
        (d / "video_preprocessor_config.json").write_text('{"patch_size": 16}')
        (d / "chat_template.jinja").write_text("{{ messages }}")
        (d / "chat_template.json").write_text('{"chat_template": "{{ messages }}"}')   # Qwen3-VL-2B réel : gabarit en .json
        (d / "merges.txt").write_text("#version: 0.2\n")
    save_file(sd, str(d / "model.safetensors"))
    return str(d)


def _controle_qvl(src: str, out: str) -> list[str]:
    """_controle_vision + le contrat Qwen3-VL : merger et deepstack tous là,
    manifeste deepstack / indices / mrope égaux à la config SOURCE."""
    defauts = _controle_vision(src, out)
    cfg = json.load(open(os.path.join(src, "config.json")))
    man = json.load(open(os.path.join(out, "acvram_manifest.json")))
    sortie = _lire(out)
    source = _lire(src)
    for fam in ("model.visual.merger.", DEEPSTACK_PREFIX):
        attendus = {k for k in source if k.startswith(fam)}
        manquants = attendus - set(sortie)
        if not attendus or manquants:
            defauts.append(f"{fam}* : attendus {len(attendus)}, manquants {sorted(manquants)}")
    idx = cfg["vision_config"]["deepstack_visual_indexes"]
    if man.get("deepstack") != "oui" or man.get("deepstack_visual_indexes") != idx:
        defauts.append(f"manifeste deepstack={man.get('deepstack')!r} indices={man.get('deepstack_visual_indexes')!r} ≠ {idx}")
    rs = cfg["text_config"]["rope_scaling"]
    if man.get("mrope_section") != rs["mrope_section"] or man.get("mrope_interleaved") is not rs["mrope_interleaved"]:
        defauts.append(f"manifeste mrope_section={man.get('mrope_section')!r} interleaved={man.get('mrope_interleaved')!r}")
    return defauts


@pytest.fixture(scope="module")
def convertis_qvl(tmp_path_factory, target_rig):
    base = tmp_path_factory.mktemp("qvl")
    srcs = {"clair": _ecrire_source_qvl(base / "src_clair", vision=True, awq=False, dequantifie=True),
            "srcawq": _ecrire_source_qvl(base / "src_awq", vision=True, awq=True),
            "texte_awq": _ecrire_source_qvl(base / "src_texte_awq", vision=False, awq=True)}
    outs = {nom: _convertir(src, str(base / f"out_{nom}"), target_rig) for nom, src in srcs.items()}
    return srcs, outs


@pytest.mark.parametrize("voie", ["clair", "srcawq"])
def test_qvl_moe_tour_merger_et_deepstack_gardes_au_bit(convertis_qvl, voie):
    srcs, outs = convertis_qvl
    src, out = srcs[voie], outs[voie]
    assert _controle_qvl(src, out) == []
    sortie = _lire(out)
    vision = {k for k in _lire(src) if k.startswith("model.visual.")}
    assert len(vision) == 3 + 12 + 6 * (1 + len(DEEPSTACK_IDX))              # 39 tenseurs, tous gardés
    assert vision <= set(sortie)
    assert len({k for k in sortie if k.startswith(DEEPSTACK_PREFIX)}) == 6 * len(DEEPSTACK_IDX)
    man = json.load(open(os.path.join(out, "acvram_manifest.json")))
    assert (man["vision"], man["deepstack"]) == ("oui", "oui")
    assert man["deepstack_visual_indexes"] == DEEPSTACK_IDX and man["mrope_section"] == MROPE
    assert man["mrope_interleaved"] is True
    cfg = json.load(open(os.path.join(out, "config.json")))
    assert cfg["model_type"] == "qwen3_vl_moe" and cfg["vision_config"]["deepstack_visual_indexes"] == DEEPSTACK_IDX
    assert cfg["text_config"]["rope_scaling"]["mrope_section"] == MROPE
    # 20/09 17:48 (Qwen3-VL-2B servi) : sans chat_template.json, AutoProcessor.apply_chat_template refusait
    # (« does not have a chat template ») ; merges.txt/vocab.json/added_tokens.json suivent aussi le tokenizer
    for fn in ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja", "chat_template.json", "merges.txt"):
        assert os.path.isfile(os.path.join(out, fn)), fn
    # le texte, lui, est quantifié : experts et projections d'attention, aucun en bf16
    fmts = {man["tensors"][k]["format"] for k in man["tensors"]
            if k.startswith("model.layers.") and "_proj.weight" in k}
    assert fmts and fmts.isdisjoint({"bf16", "fp16", "fp32"}), fmts
    assert not [k for k in sortie if k.endswith((".weight_packed", ".weight_scale", ".weight_shape"))]


def test_qvl_moe_la_voie_srcawq_rend_le_texte_du_clair_dequantifie(convertis_qvl):
    srcs, outs = convertis_qvl
    texte = lambda out: {k: v for k, v in _empreintes(out).items() if not k.startswith(VISION_PREFIXES)}
    assert texte(outs["srcawq"]) == texte(outs["clair"])       # HFQuant déquantifie exactement ce clair
    assert texte(outs["srcawq"]) == texte(outs["texte_awq"])   # la tour n'a rien changé au texte
    assert texte(outs["texte_awq"]) == TEMOIN_QVL_MOE_TEXTE, "l'alias texte ne donne plus les octets du code d'origine"
    man = json.load(open(os.path.join(outs["texte_awq"], "acvram_manifest.json")))
    assert man["vision"] == "non" and "deepstack" not in man and "mrope_section" not in man
    assert not [k for k in _lire(outs["texte_awq"]) if k.startswith(VISION_PREFIXES)]


@pytest.mark.parametrize("faute", ["deepstack_absent", "indices_faux", "mrope_faux"])
def test_qvl_moe_une_faute_construite_est_vue(convertis_qvl, tmp_path, faute):
    srcs, outs = convertis_qvl
    faux = str(tmp_path / faute)
    shutil.copytree(outs["srcawq"], faux)
    man = json.load(open(os.path.join(faux, "acvram_manifest.json")))
    if faute == "deepstack_absent":
        fn = set(man["weight_map"].values()).pop()
        sd = _lire(faux)
        sd.pop(DEEPSTACK_PREFIX + "2.linear_fc2.weight")
        save_file(sd, os.path.join(faux, fn))
    elif faute == "indices_faux":
        man["deepstack_visual_indexes"] = [8, 16]
    else:
        man["mrope_interleaved"] = False
    json.dump(man, open(os.path.join(faux, "acvram_manifest.json"), "w"))
    assert _controle_qvl(srcs["srcawq"], faux), faute


def test_qvl_moe_config_et_tenseurs_deepstack_doivent_se_repondre(tmp_path, target_rig):
    """Fusions deepstack dans les tenseurs, aucun indice dans vision_config :
    refus nommé à la conversion, pas un converti muet."""
    src = _ecrire_source_qvl(tmp_path / "src", vision=True, awq=True)
    cfg = json.load(open(os.path.join(src, "config.json")))
    del cfg["vision_config"]["deepstack_visual_indexes"]
    json.dump(cfg, open(os.path.join(src, "config.json"), "w"))
    with pytest.raises(ValueError, match="deepstack incohérent"):
        _convertir(src, str(tmp_path / "out"), target_rig)


def test_le_prefixe_deepstack_est_celui_du_contrat():
    from acvram.quant import convert
    assert tuple(convert.DEEPSTACK_PREFIXES) == (DEEPSTACK_PREFIX,)
    assert convert.est_tenseur_vision(DEEPSTACK_PREFIX + "0.norm.weight")


if __name__ == "__main__":                       # génère les témoins : TEMOIN_TEXTE, TEMOIN_QVL_MOE_TEXTE
    import pathlib
    import sys
    from acvram.hardware.profiles import load_profile
    base = pathlib.Path(sys.argv[1])
    rig = load_profile("rig-14900k-5090-3080ti")
    src = _ecrire_source(base / "src_texte", vision=False)
    out = _convertir(src, str(base / "out_texte"), rig)
    print("TEMOIN_TEXTE =", json.dumps(_empreintes(out), indent=4, sort_keys=True))
    src = _ecrire_source_qvl(base / "src_texte_awq", vision=False, awq=True)
    out = _convertir(src, str(base / "out_texte_awq"), rig)
    print("TEMOIN_QVL_MOE_TEXTE =", json.dumps(_empreintes(out), indent=4, sort_keys=True))
