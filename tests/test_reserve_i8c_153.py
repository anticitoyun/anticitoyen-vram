"""Pièce 153 (25/09, ordre chef) : la réserve de préfill (`ModelSpec.activations_prefill_bytes`,
retirée du budget KV par `loader._reserve_prefill` -> `_marge_carte`) ne comptait AUCUN tenseur promu
int8 (`_octets_marlin`, loader.py:1869, filtre explicite `format != "nvfp4"`) ni un gate+up nvfp4
fusionné en un seul tenseur -- deux OOM servis mesurés le 25/09 (37,94 Mio et 265,94 Mio libres).
Casse si l'un des deux termes est retiré."""
from acvram.engine.config import ModelSpec
from acvram.engine.loader import _plus_grosse_nvfp4_marlin_bytes, _reserve_prefill


def _spec_qwen38_i8c(lineaire=True):
    return ModelSpec(name="q38", architecture="Qwen3_5ForConditionalGeneration", hidden_size=5120,
                     intermediate_size=17408, num_layers=64, num_attention_heads=24, num_key_value_heads=4,
                     vocab_size=248320, max_position_embeddings=262144, head_dim=256,
                     linear_num_value_heads=48 if lineaire else 0, linear_num_key_heads=16,
                     linear_key_head_dim=128, linear_value_head_dim=128)


def test_octets_transitoires_i8c():
    s = _spec_qwen38_i8c()
    attn = 2 * ((24 + 2 * 4) * 256) * 5120
    gdn = s.poids_bf16_couche_lineaire_bytes() + (2 * (16 * 128) + 48 * 128) * 5120
    assert attn == 83_886_080 and gdn == 284_098_560
    assert s.octets_transitoires_i8c_bytes() == max(attn, gdn) == gdn
    assert _spec_qwen38_i8c(False).octets_transitoires_i8c_bytes() == attn      # sans GDN : ne reste que l'attention


def test_la_reserve_couvre_le_pic_i8c():
    """OOM mesuré (pièce 153, poste4-p153-25-09/decode-b1.log) : `_i8c_poids` demandait 120 Mio
    avec 37,94 Mio libres -- rien dans `activations_prefill_bytes` avant ce terme n'en approchait."""
    s = _spec_qwen38_i8c()
    assert s.activations_prefill_bytes(1) >= s.octets_transitoires_i8c_bytes()


_MANIFEST_GATE_UP_FUSIONNE = {"tensors": {
    "model.layers.0.mlp.gate_up_proj.weight": {"format": "nvfp4", "shape": [2 * 17408, 5120]},
    "model.layers.0.mlp.down_proj.weight": {"format": "nvfp4", "shape": [5120, 17408]},
    "lm_head.weight": {"format": "nvfp4", "shape": [248320, 5120]},          # exclu : couvert à part
    "mtp.layers.0.mlp.gate_up_proj.weight": {"format": "nvfp4", "shape": [2 * 17408, 5120]},  # exclu : mtp.
}}


def test_gate_up_fusionne_domine_et_exclut_tete_et_mtp():
    """OOM mesuré (decode-b8.log) : `_marlin_seul` (kernels/__init__.py:1121) demandait 340 Mio avec
    265,94 Mio libres -- `intermediate_size * hidden_size` seul (170 Mio, `plus_grosse` du config) ne
    couvrait qu'UN projecteur, pas la pile gate+up fusionnée (2×) que `depaqueter_marlin` matérialise
    en un bloc. Lu depuis le manifeste (`_plus_grosse_nvfp4_marlin_bytes`), pas depuis l'architecture
    seule (`ModelSpec`) : sinon un modèle jamais fusionné (le 70B de poste3) serait sur-réservé pour rien."""
    non_fusionne = 17408 * 5120 * 2
    fusionne = 2 * non_fusionne
    assert _plus_grosse_nvfp4_marlin_bytes(_MANIFEST_GATE_UP_FUSIONNE) == fusionne  # pas 248320*5120*2 (tête exclue)


def test_retirer_le_terme_casse_la_reserve():
    s = _spec_qwen38_i8c()
    avec = _reserve_prefill(s, 1, _MANIFEST_GATE_UP_FUSIONNE)
    sans = int(s.activations_prefill_bytes(1))                    # comme si le terme manifeste était retiré
    assert avec - sans == 2 * (17408 * 5120 * 2)


def test_manifeste_sans_gate_up_fusionne_ne_reserve_rien_de_plus():
    manifest = {"tensors": {"model.layers.0.mlp.gate_proj.weight": {"format": "nvfp4", "shape": [17408, 5120]},
                            "model.layers.0.mlp.up_proj.weight": {"format": "nvfp4", "shape": [17408, 5120]}}}
    assert _plus_grosse_nvfp4_marlin_bytes(manifest) == 17408 * 5120 * 2   # un seul projecteur, pas fusionné
