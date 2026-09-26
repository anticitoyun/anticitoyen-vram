"""Pièce 172 (question de chef, 25/09) : la réserve de préfill (`ModelSpec.activations_prefill_bytes`, retirée du
budget KV par `loader._reserve_prefill` → `_marge_carte`) compte les poids bf16 qu'une couche Gated DeltaNet garde
vivants ENSEMBLE sous `kernels.depaquetage_partage` (qkv + gate + alpha + beta + out), et plus seulement la plus
grosse matrice. Casse si le terme est retiré : sur des dimensions de Qwen3.8, la somme (232 Mio) dépasse la plus
grosse matrice (178 Mio)."""
from acvram.engine.config import ModelSpec


def _spec_qwen38(lineaire=True):
    return ModelSpec(name="q38", architecture="Qwen3_5ForConditionalGeneration", hidden_size=5120,
                     intermediate_size=17408, num_layers=64, num_attention_heads=24, num_key_value_heads=4,
                     vocab_size=248320, max_position_embeddings=262144, head_dim=256,
                     linear_num_value_heads=48 if lineaire else 0, linear_num_key_heads=16,
                     linear_key_head_dim=128, linear_value_head_dim=128)


def test_poids_de_la_couche_lineaire():
    s = _spec_qwen38()
    assert s.poids_bf16_couche_lineaire_bytes() == (10240 + 6144 + 96 + 6144) * 5120 * 2
    assert _spec_qwen38(False).poids_bf16_couche_lineaire_bytes() == 0


def test_la_reserve_couvre_les_poids_partages():
    # Pièce 153 (25/09) : `activations_prefill_bytes` a gagné un terme `gate_up_fusionne` (indépendant
    # de la couche linéaire, dominant sur ces dimensions Qwen3.8 : 356,5 Mio > 231,7) -- l'égalité au
    # bit du delta d'avant ne tient plus dans CE régime précis (le terme masque la différence), la
    # couverture du terme GDN, elle, tient toujours (`>=`, `test_poids_de_la_couche_lineaire` la prouve
    # au bit isolément).
    s, sans = _spec_qwen38(), _spec_qwen38(False)
    somme = s.poids_bf16_couche_lineaire_bytes()
    plus_grosse = 17408 * 5120 * 2
    assert somme > plus_grosse                                   # le cas où le terme compte, isolément
    assert s.activations_prefill_bytes(1) >= somme
    assert s.activations_prefill_bytes(8192) >= sans.activations_prefill_bytes(8192)
