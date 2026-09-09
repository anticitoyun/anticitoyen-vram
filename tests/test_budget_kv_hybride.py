"""Le budget KV comptait toutes les couches, l'état récurrent aucune.

Deux erreurs de sens opposé qui se compensaient par accident : le cache KV
était surestimé d'un facteur 4 à 14 sur un hybride, et l'état récurrent des
couches linéaires — qui vit sur la carte, une copie par séquence — n'était
budgété nulle part. Corriger la première seule aurait déplacé le défaut.

Le contrôle qui rend la correction sûre est ici : sur un modèle PUREMENT
QUADRATIQUE, rien ne doit bouger — ni le compte de couches, ni la provision,
ni le nombre de blocs.
"""
import pytest

from acvram.engine.config import ModelSpec

BASE = dict(name="essai", architecture="llama", hidden_size=2560,
            intermediate_size=6912, num_layers=32, num_attention_heads=20,
            num_key_value_heads=4, vocab_size=32000,
            max_position_embeddings=8192)

# 8 couches d'attention pleine sur 32, comme Agents-A1-4B
HYBRIDE = ["linear_attention"] * 3 + ["full_attention"]
HYBRIDE = HYBRIDE * 8


def _spec(**kw):
    return ModelSpec(**{**BASE, **kw})


def test_un_modele_quadratique_ne_bouge_pas():
    """Le lot de contrôle : sans layer_types, tout doit valoir l'ancien."""
    s = _spec()
    assert s.couches_avec_kv == s.num_layers, "le repli doit compter TOUTES les couches"
    assert s.couches_recurrentes == 0
    assert s.etat_recurrent_bytes(16) == 0, "aucune provision sur un quadratique"
    assert all(s.couche_a_kv(i) for i in range(s.num_layers))


def test_un_quadratique_declare_ne_bouge_pas_non_plus():
    """Même avec layer_types, un modèle 100 % attention garde son compte."""
    s = _spec(layer_types=["full_attention"] * 32)
    assert s.couches_avec_kv == 32
    assert s.etat_recurrent_bytes(16) == 0


def test_l_hybride_ne_compte_que_ses_couches_a_cache():
    s = _spec(layer_types=HYBRIDE)
    assert s.couches_avec_kv == 8, "8 full_attention sur 32"
    assert s.couches_recurrentes == 24
    # le budget par jeton suit, dans le rapport exact
    assert (_spec().kv_bytes_per_token(8)
            == 4 * s.kv_bytes_per_token(8)), "32 couches contre 8"


def test_sliding_attention_a_un_cache():
    """Piège : une fenêtre glissante borne la LONGUEUR, pas l'existence du
    cache. La ranger avec les linéaires sous-estimerait le budget."""
    s = _spec(layer_types=["sliding_attention"] * 32)
    assert s.couches_avec_kv == 32
    assert s.couches_recurrentes == 0


def test_moe_et_mlp_ne_sont_pas_des_couches_d_attention():
    """Second piège : compter « tout sauf les linéaires » donnait 29 couches à
    cache sur Nemotron là où il y en a 6."""
    s = _spec(layer_types=(["mamba"] * 20 + ["moe"] * 6 + ["full_attention"] * 6))
    assert s.couches_avec_kv == 6
    assert s.couches_recurrentes == 0, "mamba n'est pas linear_attention"


def test_l_etat_recurrent_suit_new_static_et_la_concurrence():
    """La formule doit couvrir les trois convolutions ET la matrice d'état —
    les oublier sous-provisionnait de 7 %, soit le défaut qu'on corrige."""
    s = _spec(layer_types=HYBRIDE, linear_num_value_heads=48,
              linear_value_head_dim=128, linear_conv_kernel_dim=4)
    nh, d, k1, n = 48, 128, 3, 24
    attendu = (3 * (nh * d) * k1 + nh * d * d) * 4 * n
    assert s.etat_recurrent_bytes(1) == attendu
    assert s.etat_recurrent_bytes(16) == attendu * 16, "une copie PAR SEQUENCE"


def test_la_provision_depasse_le_budget_kv_des_27b():
    """Le fait qui a arrêté la correction naïve : sur un Qwen3.x-27B, l'état à
    seize séquences dépasse le budget KV entier. Provisionner n'est donc pas
    une précaution, c'est ce qui décide."""
    s = _spec(num_layers=64, layer_types=(["linear_attention"] * 48
                                          + ["full_attention"] * 16),
              linear_num_value_heads=48, linear_value_head_dim=128,
              linear_conv_kernel_dim=4)
    assert s.etat_recurrent_bytes(16) / 2**20 > 1854, \
        "moins que le budget KV d'un 27B : la mesure du 9/09 disait le contraire"
