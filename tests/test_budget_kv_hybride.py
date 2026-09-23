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
    cache sur Nemotron là où il y en a 6.

    La seconde assertion disait « mamba n'est pas linear_attention » et
    passait : elle encodait le défaut au lieu de le prévenir. Un `mamba` ne
    porte pas de cache KV, mais il porte bien un état récurrent — les deux
    questions sont distinctes et la première ne répond pas à la seconde.
    """
    s = _spec(layer_types=(["mamba"] * 20 + ["moe"] * 6 + ["full_attention"] * 6))
    assert s.couches_avec_kv == 6, "seules les 6 full_attention ont un cache"
    assert s.couches_recurrentes == 20, "les 20 mamba portent un etat"
    assert not any(s.couche_a_kv(i) for i in range(20))


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


def test_mamba_et_conv_sont_provisionnes():
    """Onze modèles du parc étaient à découvert : la première version ne
    provisionnait que `linear_attention`, alors que quatre familles portent un
    état. `Nemotron-Nano-9B` y perdait 2,11 Gio pour 1,68 Gio de KV rendus."""
    s = _spec(num_layers=56, layer_types=(["mamba"] * 27 + ["mlp"] * 25
                                          + ["full_attention"] * 4),
              mamba_num_heads=128, mamba_head_dim=80, mamba_state_size=128,
              mamba_n_groups=8, mamba_conv_kernel=4)
    assert s.couches_avec_kv == 4
    assert s.couches_recurrentes == 27
    assert s.etat_recurrent_bytes(16) > 2 * 2**30, \
        "l'etat mamba de Nemotron-Nano-9B vaut 2,11 Gio a seize sequences"


def test_un_type_inconnu_se_signale():
    """Provisionner zéro en silence est le défaut qui a valu un commit à
    reprendre : une architecture neuve doit se dénoncer."""
    assert _spec(layer_types=["full_attention"] * 32).types_de_couche_inconnus == []
    assert _spec(layer_types=["mamba", "retention", "full_attention"]
                 ).types_de_couche_inconnus == ["retention"]


def test_les_quadratiques_restent_a_zero_apres_l_ajout():
    """Contrôle : ajouter mamba et conv ne doit pas faire déborder la
    détection sur les 61 modèles purement quadratiques du parc."""
    for lt in (None, ["full_attention"] * 32, ["sliding_attention"] * 32,
               ["full_attention"] * 16 + ["moe"] * 16):
        s = _spec(layer_types=lt) if lt else _spec()
        assert s.etat_recurrent_bytes(16) == 0, f"provision non nulle sur {lt}"
        assert s.couches_recurrentes == 0


# --- attention a latent compresse (MLA) ------------------------------------

MLA = dict(kv_lora_rank=512, qk_rope_head_dim=64)


def test_mla_stocke_sans_paginer():
    """La distinction qui décide : une couche MLA garde un cache par jeton
    mais `loader.py` ne l'enregistre jamais dans `a_allouer` — sa branche fait
    `continue` avant. Elle STOCKE sans PAGINER."""
    s = _spec(layer_types=["full_attention"] * 32, **MLA)
    assert s.est_mla
    assert s.couches_avec_kv == 32, "elles gardent bien un cache"
    assert not any(s.couche_a_kv(i) for i in range(32)), \
        "mais aucune n'alloue de bloc pagine"


def test_le_budget_mla_suit_le_latent_et_non_les_tetes():
    """Mesuré sur le parc, la formule à requêtes groupées se trompait de 0,28
    à 7,78 selon le modèle — deux sens opposés, donc pas un réglage."""
    s = _spec(num_key_value_heads=4, layer_types=["full_attention"] * 47, **MLA)
    attendu = (512 + 64) * 2 * 47
    assert s.kv_bytes_per_token(8) == attendu
    sans_mla = _spec(num_key_value_heads=4, layer_types=["full_attention"] * 47)
    assert s.kv_bytes_per_token(8) != sans_mla.kv_bytes_per_token(8)


def test_un_modele_sans_latent_garde_la_formule_groupee():
    """Contrôle : la détection ne doit pas déborder sur les 110 autres."""
    s = _spec(layer_types=["full_attention"] * 32)
    assert not s.est_mla
    assert s.kv_bytes_per_token(8) == _spec().kv_bytes_per_token(8)
    assert all(s.couche_a_kv(i) for i in range(32))


# --- tete liee : un poste que rien ne comptait -----------------------------

def test_la_tete_liee_est_comptee():
    """`lm_head_params` vaut ZERO quand la tete est liee — aucun tenseur
    `lm_head` n'existe. Mais le chargeur fabrique une copie quantifiee de la
    table d'embedding, qui S'AJOUTE sans remplacer, et qui occupe la carte."""
    lie = _spec(tie_word_embeddings=True, vocab_size=151936, hidden_size=2560)
    assert lie.lm_head_params == 0, "aucun tenseur lm_head sur un modele lie"
    assert lie.tete_liee_bytes(128) > 0, "et pourtant la copie existe"
    # ~0,369 Gio sur Qwen3-4B, mesure le 9/09/2026
    assert 0.35 < lie.tete_liee_bytes(128) / 2**30 < 0.39


def test_une_tete_separee_ne_double_pas():
    """Controle : sur un modele NON lie, la tete est deja comptee par
    `lm_head_params`. La compter deux fois surestimerait les poids et
    reserverait pour rien."""
    s = _spec(tie_word_embeddings=False, vocab_size=151936, hidden_size=2560)
    assert s.lm_head_params > 0
    assert s.tete_liee_bytes(128) == 0, "pas de copie liee a ajouter ici"
