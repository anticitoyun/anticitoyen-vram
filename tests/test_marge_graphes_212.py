"""Pièce 212 (25/09, ordre chef) : `warm_graphs` (engine/graphes.py:40 -> engine/graphs.py:1075-1086,
pool CUDA jamais libéré) grossit la VRAM APRÈS le chargement, hors de tout ce que `_reserve_prefill`
couvre. Mesuré sur 5 modèles (B=8, CTX=2048, `poste4-p212-25-09/mesure-warmgraphs.py`) : Qwen3.8-27B-
nvfp4 1 880 Mio, -unsloth-mixte-i8c 2 734 Mio, -attn-gdn-i8c 1 720 Mio dépassent la marge fixe
(1 536 Mio) ; gemma-4-31B-vision -868 (libère) et Coder-30B-A3B 266 restent dessous.

Dérivation structurelle (config.json + godets, sur le modèle de 172/201) tentée et ABANDONNÉE (voir
`loader.py`, commentaire de `_KV_MARGE_MIN_GDN`) : les trois Qwen3.8 partagent un `config.json`
identique mais divergent de plus de 25 % (mixte 2 734 contre i8c 1 720) -- aucune fonction de la seule
architecture ne peut rendre `E >= 2734` ET `E <= 1,25*1720 = 2150` à la fois. Repli (a), autorisé par
chef : constante nommée, conditionnée à la présence de couches à récurrence linéaire (lue au
manifeste, `_a_des_couches_lineaires`)."""
from acvram.engine.loader import _KV_MARGE_MIN, _KV_MARGE_MIN_GDN, _a_des_couches_lineaires, _marge_carte

_MESURES = {  # modele: cout Mio mesure (libre_apres_engine - libre_apres_graphes)
    "Qwen3.8-27B-nvfp4": 1880,
    "Qwen3.8-27B-unsloth-mixte-i8c": 2734,
    "Qwen3.8-27B-nvfp4-attn-gdn-i8c": 1720,
}

_MANIFEST_GDN = {"tensors": {"model.layers.0.linear_attn.qkv.weight": {"format": "int8"}}}
_MANIFEST_SANS_GDN = {"tensors": {"model.layers.0.self_attn.q_proj.weight": {"format": "nvfp4"}}}


def test_detection_couches_lineaires():
    assert _a_des_couches_lineaires(_MANIFEST_GDN)
    assert not _a_des_couches_lineaires(_MANIFEST_SANS_GDN)
    assert not _a_des_couches_lineaires(None) and not _a_des_couches_lineaires({})


def test_la_marge_gdn_couvre_les_trois_modeles_qui_depassaient():
    """`>= mesure` pour les trois -- pas de contrainte `<= 1,25x` PAR MODELE ICI (impossible, voir
    l'en-tete de ce fichier) : la constante couvre le pire cas avec une marge de sécurité globale."""
    marge = _marge_carte(0, manifest=_MANIFEST_GDN)
    for nom, cout_mio in _MESURES.items():
        assert marge >= cout_mio * 2**20, nom


def test_la_marge_gdn_ne_gaspille_pas_trop_le_kv():
    """La constante ne dépasse pas +25 % du pire mesuré (2 734 Mio) -- sur CE point, celui que la
    dérivation structurelle visait, la constante calibrée fait aussi bien."""
    marge = _marge_carte(0, manifest=_MANIFEST_GDN)
    assert marge <= 1.25 * 2734 * 2**20


def test_les_modeles_sans_gdn_gardent_l_ancienne_marge():
    assert _marge_carte(0, manifest=_MANIFEST_SANS_GDN) == _KV_MARGE_MIN
    assert _marge_carte(0, manifest=None) == _KV_MARGE_MIN


def test_retirer_le_terme_gdn_casse():
    """Cassant (demandé) : sans la détection GDN, les trois Qwen3.8 repassent sous la mesure."""
    marge_sans_terme = _KV_MARGE_MIN                              # comme si `_a_des_couches_lineaires` rendait toujours faux
    for nom, cout_mio in _MESURES.items():
        assert marge_sans_terme < cout_mio * 2**20, nom           # repasse sous la mesure : le terme comptait
    assert _KV_MARGE_MIN_GDN > _KV_MARGE_MIN
