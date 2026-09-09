"""La table de plongements va sur la carte quand la place existe.

Defaut du 9/09 : `plan.embed_device = "cpu" if host_tier else fastest` testait
l'EXISTENCE de l'etage hote, or il existe des que `allow_host_tier` est vrai —
son defaut — meme quand rien n'y est exile. Un modele qui tient entierement sur
la carte se voyait donc placer sa table cote hote, ce qui coute deux traversees
PCIe par jeton et cree une asymetrie face a llama.cpp en `-ngl 999`.

Releve : Qwen2.5-Coder-14B-bf16-pur, 0 couche exilee, `embed_device: cpu`.
"""
import inspect
from acvram.memory import tiering


def test_la_condition_porte_sur_la_place_pas_sur_l_etage():
    src = inspect.getsource(tiering.plan_placement)
    assert "remaining[fastest] > embed_bytes" in src, (
        "la condition doit tester la PLACE disponible")
    assert '"cpu" if host_tier else fastest' not in src, (
        "l'ancienne condition, qui testait l'existence de l'etage, est restee")


def test_symetrie_avec_lm_head():
    """lm_head et embed decidaient la meme chose par deux logiques opposees.

    Deux champs voisins, deux defauts contraires : celui qui verifiait l'un et
    le trouvait juste n'avait aucune raison de soupconner l'autre.
    """
    src = inspect.getsource(tiering.plan_placement)
    assert src.count("if gpu_tiers and remaining[fastest] > ") == 2, (
        "lm_head_device et embed_device doivent tester la place de la meme facon")


def test_le_defaut_de_la_dataclasse_reste_prudent():
    """Le defaut de `Plan` reste 'cpu' : c'est le placement CALCULE qui change.

    Un plan construit a la main sans passer par le placeur ne doit pas supposer
    une carte qui n'existe peut-etre pas.
    """
    assert tiering.Plan(model="x").embed_device == "cpu"
