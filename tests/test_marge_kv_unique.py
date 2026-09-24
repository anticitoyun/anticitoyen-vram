"""Pièce 156 (bead ba9) : une seule marge hors poids et hors KV, `loader._marge_carte`, lue par la borne du KV
(`_borner_kv_par_la_vram`) ET par l'exil des poids (`_reajuster_plan`). Bras cassants : remettre max(2 Gio, 7 %) en dur
dans `_reajuster_plan` rend rouge `test_l_exil_suit_la_marge_unique` (le montage au bord de 5 % exilerait) ; un site qui
ignore la constante rend rouge la seconde moitié de chaque test (la constante relevée ne change rien)."""
import torch

from acvram.engine import loader as LD
from acvram.memory.tiering import LayerPlacement, Plan, Tier

GIB = 2 ** 30


def _plan(capacite: int, n: int, mlp_gib: float, attn_gib: float) -> Plan:
    tier = Tier(name="gpu-test", kind="gpu", device_index=0, capacity=capacite,
                weight_format="nvfp4", kv_format="int8", read_bandwidth=1790.0, link_bandwidth=21.0)
    couches = [LayerPlacement(index=i, exec_device="gpu-test", attn_storage="gpu-test", mlp_storage="gpu-test",
                              fmt="nvfp4", attn_bytes=int(attn_gib * GIB), mlp_bytes=int(mlp_gib * GIB),
                              mlp_active_bytes=0, is_moe=False) for i in range(n)]
    return Plan(model="synthetique", tiers=[tier], layers=couches)


def test_la_marge_vaut_le_plus_grand_de_1_5_gio_et_5_pour_cent_plus_la_reserve():
    assert LD._marge_carte(20 * GIB) == int(1.5 * GIB)
    assert LD._marge_carte(40 * GIB) == int(0.05 * 40 * GIB)
    assert LD._marge_carte(40 * GIB, reserve=GIB) == int(0.05 * 40 * GIB) + GIB


def test_l_exil_suit_la_marge_unique(monkeypatch):
    capacite = 30 * GIB                          # marge 1,5 Gio : budget des poids 28,5 Gio
    plan = _plan(capacite, 20, 1.0, 0.4245)      # 28,49 Gio : sous le budget de 5 %, au-dessus de celui de 7 %
    LD._reajuster_plan(plan, {"tensors": {}}, top_k=8, reserve=0)
    assert sum(l.mlp_storage == "cpu" for l in plan.layers) == 0, "au bord de la marge unique, rien ne part"
    monkeypatch.setattr(LD, "_KV_MARGE_PART", 0.10)
    plan = _plan(capacite, 20, 1.0, 0.4245)
    LD._reajuster_plan(plan, {"tensors": {}}, top_k=8, reserve=0)
    assert sum(l.mlp_storage == "cpu" for l in plan.layers) >= 1, "l'exil doit lire la constante"


def test_la_borne_du_kv_suit_la_marge_unique(monkeypatch):
    libre, capacite = 28 * GIB, 32 * GIB
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (libre, capacite))

    def borne():
        plan = _plan(capacite, 10, 1.0, 0.5)     # 15 Gio de poids résidents
        plan.kv_budget = {"gpu-test": 64 * GIB}
        LD._borner_kv_par_la_vram(plan, {"tensors": {}}, lambda nom: 0)
        return plan.kv_budget["gpu-test"]

    assert borne() == libre - 15 * GIB - LD._marge_carte(capacite)
    monkeypatch.setattr(LD, "_KV_MARGE_PART", 0.10)
    assert borne() == libre - 15 * GIB - int(0.10 * capacite), "la borne doit lire la constante"


def test_la_borne_rend_son_deficit_brut(monkeypatch):
    """Poids au-delà de libre − marge : le budget s'arrête à 0, la borne rendue reste négative — c'est elle que
    `_borner_kv_avec_exil` exile d'un coup (sans elle : 4 tours de 325 Mio et refus du 70B, test_kv_plancher_exil)."""
    libre, capacite = 16 * GIB, 32 * GIB
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (libre, capacite))
    plan = _plan(capacite, 10, 1.0, 0.5)         # 15 Gio de poids : 16 − 15 − 1,6 < 0
    plan.kv_budget = {"gpu-test": GIB}
    bornes = LD._borner_kv_par_la_vram(plan, {"tensors": {}}, lambda nom: 0)
    assert plan.kv_budget["gpu-test"] == 0
    assert bornes["gpu-test"] == libre - 15 * GIB - LD._marge_carte(capacite) < 0
