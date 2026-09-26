"""Pièce 201 : les poids chargés APRÈS la borne du KV (tour de vision, têtes MTP, tout bloc hors couches/embed/tête)
entrent dans `libre − poids − marge`. Sur Qwen3.8-27B-nvfp4-attn-gdn-i8c, 1,19 Gio invisibles au planificateur :
0/64 couches exilées, 148 Mio libres après chargement, OOM au premier pas. Un modèle qui ne tient qu'en les oubliant
doit être exilé ou refusé, jamais chargé."""
import pytest
import torch

from acvram.engine import loader as LD
from acvram.memory.tiering import LayerPlacement, Plan, Tier

G, M = 2**30, 2**20


def _t(shape, fmt="bf16"):
    return {"format": fmt, "shape": list(shape)}


def test_tout_bloc_hors_couches_est_compte(monkeypatch):
    """Par EXCLUSION, pas par une liste de préfixes : un bloc au nom jamais vu est compté d'office."""
    monkeypatch.delenv("ACVRAM_MTP", raising=False)
    man = {"vision": "oui", "tensors": {
        "model.layers.0.self_attn.q_proj.weight": _t((1024, 1024)),
        "model.embed_tokens.weight": _t((4096, 1024)), "lm_head.weight": _t((4096, 1024)),
        "model.visual.blocks.0.attn.qkv.weight": _t((1024, 1024)),       # 2 Mio
        "mtp.layers.0.mlp.gate_proj.weight": _t((512, 1024)),            # 1 Mio
        "zzz.bloc_inconnu.weight": _t((256, 1024))}}                     # 0,5 Mio
    assert LD._octets_annexes(man) == 2 * M + M + M // 2


def test_les_chargeurs_qui_refusent_retirent_leur_bloc(monkeypatch):
    """Retirés seulement sur la déclaration que leur chargeur lit : manifeste ``vision: non``, ``ACVRAM_MTP=non``."""
    man = {"vision": "non", "tensors": {"model.visual.x.weight": _t((1024, 1024)),
                                        "mtp.fc.weight": _t((512, 1024)), "model.mtp.0.y.weight": _t((512, 1024))}}
    monkeypatch.delenv("ACVRAM_MTP", raising=False)
    assert LD._octets_annexes(man) == 2 * M                                # MTP seule, dans ses deux conventions
    monkeypatch.setenv("ACVRAM_MTP", "non")
    assert LD._octets_annexes(man) == 0
    man["vision"] = "oui"
    assert LD._octets_annexes(man) == 2 * M


def _plan(cap, n_couches, octets_couche):
    p = Plan(model="synthetique", tiers=[
        Tier(name="cuda:0", kind="gpu", device_index=0, capacity=cap, weight_format="nvfp4", kv_format="int8",
             read_bandwidth=1790.0, link_bandwidth=21.0),
        Tier(name="cpu", kind="host", device_index=-1, capacity=128 * G, weight_format="nvfp4", kv_format="int8",
             read_bandwidth=60.0, link_bandwidth=21.0)],
        layers=[LayerPlacement(index=i, exec_device="cuda:0", attn_storage="cuda:0", mlp_storage="cuda:0", fmt="nvfp4",
                               attn_bytes=octets_couche // 2, mlp_bytes=octets_couche // 2,
                               mlp_active_bytes=octets_couche // 2) for i in range(n_couches)])
    p.embed_device, p.lm_head_device, p.kv_budget = "cuda:0", "cuda:0", {"cuda:0": 256 * M}
    return p


@pytest.mark.parametrize("vision", ["non", "oui"])
def test_un_modele_qui_ne_tient_qu_en_oubliant_la_tour_est_exile(monkeypatch, vision):
    """Carte de 24 Gio, 19 Gio libres ; 16 couches de 1 Gio ; plancher KV 256 Mio ; marge 1,5 Gio ; tour de vision
    de 2 Gio chargée après la borne. Sans la tour : 19 − 16 − 1,5 = 1,5 ≥ 0,25, tout tient, rien n'est exilé.
    Avec : 19 − 16 − 2 − 1,5 = −0,5, il manque 0,75 Gio, des couches DOIVENT partir à l'hôte. Le témoin « non » prouve que seule la tour
    fait la différence ; le bras cassant (tour non comptée) laisse « oui » sans exil."""
    monkeypatch.delenv("ACVRAM_MTP", raising=False)
    cap, libre = 24 * G, 19 * G
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (libre, cap))
    n, couche = 16, G
    tenseurs = {f"model.layers.{i}.{r}.weight": _t((couche // 2 // 2 // 1024, 1024)) for i in range(n)
                for r in ("self_attn.o_proj", "mlp.down_proj")}
    tenseurs["model.visual.blocks.0.mlp.weight"] = _t((G // 1024, 1024))    # 2 Gio en bf16
    man = {"vision": vision, "tensors": tenseurs, "model": {}}
    p = _plan(cap, n, couche)
    spec = type("S", (), {"couche_a_kv": staticmethod(lambda i: True), "num_experts_per_tok": None,
                          "kv_bytes_per_token": staticmethod(lambda: 256 * M // 1024)})()
    try:
        LD._borner_kv_avec_exil(p, man, lambda nom: nom, spec, max_model_len=1024)
    except RuntimeError as e:                                  # refus nommé : acceptable, jamais un chargement muet
        assert vision == "oui" and "refus" in str(e)
        return
    exilees = sum(1 for l in p.layers if l.attn_storage == "cpu" or l.mlp_storage == "cpu")
    if vision == "non":
        assert exilees == 0, f"témoin : sans tour, rien ne devait partir ({exilees} couches exilées)"
    else:
        assert exilees > 0, "tour de vision chargée après la borne et non comptée : modèle chargé sans exil"
