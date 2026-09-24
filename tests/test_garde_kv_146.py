"""Pièce 146, décisions de chef : (a) la garde Marlin compare la capacité KV de B à celle du chemin par défaut au même
chargement, pas à la demande — gemma4 31B servait 3 824 jetons en A comme en B, B seul était refusé ; (b) une capacité
sous la demande est NOMMÉE sur la ligne de régime (kv_sous_demande=capacité/demande), sans refus pour le défaut."""
import pytest
import torch

from acvram.engine.loader import _verifier_memoire_marlin

BILAN = {"en_flux": 0, "doubles": 0, "octets_doubles": 0}


def test_garde_contre_la_capacite_du_defaut(monkeypatch):
    monkeypatch.delenv("ACVRAM_PROJ_MARLIN_CAPACITE", raising=False)
    _verifier_memoire_marlin([], {"cuda:0": 239}, BILAN, demande=20480, reference={"cuda:0": 239})   # = A : accepté
    with pytest.raises(RuntimeError, match="chemin par défaut"):
        _verifier_memoire_marlin([], {"cuda:0": 238}, BILAN, demande=20480, reference={"cuda:0": 239})
    with pytest.raises(RuntimeError, match="20480"):                                             # sans référence
        _verifier_memoire_marlin([], {"cuda:0": 239}, BILAN, demande=20480)
    monkeypatch.setenv("ACVRAM_PROJ_MARLIN_CAPACITE", "8192")                                     # imposée : prime
    with pytest.raises(RuntimeError, match="8192"):
        _verifier_memoire_marlin([], {"cuda:0": 239}, BILAN, demande=20480, reference={"cuda:0": 239})


@pytest.fixture(scope="module")
def moteur(converted):
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu", max_concurrent_seqs=4)
    loaded.plan.kv_planned_seqs = 4
    return Engine(loaded, None, max_batch_size=4, max_model_len=256, enable_cuda_graphs=False)


def test_regime_nomme_la_capacite_sous_la_demande(moteur, monkeypatch):
    monkeypatch.setattr(moteur.allocator, "num_blocks", 2)                  # 32 jetons pour 4 × 256
    assert "kv_sous_demande=32/1024" in moteur.regime_ligne()
    monkeypatch.setattr(moteur.allocator, "num_blocks", 64)                 # 1 024 : la demande tient
    assert "kv_sous_demande" not in moteur.regime_ligne()


def test_embed_deja_charge_compte_une_fois(monkeypatch):
    """Pièce 146 (i) : load_model borne le KV APRÈS avoir mis embed_tokens sur la carte — `libre` l'a déjà soustrait.
    Budget attendu = libre − marge (aucun autre poids ici) ; l'embed recompté le retirait une seconde fois."""
    from acvram.engine import loader as LD
    from acvram.memory.tiering import Plan, Tier
    G = 2**30
    libre, cap = 10 * G, 32 * G
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (libre, cap))
    man = {"tensors": {"model.embed_tokens.weight": {"format": "bf16", "shape": [262144, 5376]}}, "model": {}}

    def plan():
        p = Plan(model="synthetique", tiers=[Tier(name="cuda:0", kind="gpu", device_index=0, capacity=cap,
                                                    weight_format="nvfp4", kv_format="int8", read_bandwidth=1790.0,
                                                    link_bandwidth=21.0)], layers=[])
        p.embed_device, p.lm_head_device, p.kv_budget = "cuda:0", "cpu", {"cuda:0": 20 * G}
        return p

    marge = max(LD._KV_MARGE_MIN, int(LD._KV_MARGE_PART * cap))
    p = plan()
    LD._borner_kv_par_la_vram(p, man, lambda n: n, reserve=0, embed_charge=True)
    assert p.kv_budget["cuda:0"] == libre - marge, "embed déjà dans libre : il ne doit pas être retiré une 2e fois"
    p = plan()
    LD._borner_kv_par_la_vram(p, man, lambda n: n, reserve=0)             # embed PAS encore chargé : compté
    assert p.kv_budget["cuda:0"] == libre - marge - 262144 * 5376 * 2
