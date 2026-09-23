"""Les poids denses exilés passent par UN pool GPU partagé par appareil
(loader `_pool_dense`, layers `ExpertPool(dense=True)`), pas par deux copies
privées par poids (`StreamedWeight._ensure`) : sur Llama-3.3-70B-nvfp4, 52/80
couches exilées réclamaient ~37 Gio à la carte au premier préfill (2 × 18,6
Gio épinglés), OOM 448 Mio quel que soit le budget (Laure,
verdict-palier2-nemotron-17-09) — exiler une couche dense DOUBLAIT son
empreinte VRAM au lieu de la libérer.

À sec : la réserve du plan compte les tampons (`_DENSE_SLOTS` × plus grosse
couche) ; `QuantLinear.prefetch` précharge depuis un pool dense et jamais
depuis un pool d'experts. Sur carte : N poids de même forme dans un pool dense
occupent ≤ `_DENSE_SLOTS` tampons (pas 2 × N)."""
import pytest
import torch

from acvram.engine import layers as L
from acvram.engine import loader as LD
from acvram.engine.config import ModelSpec
from acvram.memory.tiering import LayerPlacement, Plan, Tier

GIB = 2 ** 30


def _spec():
    return ModelSpec(name="l70", architecture="LlamaForCausalLM", hidden_size=8192, intermediate_size=28672,
                     num_layers=80, num_attention_heads=64, num_key_value_heads=8, vocab_size=128256,
                     max_position_embeddings=131072, head_dim=128)


def test_la_reserve_compte_les_tampons_du_streaming_dense():
    plan = Plan(model="x", tiers=[Tier(name="gpu-test", kind="gpu", device_index=0, capacity=32 * GIB,
                                       weight_format="nvfp4", kv_format="int8", read_bandwidth=1790.0, link_bandwidth=21.0)],
                layers=[LayerPlacement(index=i, exec_device="gpu-test", attn_storage="gpu-test", mlp_storage="gpu-test",
                                       fmt="nvfp4", attn_bytes=int(0.07 * GIB), mlp_bytes=int(0.35 * GIB),
                                       mlp_active_bytes=0, is_moe=False) for i in range(80)])
    sans = LD._reserve_prefill(_spec(), 2048, {})
    avec = LD._reserve_prefill(_spec(), 2048, {}, plan)
    assert avec - sans == LD._DENSE_SLOTS * (int(0.07 * GIB) + int(0.35 * GIB))
    assert 1.5 * GIB < avec < 4 * GIB, avec / GIB


class _Streamed:
    def __init__(self, pool):
        self.pool = pool
        self.appels = 0

    def prefetch(self):
        self.appels += 1
        return 7


def test_prefetch_depuis_un_pool_dense_jamais_depuis_un_pool_d_experts():
    q = L.QuantLinear.__new__(L.QuantLinear)
    q._pending_slot = None
    for pool, attendu in ((None, 1), (type("PoolDense", (), {"dense": True})(), 1),
                          (type("PoolExperts", (), {"dense": False})(), 0), (type("PoolAncien", (), {})(), 0)):
        q.streamed = _Streamed(pool)
        q._pending_slot = None
        q.prefetch()
        assert q.streamed.appels == attendu and (q._pending_slot == 7) == bool(attendu), (pool, attendu)


def test_pas_de_pool_dense_sans_carte():
    """Le pool ne se construit que sur cuda (sur cpu l'ancien chemin reste) et
    ses emplacements suffisent au recouvrement attn + mlp d'une couche."""
    assert LD._DENSE_SLOTS >= 4
    assert L.ExpertPool(torch.device("cpu"), 2, dense=True).dense is True
    assert L.ExpertPool(torch.device("cpu"), 2).dense is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")
def test_sur_carte_n_poids_partagent_les_tampons_du_pool():
    dev = torch.device("cuda")
    pool = L.ExpertPool(dev, LD._DENSE_SLOTS, dense=True)
    poids = [L.StreamedWeight({"w": torch.randn(1024, 1024, dtype=torch.bfloat16)}, dev, pool=pool)
             for _ in range(12)]
    slots = []
    for i, p in enumerate(poids):
        slots.append(p.prefetch())
        if len(slots) >= 2:                     # comme une couche qui rend son emplacement après calcul
            p_prec = poids[i - 1]
            p_prec.wait(slots[-2]); p_prec.release(slots[-2])
    assert pool.nbytes <= LD._DENSE_SLOTS * 1024 * 1024 * 2, "le pool a alloué plus que ses emplacements"
    avant = torch.cuda.memory_allocated(dev)
    prive = L.StreamedWeight({"w": torch.randn(1024, 1024, dtype=torch.bfloat16)}, dev)
    prive.prefetch(); torch.cuda.synchronize(dev)
    assert torch.cuda.memory_allocated(dev) - avant >= 2 * 1024 * 1024 * 2, "sans pool : deux copies privées (le bras qui doit différer)"


class _EvenementFactice:
    def record(self, *a): pass
    def wait(self, *a): pass


def test_le_pool_prend_un_emplacement_libre_pas_le_tour_de_role(monkeypatch):
    """Exil total (36/36, Laure 02b316d) : « ExpertPool saturé : 4 emplacements »
    alors que deux étaient rendus. Séquence réelle du forward (model.py :
    précharge i+1 puis exécute i) quand la couche 0 est elle-même en flux :
    précharge(1) prend s0,s1 ; la couche 0 copie à la demande s2 et s3 et les
    rend ; précharge(2) tombait sur s0 (tour de rôle), en vol."""
    monkeypatch.setattr(torch.cuda, "Event", _EvenementFactice)
    monkeypatch.setattr(L.ExpertPool, "SYNC", True)
    pool = L.ExpertPool(torch.device("cpu"), 4, dense=True)
    plat = torch.zeros(64, dtype=torch.uint8)
    dec = {"w": (0, (64,), torch.uint8, 64)}
    s0, s1 = pool.copier(plat, dec), pool.copier(plat, dec)        # précharge(1)
    s2 = pool.copier(plat, dec); pool.liberer(s2)                   # couche 0, q à la demande
    s3 = pool.copier(plat, dec); pool.liberer(s3)                   # couche 0, o à la demande
    a, b = pool.copier(plat, dec), pool.copier(plat, dec)           # précharge(2) : ne doit pas saturer
    assert {a, b} == {s2, s3} and len({s0, s1, a, b}) == 4
    with pytest.raises(RuntimeError, match="ExpertPool sature"):    # tout en vol : le refus reste
        pool.copier(plat, dec)
