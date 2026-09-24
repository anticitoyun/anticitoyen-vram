"""Le budget KV ne descend jamais sous le plancher d'UNE séquence de
``max_model_len`` jetons : la borne par la VRAM libre et la boucle d'exil se
répondent (`loader._borner_kv_avec_exil`), sinon le chargement refuse avec les
chiffres ; et une requête qui ne tiendra jamais dans les blocs est refusée
par l'ordonnanceur, pas gardée en file (`runner._admit`).

Llama-3.3-70B-nvfp4 (poste3, 17/09, trois essais lus comme un blocage CUDA) :
« budget KV de cuda:0 borné par la VRAM libre : 0,32 → 0,00 Gio » — un bloc
de 16 jetons pour une invite de 256, jamais admise, moteur à vide 33 min."""
import pytest
import torch

from acvram.engine import loader as LD
from acvram.engine.runner import Engine, SamplingParams
from acvram.engine.loader import load_model
from acvram.memory.tiering import LayerPlacement, Plan, Tier

GIB = 2 ** 30


def _spec_70b():
    from acvram.engine.config import ModelSpec
    return ModelSpec(name="l70", architecture="LlamaForCausalLM", hidden_size=8192, intermediate_size=28672,
                     num_layers=80, num_attention_heads=64, num_key_value_heads=8, vocab_size=128256,
                     max_position_embeddings=131072, head_dim=128)


def _plan(n: int, capacite: int, kv: int) -> Plan:
    tier = Tier(name="gpu-test", kind="gpu", device_index=0, capacity=capacite,
                weight_format="nvfp4", kv_format="int8", read_bandwidth=1790.0, link_bandwidth=21.0)
    couches = [LayerPlacement(index=i, exec_device="gpu-test", attn_storage="gpu-test", mlp_storage="gpu-test",
                              fmt="nvfp4", attn_bytes=int(0.07 * GIB), mlp_bytes=int(0.35 * GIB),
                              mlp_active_bytes=0, is_moe=False) for i in range(n)]
    p = Plan(model="synthetique", tiers=[tier], layers=couches)
    p.kv_budget = {"gpu-test": kv}
    return p


def _borne_factice(libre: int):
    """Rejoue `_borner_kv_par_la_vram` à sec : budget = libre − poids résidents − marge."""
    def borner(plan, manifest, dev, reserve=0, embed_charge=False):
        poids = sum(l.attn_bytes for l in plan.layers if l.attn_storage == "gpu-test") \
            + sum(l.mlp_bytes for l in plan.layers if l.mlp_storage == "gpu-test")
        marge = LD._KV_MARGE_MIN + reserve
        borne = libre - poids - marge
        if borne < plan.kv_budget["gpu-test"]:
            plan.kv_budget["gpu-test"] = max(0, borne)
        return {"gpu-test": borne}                      # contrat de la vraie borne (pièce 156)
    return borner


def test_le_plancher_est_une_sequence_de_max_model_len():
    spec = _spec_70b()
    plan = _plan(80, 30 * GIB, 0)
    plan.kv_bytes_per_token = spec.kv_bytes_per_token()
    assert plan.kv_bytes_per_token == 166400, plan.kv_bytes_per_token   # int8 + échelles : 8,125 bits × 80 × 2 × 8 × 128
    plancher = LD._kv_plancher(plan, spec, 2048, "gpu-test")
    assert plancher == plan.kv_bytes_per_token * 2048, "exactement la cible du planificateur pour une séquence"
    assert 0.3 * GIB < plancher < 0.35 * GIB


def test_un_plan_dont_la_cible_est_sous_le_plancher_est_releve_puis_borne(monkeypatch):
    """Essai a803254 (poste3) : cible 0,32 Gio, plancher 3 Mio plus haut, la
    borne ne fait que réduire → quatre tours « 3 Mio manquants » puis refus.
    La cible est d'abord relevée au plancher ; la borne fait le reste."""
    spec = _spec_70b()
    plan = _plan(80, 30 * GIB, 100 * 2 ** 20)                 # cible bien sous le plancher
    plan.kv_bytes_per_token = spec.kv_bytes_per_token()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(LD, "_borner_kv_par_la_vram", _borne_factice(28 * GIB))
    LD._reajuster_plan(plan, {"tensors": {}}, top_k=8, reserve=3 * GIB)
    LD._borner_kv_avec_exil(plan, {"tensors": {}}, None, spec, 2048, reserve=3 * GIB)
    assert plan.kv_budget["gpu-test"] == LD._kv_plancher(plan, spec, 2048, "gpu-test")


def test_sous_le_plancher_on_exile_encore_jusqu_a_ce_que_le_kv_tienne(monkeypatch):
    spec = _spec_70b()
    reserve = 3 * GIB
    # 80 couches × 0,42 = 33,6 Gio de poids, capacité 30 : l'exil descend
    # au bord du budget ; la « VRAM libre » vue ensuite est plus petite de
    # 2 Gio (contexte CUDA, JIT) — le montage du 70B : la borne mettait le KV à 0
    plan = _plan(80, 30 * GIB, int(0.32 * GIB))
    plan.kv_bytes_per_token = spec.kv_bytes_per_token()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(LD, "_borner_kv_par_la_vram", _borne_factice(28 * GIB))
    LD._reajuster_plan(plan, {"tensors": {}}, top_k=8, reserve=reserve)
    avant = sum(l.mlp_storage == "cpu" for l in plan.layers)
    temoin = _plan(80, 30 * GIB, int(0.32 * GIB)); temoin.kv_bytes_per_token = plan.kv_bytes_per_token
    LD._reajuster_plan(temoin, {"tensors": {}}, top_k=8, reserve=reserve)
    _borne_factice(28 * GIB)(temoin, {}, None, reserve=reserve)
    assert temoin.kv_budget["gpu-test"] == 0, "le montage doit reproduire le KV à zéro (le bras qui doit casser)"
    LD._borner_kv_avec_exil(plan, {"tensors": {}}, None, spec, 2048, reserve=reserve)
    apres = sum(l.mlp_storage == "cpu" for l in plan.layers)
    assert apres > avant, (avant, apres)
    assert plan.kv_budget["gpu-test"] >= LD._kv_plancher(plan, spec, 2048, "gpu-test")


def test_si_rien_ne_suffit_le_chargement_refuse_avec_les_chiffres(monkeypatch):
    spec = _spec_70b()
    plan = _plan(80, 30 * GIB, int(0.32 * GIB))
    plan.kv_bytes_per_token = spec.kv_bytes_per_token()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(LD, "_borner_kv_par_la_vram", _borne_factice(6 * GIB))   # attention seule = 5,6 Gio
    with pytest.raises(RuntimeError, match="refus : budget KV insuffisant"):
        LD._borner_kv_avec_exil(plan, {"tensors": {}}, None, spec, 2048, reserve=3 * GIB)


def test_une_requete_qui_ne_tiendra_jamais_est_refusee_pas_mise_en_attente(converted):
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    engine = Engine(loaded, None, max_batch_size=2, max_model_len=256)
    engine.allocator.num_blocks = 1                      # le 70B : un bloc pour tout
    engine.allocator._free = engine.allocator._free[:1]
    params = SamplingParams(temperature=0.0, max_tokens=4)
    sorties = []
    engine.add_request(list(range(40)), params, request_id="trop-long")
    for _ in range(3):
        sorties += engine.step()
    assert sorties and sorties[0].finished and sorties[0].finish_reason == "refus", sorties
    assert not engine.waiting and not engine.running
    engine.allocator.num_blocks = 4; engine.allocator._free = list(range(4))
    sorties = [o for o in engine.generate([1, 2, 3], params)]        # une courte passe encore
    assert sorties[-1].finish_reason == "length"
