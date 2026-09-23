"""Le budget d'exil compte les activations du plus grand préfill
(verdict-palier1-bloc6-17-09, Laure : Llama-3.3-70B-nvfp4 chargé DÉGRADÉ à
32/80 couches exilées — la boucle `_reajuster_plan` (engine/loader.py,
`while utilise() > capacite - marge`) ne comptait que résidents + KV + marge —
puis OOM de 448 Mio au tout premier préfill : la matrice déquantifiée en bf16
du chemin W4A16, 28672 × 8192 × 2 = 470 Mio, plus les activations).

Deux contrôles : (1) l'estimation `ModelSpec.activations_prefill_bytes` donne
l'ordre de grandeur du 70B (≈ 1,15 Gio à 2 048 jetons, dont 0,44 Gio de
déquant) et croît avec la longueur ; (2) sur un plan synthétique posé
EXACTEMENT sous `capacité − marge`, le réajustement sans réserve n'exile rien
(l'ancien comportement, celui qui menait à l'OOM) et le réajustement avec la
réserve exile — le test casse si le terme est retiré. Tier hors `cuda:N`
pour que `mem_get_info` échoue et laisse la capacité posée."""
from acvram.engine.config import ModelSpec
from acvram.engine.loader import _reajuster_plan, _reserve_prefill
from acvram.memory.tiering import LayerPlacement, Plan, Tier

GIB = 2 ** 30


def _spec_70b():
    return ModelSpec(name="l70", architecture="LlamaForCausalLM", hidden_size=8192, intermediate_size=28672,
                     num_layers=80, num_attention_heads=64, num_key_value_heads=8, vocab_size=128256,
                     max_position_embeddings=131072, head_dim=128)


def test_l_estimation_donne_l_ordre_de_grandeur_du_70b_et_croit_avec_la_longueur():
    s = _spec_70b()
    a2k, a8k = s.activations_prefill_bytes(2048), s.activations_prefill_bytes(8192)
    assert 0.9 * GIB < a2k < 1.5 * GIB, a2k / GIB
    assert a8k > 2.5 * a2k
    assert s.activations_prefill_bytes(1) >= 28672 * 8192 * 2, "la déquant bf16 de la plus grosse projection y est"
    assert _reserve_prefill(s, 2048, {}) == a2k and _reserve_prefill(None, 2048, {}) == 0
    assert _reserve_prefill(s, None, {"plan": {"kv_max_tokens": 8192}}) == a8k
    # la capacité KV planifiée n'est pas une longueur d'invite (Qwen3.8 : 56 401
    # jetons → 13 Gio réservés, 19/64 couches exilées pour rien)
    assert _reserve_prefill(s, None, {"plan": {"kv_max_tokens": 56401}}) == a8k
    assert _reserve_prefill(s, 4096, {"plan": {"kv_max_tokens": 56401}}) == s.activations_prefill_bytes(4096)


def _plan(n_couches: int, mlp_gib: float, attn_gib: float, capacite: int) -> Plan:
    tier = Tier(name="gpu-test", kind="gpu", device_index=0, capacity=capacite,
                weight_format="nvfp4", kv_format="int8", read_bandwidth=1790.0, link_bandwidth=21.0)
    couches = [LayerPlacement(index=i, exec_device="gpu-test", attn_storage="gpu-test", mlp_storage="gpu-test",
                              fmt="nvfp4", attn_bytes=int(attn_gib * GIB), mlp_bytes=int(mlp_gib * GIB),
                              mlp_active_bytes=0, is_moe=False) for i in range(n_couches)]
    return Plan(model="synthetique", tiers=[tier], layers=couches)


def test_sans_la_reserve_rien_n_est_exile_avec_elle_le_prefill_a_sa_place():
    """Poids résidents = capacité − marge exactement (marge = max(2 Gio, 7 %)) :
    l'ancien budget est satisfait, le préfill n'a plus un octet."""
    capacite = 30 * GIB
    marge = max(2 * GIB, int(0.07 * capacite))
    n, mlp, attn = 20, 1.0, 0.395
    poids = n * (mlp + attn) * GIB
    assert 0 <= (capacite - marge) - poids < 0.05 * GIB, "le montage doit poser les poids au bord du budget, dessous"
    plan = _plan(n, mlp, attn, capacite)
    _reajuster_plan(plan, {"tensors": {}}, top_k=8, reserve=0)
    assert sum(l.mlp_storage == "cpu" for l in plan.layers) == 0, "sans réserve, rien n'est exilé (l'OOM au préfill)"
    reserve = _spec_70b().activations_prefill_bytes(2048)
    plan = _plan(n, mlp, attn, capacite)
    _reajuster_plan(plan, {"tensors": {}}, top_k=8, reserve=reserve)
    exilees = sum(l.mlp_storage == "cpu" for l in plan.layers)
    assert exilees >= 1, "la réserve de préfill doit faire descendre au moins une couche"
    assert exilees <= 3, exilees                  # 1,15 Gio ≈ une à deux couches de 1 Gio, pas plus
