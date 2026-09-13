"""`_reajuster_plan` pose `experts_residents` PAR COUCHE (bead pds, point 1,
poste7 §4) au lieu d'exiler la couche entière, quand assez d'experts restent
résidents (`>= top_k`) — sinon le comportement d'avant ce bead (exil complet).

Aucune carte : le nom du tier n'est délibérément PAS `cuda:N`, pour que
`torch.cuda.mem_get_info` échoue dans le `try/except` de `_reajuster_plan` et
laisse `capacite` telle que le test l'a posée — sinon la capacité RÉELLE de
la carte du moment s'y substituerait, rendant le test dépendant de l'état
ambiant de la machine.
"""
from __future__ import annotations

from acvram.engine.loader import _compter_experts_manifest, _reajuster_plan
from acvram.memory.tiering import LayerPlacement, Plan, Tier

_GIB = 2 ** 30
N_EXPERTS = 16
PAR_EXPERT_GIB = 1


def _manifest(n_experts: int = N_EXPERTS, par_expert_gib: int = PAR_EXPERT_GIB) -> dict:
    tenseurs = {}
    for e in range(n_experts):
        tenseurs[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = {
            "shape": [par_expert_gib * _GIB], "bpw": 8.0, "format": "int8"}
    return {"tensors": tenseurs}


def _plan(n_experts: int = N_EXPERTS,
         par_expert_gib: int = PAR_EXPERT_GIB) -> Plan:
    mlp_bytes = n_experts * par_expert_gib * _GIB
    tier = Tier(name="gpu-test", kind="gpu", device_index=0,
               capacity=0,             # posé par le test, une fois par cas
               weight_format="nvfp4", kv_format="fp8",
               read_bandwidth=1050.0, link_bandwidth=21.0)
    couche = LayerPlacement(
        index=0, exec_device="gpu-test", attn_storage="gpu-test",
        mlp_storage="gpu-test", fmt="nvfp4",
        attn_bytes=0, mlp_bytes=mlp_bytes, mlp_active_bytes=0, is_moe=True)
    return Plan(model="synthetique", tiers=[tier], layers=[couche])


def test_compter_experts_manifest():
    assert _compter_experts_manifest(_manifest()) == {0: N_EXPERTS}


def test_compter_experts_manifest_absent_pour_couche_dense():
    assert _compter_experts_manifest({"tensors": {
        "model.layers.0.mlp.gate_proj.weight": {"shape": [1], "bpw": 16.0}}}) == {}


def test_manque_modeste_pose_experts_residents():
    manifest = _manifest()
    plan = _plan()
    # 16 Gio d'experts ; capacité posée pour forcer un manque modeste.
    plan.tiers[0].capacity = 12 * _GIB
    _reajuster_plan(plan, manifest, top_k=8)
    couche = plan.layers[0]
    assert couche.mlp_storage == "gpu-test"        # jamais exilée entièrement
    assert couche.experts_residents is not None
    assert couche.experts_residents >= 8            # au-dessus de top_k
    assert couche.experts_residents < N_EXPERTS      # une vraie réduction


def test_manque_severe_retombe_sur_l_exil_complet():
    manifest = _manifest()
    plan = _plan()
    # Capacité si petite que résider ne serait-ce que top_k experts ne
    # tiendrait pas : le placement par expert n'a plus de sens (docstring).
    plan.tiers[0].capacity = 2 * _GIB
    _reajuster_plan(plan, manifest, top_k=8)
    couche = plan.layers[0]
    assert couche.mlp_storage == "cpu"
    assert couche.experts_residents is None


def test_peu_d_experts_saute_directement_le_placement_par_expert():
    # 4 experts < top_k=8 : le geste par expert ne s'engage jamais, même pour
    # un manque modeste — la docstring de `_reajuster_plan` l'exige.
    manifest = _manifest(n_experts=4)
    plan = _plan(n_experts=4)
    plan.tiers[0].capacity = int(3.5 * _GIB)        # manque léger (4 Gio total)
    _reajuster_plan(plan, manifest, top_k=8)
    couche = plan.layers[0]
    assert couche.mlp_storage == "cpu"
    assert couche.experts_residents is None


def test_rien_a_reajuster_ne_touche_a_rien():
    manifest = _manifest()
    plan = _plan()
    plan.tiers[0].capacity = 32 * _GIB              # largement assez
    _reajuster_plan(plan, manifest, top_k=8)
    couche = plan.layers[0]
    assert couche.mlp_storage == "gpu-test"
    assert couche.experts_residents is None
