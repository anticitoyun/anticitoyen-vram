"""a5v : MLP dense par tranches au-delà du seuil (Devstral 24B à 32 k : 10/40 MLP exilés par la réserve d un seul tenant).
Cassants sur l ancien code : `ModelSpec.mlp_prefill_plafond`, `loader._plafonner_mlp_prefill` et `attention.definir_seuil`
n y existent pas ; la réserve d un seul tenant y exilait."""
import types

import pytest
import torch

import acvram.engine.attention as A
from acvram.engine import loader as L
from acvram.engine.config import ModelSpec
from acvram.engine.layers import QuantLinear
from acvram.quant.formats import PlainTensor

H, I = 64, 96


def _lin(n, k):
    t = torch.randn(n, k, dtype=torch.bfloat16) * 0.05
    return QuantLinear(PlainTensor(t, tuple(t.shape), "bf16"), out_features=n, in_features=k)


def _spec(**k):
    base = dict(name="devstral-factice", architecture="MistralForCausalLM", num_layers=40, max_position_embeddings=131072,
                hidden_size=5120, intermediate_size=32768, num_attention_heads=32, num_key_value_heads=8, head_dim=128,
                num_hidden_layers=40, vocab_size=131072)
    base.update(k)
    return ModelSpec(**{a: b for a, b in base.items() if a in ModelSpec.__dataclass_fields__})


@pytest.mark.parametrize("seuil,appels", [(None, [30]), (30, [30]), (8, [8, 8, 8, 6])])
def test_mlp_tranches_seulement_au_dela_du_seuil(monkeypatch, seuil, appels):
    torch.manual_seed(0)
    mlp = A.MLP(_lin(I, H), _lin(I, H), _lin(H, I))
    x = torch.randn(30, H, dtype=torch.bfloat16)
    ref = mlp(x)
    vus = []
    vrai = A.MLP._forward_un
    monkeypatch.setattr(A.MLP, "_forward_un", lambda self, x: vus.append(x.shape[0]) or vrai(self, x))
    monkeypatch.setattr(A, "_MLP_MORCEAU", 8)
    A.definir_seuil(seuil)
    try:
        y = mlp(x)
    finally:
        A.definir_seuil(None)
    assert vus == appels and torch.allclose(y.float(), ref.float(), atol=1e-3)


def test_reserve_inchangee_sans_plafond_et_sous_le_plafond():
    s0, s1 = _spec(), _spec(mlp_prefill_plafond=20480)
    for T in (1, 4096, 20480):
        assert s1.activations_prefill_bytes(T) == s0.activations_prefill_bytes(T)
    mlp_jeton = 3 * 32768 * 2 + 32768 * 4
    ecart = s0.activations_prefill_bytes(34816) - s1.activations_prefill_bytes(34816)
    assert abs(ecart - (34816 - 20480) * mlp_jeton) < 34816                   # arrondi entier < T octets


def _sans_carte(monkeypatch):
    """Les tests à plan factice (`_plan` : couches sans octets ni budget KV) jugent le seul critère a5v « le plus grand
    plafond sans exil de plus ». Sous CUDA, `_plafonner_mlp_prefill` (kv31b) demande aussi si le plancher KV tient —
    `_reserve_prefill` et `_borner_kv_par_la_vram` lisent alors `attn_bytes`, `mlp_bytes`, les tiers — ce que le plan
    factice n'a pas : la suite GPU de la 0.7.17 cassait ici (AttributeError, chef 01/10). Le régime est donc posé par
    le test, pas par la machine ; le chemin « plancher KV sous VRAM simulée » a ses propres tests
    (test_reserve_prefill_tranches_kv31b.py, is_available et mem_get_info forcés)."""
    monkeypatch.setattr(A, "_MLP_MORCEAU", 4096)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _plan(n_exiles):
    return types.SimpleNamespace(layers=[types.SimpleNamespace(mlp_storage="cpu" if i < n_exiles else "cuda:0")
                                         for i in range(40)])


def test_plafond_le_plus_grand_sans_exil(monkeypatch):
    """Planificateur factice : un MLP exilé par 1 Gio de réserve au-delà de 10 Gio. Le plafond rendu est le plus grand
    multiple de 1 024 sans exil ; un plafond de 1 024 de plus en exilerait un."""
    _sans_carte(monkeypatch)
    spec, ctx = _spec(), 34816

    def planifier():
        r = spec.activations_prefill_bytes(ctx) / 2**30
        return _plan(max(0, int(r - 10.0) + (r > 10.0)))
    assert L._plafonner_mlp_prefill(spec, ctx, planifier(), planifier)
    c = spec.mlp_prefill_plafond
    assert c is not None and c % 1024 == 0 and L._mlp_exiles(planifier()) == 0
    spec.mlp_prefill_plafond = c + 1024
    assert L._mlp_exiles(planifier()) > 0


def test_plafond_jamais_pose_si_rien_n_est_exile_ni_pour_un_moe_ni_un_hybride(monkeypatch):
    _sans_carte(monkeypatch)
    for spec, plan in ((_spec(), _plan(0)), (_spec(num_experts=128, moe_intermediate_size=768), _plan(10)),
                       (_spec(layer_types=["linear_attention", "full_attention"]), _plan(10))):
        assert not L._plafonner_mlp_prefill(spec, 34816, plan, lambda: _plan(0))
        assert spec.mlp_prefill_plafond is None
    monkeypatch.setattr(A, "_MLP_MORCEAU", 0)                                  # témoin
    spec = _spec()
    assert not L._plafonner_mlp_prefill(spec, 34816, _plan(10), lambda: _plan(0)) and spec.mlp_prefill_plafond is None


def test_chauffe_dense_pose_le_seuil_mlp_au_dela_du_seul_tenant(converted, monkeypatch):
    """Modèle tiny DENSE : OOM simulé au-delà de 40 jetons d un seul tenant tant que le MLP n a pas de seuil ; la chauffe
    prouve 40, pose le seuil MLP à 40 et tient 64. Témoin ACVRAM_MLP_MORCEAU=0 : refus à 40 comme avant."""
    from acvram.engine.runner import ContexteNonTenu
    from tests.test_engine import _engine_cpu
    monkeypatch.delenv("ACVRAM_CHAUFFE_CTX", raising=False)
    eng = _engine_cpu(converted)
    assert eng._dense_pur()

    def faux(e):
        vrai = e.generate

        def gen(prompt_ids, params, images=None):
            if len(prompt_ids) + 2 > 40 and A._MLP_SEUIL is None:
                raise torch.OutOfMemoryError("CUDA out of memory (simulé)")
            return vrai(prompt_ids, params, images=images)
        monkeypatch.setattr(e, "generate", gen)
    faux(eng)
    monkeypatch.setattr(A, "_MLP_MORCEAU", 8)
    try:
        assert eng.chauffer_contexte(pas=8) == 64 and A._MLP_SEUIL == 40
        assert " tranches>40" in eng.regime_ligne() + " "
        monkeypatch.setattr(A, "_MLP_MORCEAU", 0)
        import acvram.engine.runner as _R
        monkeypatch.setattr(_R, "_MORCEAU_AU_DELA", 0)             # d19 : sans les morceaux d'attention non plus (témoin complet)
        eng2 = _engine_cpu(converted); faux(eng2)
        with pytest.raises(ContexteNonTenu, match="40 jetons tenus"):
            eng2.chauffer_contexte(pas=8)
        assert A._MLP_SEUIL is None
    finally:
        A.definir_seuil(None)
        import acvram.engine.gdn as gdn
        import acvram.engine.moe as moe
        gdn.definir_seuil(None); moe.definir_seuil(None)
