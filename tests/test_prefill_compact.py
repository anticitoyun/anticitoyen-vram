"""C15-prefill (revue/chantier-c15-prefill-20-09) : la glue du préfill eager sous
`ACVRAM_PREFILL_COMPACT=1`, fusion par fusion, AU BIT contre le témoin (0, défaut).
Chaque test : le chemin compact rend les MÊMES octets que le chemin d'avant, et un
témoin cassant montre que la comparaison peut rendre « faux » (REGLES § 7)."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

from acvram import kernels, regime


def _var(nom):
    return next(v for v in regime.VARIABLES if v.nom == nom)


# --- régime ----------------------------------------------------------------

def test_le_defaut_est_le_temoin():
    """Défaut 0 tant que le scellé n'est pas mesuré sur carte (poste7, C15-prefill)."""
    assert _var("PREFILL_COMPACT").defaut == "0" and _var("PREFILL_COMPACT").torch == "0"
    assert _var("PREFILL_COMPACT_ITEMS").defaut == ""


def test_le_module_lit_le_defaut_sans_variable():
    env = {k: v for k, v in os.environ.items() if not k.startswith("ACVRAM_PREFILL_COMPACT")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    out = subprocess.run([sys.executable, "-c", "from acvram import kernels; print(kernels._PREFILL_COMPACT)"],
                         env=env, capture_output=True, text=True, timeout=120)
    assert out.stdout.split() == ["0"], out.stdout + out.stderr[-500:]


def test_la_ligne_de_regime_nomme_la_glue_du_prefill(monkeypatch):
    """Défaut compris : un chiffre de préfill sans cette étiquette ne dit pas son chemin."""
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT", 0)
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT_ITEMS", "")
    assert regime.prefill_glue_texte() == "prefill_glue=temoin"
    assert "prefill_glue=temoin" in regime.regime_ligne()
    assert not kernels.prefill_compact() and not kernels.prefill_compact("epilogue")
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT", 1)
    assert regime.prefill_glue_texte() == "prefill_glue=compact"
    assert kernels.prefill_compact() and kernels.prefill_compact("permut")
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT_ITEMS", "epilogue,a8")
    assert regime.prefill_glue_texte() == "prefill_glue=compact(items=epilogue,a8)"
    assert kernels.prefill_compact("epilogue") and not kernels.prefill_compact("residu")
    with pytest.raises(AssertionError):
        kernels.prefill_compact("inconnue")


# --- fusion 1 : épilogue i8c en un noyau ---------------------------------------

def _acc_et_echelles(M=200, N=520, seed=3):
    torch.manual_seed(seed)
    # amplitudes du produit entier réel (|Σ| ≤ K·127·127 ≈ 3,3·10⁷ à K = 2048) :
    # au-delà de 2²⁴ la conversion int32 → fp32 ARRONDIT, c'est cet arrondi-là
    # que l'épilogue doit reproduire
    acc = torch.randint(-40_000_000, 40_000_000, (M, N), dtype=torch.int32)
    sx = (torch.rand(M) * 0.02 + 1e-4)
    sw = (torch.rand(N) * 0.01 + 1e-5)
    return acc, sx, sw


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_epilogue_i8c_au_bit_avec_la_chaine_torch(dtype):
    from acvram.kernels import gemm_w8a8 as g
    if not g.disponible():
        pytest.skip("Triton absent")
    acc, sx, sw = _acc_et_echelles()
    y = g.epilogue_i8c(acc, sx, sw, dtype)
    ref = g.epilogue_i8c_torch(acc, sx, sw, dtype)
    assert y.dtype == dtype and y.shape == ref.shape
    assert torch.equal(y, ref), (y.view(torch.int16 if dtype == torch.bfloat16 else torch.int32)
                                 != ref.view(torch.int16 if dtype == torch.bfloat16 else torch.int32)).sum()


def test_epilogue_temoins_cassants_ordre_des_produits_et_arrondi_bf16():
    """La comparaison peut rendre faux : (1) en fp32, (f32(acc)·s_w)·s_x n'est pas
    (f32(acc)·s_x)·s_w au bit (un tiers des éléments) ; (2) en bf16, l'arrondi
    par troncature (ce que `.to(tl.bfloat16)` fait sous TRITON_INTERPRET, d'où
    la formule entière du noyau) diverge de l'arrondi au plus proche sur la
    moitié des éléments. Une régression sur l'un ou l'autre casse le test au bit."""
    from acvram.kernels import gemm_w8a8 as g
    acc, sx, sw = _acc_et_echelles()
    ref32 = g.epilogue_i8c_torch(acc, sx, sw, torch.float32)
    inverse = acc.to(torch.float32) * sw[None, :] * sx[:, None]
    assert (inverse != ref32).float().mean() > 0.1
    ref16 = g.epilogue_i8c_torch(acc, sx, sw, torch.bfloat16)
    tronque = (ref32.view(torch.int32) >> 16).to(torch.int16).view(torch.bfloat16)
    assert (tronque.view(torch.int16) != ref16.view(torch.int16)).float().mean() > 0.3


def test_gemm_i8c_cublas_compact_egale_le_temoin(monkeypatch):
    """Le dispatcher : sous PREFILL_COMPACT=1 le chemin passe par epilogue_i8c
    (compté), et rend les mêmes octets que sous 0 — bf16 et fp32."""
    from acvram.kernels import gemm_w8a8 as g
    from acvram.quant.formats import _quantize_int8
    if not g.disponible():
        pytest.skip("Triton absent")
    torch.manual_seed(5)
    w = torch.randn(96, 256) * 0.02
    t = _quantize_int8(w, group_size=256, symmetric=True)
    x = (torch.randn(40, 256) * 0.5).to(torch.bfloat16)
    appels = []
    vrai = g.epilogue_i8c
    monkeypatch.setattr(g, "epilogue_i8c", lambda *a, **k: appels.append(1) or vrai(*a, **k))
    refs = {}
    for fp32 in (False, True):
        monkeypatch.setattr(kernels, "_PREFILL_COMPACT", 0)
        refs[fp32] = ref = kernels.gemm_i8c_cublas(x, t, sortie_fp32=fp32)
        monkeypatch.setattr(kernels, "_PREFILL_COMPACT", 1)
        monkeypatch.setattr(kernels, "_PREFILL_COMPACT_ITEMS", "")
        y = kernels.gemm_i8c_cublas(x, t, sortie_fp32=fp32)
        assert ref is not None and y is not None and y.dtype == ref.dtype and torch.equal(y, ref)
    assert len(appels) == 2, appels
    # bissection : la fusion écartée suit le témoin (aucun appel de plus, mêmes octets)
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT_ITEMS", "a8")
    y = kernels.gemm_i8c_cublas(x, t)
    assert len(appels) == 2 and torch.equal(y, refs[False])


# --- fusion 2 : A8 quantifiée une fois pour q/k/v ----------------------------

def _trois_i8c(K=256, seed=7):
    from acvram.quant.formats import _quantize_int8
    torch.manual_seed(seed)
    return [_quantize_int8(torch.randn(N, K) * 0.02, group_size=K, symmetric=True) for N in (128, 32, 32)]


def test_a8_partagee_egale_trois_int8_matmul(monkeypatch):
    """Les trois sorties du chemin partagé sont les octets des trois `int8_matmul`
    séparés (même A8, même produit entier, même épilogue) ; un seul quantificateur."""
    from acvram.kernels import gemm_w8a8 as g
    ts = _trois_i8c()
    x = (torch.randn(3, 20, 256) * 0.5).to(torch.bfloat16)          # [b, t, K] : la forme est rendue
    appels = []
    vrai = g.quantifier_a8_torch
    monkeypatch.setattr(g, "quantifier_a8_torch", lambda *a, **k: appels.append(1) or vrai(*a, **k))
    refs = [kernels.int8_matmul(x, t) for t in ts]
    assert len(appels) == 3
    sorties = kernels.int8_matmul_partage(x, ts)
    assert sorties is not None and len(appels) == 4
    for y, r, t in zip(sorties, refs, ts):
        assert y.shape == (3, 20, t.shape[0]) and y.dtype == r.dtype and torch.equal(y, r)
    assert kernels.CHEMINS_INT8["cublas_partage"] >= 3


def test_a8_partagee_decline_ce_que_le_dispatcher_ne_prendrait_pas(monkeypatch):
    """Témoins cassants : M ≤ 16 (cuBLASLt refuse), un poids affine par groupes,
    le régime bf16 — dans chaque cas None, et les projections suivent le
    chemin d'avant une par une."""
    from acvram.quant.formats import _quantize_int8
    ts = _trois_i8c()
    x = (torch.randn(20, 256) * 0.5).to(torch.bfloat16)
    assert kernels.int8_matmul_partage(x[:16], ts) is None
    t_aff = _quantize_int8(torch.randn(32, 256) * 0.02, group_size=128, symmetric=False)
    assert kernels.int8_matmul_partage(x, ts[:2] + [t_aff]) is None
    monkeypatch.setattr(kernels, "_PREFILL_INT8", "bf16")
    assert kernels.int8_matmul_partage(x, ts) is None


def test_proj_partagee_de_l_attention_suit_le_temoin(monkeypatch):
    """`Attention._proj_i8c_partage` sur trois QuantLinear INT8 : mêmes octets que
    q_proj(x), k_proj(x), v_proj(x) ; un biais ou une échelle de canal rend None."""
    from acvram.engine.layers import QuantLinear
    from acvram.engine.model import Attention

    class Faux:
        k_eq_v = False

    ts = _trois_i8c()
    f = Faux()
    f.q_proj, f.k_proj, f.v_proj = (QuantLinear(t) for t in ts)
    x = (torch.randn(24, 256) * 0.5).to(torch.bfloat16)
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT", 1)
    qr, kr, vr = Attention._proj_i8c_partage(f, x)
    # à sec, `QuantLinear.forward` passe par le backend de référence (déquant fp32) :
    # le témoin du chemin cublas est `int8_matmul` appelé directement
    for y, t in zip((qr, kr, vr), ts):
        assert torch.equal(y, kernels.int8_matmul(x, t))
    f.k_proj = QuantLinear(ts[1], bias=torch.zeros(32, dtype=torch.bfloat16))
    assert Attention._proj_i8c_partage(f, x) is None


# --- fusion 3 : résidu différé au préfill -------------------------------------

def _prefill(model, prompt):
    from acvram.engine.model import ForwardBatch
    from acvram.memory.kvcache import BLOCK_SIZE, BlockAllocator
    n = len(prompt)
    alloc = BlockAllocator(model.caches[0].cfg.num_blocks)
    blocks = alloc.allocate((n + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)
    slots = torch.tensor([blocks[i // BLOCK_SIZE] * BLOCK_SIZE + i % BLOCK_SIZE for i in range(n)])
    batch = ForwardBatch(torch.tensor(prompt), torch.arange(n), [n], [n], [torch.tensor(blocks)], slots, True)
    return model(batch, logits_positions=torch.arange(n))


def test_residu_differe_au_prefill_rend_les_logits_du_temoin(converted, monkeypatch):
    """Tiny llama à sec : sous PREFILL_COMPACT=1 le préfill passe par `forward_res`
    (compté) et rend, sur TOUS les jetons, les octets des logits du témoin ; le
    dernier delta va dans la norme finale. À sec `add_norm` est le repli torch
    (add puis norme = le chemin d'avant) : ce test tient la plomberie ; l'égalité
    du noyau rmsnorm_bf16 avec résidu est celle du décodage (model.py, docstring
    de decode_fixed_res) et se relit sur carte par la PPL au bit de la chaîne."""
    from acvram.engine.loader import load_model
    from acvram.engine.model import DecoderLayer
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    model = loaded.model
    prompt = [5, 42, 7, 99, 13, 8, 21, 3]
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT", 0)
    ref = _prefill(model, prompt)
    appels = []
    vrai = DecoderLayer.forward_res
    monkeypatch.setattr(DecoderLayer, "forward_res", lambda self, *a: appels.append(1) or vrai(self, *a))
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT", 1)
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT_ITEMS", "")
    y = _prefill(model, prompt)
    assert len(appels) == len(model.layers) and y.shape == ref.shape and torch.equal(y, ref)
    # bissection : sans « residu », le chemin d'avant, aucun forward_res
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT_ITEMS", "epilogue")
    y = _prefill(model, prompt)
    assert len(appels) == len(model.layers) and torch.equal(y, ref)
    # témoin cassant : un multiplicateur résiduel ≠ 1 (granite) écarte le chemin
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT_ITEMS", "")
    monkeypatch.setattr(model.layers[1], "residual_multiplier", 0.5)
    _prefill(model, prompt)
    assert len(appels) == len(model.layers)


def test_une_valeur_hors_domaine_est_refusee():
    env = dict(os.environ, ACVRAM_PREFILL_COMPACT="2", CUDA_VISIBLE_DEVICES="")
    out = subprocess.run([sys.executable, "-c", "import acvram.kernels"], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode != 0 and "ACVRAM_PREFILL_COMPACT" in out.stderr
    env = dict(os.environ, ACVRAM_PREFILL_COMPACT_ITEMS="rmsnorm", CUDA_VISIBLE_DEVICES="")
    out = subprocess.run([sys.executable, "-c", "import acvram.kernels"], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode != 0 and "PREFILL_COMPACT_ITEMS" in out.stderr
