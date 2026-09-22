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

def test_le_defaut_est_compact_sans_norm():
    """DÉFAUT 1 depuis le 20/09 (verdict-c15-prefill-19-09, poste2 bcd7a73b : au bit 3 tranches, capture 4/4,
    prefill servi × 1,29-1,34) ; le témoin reste 0 ; `norm` hors défaut (3,82 ms contre 2,12 au bloc)."""
    assert _var("PREFILL_COMPACT").defaut == "1" and _var("PREFILL_COMPACT").torch == "0"
    assert _var("PREFILL_COMPACT_ITEMS").defaut == ""
    assert "norm" not in kernels.PREFILL_COMPACT_DEFAUT and set(kernels.PREFILL_COMPACT_DEFAUT) >= {"epilogue", "a8", "residu", "permut", "attn"}


def test_le_module_lit_le_defaut_sans_variable():
    env = {k: v for k, v in os.environ.items() if not k.startswith("ACVRAM_PREFILL_COMPACT")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    out = subprocess.run([sys.executable, "-c", "from acvram import kernels; print(kernels._PREFILL_COMPACT)"],
                         env=env, capture_output=True, text=True, timeout=120)
    assert out.stdout.split() == ["1"], out.stdout + out.stderr[-500:]


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
    # « norm » hors défaut (3,82 ms contre 2,12 pour le bloc, poste2 20/09) : opt-in seulement
    assert not kernels.prefill_compact("norm")
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT_ITEMS", "permut,norm")
    assert kernels.prefill_compact("norm") and not kernels.prefill_compact("epilogue")
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT_ITEMS", "epilogue,a8")
    assert regime.prefill_glue_texte() == "prefill_glue=compact(items=epilogue,a8)"
    assert kernels.prefill_compact("epilogue") and not kernels.prefill_compact("residu")
    with pytest.raises(AssertionError):
        kernels.prefill_compact("inconnue")


# --- fusion 1 : épilogue i8c en un noyau ---------------------------------------

# Sur carte le noyau Triton est RÉEL et lit des tenseurs CUDA ; à sec (conftest :
# CUDA_VISIBLE_DEVICES vide → TRITON_INTERPRET=1) l'interpréteur lit des tenseurs
# CPU. Un test qui donne des tenseurs CPU au noyau avec une carte visible est
# faux par construction (verdict poste2 07 h 20 : « Pointer argument cannot be
# accessed from Triton »).
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _acc_et_echelles(M=200, N=520, seed=3):
    torch.manual_seed(seed)
    # amplitudes du produit entier réel (|Σ| ≤ K·127·127 ≈ 3,3·10⁷ à K = 2048) :
    # au-delà de 2²⁴ la conversion int32 → fp32 ARRONDIT, c'est cet arrondi-là
    # que l'épilogue doit reproduire
    acc = torch.randint(-40_000_000, 40_000_000, (M, N), dtype=torch.int32, device=DEV)
    sx = (torch.rand(M, device=DEV) * 0.02 + 1e-4)
    sw = (torch.rand(N, device=DEV) * 0.01 + 1e-5)
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
    # éligible au chemin cublas sur carte comme à sec : M = 40 > 16, K = 256 et
    # N = 96 multiples de 8, groupe = K, zéros = 128 (symétrique par canal)
    t = _quantize_int8(w, group_size=256, symmetric=True).to(DEV)
    x = (torch.randn(40, 256, device=DEV) * 0.5).to(torch.bfloat16)
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
    return [_quantize_int8(torch.randn(N, K) * 0.02, group_size=K, symmetric=True).to(DEV) for N in (128, 32, 32)]


def _sous_le_seuil_gemv(monkeypatch):
    """Sur carte `int8_matmul` prend le GEMV jusqu'à 80 lignes (ACVRAM_INT8_GEMV_MAX)
    et le chemin partagé décline alors, comme il doit : les tests du chemin cublas
    abaissent le seuil pour l'atteindre avec quelques dizaines de lignes."""
    monkeypatch.setattr(kernels, "_INT8_GEMV_MAX", 16)


def test_a8_partagee_egale_trois_int8_matmul(monkeypatch):
    """Les trois sorties du chemin partagé sont les octets des trois `int8_matmul`
    séparés (même A8, même produit entier, même épilogue) ; un seul quantificateur."""
    from acvram.kernels import gemm_w8a8 as g
    _sous_le_seuil_gemv(monkeypatch)
    ts = _trois_i8c()
    x = (torch.randn(3, 20, 256, device=DEV) * 0.5).to(torch.bfloat16)   # [b, t, K] : la forme est rendue
    appels = []
    vrai = g.quantifier_a8_torch if DEV == "cpu" else g.quantifier_a8
    monkeypatch.setattr(g, "quantifier_a8_torch" if DEV == "cpu" else "quantifier_a8",
                        lambda *a, **k: appels.append(1) or vrai(*a, **k))
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
    _sous_le_seuil_gemv(monkeypatch)
    ts = _trois_i8c()
    x = (torch.randn(20, 256, device=DEV) * 0.5).to(torch.bfloat16)
    assert kernels.int8_matmul_partage(x, ts) is not None                # le cas nominal, d'abord
    assert kernels.int8_matmul_partage(x[:16], ts) is None
    t_aff = _quantize_int8(torch.randn(32, 256) * 0.02, group_size=128, symmetric=False).to(DEV)
    assert kernels.int8_matmul_partage(x, ts[:2] + [t_aff]) is None
    monkeypatch.setattr(kernels, "_INT8_GEMV_MAX", 80)                   # sous le seuil GEMV : le dispatcher prendrait le GEMV
    if DEV == "cuda":
        assert kernels.int8_matmul_partage(x, ts) is None
    monkeypatch.setattr(kernels, "_PREFILL_INT8", "bf16")
    assert kernels.int8_matmul_partage(x, ts) is None


def test_proj_partagee_de_l_attention_suit_le_temoin(monkeypatch):
    """`Attention._proj_i8c_partage` sur trois QuantLinear INT8 : mêmes octets que
    q_proj(x), k_proj(x), v_proj(x) ; un biais ou une échelle de canal rend None."""
    from acvram.engine.layers import QuantLinear
    from acvram.engine.model import Attention

    class Faux:
        k_eq_v = False

    _sous_le_seuil_gemv(monkeypatch)
    ts = _trois_i8c()
    f = Faux()
    f.q_proj, f.k_proj, f.v_proj = (QuantLinear(t) for t in ts)
    x = (torch.randn(24, 256, device=DEV) * 0.5).to(torch.bfloat16)
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT", 1)
    qr, kr, vr = Attention._proj_i8c_partage(f, x)
    # à sec, `QuantLinear.forward` passe par le backend de référence (déquant fp32) :
    # le témoin du chemin cublas est `int8_matmul` appelé directement ; sur carte
    # c'est aussi ce que q_proj(x) fait (backend cuda-fusionne)
    for y, t, lin in zip((qr, kr, vr), ts, (f.q_proj, f.k_proj, f.v_proj)):
        assert torch.equal(y, kernels.int8_matmul(x, t))
        if DEV == "cuda":
            assert torch.equal(y, lin(x))
    f.k_proj = QuantLinear(ts[1], bias=torch.zeros(32, dtype=torch.bfloat16, device=DEV))
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


# --- fusion 4 : permutations MoE du préfill sans second tri ------------------

def _paires(t=300, k=8, E=128, seed=11, vides=True):
    torch.manual_seed(seed)
    topi = torch.randint(0, E, (t, k), dtype=torch.int32)
    if vides:                                           # des experts sans aucune paire
        topi[topi % 7 == 3] = 5
    return topi


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_comptes_tries_egalent_bincount_et_cassent_sans_tri(dtype):
    from acvram.engine.model import _comptes_tries
    E = 128
    flat_e = _paires().reshape(-1).to(dtype)
    e_sorted = flat_e[torch.argsort(flat_e, stable=True)]
    cnt = _comptes_tries(e_sorted, E)
    ref = torch.bincount(flat_e, minlength=E)
    assert cnt.dtype == ref.dtype == torch.int64 and torch.equal(cnt, ref)
    assert int((cnt == 0).sum()) > 0                    # les experts vides comptent zéro
    assert not torch.equal(_comptes_tries(flat_e, E), ref)   # une liste non triée rend faux


def test_ordre_sur_k_egale_flat_t_ordre():
    """flat_t[ordre] (arange.repeat_interleave puis gather) = ordre // k."""
    t, k = 300, 8
    flat_e = _paires(t, k).reshape(-1)
    ordre = torch.argsort(flat_e, stable=True)
    flat_t = torch.arange(t).repeat_interleave(k)
    assert torch.equal(flat_t[ordre], torch.div(ordre, k, rounding_mode="floor"))
    # le tri sur les clés int32 rend le même ordre que sur leur copie int64
    assert torch.equal(ordre, torch.argsort(flat_e.to(torch.int64), stable=True))


@pytest.mark.parametrize("bloc", [8, 16, 64])
@pytest.mark.parametrize("vides", [False, True])
def test_aligner_blocs_tries_egale_aligner_blocs(bloc, vides):
    """Mêmes sorted_ids / expert_ids / num_post que `aligner_blocs` sur la
    liste triée (ce que le préfill lui donnait) sur les `total` premières
    entrées ; au-delà (taille fixe P, aucun scalaire hôte) : sentinelle G et
    −1, ce que le noyau ne visite jamais. Témoin cassant : des comptes faux
    (décalés d'un expert) changent les sorties."""
    from acvram.kernels import marlin_port as MP
    E = 128
    flat_e = _paires(vides=vides).reshape(-1)
    G = flat_e.numel()
    e_sorted = flat_e[torch.argsort(flat_e, stable=True)]
    cnt = torch.bincount(e_sorted, minlength=E)
    s, e, n = MP.aligner_blocs(e_sorted.unsqueeze(1), bloc, E)
    s2, e2, n2 = MP.aligner_blocs_tries(e_sorted, cnt, bloc, E)
    total = int(n)
    assert torch.equal(n, n2) and s2.numel() >= total and s2.numel() % bloc == 0
    assert torch.equal(s[:total], s2[:total]) and torch.equal(e[: total // bloc], e2[: total // bloc])
    assert bool((s2[total:] == G).all()) and bool((e2[total // bloc:] == -1).all())
    assert s2.dtype == e2.dtype == n2.dtype == torch.int32
    faux = torch.roll(cnt, 1)
    s3, e3, _ = MP.aligner_blocs_tries(e_sorted, faux, bloc, E)
    assert not (torch.equal(s3[:total], s[:total]) and torch.equal(e3[: total // bloc], e[: total // bloc]))


def test_a_sec_le_compte_d_usage_n_est_jamais_differe(monkeypatch):
    """Sans carte le chemin groupé n'existe pas : sous PREFILL_COMPACT=1 le compte
    d'usage reste celui de `_compter_routage`, égal au témoin, sans drapeau laissé."""
    from test_marlin_prefill_p1 import _bloc_moe_jouet
    bloc = _bloc_moe_jouet(4, 64, 32, 2, dev="cpu")
    torch.manual_seed(1)
    x = (torch.randn(40, 64) * 0.5).to(torch.bfloat16)
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT", 0)
    ref = bloc(x)
    usage = bloc._usage_routage.clone()
    bloc._usage_routage.zero_()
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT", 1)
    y = bloc(x)
    assert torch.equal(y, ref) and torch.equal(bloc._usage_routage, usage) and int(usage.sum()) == 80
    assert "_compte_en_attente" not in bloc.__dict__


@pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise (noyaux Marlin)")
def test_prefill_moe_marlin_compact_au_bit_avec_le_temoin(monkeypatch):
    """Sur carte : `_forward_prefill_grouped` (marlin) sous PREFILL_COMPACT=1
    rend les OCTETS du témoin — mêmes lignes triées, mêmes blocs, même
    réduction ; seule la glue change."""
    from conftest import attendre_chemin
    from test_marlin_prefill_p1 import _bloc_moe_jouet
    from acvram.engine import model as MD
    from acvram.engine import moe as MOE_D
    from acvram.kernels import marlin_port as MP
    if MP.charger(compiler=False) is None:
        pytest.skip("extension Marlin non compilée")
    E, H, I, top_k, T = 8, 256, 128, 2, 1024
    bloc = _bloc_moe_jouet(E, H, I, top_k)
    torch.manual_seed(T)
    x = (torch.randn(T, H, device="cuda") * 0.5).to(torch.bfloat16)
    logits = bloc.router(x).float()
    topw, topi = torch.topk(torch.softmax(logits, -1), top_k, dim=-1)
    topw = topw / topw.sum(-1, keepdim=True)
    topi32 = topi.to(torch.int32)
    monkeypatch.setattr(MOE_D, "_PREFILL_GROUPED", "marlin")
    assert bloc._try_build_stacks()
    assert getattr(bloc, "_stacks_marlin", None) is not None
    # `forward` reconstruit les piles tant que `_stack_state` vaut « ? » : la
    # seconde construction voyait la pile naturelle déjà RENDUE (vide, hors CUDA)
    # après le repack Marlin, refusait Marlin et partait en MMA sur une pile
    # vide (verdict poste2 07 h 20, model.py:1301). L'état est posé comme
    # `forward` le pose après une construction réussie.
    bloc._stack_state = "oui"
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT", 0)
    n0 = bloc.chemins.get("marlin", 0) if hasattr(bloc, "chemins") else 0
    ref = bloc._forward_prefill_grouped(x, topw, topi32)
    attendre_chemin(bloc, "marlin", avant=n0)
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT", 1)
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT_ITEMS", "permut")
    y = bloc._forward_prefill_grouped(x, topw, topi32)
    attendre_chemin(bloc, "marlin", avant=n0 + 1)
    assert torch.equal(y, ref)
    # par `forward` (routage réel, compte d'usage différé sous « permut ») : mêmes
    # octets et le même compteur d'usage que le témoin
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT", 0)
    bloc._usage_routage = None
    ref_f = bloc(x)
    usage_a = bloc._usage_routage.clone()
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT", 1)
    bloc._usage_routage.zero_()
    y_f = bloc(x)
    assert torch.equal(y_f, ref_f) and torch.equal(bloc._usage_routage, usage_a)
    assert int(usage_a.sum()) == T * top_k and "_compte_en_attente" not in bloc.__dict__


# --- fusion 5 : rmsnorm un warp par ligne (carte) -----------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise (noyau CUDA)")
@pytest.mark.parametrize("H", [2048, 1024, 1536, 1000, 256, 200, 4096])
def test_rmsnorm_warp_au_bit_avec_le_noyau_a_bloc(H):
    """Sur carte : `rmsnorm_bf16_warp` (un warp par ligne, ordre de somme rejoué)
    rend les octets de `rmsnorm_bf16` (bloc par ligne), sans et avec résidu, sur
    les formes du préfill et les autres découpes TH / EPT (H non multiple de TH
    compris) ; témoin cassant : eps décalé ; H > 2 048 est refusé (le bloc)."""
    ext = kernels.get_extension()
    if ext is None or not hasattr(ext, "rmsnorm_bf16_warp"):
        pytest.skip("extension sans rmsnorm_bf16_warp")
    torch.manual_seed(H)
    if H > 2048:
        x = torch.randn(4, H, device="cuda").to(torch.bfloat16)
        with pytest.raises(RuntimeError):
            ext.rmsnorm_bf16_warp(x, torch.ones(H, device="cuda", dtype=torch.bfloat16), 1e-6)
        return
    for R in (2047, 300, 9):
        x = (torch.randn(R, H, device="cuda") * 2.0).to(torch.bfloat16)
        res = (torch.randn(R, H, device="cuda") * 2.0).to(torch.bfloat16)
        w = (1.0 + 0.1 * torch.randn(H, device="cuda")).to(torch.bfloat16)
        y_bloc = ext.rmsnorm_bf16(x, w, 1e-6)[0]
        y_warp = ext.rmsnorm_bf16_warp(x, w, 1e-6)[0]
        assert torch.equal(y_bloc, y_warp), (H, R, int((y_bloc != y_warp).sum()))
        hb, xb = ext.rmsnorm_bf16(x, w, 1e-6, res, 1.0)
        hw, xw = ext.rmsnorm_bf16_warp(x, w, 1e-6, res, 1.0)
        assert torch.equal(xb, xw) and torch.equal(hb, hw), (H, R)
        assert not torch.equal(ext.rmsnorm_bf16_warp(x, w, 1e-3)[0], y_bloc)


def test_norme_warp_reservee_au_prefill(monkeypatch):
    """Le seuil : sous 256 lignes (décodage, graphes) le noyau à bloc reste."""
    from acvram.engine import layers as L

    class Ext:
        rmsnorm_bf16_warp = True

    monkeypatch.setattr(kernels, "_PREFILL_COMPACT", 1)
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT_ITEMS", "")
    assert not L._norme_warp(Ext(), torch.empty(2047, 2048, dtype=torch.bfloat16))   # hors défaut
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT_ITEMS", "norm")
    assert L._norme_warp(Ext(), torch.empty(2047, 2048, dtype=torch.bfloat16))
    assert not L._norme_warp(Ext(), torch.empty(16, 2048, dtype=torch.bfloat16))
    assert not L._norme_warp(Ext(), torch.empty(2047, 4096, dtype=torch.bfloat16))   # GLM : le bloc
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT", 0)
    assert not L._norme_warp(Ext(), torch.empty(2047, 2048, dtype=torch.bfloat16))


# --- fusion 6 : attention du préfill sans copie ×n_rep de K/V ----------------

@pytest.mark.parametrize("cas", ["causal", "decale", "fenetre"])
def test_attention_gqa_egale_repeat_kv(cas):
    """`attention(..., n_rep)` (SDPA enable_gqa) rend les octets de
    `attention(q, repeat_kv(k), repeat_kv(v))` — causal, requête décalée
    (masque), fenêtre glissante ; témoin cassant : n_rep faux."""
    from acvram.engine.layers import attention, repeat_kv
    torch.manual_seed(3)
    t, hq, hkv, d = 40, 8, 2, 32
    q = torch.randn(t, hq, d); k = torch.randn(t + 8, hkv, d); v = torch.randn(t + 8, hkv, d)
    kw = dict(causal=True, scale=d ** -0.5)
    if cas == "causal":
        k, v = k[:t], v[:t]
    elif cas == "decale":
        kw["q_offset"] = 8
    else:
        k, v = k[:t], v[:t]
        kw["window"] = 16
    ref = attention(q, repeat_kv(k, hq // hkv), repeat_kv(v, hq // hkv), **kw)
    y = attention(q, k, v, n_rep=hq // hkv, **kw)
    assert y.shape == ref.shape and torch.equal(y, ref)
    faux = attention(q, k.flip(1), v.flip(1), n_rep=hq // hkv, **kw)   # têtes KV échangées
    assert not torch.equal(faux, ref)


def test_prefill_attn_compact_rend_les_logits_du_temoin(converted, monkeypatch):
    """Tiny llama (8 têtes, 2 KV) à sec : fusion « attn » seule, logits de tous
    les jetons au bit ; une séquence (sans tampon `out`) et deux séquences."""
    from acvram.engine.loader import load_model
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
    model = loaded.model
    prompt = [5, 42, 7, 99, 13, 8, 21, 3, 77, 1]
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT", 0)
    ref = _prefill(model, prompt)
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT", 1)
    monkeypatch.setattr(kernels, "_PREFILL_COMPACT_ITEMS", "attn")
    assert torch.equal(_prefill(model, prompt), ref)


def test_une_valeur_hors_domaine_est_refusee():
    env = dict(os.environ, ACVRAM_PREFILL_COMPACT="2", CUDA_VISIBLE_DEVICES="")
    out = subprocess.run([sys.executable, "-c", "import acvram.kernels"], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode != 0 and "ACVRAM_PREFILL_COMPACT" in out.stderr
    env = dict(os.environ, ACVRAM_PREFILL_COMPACT_ITEMS="rmsnorm", CUDA_VISIBLE_DEVICES="")
    out = subprocess.run([sys.executable, "-c", "import acvram.kernels"], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode != 0 and "PREFILL_COMPACT_ITEMS" in out.stderr
