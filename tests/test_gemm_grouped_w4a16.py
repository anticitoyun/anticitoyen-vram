"""Juge d'équivalence pour une GEMM groupée W4A16 (poids NVFP4 déquantifiés
dans la tuile, activations bf16 lues telles quelles), écrit AVANT le noyau
(sage-objectif-suivant-17-09) — pour les DEUX cibles, décodage b = 12 et
prefill : même juge, mêmes fautes fabriquées, seules les formes changent.

Contrat du candidat : ``f(qw, bs, gs, x, comptes, M, K) -> y`` bf16 [G, M],
lignes triées par expert (``comptes[e]`` lignes pour l'expert e, comme
``_tuiles`` / ``moe_route_pack``), ``y[g] = x[g] · dequant(W[e])ᵀ · gs[e]``.
Tolérance : 2⁻⁷ × Σ|x_i·w_i| par valeur — la borne d'un produit scalaire
dont chaque terme est perturbé de 2⁻⁷ relatif (poids en bf16, accumulation
fp32 dans un ordre libre, sortie bf16). Un « 2⁻⁷ de la sortie » jugerait
faux le prefill d'aujourd'hui sur les lignes qui se compensent (2 valeurs
sur 12 288 en décodage, 15 sur 262 144 en prefill, écart 0,17 sur des
termes de 40) : le juge doit borner ce que l'arithmétique perturbe, les
termes, pas ce qu'elle rend.

Trois paliers :
1. la référence torch (déquant bf16 + matmul fp32, l'arithmétique de
   `_pile_bf16` + `torch._grouped_mm` du prefill) passe le juge ;
2. le juge SAIT dire « faux » (REGLES § 5) : un bloc de 16 dont la déquant
   est sautée, une échelle globale oubliée, une échelle de bloc décalée d'un
   bloc — trois fautes de noyau plausibles, chacune doit sortir de tolérance
   sur les deux formes ;
3. les noyaux CUDA quand ils existent, sautés proprement sinon :
   `nvfp4_gemv_grouped` (le chemin b = 12 d'aujourd'hui) et
   `nvfp4_gemm_grouped_w4a16` (le noyau à écrire).
"""
import pytest
import torch

from acvram.quant.nvfp4 import NVFP4Tensor, dequantize_nvfp4, quantize_nvfp4

TOL_REL = 2 ** -7
# décodage : t = 12, top_k = 8 → 96 lignes sur 8 experts, dont un vide ;
# prefill : 2 048 lignes sur 16 experts, très inégales
FORMES = {
    "decodage": ([20, 0, 7, 31, 3, 15, 12, 8], 128, 256),
    "prefill": ([512, 3, 0, 129, 64, 200, 17, 300, 1, 96, 88, 250, 41, 100, 150, 97], 128, 256),
}


def _pile(E, M, K, amplitude, graine, device="cpu"):
    """Pile NVFP4 comme `Layer._try_build_stacks` : qw [E, M, K/2] uint8,
    bs [E, M, K/16] octets E4M3, gs [E] fp32."""
    g = torch.Generator().manual_seed(graine)
    ts = [quantize_nvfp4((torch.randn(M, K, generator=g) * amplitude * (1 + e / E)).to(torch.bfloat16))
          for e in range(E)]
    qw = torch.stack([t.qweight for t in ts]).contiguous().to(device)
    bs = torch.stack([t.block_scale.view(torch.uint8) for t in ts]).contiguous().to(device)
    gs = torch.stack([t.global_scale.reshape(()) for t in ts]).to(torch.float32).contiguous().to(device)
    return qw, bs, gs


def _x(G, K, graine, device="cpu"):
    g = torch.Generator().manual_seed(graine)
    x = torch.randn(G, K, generator=g) * torch.logspace(-2, 1, G).unsqueeze(1)
    return x.to(torch.bfloat16).contiguous().to(device)


def _dequant(qw, bs, gs, e, dtype=torch.float64):
    t = NVFP4Tensor.__new__(NVFP4Tensor)
    t.qweight, t.block_scale = qw[e], bs[e].view(torch.float8_e4m3fn)
    t.global_scale = gs[e].reshape(())
    t.padded_in = qw.shape[-1] * 2
    t.shape = (qw.shape[1], t.padded_in)
    return dequantize_nvfp4(t, torch.float32).to(dtype)


def reference(qw, bs, gs, x, comptes, M, K):
    """float64 : la vérité contre laquelle tout candidat est jugé, et la
    somme des termes en valeur absolue qui borne sa tolérance."""
    y = torch.zeros(x.shape[0], M, dtype=torch.float64, device=x.device)
    borne = torch.zeros_like(y)
    debut = 0
    for e, n in enumerate(comptes):
        if n:
            w = _dequant(qw, bs, gs, e)
            xe = x[debut:debut + n].double()
            y[debut:debut + n] = xe @ w.T
            borne[debut:debut + n] = xe.abs() @ w.abs().T
            debut += n
    return y, borne


def _hors(y, attendu):
    y_, borne = attendu
    return (y.double() - y_).abs() > TOL_REL * borne


def juger(y, attendu):
    """Nombre de valeurs hors tolérance, et l'écart maximal."""
    h = _hors(y, attendu)
    return int(h.sum()), float((y.double() - attendu[0]).abs().max())


def candidat_torch(qw, bs, gs, x, comptes, M, K):
    """L'arithmétique du prefill d'aujourd'hui : déquant bf16 par expert,
    produit fp32, sortie bf16."""
    y = torch.empty(x.shape[0], M, dtype=torch.bfloat16, device=x.device)
    debut = 0
    for e, n in enumerate(comptes):
        if n:
            w = _dequant(qw, bs, gs, e, torch.bfloat16)
            y[debut:debut + n] = (x[debut:debut + n].float() @ w.float().T).to(torch.bfloat16)
            debut += n
    return y


# ---- fautes de noyau fabriquées : chacune DOIT sortir de tolérance

def _faute_bloc_sans_dequant(qw, bs, gs, x, comptes, M, K):
    """Une tuile (expert 3, lignes 0-31, bloc de colonnes 16-31) lue en codes
    bruts : échelle de bloc remplacée par 1,0 (E4M3 0x38)."""
    bs = bs.clone()
    bs[3, 0:32, 1] = 0x38
    return candidat_torch(qw, bs, gs, x, comptes, M, K)


def _faute_echelle_globale_oubliee(qw, bs, gs, x, comptes, M, K):
    gs = gs.clone()
    gs[3] = 1.0
    return candidat_torch(qw, bs, gs, x, comptes, M, K)


def _faute_echelle_de_bloc_decalee(qw, bs, gs, x, comptes, M, K):
    """Indexation des échelles décalée d'un bloc sur l'expert 3 (le bogue
    d'adresse le plus probable d'un chargeur de tuile)."""
    bs = bs.clone()
    bs[3] = torch.roll(bs[3], 1, dims=-1)
    return candidat_torch(qw, bs, gs, x, comptes, M, K)


FAUTES = {"bloc-sans-dequant": _faute_bloc_sans_dequant,
          "echelle-globale-oubliee": _faute_echelle_globale_oubliee,
          "echelle-de-bloc-decalee": _faute_echelle_de_bloc_decalee}


def _cas(forme, device="cpu"):
    comptes, M, K = FORMES[forme]
    qw, bs, gs = _pile(len(comptes), M, K, 0.05, 17 + len(comptes), device)
    x = _x(sum(comptes), K, 4 + sum(comptes), device)
    return qw, bs, gs, x, comptes, M, K


@pytest.mark.parametrize("forme", list(FORMES))
def test_la_reference_torch_passe_le_juge(forme):
    cas = _cas(forme)
    hors, ecart = juger(candidat_torch(*cas), reference(*cas))
    assert hors == 0, f"{forme} : {hors} valeurs hors tolérance, écart max {ecart:.3e}"


@pytest.mark.parametrize("forme", list(FORMES))
@pytest.mark.parametrize("faute", list(FAUTES))
def test_le_juge_sait_dire_faux(forme, faute):
    """Sans ce palier, « 0 hors tolérance » ne se distingue pas d'un juge
    qui ne voit rien : la faute doit toucher l'expert 3 (non vide dans les
    deux formes) et lui seul."""
    cas = _cas(forme)
    attendu = reference(*cas)
    y = FAUTES[faute](*cas)
    hors, ecart = juger(y, attendu)
    assert hors > 0, f"{forme}/{faute} : le juge ne voit pas la faute"
    comptes = cas[4]
    d3 = sum(comptes[:3])
    lignes = _hors(y, attendu).any(1)
    assert lignes[d3:d3 + comptes[3]].any() and not lignes[:d3].any() and not lignes[d3 + comptes[3]:].any(), \
        f"{forme}/{faute} : la faute déborde de l'expert 3"


# ---- noyaux CUDA, quand ils existent

def _ext():
    if not torch.cuda.is_available():
        pytest.skip("pas de GPU")
    from acvram.kernels import get_extension
    ext = get_extension()
    if ext is None:
        pytest.skip("extension absente")
    return ext


def _via_gemv(ext):
    """Le chemin b = 12 d'aujourd'hui (`nvfp4_gemv_grouped`, activation fp32,
    une ligne par (jeton, expert)) : il doit passer le même juge — c'est
    l'équivalence que le futur noyau doit conserver."""
    def f(qw, bs, gs, x, comptes, M, K):
        eid = torch.cat([torch.full((n,), e, dtype=torch.int32) for e, n in enumerate(comptes)]).cuda()
        tok = torch.arange(x.shape[0], dtype=torch.int32, device="cuda")
        return ext.nvfp4_gemv_grouped(qw, bs, gs, eid, tok, x.float().contiguous(), K)[:, :M].to(torch.bfloat16)
    return f


def _via_gemm_w4a16(ext):
    if not hasattr(ext, "nvfp4_gemm_grouped_w4a16"):
        pytest.skip("nvfp4_gemm_grouped_w4a16 : noyau pas encore écrit")

    def f(qw, bs, gs, x, comptes, M, K):
        from acvram.engine.model import MoEBlock
        cnt = torch.tensor(comptes, device="cuda")
        te, t0, tn = MoEBlock._tuiles(cnt, 32)
        E = qw.shape[0]
        tq = (qw.data_ptr() + torch.arange(E, dtype=torch.int64) * qw.stride(0)).cuda()
        tb = (bs.data_ptr() + torch.arange(E, dtype=torch.int64) * bs.stride(0)).cuda()
        return ext.nvfp4_gemm_grouped_w4a16(tq, tb, gs, x.contiguous(), te, t0, tn, M, K, 32)
    return f


@pytest.mark.parametrize("forme", list(FORMES))
@pytest.mark.parametrize("noyau", ["gemv", "gemm_w4a16"])
def test_le_noyau_cuda_passe_le_juge(forme, noyau):
    ext = _ext()
    f = _via_gemv(ext) if noyau == "gemv" else _via_gemm_w4a16(ext)
    cas = _cas(forme, "cuda")
    hors, ecart = juger(f(*cas), reference(*cas))
    assert hors == 0, f"{noyau}/{forme} : {hors} valeurs hors tolérance, écart max {ecart:.3e}"


# ---- commit A (sage-profil-verdict-17-09) : la GEMM groupée bf16 par seaux de bmm

def candidat_bmm(qw, bs, gs, x, comptes, M, K):
    """`MoEBlock._plan_bmm` + `_grouped_bmm` sur la pile déquantifiée en bf16 :
    le chemin `else` de `_forward_prefill_grouped` (`ACVRAM_PREFILL_GROUPED=bmm`,
    réfuté sur carte : −36 %, Laure 0edc3b9 ; le défaut est `grouped_mm`)."""
    from acvram.engine.model import MoEBlock
    w = torch.stack([_dequant(qw, bs, gs, e, torch.bfloat16) for e in range(qw.shape[0])])
    plan = MoEBlock._plan_bmm(torch.tensor(comptes, device=x.device))
    return MoEBlock._grouped_bmm(x, w, plan)


@pytest.mark.parametrize("forme", list(FORMES))
def test_la_gemm_groupee_par_seaux_passe_le_juge(forme):
    cas = _cas(forme)
    hors, ecart = juger(candidat_bmm(*cas), reference(*cas))
    assert hors == 0, f"{forme} : {hors} valeurs hors tolérance, écart max {ecart:.3e}"


@pytest.mark.parametrize("comptes", [FORMES["decodage"][0], FORMES["prefill"][0],
                                     [1000, 1, 1, 1, 1, 1, 1, 1], [0] * 7 + [5], [64] * 128])
def test_le_plan_bmm_borne_le_travail_inutile_et_couvre_chaque_ligne(comptes):
    from acvram.engine.model import MoEBlock
    seaux, src, dst, G = MoEBlock._plan_bmm(torch.tensor(comptes))
    n = sum(comptes)
    assert sorted(src.tolist()) == list(range(n)) and dst.unique().numel() == n
    assert G <= 1.5 * n + 32 * len(seaux), (G, n, len(seaux))
    couverts = set()
    for experts, cap, b0 in seaux:
        for r, e in enumerate(experts.tolist()):
            assert comptes[e] <= cap and (r == 0 or comptes[e] * 1.5 >= cap), (e, comptes[e], cap)
            couverts.add(e)
    assert couverts == {e for e, c in enumerate(comptes) if c}
    if hasattr(torch, "_grouped_mm"):
        x = torch.randn(n, 64).to(torch.bfloat16)
        w = torch.randn(len(comptes), 48, 64).to(torch.bfloat16)
        y = MoEBlock._grouped_bmm(x, w, (seaux, src, dst, G))
        attendu = torch.cat([x[sum(comptes[:e]):sum(comptes[:e + 1])].float() @ w[e].float().T
                             for e in range(len(comptes))]).to(torch.bfloat16)
        assert torch.allclose(y.float(), attendu.float(), rtol=2 ** -6, atol=1e-2)


from test_moe_grouped import tiny_moe  # noqa: E402,F401  (fixture de session, réutilisée)


@pytest.mark.sans_extension
def test_prefill_groupe_bmm_egale_grouped_mm_et_la_boucle_sur_le_mini_moe(tiny_moe, monkeypatch):
    """Bout en bout sur le mini-MoE du conftest, processeur : le chemin
    `_forward_prefill_grouped` en `bmm` (réfuté, témoin) contre le défaut
    `grouped_mm` et contre la boucle par expert (`forward` sur CPU), même
    routage, t = 64 > _MOE_GROUPED_MAX."""
    from acvram.engine import model as M
    from acvram.engine.loader import load_model
    loaded = load_model(tiny_moe, dtype=torch.float32, device_override="cpu")
    blocs = [m for m in loaded.model.modules() if isinstance(m, M.MoEBlock)]
    bloc = blocs[0]
    assert bloc._try_build_stacks(), "pile refusée sur le mini-MoE"
    torch.manual_seed(3)
    x = torch.randn(64, loaded.spec.hidden_size) * 0.3
    x = x.to(torch.bfloat16).float()
    scores = bloc.router(x) if hasattr(bloc, "router") else None
    if scores is None:
        pytest.skip("routeur introuvable sur ce bloc")
    topw, topi = torch.topk(torch.softmax(scores.float(), -1), bloc.top_k, dim=-1)
    topw = topw / topw.sum(-1, keepdim=True)
    sorties = {}
    for chemin in ("bmm", "grouped_mm"):
        if chemin == "grouped_mm" and not hasattr(torch, "_grouped_mm"):
            continue
        monkeypatch.setattr(M, "_PREFILL_GROUPED", chemin)
        y = bloc._forward_prefill_grouped(x, topw, topi)
        assert y is not None, f"chemin {chemin} non pris"
        sorties[chemin] = y.float()
    boucle = bloc.forward(x).float()
    if bloc.shared is not None:
        boucle = boucle - bloc._shared_out(x).float()
    for chemin, y in sorties.items():
        ecart = (y - boucle).norm() / boucle.norm()
        print(f"{chemin} contre la boucle : {ecart:.5f}")
        assert ecart < 2 ** -6, f"{chemin} contre la boucle : {ecart:.4f}"
    if "grouped_mm" in sorties:
        ecart = (sorties["bmm"] - sorties["grouped_mm"]).norm() / sorties["grouped_mm"].norm()
        print(f"bmm contre grouped_mm : {ecart:.6f}")
        assert ecart < 2 ** -8, f"bmm contre grouped_mm : {ecart:.5f}"


# ---- B0 (sage-prefill-b-plan-17-09) : GEMM groupée persistante Triton

def _gg():
    """Le module Triton ; sans carte, l'interpréteur (`TRITON_INTERPRET=1`,
    posé AVANT l'import : le choix se fait à la décoration) en fp16 — son
    numpy n'a pas de bf16."""
    import importlib
    import os
    if not torch.cuda.is_available():
        os.environ.setdefault("TRITON_INTERPRET", "1")
    gg = importlib.import_module("acvram.kernels.gemm_groupe")
    if not gg.disponible():
        pytest.skip("Triton indisponible")
    return gg


def _tiles_pour(comptes, bt, device, decalage=0):
    from acvram.engine.model import MoEBlock
    cnt = torch.tensor(comptes, device=device)
    te, t0, tn = MoEBlock._tuiles(cnt, bt, t_max=-(-sum(comptes) // bt) + len(comptes))
    if decalage:
        t0 = t0.clone()
        t0[2] += decalage                                  # une tuile décalée d'une ligne
    return te, t0, tn


def candidat_groupe(qw, bs, gs, x, comptes, M, K, decalage=0):
    gg = _gg()
    dt = torch.bfloat16 if x.is_cuda else torch.float16
    w = torch.stack([_dequant(qw, bs, gs, e, dt) for e in range(qw.shape[0])])
    y = gg.gemm_groupe(x.to(dt), w, _tiles_pour(comptes, gg.BT, x.device, decalage))
    return y.to(torch.bfloat16) if not x.is_cuda else y


@pytest.mark.parametrize("forme", list(FORMES))
def test_b0_la_gemm_groupee_persistante_passe_le_juge(forme):
    cas = _cas(forme, "cuda" if torch.cuda.is_available() else "cpu")
    hors, ecart = juger(candidat_groupe(*cas), reference(*cas))
    assert hors == 0, f"{forme} : {hors} valeurs hors tolérance, écart max {ecart:.3e}"


def test_b0_un_offset_decale_d_une_ligne_casse():
    """Le juge doit voir une tuile qui commence une ligne trop tard : les
    lignes de l'expert touché lisent celles du voisin."""
    cas = _cas("decodage", "cuda" if torch.cuda.is_available() else "cpu")
    hors, _ = juger(candidat_groupe(*cas, decalage=1), reference(*cas))
    assert hors > 0, "un offset décalé d'une ligne passe le juge : il ne voit rien"


# ---- B1 : la même grille lisant NVFP4 dans la tuile

def test_b1_les_tables_de_decodage_sont_exactes_en_bf16():
    """Les deux tables du noyau (E2M1 → bf16, E4M3 → bf16) : exactes contre
    torch (E4M3 fn : 0x7F/0xFF = NaN → 0, jamais produits par quantize_nvfp4),
    et le produit code × échelle est exact en bf16 (1 + 3 bits ≤ 7)."""
    gg = _gg()
    lut4, lut8 = gg.tables("cpu", torch.bfloat16)
    att4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6] * 2) * torch.tensor([1.] * 8 + [-1.] * 8)
    assert torch.equal(lut4.float(), att4)
    att8 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    ok = ~torch.isnan(att8)
    assert torch.equal(lut8.float()[ok], att8[ok]) and (lut8.float()[~ok] == 0).all()
    prod = (lut4.float()[:, None] * lut8.float()[None, ok])
    assert torch.equal(prod.to(torch.bfloat16).float(), prod)


def candidat_groupe_nvfp4(qw, bs, gs, x, comptes, M, K, decalage=0, gs_par_ligne=False):
    gg = _gg()
    dt = torch.bfloat16 if x.is_cuda else torch.float16
    if gs_par_ligne:
        gs = gs.unsqueeze(1).expand(qw.shape[0], qw.shape[1]).contiguous()
    y = gg.gemm_groupe_nvfp4(x.to(dt), qw, bs, gs, _tiles_pour(comptes, gg.BT, x.device, decalage))
    return y.to(torch.bfloat16) if not x.is_cuda else y


@pytest.mark.parametrize("forme", list(FORMES))
@pytest.mark.parametrize("gs_par_ligne", [False, True])
def test_b1_la_gemm_nvfp4_dans_la_tuile_passe_le_juge(forme, gs_par_ligne):
    cas = _cas(forme, "cuda" if torch.cuda.is_available() else "cpu")
    hors, ecart = juger(candidat_groupe_nvfp4(*cas, gs_par_ligne=gs_par_ligne), reference(*cas))
    assert hors == 0, f"{forme} : {hors} valeurs hors tolérance, écart max {ecart:.3e}"


@pytest.mark.parametrize("faute", list(FAUTES))
def test_b1_les_fautes_fabriquees_cassent_aussi_le_noyau(faute):
    """Les trois fautes (bloc sans déquant, échelle globale oubliée, échelle
    décalée d'un bloc) injectées dans les OCTETS que lit le noyau."""
    cas = _cas("decodage", "cuda" if torch.cuda.is_available() else "cpu")
    qw, bs, gs, x, comptes, M, K = cas
    bs2, gs2 = bs.clone(), gs.clone()
    if faute == "bloc-sans-dequant":
        bs2[3, 0:32, 1] = 0x38
    elif faute == "echelle-globale-oubliee":
        gs2[3] = 1.0
    else:
        bs2[3] = torch.roll(bs2[3], 1, dims=-1)
    hors, _ = juger(candidat_groupe_nvfp4(qw, bs2, gs2, x, comptes, M, K), reference(*cas))
    assert hors > 0, f"{faute} : le noyau fauté passe le juge"


def test_b1_un_offset_decale_d_une_ligne_casse():
    cas = _cas("decodage", "cuda" if torch.cuda.is_available() else "cpu")
    hors, _ = juger(candidat_groupe_nvfp4(*cas, decalage=1), reference(*cas))
    assert hors > 0


@pytest.mark.a_sec
def test_b1_nvfp4_linear_egale_la_dequant_bf16_sur_un_tenseur_fusionne():
    """Le chemin non groupé (`ACVRAM_PREFILL=w4a16`, nvfp4_matmul) : un
    NVFP4Tensor avec échelle globale PAR LIGNE (q/k/v fusionnés) et une
    entrée plus courte que `padded_in` — contre dequantize_nvfp4 + linear."""
    gg = _gg()
    from acvram.quant.nvfp4 import dequantize_nvfp4, quantize_nvfp4
    g = torch.Generator().manual_seed(5)
    w = (torch.randn(96, 200, generator=g) * 0.05).to(torch.bfloat16)
    t = quantize_nvfp4(w)
    assert t.padded_in > 200
    t.global_scale_rows = (t.global_scale.float() * torch.linspace(0.5, 2.0, 96)).contiguous()
    x = (torch.randn(300, 200, generator=g)).to(torch.float16 if not torch.cuda.is_available() else torch.bfloat16)
    y = gg.nvfp4_linear(x, t)
    wd = dequantize_nvfp4(t, torch.float32)
    attendu = x.float() @ wd.T
    borne = x.float().abs() @ wd.abs().T
    hors = ((y.float() - attendu).abs() > TOL_REL * borne).sum()
    assert y.shape == (300, 96) and int(hors) == 0, int(hors)
