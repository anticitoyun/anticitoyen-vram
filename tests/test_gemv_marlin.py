"""Forme (b) de P1 disposition unique (poste7-p1-disposition-unique-18-09,
chiffrage revue/p1-disposition-unique-chiffrage-18-09) : le GEMV du décodage
lit la DISPOSITION MARLIN des experts (`nvfp4_gemv_marlin[_gateup]`,
acvram_kernels.cu). Juge : fp32 par ligne, |Δ| ≤ 2⁻⁷·max|y| (le critère de
P1 ; pas identique au bit à v1, l'ordre fp32 diffère). REGLES § 7 : le test de
bloc asserte le chemin `gemv_marlin` (compteur `_chemin`) AVANT de comparer.

À sec (sans carte) : la place et le décodage des échelles S0E5M3 telles que le
noyau les lit — les fonctions de permutation sont en torch CPU.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from acvram.kernels import get_extension                                     # noqa: E402
from acvram.kernels import marlin_port as MP                                 # noqa: E402
from acvram.quant.nvfp4 import dequantize_nvfp4, quantize_nvfp4              # noqa: E402

TOL_HORS = 5e-4                       # part de valeurs hors 2⁻⁷ par ligne tolérée (P1)
# octet d'échelle (dans les 8 contigus de la voie) de la colonne w·16 + t/4 + 8h
OCTET_ECHELLE = {(0, 0): 0, (0, 1): 2, (1, 0): 1, (1, 1): 3, (2, 0): 4, (2, 1): 6, (3, 0): 5, (3, 1): 7}


def _decoder_s0e5m3(b: torch.Tensor) -> torch.Tensor:
    """Ce que fait `s0e5m3_octet` : bits fp32 = (b << 20) + 0x34800000 (b = 0 → 2⁻²²)."""
    return ((b.to(torch.int32) << 20) + 0x34800000).view(torch.float32)


def test_echelles_position_et_decodage_a_sec():
    """Pour chaque colonne n et tuile k : l'octet lu par le noyau (8·(t/4) +
    OCTET_ECHELLE[w, h], t/4 = n % 8, w = (n % 64) // 16, h = (n % 16) // 8)
    décodé × g_marlin·2⁻¹¹⁹ vaut l'échelle naturelle × échelle globale —
    exactement (puissances de deux et 3 bits de mantisse), sauf les échelles
    que la pile annule (s·facteur·2⁷ < 2)."""
    K, N = 128, 128
    g = torch.Generator().manual_seed(7)
    t = quantize_nvfp4((torch.randn(N, K, generator=g) * 0.05).to(torch.bfloat16))
    bs = t.block_scale.view(torch.uint8).unsqueeze(0)                        # [1, N, K/16]
    gs = t.global_scale.float().reshape(1)
    scales = bs.view(torch.float8_e4m3fn).to(torch.bfloat16)
    facteur = MP.facteur_nvfp4(scales)
    s = MP.traiter_echelles_nvfp4(MP.permuter_echelles(scales[0].T, K, N, MP.GROUP_SIZE), facteur)
    s8 = s.view(torch.uint8)                                                # [K/16, N]
    g_marlin = MP.traiter_echelle_globale(gs, facteur)
    naturel = scales[0].float() * gs                                        # [N, K/16]
    for kt in range(K // 16):
        for n in range(N):
            nt, nl = n // 64, n % 64
            c, w, h = nl % 8, nl // 16, (nl % 16) // 8
            b = s8[kt, nt * 64 + 8 * c + OCTET_ECHELLE[(w, h)]]
            lu = _decoder_s0e5m3(b) * g_marlin * 2.0 ** -119
            attendu = naturel[n, kt]
            if int(b) == 0:
                assert float(attendu) * facteur * 128 < 2, (n, kt, float(attendu))
            else:
                assert torch.isclose(lu, attendu, rtol=1e-6, atol=0), (n, kt, float(lu), float(attendu))


def test_facteur_et_globale_se_compensent_a_sec():
    """g_marlin·2⁻¹¹⁹ = gs / facteur, exactement (puissances de deux)."""
    gs = torch.tensor([0.0031, 1.7, 12.0])
    for facteur in (1.0, 4.0, 64.0):
        assert torch.equal(MP.traiter_echelle_globale(gs, facteur) * 2.0 ** -119, gs / facteur)


# ---- carte ----------------------------------------------------------------
CARTE = pytest.mark.skipif(not torch.cuda.is_available() or get_extension() is None
                           or not hasattr(get_extension(), "nvfp4_gemv_marlin_gateup"),
                           reason="carte et extension avec nvfp4_gemv_marlin requises")
DEV = "cuda"


def _pile(E, N, K, seed):
    g = torch.Generator().manual_seed(seed)
    ts = [quantize_nvfp4((torch.randn(N, K, generator=g) * 0.05).to(torch.bfloat16)) for _ in range(E)]
    qw = torch.stack([t.qweight for t in ts]).contiguous().to(DEV)
    bs = torch.stack([t.block_scale for t in ts]).contiguous().to(DEV)
    gs = torch.stack([t.global_scale.float().reshape(()) for t in ts]).contiguous().to(DEV)
    w32 = torch.stack([dequantize_nvfp4(t, torch.float32) for t in ts]).to(DEV)
    return qw, bs, gs, w32


def _routage(b, k, E, seed, mode="aleatoire"):
    g = torch.Generator().manual_seed(seed)
    if mode == "un_expert":
        topi = torch.full((b, k), 3); topi[:, 1:] = torch.stack([torch.randperm(E, generator=g)[:k - 1] for _ in range(b)])
    else:
        topi = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(b)])
    eid = topi.reshape(-1).to(torch.int32)
    if mode == "fantomes":
        eid[-2 * k:] = -1
    tok = torch.arange(b, dtype=torch.int32).repeat_interleave(k)
    return eid.to(DEV), tok.to(DEV)


def _hors_par_ligne(y, ref):
    return int(((y.float() - ref).abs() > 2 ** -7 * ref.abs().amax(1, keepdim=True)).sum())


def _marlin(qw, bs, gs):
    if MP.charger(compiler=False) is None:
        pytest.skip("extension Marlin non compilée à sec")
    return MP.preparer_pile(qw, bs, gs)


@CARTE
@pytest.mark.parametrize("mode", ["aleatoire", "un_expert", "fantomes"])
@pytest.mark.parametrize("K,N", [(2048, 768), (768, 2048), (128, 64), (2048, 1536), (1536, 2048)])   # gate/up et down de Coder, une tuile, gate/up et down de GLM (C10, contrôle (b))
def test_b_down_contre_fp32_par_ligne(mode, K, N):
    ext = get_extension()
    E, b, k = 8, 6, 2
    qw, bs, gs, w32 = _pile(E, N, K, 1)
    w, s, g = _marlin(qw, bs, gs)
    eid, tok = _routage(b, k, E, 3, mode)
    x = (torch.randn(b, K, device=DEV) * 0.5).to(torch.bfloat16)
    y = ext.nvfp4_gemv_marlin(w, s, g, eid, tok, x, K, N)
    assert y.shape == (b * k, N)
    ref = torch.stack([x[t].float() @ w32[e].T if e >= 0 else torch.zeros(N, device=DEV)
                       for e, t in zip(eid.tolist(), tok.tolist())])
    if mode == "fantomes":
        assert torch.equal(y[-2 * k:], torch.zeros(2 * k, N, device=DEV))
    assert _hors_par_ligne(y, ref) <= TOL_HORS * ref.numel()
    # le GEMV v1 (pile naturelle) passe le même juge : la référence est jugeable
    y1 = ext.nvfp4_gemv_grouped(qw, bs.view(torch.uint8), gs, eid, tok, x, K)[:, :N]
    assert _hors_par_ligne(y1, ref) <= TOL_HORS * ref.numel()
    # déterminisme : deux appels, même sortie au bit
    assert torch.equal(y, ext.nvfp4_gemv_marlin(w, s, g, eid, tok, x, K, N))


@CARTE
@pytest.mark.parametrize("act", [0, 1])
@pytest.mark.parametrize("K,N", [(2048, 768), (2048, 1536)])                  # Coder ; GLM (C10, contrôle (b))
def test_b_gateup_contre_fp32_par_ligne(act, K, N):
    ext = get_extension()
    E, b, k = 8, 6, 2
    qg, bg, gsg, wg32 = _pile(E, N, K, 11); qu, bu, gsu, wu32 = _pile(E, N, K, 12)
    mg, mu = _marlin(qg, bg, gsg), _marlin(qu, bu, gsu)
    eid, tok = _routage(b, k, E, 5)
    x = (torch.randn(b, K, device=DEV) * 0.5).to(torch.bfloat16)
    y = ext.nvfp4_gemv_marlin_gateup(*mg, *mu, eid, tok, x, K, N, act)
    f = torch.nn.functional.silu if act == 0 else (lambda v: torch.nn.functional.gelu(v, approximate="tanh"))
    ref = torch.stack([f(x[t].float() @ wg32[e].T) * (x[t].float() @ wu32[e].T)
                       for e, t in zip(eid.tolist(), tok.tolist())])
    assert _hors_par_ligne(y, ref) <= TOL_HORS * ref.numel()
    y1 = ext.nvfp4_gemv_grouped_gateup(qg, bg.view(torch.uint8), gsg, qu, bu.view(torch.uint8), gsu, eid, tok, x, K, act)[:, :N]
    assert _hors_par_ligne(y1, ref) <= TOL_HORS * ref.numel()


@CARTE
def test_bras_cassant_echelles_decalees_et_poids_permutes():
    """Un octet d'échelle de décalage, ou les mots d'une tuile permutés :
    le juge doit dire faux — sinon il ne juge rien."""
    ext = get_extension()
    E, b, k, K, N = 4, 4, 2, 768, 2048
    qw, bs, gs, w32 = _pile(E, N, K, 21)
    w, s, g = _marlin(qw, bs, gs)
    eid, tok = _routage(b, k, E, 9)
    x = (torch.randn(b, K, device=DEV) * 0.5).to(torch.bfloat16)
    ref = torch.stack([x[t].float() @ w32[e].T for e, t in zip(eid.tolist(), tok.tolist())])
    assert _hors_par_ligne(ext.nvfp4_gemv_marlin(w, s, g, eid, tok, x, K, N), ref) <= TOL_HORS * ref.numel()
    s_faux = torch.roll(s.view(torch.uint8), 1, dims=2).contiguous().view(torch.float8_e4m3fn)
    assert _hors_par_ligne(ext.nvfp4_gemv_marlin(w, s_faux, g, eid, tok, x, K, N), ref) > TOL_HORS * ref.numel()
    w_faux = torch.roll(w, 1, dims=2).contiguous()
    assert _hors_par_ligne(ext.nvfp4_gemv_marlin(w_faux, s, g, eid, tok, x, K, N), ref) > TOL_HORS * ref.numel()


@CARTE
def test_le_bloc_moe_decode_prend_gemv_marlin(monkeypatch):
    """Intégration : sous ACVRAM_GEMV_LAYOUT=marlin, `_forward_grouped` prend
    `gemv_marlin` (asserté, REGLES § 7) et sa sortie passe le juge fp32 par
    ligne comme celle de `gemv_v1` (témoin gardé)."""
    from conftest import attendre_chemin
    from test_marlin_prefill_p1 import _bloc_moe_jouet
    from acvram.engine import model as MD
    from acvram.engine import moe as MOE_D
    if MP.charger(compiler=False) is None:
        pytest.skip("extension Marlin non compilée à sec")
    E, H, I, top_k, T = 8, 256, 128, 2, 6
    bloc = _bloc_moe_jouet(E, H, I, top_k)
    monkeypatch.setattr(MOE_D, "_GEMV_LAYOUT", "naturel")            # témoin naturel (défaut marlin depuis le 18/09)
    monkeypatch.setattr(MOE_D, "_PREFILL_GROUPED", "groupe")
    assert bloc._try_build_stacks()
    assert bloc._stacks_marlin is None
    torch.manual_seed(T)
    x = (torch.randn(T, H, device=DEV) * 0.5).to(torch.bfloat16)
    topw, topi = torch.topk(torch.softmax(bloc.router(x).float(), -1), top_k, dim=-1)
    topw = (topw / topw.sum(-1, keepdim=True)).float()
    # référence fp32 : experts déquantifiés
    from acvram.quant.nvfp4 import dequantize_nvfp4 as dq
    ref = torch.zeros(T, H, device=DEV)
    for t in range(T):
        for j in range(top_k):
            e = int(topi[t, j]); m = bloc.experts[e]
            wg, wu, wd = (dq(getattr(m, n).qweight, torch.float32).to(DEV) for n in ("gate_proj", "up_proj", "down_proj"))
            a = torch.nn.functional.silu(x[t].float() @ wg.T) * (x[t].float() @ wu.T)
            ref[t] += topw[t, j] * (a.to(torch.bfloat16).float() @ wd.T)
    y1 = bloc._forward_grouped(x, topw, topi.to(torch.int32))
    attendre_chemin(bloc, "gemv_v1")
    monkeypatch.setattr(MOE_D, "_GEMV_LAYOUT", "marlin")
    bloc._stacks_marlin = bloc._construire_marlin(bloc._stacks, bloc._stacks_awq, bloc._stacks_awq.get("hadamard", {}))
    assert bloc._stacks_marlin is not None
    n_v1 = bloc.chemins["gemv_v1"]
    yb = bloc._forward_grouped(x, topw, topi.to(torch.int32))
    attendre_chemin(bloc, "gemv_marlin")
    # un seul chemin par appel : la première version prenait gemv_marlin PUIS
    # recalculait gate/up par v1 (chaîne if/elif coupée en deux — poste3, 18/09)
    assert bloc.chemins["gemv_v1"] == n_v1, bloc.chemins
    h1, hb = _hors_par_ligne(y1, ref), _hors_par_ligne(yb, ref)
    assert h1 <= TOL_HORS * ref.numel() and hb <= TOL_HORS * ref.numel(), (h1, hb)
    # disposition unique : pile naturelle rendue, le chemin marlin reste le seul
    # — même sous un témoin naturel demandé après coup (rien d'autre à lire)
    bloc._liberer_pile_naturelle()
    assert bloc._stacks["gate_proj"][1] is None and bloc.experts_layout in ("marlin", "marlin-w13")
    monkeypatch.setattr(MOE_D, "_GEMV_LAYOUT", "naturel")
    n_m = bloc.chemins["gemv_marlin"]
    yu = bloc._forward_grouped(x, topw, topi.to(torch.int32))
    attendre_chemin(bloc, "gemv_marlin", avant=n_m)
    assert torch.equal(yu, yb)


@CARTE
def test_splitk_vaut_un_au_defaut():
    """Pièce 70 (0.6.37) : ACVRAM_GEMV_SPLITK=1 est le défaut (S auto).
    Sans variable, S est calculé selon la forme : gate/up b=1 → S=4,
    down b=1 → S=2, b=12 → S=1 (grille ≥ MB_BLOCS_MIN=384). Ce test casse si
    le défaut change de sortie."""
    ext = get_extension()
    assert "ACVRAM_GEMV_SPLITK" not in os.environ or os.environ["ACVRAM_GEMV_SPLITK"] == "1"
    assert ext.nvfp4_gemv_marlin_splitk(2048, 768, 8) == 4      # gate/up b=1 : NT=12, G=8, S auto=4
    assert ext.nvfp4_gemv_marlin_splitk(768, 2048, 8) == 2      # down b=1 : NT=32, G=8, S auto=2
    assert ext.nvfp4_gemv_marlin_splitk(2048, 768, 96) == 1     # b=12 : grille 1152 ≥ 384, S=1


@CARTE
def test_splitk_opt_in_sous_processus():
    """ACVRAM_GEMV_SPLITK=1 : S auto (4 à b=1 gate/up, 2 à b=2, 1 dès b=4) et le
    noyau passe le juge sur ces lots — dans un PROCESSUS à part, la variable
    étant lue une fois par processus (comme GROUPED_RPW). Le verrou est celui
    de cette session (ACVRAM_CARTE_TENUE hérité)."""
    import subprocess
    env = dict(os.environ, ACVRAM_GEMV_SPLITK="1", ACVRAM_CARTE_TENUE=os.environ.get("ACVRAM_CARTE_TENUE", str(os.getpid())))
    r = subprocess.run([sys.executable, "-m", "pytest", __file__, "-q", "-p", "no:cacheprovider",
                        "-k", "_splitk_auto_"], env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert "passed" in r.stdout


@CARTE
@pytest.mark.skipif(os.environ.get("ACVRAM_GEMV_SPLITK") != "1", reason="opt-in : lancé par test_splitk_opt_in_sous_processus")
@pytest.mark.parametrize("b,mode", [(1, "aleatoire"), (2, "aleatoire"), (4, "aleatoire"), (2, "fantomes")])
def test_b_splitk_auto_par_lot(b, mode):
    """Split-K par lot (poste b=1, verdict-profil-b1-b-18-09) : gate/up Coder
    (K=2048, N=768, top_k=8) à b=1 → 96 blocs → S=4 ; b=2 → S=2 ; b=4 → S=1
    (chemin d'origine). Chaque S passe le juge fp32 par ligne, les fantômes
    rendent zéro sous S>1, et deux appels sont identiques au bit (réduction
    du dernier bloc dans l'ordre z fixe)."""
    ext = get_extension()
    attendu = {1: 4, 2: 2, 4: 1}[b]
    assert ext.nvfp4_gemv_marlin_splitk(2048, 768, b * 8) == attendu
    E, k, K, N = 16, 8, 2048, 768
    qg, bg, gsg, wg32 = _pile(E, N, K, 31); qu, bu, gsu, wu32 = _pile(E, N, K, 32)
    mg, mu = _marlin(qg, bg, gsg), _marlin(qu, bu, gsu)
    eid, tok = _routage(b + (2 if mode == "fantomes" else 0), k, E, 7, mode)
    x = (torch.randn(int(tok.max()) + 1, K, device=DEV) * 0.5).to(torch.bfloat16)
    y = ext.nvfp4_gemv_marlin_gateup(*mg, *mu, eid, tok, x, K, N, 0)
    f = torch.nn.functional.silu
    ref = torch.stack([f(x[t].float() @ wg32[e].T) * (x[t].float() @ wu32[e].T) if e >= 0
                       else torch.zeros(N, device=DEV) for e, t in zip(eid.tolist(), tok.tolist())])
    if mode == "fantomes":
        assert torch.equal(y[-2 * k:], torch.zeros(2 * k, N, device=DEV))
    assert _hors_par_ligne(y, ref) <= TOL_HORS * ref.numel()
    assert torch.equal(y, ext.nvfp4_gemv_marlin_gateup(*mg, *mu, eid, tok, x, K, N, 0))
    # down (K=768, N=2048) au même lot : S=2 à b=1 (256 blocs), S=1 dès b=2
    qd, bd, gsd, wd32 = _pile(E, K, N, 33)
    md = _marlin(qd, bd, gsd)
    xd = (torch.randn(int(tok.max()) + 1, N, device=DEV) * 0.5).to(torch.bfloat16)
    yd = ext.nvfp4_gemv_marlin(*md, eid, tok, xd, N, K)
    refd = torch.stack([xd[t].float() @ wd32[e].T if e >= 0 else torch.zeros(K, device=DEV)
                        for e, t in zip(eid.tolist(), tok.tolist())])
    assert _hors_par_ligne(yd, refd) <= TOL_HORS * refd.numel()
    assert torch.equal(yd, ext.nvfp4_gemv_marlin(*md, eid, tok, xd, N, K))


# ---------------------------------------------------------------------------
# Pièce 47 : l'échelle AWQ par expert portée par le GEMV (`xscale`), AU BIT
# contre la division en torch qu'elle remplace (poste1-piece42-glue-22-09 :
# 8 lancements et 0,473 ms/pas à b=12). Le juge n'est pas une tolérance mais
# `torch.equal` : une division faite dans un autre ordre ou sans l'arrondi bf16
# de PyTorch le fait échouer — c'est ce que ce test doit pouvoir rendre.
# ---------------------------------------------------------------------------
PORTE_ECHELLE = pytest.mark.skipif(
    get_extension() is None or "xscale" not in (getattr(getattr(get_extension(), "nvfp4_gemv_marlin", None),
                                                        "__doc__", "") or ""),
    reason="extension compilée sans `xscale` (pièce 47)")


def _table_awq(E, K, seed):
    """Comme `MoEBlock._table_pile` : bf16, [E, K], jamais nulle."""
    g = torch.Generator().manual_seed(seed)
    return (0.5 + torch.rand(E, K, generator=g)).to(torch.bfloat16).to(DEV)


@CARTE
@PORTE_ECHELLE
@pytest.mark.parametrize("K,N", [(2048, 768), (768, 2048)])
def test_xscale_down_au_bit_contre_la_division_torch(K, N):
    """`act` arrive en fp32 : moe.py:1296 l'arrondit en bf16, divise, et
    reconvertit — le noyau doit faire les deux arrondis dans le même ordre."""
    ext = get_extension()
    E, b, k = 8, 6, 2
    qw, bs, gs, _ = _pile(E, N, K, 21)
    w, s, g = _marlin(qw, bs, gs)
    eid, tok = _routage(b, k, E, 23)
    table = _table_awq(E, K, 24)
    act = torch.randn(b * k, K, device=DEV) * 0.5                            # fp32, comme act(gate)·up
    seq = torch.arange(b * k, device=DEV, dtype=torch.int32)
    # chemin actuel : gather + division + cast en torch, puis GEMV sans échelle
    ref_x = (act.to(torch.bfloat16) / table[eid.long(), :K]).to(act.dtype)
    ref = ext.nvfp4_gemv_marlin(w, s, g, eid, seq, ref_x.contiguous(), K, N)
    # chemin fusionné : le noyau lit la table
    y = ext.nvfp4_gemv_marlin(w, s, g, eid, seq, act.contiguous(), K, N, table)
    assert torch.equal(y, ref)
    # et l'échelle n'est pas ignorée : sans elle la sortie diffère
    assert not torch.equal(y, ext.nvfp4_gemv_marlin(w, s, g, eid, seq, act.contiguous(), K, N))


@CARTE
@PORTE_ECHELLE
@pytest.mark.parametrize("act_code", [0, 1])
def test_xscale_gateup_au_bit_contre_la_division_torch(act_code):
    """gate et up partagent leur table (`not distinct`) : le noyau n'a qu'un x.
    Ici `x` est bf16 et le GEMV lit les vrais index de jetons — le gather
    `x[tok]` de moe.py:1199 disparaît aussi."""
    ext = get_extension()
    K, N, E, b, k = 2048, 768, 8, 6, 2
    qg, bg, gsg, _ = _pile(E, N, K, 31); qu, bu, gsu, _ = _pile(E, N, K, 32)
    mg, mu = _marlin(qg, bg, gsg), _marlin(qu, bu, gsu)
    eid, tok = _routage(b, k, E, 33)
    table = _table_awq(E, K, 34)
    x = (torch.randn(b, K, device=DEV) * 0.5).to(torch.bfloat16)
    seq = torch.arange(b * k, device=DEV, dtype=torch.int32)
    ref_x = (x[tok.long()].to(torch.bfloat16) / table[eid.long(), :K]).to(x.dtype)
    ref = ext.nvfp4_gemv_marlin_gateup(mg[0], mg[1], mg[2], mu[0], mu[1], mu[2],
                                       eid, seq, ref_x.contiguous(), K, N, act_code)
    y = ext.nvfp4_gemv_marlin_gateup(mg[0], mg[1], mg[2], mu[0], mu[1], mu[2],
                                     eid, tok, x.contiguous(), K, N, act_code, table)
    assert torch.equal(y, ref)
    assert not torch.equal(y, ext.nvfp4_gemv_marlin_gateup(
        mg[0], mg[1], mg[2], mu[0], mu[1], mu[2], eid, tok, x.contiguous(), K, N, act_code))


@CARTE
@PORTE_ECHELLE
def test_xscale_table_plus_large_que_K():
    """La table est en `padded_in` (K_in ≥ K) et moe.py la tranche `[:, :K]` :
    le noyau doit lire la ligne avec le pas de la table, pas avec K."""
    ext = get_extension()
    K, N, E, b, k = 768, 2048, 8, 4, 2
    qw, bs, gs, _ = _pile(E, N, K, 41)
    w, s, g = _marlin(qw, bs, gs)
    eid, tok = _routage(b, k, E, 43)
    large = _table_awq(E, K + 256, 44)
    x = (torch.randn(b, K, device=DEV) * 0.5).to(torch.bfloat16)
    seq = torch.arange(b * k, device=DEV, dtype=torch.int32)
    ref_x = (x[tok.long()] / large[eid.long(), :K]).to(x.dtype)
    ref = ext.nvfp4_gemv_marlin(w, s, g, eid, seq, ref_x.contiguous(), K, N)
    assert torch.equal(ext.nvfp4_gemv_marlin(w, s, g, eid, tok, x.contiguous(), K, N, large), ref)


def test_extension_sans_xscale_garde_la_division_torch_a_sec():
    """Garde de compatibilité : sur une extension d'avant la pièce 47, le bloc
    MoE doit refuser la fusion — sinon il sauterait la division sans que
    personne ne la fasse (sorties non échelonnées, en silence)."""
    from acvram.engine.moe import _gemv_marlin_porte_echelle

    class _Vieille:
        def nvfp4_gemv_marlin(self):
            """nvfp4_gemv_marlin(w, s, g, expert_ids, token_ids, x, K, N)"""

        def nvfp4_gemv_marlin_gateup(self):
            """nvfp4_gemv_marlin_gateup(wg, sg, gg, wu, su, gu, expert_ids, token_ids, x, K, N, act)"""

    class _Neuve:
        def nvfp4_gemv_marlin(self):
            """nvfp4_gemv_marlin(w, s, g, expert_ids, token_ids, x, K, N, xscale)"""

        def nvfp4_gemv_marlin_gateup(self):
            """nvfp4_gemv_marlin_gateup(wg, sg, gg, wu, su, gu, expert_ids, token_ids, x, K, N, act, xscale)"""

    assert _gemv_marlin_porte_echelle(_Vieille()) is False
    assert _gemv_marlin_porte_echelle(_Neuve()) is True
