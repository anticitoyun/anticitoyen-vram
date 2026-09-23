"""P1 — GEMM groupée classe Marlin au préfill (`ACVRAM_PREFILL_GROUPED=marlin`,
port vLLM v0.29.0). À sec : le Plan compte la seconde disposition ; le
régime la nomme (`experts_layout` marlin | naturel). Sur carte : la sortie de
`MoEBlock._forward_prefill_grouped` sous « marlin » égale celle de « groupe »
(B0) à 2⁻⁷ × Σ|x·w| ; bras cassant : échelle de bloc décalée d'un rang → rouge."""
import os

import pytest
import torch

from acvram.engine import loader as LD
from acvram.engine.config import ModelSpec
from acvram.memory.tiering import LayerPlacement, Plan, Tier

GIB = 2 ** 30


def _spec():
    return ModelSpec(name="c30", architecture="Qwen3MoeForCausalLM", hidden_size=2048, intermediate_size=6144,
                     num_layers=48, num_attention_heads=32, num_key_value_heads=4, vocab_size=151936,
                     max_position_embeddings=32768, head_dim=128, num_experts=128, num_experts_per_tok=8,
                     moe_intermediate_size=768)


def _plan():
    tier = Tier(name="gpu-test", kind="gpu", device_index=0, capacity=30 * GIB,
                weight_format="nvfp4", kv_format="int8", read_bandwidth=1790.0, link_bandwidth=21.0)
    couches = [LayerPlacement(index=i, exec_device="gpu-test", attn_storage="gpu-test", mlp_storage="gpu-test",
                              fmt="nvfp4", attn_bytes=int(0.02 * GIB), mlp_bytes=int(0.3 * GIB),
                              mlp_active_bytes=0, is_moe=True) for i in range(48)]
    return Plan(model="synthetique", tiers=[tier], layers=couches)


def test_le_plan_ne_reserve_pas_de_seconde_disposition(monkeypatch):
    """Disposition unique (poste7-p1-disposition-unique-18-09) : la réserve du
    Plan est la même sous marlin et sous naturel — Σ mlp_bytes × 1, jamais
    × 2. Bras cassant : réintroduire « reserve += Σ mlp_bytes » sous
    PREFILL_GROUPED=marlin dans loader._reserve_prefill (5a0f6c4) rend
    l'écart égal à Σ (14,4 Gio ici) et ce test rouge."""
    spec, plan = _spec(), _plan()
    somme = sum(l.mlp_bytes for l in plan.layers)
    monkeypatch.delenv("ACVRAM_PREFILL_GROUPED", raising=False)
    monkeypatch.delenv("ACVRAM_GEMV_LAYOUT", raising=False)
    naturel = LD._reserve_prefill(spec, 2048, {}, plan)
    monkeypatch.setenv("ACVRAM_PREFILL_GROUPED", "marlin")
    monkeypatch.setenv("ACVRAM_GEMV_LAYOUT", "marlin")
    marlin = LD._reserve_prefill(spec, 2048, {}, plan)
    assert marlin == naturel, (marlin - naturel, somme)
    assert marlin < somme                                          # la réserve ne contient aucune copie des experts


def test_la_pile_naturelle_est_rendue_apres_le_repack_a_sec(monkeypatch):
    """`_try_build_stacks` : quand `_construire_marlin` rend des piles, la pile
    NVFP4 est libérée (qw/bs None, experts au gabarit vide, experts_layout
    marlin) ; quand il rend None, elle reste (témoin naturel). Sans carte :
    bloc jouet sur CPU, repack remplacé par un stub."""
    from acvram.engine.model import MoEBlock
    bloc = _bloc_moe_jouet(4, 128, 64, 2, dev="cpu")
    monkeypatch.setattr(MoEBlock, "_construire_marlin", lambda self, p, a, h: None)   # extension absente → naturel
    assert bloc._try_build_stacks()
    assert bloc._stacks["gate_proj"][1] is not None and bloc.experts[0].gate_proj.qweight.qweight.numel() > 0
    assert getattr(bloc, "experts_layout", "naturel") == "naturel"
    bloc2 = _bloc_moe_jouet(4, 128, 64, 2, dev="cpu")
    monkeypatch.setattr(MoEBlock, "_construire_marlin",
                        lambda self, p, a, h: {n: ("w", "s", "g", p[n][4], p[n][5]) for n in p})
    assert bloc2._try_build_stacks()
    for n in ("gate_proj", "up_proj", "down_proj"):
        assert bloc2._stacks[n][1] is None and bloc2._stacks[n][2] is None
        assert all(getattr(e, n).qweight.qweight.numel() == 0 for e in bloc2.experts)
        assert getattr(bloc2.experts[0], n).qweight.shape == getattr(bloc.experts[0], n).qweight.shape
    assert bloc2.experts_layout in ("marlin", "marlin-w13")


def test_le_regime_nomme_la_disposition(converted):
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    engine = Engine(load_model(converted, dtype=torch.float32, device_override="cpu"), None,
                    max_batch_size=2, max_model_len=256)
    # 790c995c (couverture Marlin PAR COUCHE) : un modèle sans couche MoE dit « aucun »,
    # pas « naturel » — le jouet `converted` est dense ; la disposition « naturel » se
    # nomme sur un MoE (test_glue_compact / marlin_prefill sur carte)
    assert "experts_layout=aucun" in engine.regime_ligne()


CARTE = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise (noyaux Marlin)")


def _bloc_moe_jouet(E=8, H=256, I=128, top_k=2, dev="cuda", awq=False):
    """Un MoEBlock à experts NVFP4 aléatoires, construit comme dans
    test_moe_hadamard_pile (QuantLinear → to_device, MLP, routeur) ; ``awq`` :
    une échelle AWQ par expert (ChannelScaler, table [E, K] côté activation
    comme le Coder classé 302025e), gate/up partagée, down distincte."""
    from acvram.engine.layers import QuantLinear
    from acvram.engine.model import MLP, MoEBlock
    from acvram.quant.calibrate import ChannelScaler
    from acvram.quant.nvfp4 import quantize_nvfp4
    from acvram.quant.formats import _quantize_int8

    def lin(o, i, graine, sc=None):
        g = torch.Generator().manual_seed(graine)
        w = (torch.randn(o, i, generator=g) * 0.05).to(torch.bfloat16)
        return QuantLinear(quantize_nvfp4(w), out_features=o, in_features=i, scaler=sc).to_device(dev)

    def scaler(i, graine):
        if not awq:
            return None
        g = torch.Generator().manual_seed(graine)
        return ChannelScaler((0.5 + torch.rand(i, generator=g) * 1.5).to(torch.bfloat16), 0)
    experts = []
    for e in range(E):
        s_x, s_d = scaler(H, 100 + e), scaler(I, 200 + e)
        experts.append(MLP(lin(I, H, 10 * e + 1, s_x), lin(I, H, 10 * e + 2, s_x), lin(H, I, 10 * e + 3, s_d)))
    gen = torch.Generator().manual_seed(5)
    routeur = QuantLinear(_quantize_int8((torch.randn(E, H, generator=gen) * 0.02).to(torch.bfloat16), 128),
                          out_features=E, in_features=H).to_device(dev)
    return MoEBlock(routeur, experts, top_k).to(dev)


def _reference_fp32(bloc, x, topw, topi):
    """Le juge : tout en fp32 sur la déquantification fp32 (codes × bloc ×
    globale sans arrondi bf16) — chaque chemin se compare À LUI, jamais l'un
    à l'autre (deux approximations bf16 se comparent à 2⁻⁷ de près ~3 % du
    temps sous un critère relatif près de zéro : hypothèse (a) de poste7,
    tranchée à sec par émulation). L'activation mise à l'échelle AWQ est LA
    MÊME que celle que le moteur donne aux GEMM : la table bf16 `_stacks_awq`
    (construite par `_try_build_stacks`, à appeler avant), division en bf16
    (`xs / awq_g[e_sorted]`), et pour down `act / awq_d[e]` arrondi une fois
    en bf16 (comme `moe_act`) — la référence ne recalcule pas l'AWQ à part
    (poste7, 18/09 : « la référence passe son propre critère »)."""
    from acvram.quant.nvfp4 import dequantize_nvfp4
    N, H = x.shape
    awq = getattr(bloc, "_stacks_awq", {}) or {}
    tg, tu, td = awq.get("gate_proj"), awq.get("up_proj"), awq.get("down_proj")
    ref = torch.zeros(N, H, dtype=torch.float32, device=x.device)
    for e, ex in enumerate(bloc.experts):
        wg = dequantize_nvfp4(ex.gate_proj.qweight, torch.float32).to(x.device)
        wu = dequantize_nvfp4(ex.up_proj.qweight, torch.float32).to(x.device)
        wd = dequantize_nvfp4(ex.down_proj.qweight, torch.float32).to(x.device)
        for j in range(topi.shape[1]):
            sel = topi[:, j] == e
            if not bool(sel.any()):
                continue
            h = x[sel]                                                     # bf16, comme xs
            hg = (h / tg[e, :H]).float() if tg is not None else h.float()  # division bf16 → bf16, puis fp32
            hu = (h / tu[e, :H]).float() if tu is not None else h.float()
            a = torch.nn.functional.silu(hg @ wg.T) * (hu @ wu.T)          # fp32
            if td is not None:
                a = (a / td[e, : a.shape[1]].float()).to(torch.bfloat16).float()   # un arrondi, comme moe_act
            else:
                a = a.to(torch.bfloat16).float()                           # l'act est bf16 à l'entrée de down
            ref[sel] += (a @ wd.T) * topw[sel, j:j + 1].float()
    return ref


def _hors_par_ligne(y, ref):
    """|Δ| > 2⁻⁷ · max|y| de la ligne — le critère retenu."""
    return int(((y.float() - ref).abs() > 2 ** -7 * ref.abs().amax(1, keepdim=True)).sum())


# Part tolérée hors 2⁻⁷·max|y| par ligne : B0 lui-même (déquant arrondie en
# bf16) en laisse 3·10⁻⁵ sans AWQ et 1,2·10⁻⁴ avec une table AWQ large (émulé
# à sec, T = 1 024) — la référence doit passer son propre critère (poste7), le
# seuil est posé au-dessus de ce que B0 rend, et Marlin doit faire ≤ B0.
TOL_HORS = 5e-4


@CARTE
@pytest.mark.parametrize("awq", [False, True], ids=["sans_awq", "awq_par_expert"])
@pytest.mark.parametrize("T", [96, 1024])
def test_marlin_et_groupe_contre_fp32_et_le_bras_casse(monkeypatch, awq, T):
    """Chaque chemin contre la référence fp32 (jamais l'un contre l'autre),
    critère 2⁻⁷ · max|y| par ligne, le CHEMIN asserté avant toute
    comparaison (conftest.attendre_chemin : à T = 96 la branche `direct`
    passait devant Marlin et un test comparait sans l'atteindre — poste3
    5803c1e). Avec la table AWQ par expert (le Coder classé) : l'échelle est
    côté activation, Marlin la voit comme groupe. Bras cassant : échelles de
    bloc de gate décalées d'un rang (vue uint8) → rouge."""
    from conftest import attendre_chemin
    from acvram.engine import model as MD
    from acvram.engine import moe as MOE_D
    from acvram.kernels import marlin_port as MP
    if MP.charger(compiler=False) is None:
        pytest.skip("extension Marlin non compilée à sec")
    E, H, I, top_k = 8, 256, 128, 2
    bloc = _bloc_moe_jouet(E, H, I, top_k, awq=awq)
    torch.manual_seed(T)
    x = (torch.randn(T, H, device="cuda") * 0.5).to(torch.bfloat16)
    logits = bloc.router(x).float()
    topw, topi = torch.topk(torch.softmax(logits, -1), top_k, dim=-1)
    topw = topw / topw.sum(-1, keepdim=True)
    topi32 = topi.to(torch.int32)
    monkeypatch.setattr(MOE_D, "_PREFILL_GROUPED", "groupe")
    monkeypatch.setattr(MOE_D, "_GEMV_LAYOUT", "naturel")            # témoin : pile naturelle gardée (défaut marlin depuis le 18/09)
    monkeypatch.setattr(MOE_D, "_MOE_MMA", False)                     # sinon la MMA FP4 prime sur « groupe » (T4 20/09 : chemin pris mma)
    # à petit T (96 : 24 lignes par expert ≤ _MOE_GEMM_MAX 48) `direct` prend
    # le pas sur `groupe` : le témoin serait inatteignable — on force groupe
    monkeypatch.setattr(MOE_D, "_MOE_GEMM_MAX", 0)
    assert bloc._try_build_stacks()
    assert (bloc._stacks_awq.get("gate_proj") is not None) == awq
    ref = _reference_fp32(bloc, x, topw, topi)                    # après les piles : consomme la table AWQ du moteur
    y_groupe = bloc._forward_prefill_grouped(x, topw, topi32)
    attendre_chemin(bloc, "groupe")
    hors_groupe = _hors_par_ligne(y_groupe, ref)
    assert hors_groupe <= TOL_HORS * ref.numel(), hors_groupe
    monkeypatch.setattr(MOE_D, "_PREFILL_GROUPED", "marlin")
    monkeypatch.setattr(MOE_D, "_MOE_W13", False)   # le bras cassant décale les échelles de la pile gate SÉPARÉE (82 ter : w13 par défaut)
    bloc._stacks_marlin = bloc._construire_marlin(bloc._stacks, bloc._stacks_awq, bloc._stacks_awq.get("hadamard", {}))
    assert bloc._stacks_marlin is not None
    n0 = bloc.chemins.get("marlin", 0)
    y_marlin = bloc._forward_prefill_grouped(x, topw, topi32)
    attendre_chemin(bloc, "marlin", avant=n0)
    hors_marlin = _hors_par_ligne(y_marlin, ref)
    assert hors_marlin <= TOL_HORS * ref.numel() and hors_marlin <= max(hors_groupe, 1), (hors_marlin, hors_groupe)
    # bras cassant : échelles de bloc de gate décalées d'un rang (roll sur la vue uint8 :
    # roll_cuda n'existe pas pour Float8_e4m3fn — poste3 5803c1e)
    w, sc, g, k, m = bloc._stacks_marlin["gate_proj"]
    sc_faux = torch.roll(sc.view(torch.uint8), 1, dims=2).contiguous().view(torch.float8_e4m3fn)
    bloc._stacks_marlin["gate_proj"] = (w, sc_faux, g, k, m)
    y_faux = bloc._forward_prefill_grouped(x, topw, topi32)
    attendre_chemin(bloc, "marlin", avant=n0 + 1)
    assert _hors_par_ligne(y_faux, ref) > TOL_HORS * ref.numel()


def test_le_critere_a_sec_deux_approximations_bf16_ne_se_comparent_pas_entre_elles():
    """Hypothèse (a) de poste7, tranchée par émulation torch (CPU) : B0
    (déquant bf16 : code × bloc × globale arrondi en bf16) et Marlin (code ×
    bloc exact en bf16, globale fp32 dans l'épilogue) contre fp32, sous le
    critère relatif 2⁻⁷·(moy|y| + |y|) : B0 ≈ 2,4 % hors, Marlin ≈ 0,6 %,
    Marlin-contre-B0 ≈ 3 % (ce que poste3 a mesuré, 3,7-4,3 %) ; sous
    2⁻⁷·max|y| par ligne : ≤ 10⁻⁴ pour les deux contre fp32."""
    from acvram.quant.nvfp4 import dequantize_nvfp4, quantize_nvfp4
    torch.manual_seed(0)
    E, H, I, top_k, N = 8, 256, 128, 2, 256
    W = [tuple(quantize_nvfp4((torch.randn(o, i) * 0.05).to(torch.bfloat16)) for o, i in ((I, H), (I, H), (H, I)))
         for _ in range(E)]
    x = (torch.randn(N, H) * 0.5).to(torch.bfloat16)
    topi = torch.stack([torch.randperm(E)[:top_k] for _ in range(N)]); topw = torch.rand(N, top_k); topw = topw / topw.sum(-1, keepdim=True)
    ref = torch.zeros(N, H); b0 = torch.zeros(N, H); marl = torch.zeros(N, H)
    for e in range(E):
        g32, u32, d32 = (dequantize_nvfp4(t, torch.float32) for t in W[e])
        g16, u16, d16 = (dequantize_nvfp4(t, torch.bfloat16) for t in W[e])
        gm, um, dm = ((dequantize_nvfp4(t, torch.float32) / t.global_scale.float()).to(torch.bfloat16).float() * t.global_scale.float() for t in W[e])
        for j in range(top_k):
            sel = topi[:, j] == e
            h = x[sel]
            a = torch.nn.functional.silu(h.float() @ g32.T) * (h.float() @ u32.T); ref[sel] += (a @ d32.T) * topw[sel, j:j + 1]
            ab = (torch.nn.functional.silu((h @ g16.T).float()) * (h @ u16.T).float()).to(torch.bfloat16)
            b0[sel] += (ab @ d16.T).float() * topw[sel, j:j + 1]
            aq = (torch.nn.functional.silu((h.float() @ gm.T).to(torch.bfloat16).float()) * (h.float() @ um.T).to(torch.bfloat16).float()).to(torch.bfloat16)
            marl[sel] += (aq.float() @ dm.T).to(torch.bfloat16).float() * topw[sel, j:j + 1]
    poste3 = lambda y, r: ((y - r).abs() > 2 ** -7 * (r.abs().mean() + r.abs())).float().mean().item()
    assert poste3(b0, ref) > 0.01 and poste3(marl, b0) > 0.01, (poste3(b0, ref), poste3(marl, b0))   # le critère relatif condamne B0 lui-même
    assert _hors_par_ligne(b0, ref) <= TOL_HORS * ref.numel() and _hors_par_ligne(marl, ref) <= TOL_HORS * ref.numel()   # le critère par ligne tient pour les deux
    assert (marl - ref).norm() <= (b0 - ref).norm()                                                # Marlin n'est pas moins exact que B0


def _b0_emule(bloc, x, topw, topi):
    """Émulation torch du chemin « groupe » (B0) : déquant bf16 (code × bloc ×
    globale arrondi en bf16), GEMM bf16 à accumulation fp32, act fp32 → bf16
    (avec la division AWQ de down avant l'arrondi, comme moe_act) — sur la
    MÊME activation mise à l'échelle que le moteur (table bf16)."""
    from acvram.quant.nvfp4 import dequantize_nvfp4
    N, H = x.shape
    awq = getattr(bloc, "_stacks_awq", {}) or {}
    tg, tu, td = awq.get("gate_proj"), awq.get("up_proj"), awq.get("down_proj")
    out = torch.zeros(N, H, dtype=torch.float32)
    for e, ex in enumerate(bloc.experts):
        wg, wu, wd = (dequantize_nvfp4(getattr(ex, n).qweight, torch.bfloat16) for n in ("gate_proj", "up_proj", "down_proj"))
        for j in range(topi.shape[1]):
            sel = topi[:, j] == e
            if not bool(sel.any()):
                continue
            h = x[sel]
            hg = (h / tg[e, :H]) if tg is not None else h
            hu = (h / tu[e, :H]) if tu is not None else h
            g, u = (hg @ wg.T).float(), (hu @ wu.T).float()
            a = torch.nn.functional.silu(g) * u
            a = ((a / td[e, : a.shape[1]].float()) if td is not None else a).to(torch.bfloat16)
            out[sel] += (a @ wd.T).float() * topw[sel, j:j + 1].float()
    return out


@pytest.mark.parametrize("awq", [False, True], ids=["sans_awq", "awq_par_expert"])
@pytest.mark.parametrize("T", [96, 1024])
def test_a_sec_la_reference_passe_son_propre_critere_avec_awq(monkeypatch, awq, T):
    """Les 4 cas de la carte, émulés à sec (poste7 : « 4/4 verts avant que poste3
    ne reprenne ») : B0 émulé sur la même activation AWQ que le moteur passe
    le critère par ligne contre `_reference_fp32` — la référence est juste
    avec AWQ, le critère tient."""
    from acvram.engine import model as MD
    from acvram.engine import moe as MOE_D
    E, H, I, top_k = 8, 256, 128, 2
    bloc = _bloc_moe_jouet(E, H, I, top_k, dev="cpu", awq=awq)
    torch.manual_seed(T)
    x = (torch.randn(T, H) * 0.5).to(torch.bfloat16)
    logits = bloc.router(x).float()
    topw, topi = torch.topk(torch.softmax(logits, -1), top_k, dim=-1)
    topw = topw / topw.sum(-1, keepdim=True)
    monkeypatch.setattr(MOE_D, "_PREFILL_GROUPED", "groupe")
    with torch.no_grad():
        assert bloc._try_build_stacks()
        assert (bloc._stacks_awq.get("gate_proj") is not None) == awq
        ref = _reference_fp32(bloc, x, topw, topi)
        b0 = _b0_emule(bloc, x, topw, topi)
    hors = _hors_par_ligne(b0, ref)
    assert hors <= TOL_HORS * ref.numel(), (hors, ref.numel())
