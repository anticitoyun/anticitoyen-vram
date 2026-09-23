"""Pièce 123 (23/09) : tables AWQ des experts portées par le chemin tensor-core.

(a) `moe_aligner_petit_xs` : mêmes tampons que `moe_aligner_petit`, et xs = bf16(x[g / top_k] / s[e(g)]) au bit du
    torch (division IEEE, fantômes → expert 0), table plus large que K ; carte requise.
(b) `gemm_experts_tensor` avec xs du noyau = avec xs du torch (extension sans `moe_aligner_petit_xs`), au bit, et
    différent du même appel sans tables (les tables sont appliquées) ; carte requise.
(c) `forme_tensor_refus` n oppose plus les tables non unité quand l aligneur xs existe (sauf gate/up distinctes) ;
    à sec — casse si le repli « tables AWQ non unité » revient.
(d) témoin glue A4 (fusion=False, xs en torch) à sec, sur un port Marlin simulé en torch : d = référence écrite ici,
    avec et sans w13 — casse si le chemin tensor lit x au lieu de xs, ou si le témoin oublie la division.
Bras cassants : `scratchpad/poste1-123-24-09/casser-123.sh`."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

carte = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")


def _eid(t, k, E, graine, fantomes=True):
    from test_moe_tensor_glue_fusee import _eid as eid_glue
    return eid_glue(t, k, E, graine, fantomes)


def _table(E, K, graine, dev="cpu"):
    g = torch.Generator().manual_seed(graine)
    return (0.5 + 1.5 * torch.rand(E, K, generator=g)).to(torch.bfloat16).to(dev)


def _xs_torch(x, eid, table, k):
    return (x.repeat_interleave(k, dim=0).to(torch.bfloat16) / table[eid.long().clamp(min=0), :x.shape[1]]).to(torch.bfloat16)


# ---------------------------------------------------------------- (c) à sec

def test_forme_refus_tables_awq_fondues():
    from acvram.engine.moe import forme_tensor_refus
    assert forme_tensor_refus({"nvfp4"}, 2048, 768, 2048, 128, False, awq_fondue=True) == ""
    assert "moe_aligner_petit_xs" in forme_tensor_refus({"nvfp4"}, 2048, 768, 2048, 128, False)
    assert "distinctes" in forme_tensor_refus({"nvfp4"}, 2048, 768, 2048, 128, False, awq_fondue=True, awq_distinct=True)
    assert "Hadamard" in forme_tensor_refus({"nvfp4"}, 2048, 768, 2048, 128, False, 512, awq_fondue=True)
    assert forme_tensor_refus({"nvfp4"}, 2048, 768, 2048, 128, True, awq_fondue=False) == ""


# ---------------------------------------------------------------- (d) à sec : port Marlin simulé

class _PortSimule:
    """gemm_moe = une ligne par paire, c[p] = a[p // top_k] @ W[e(p)] (s_ids = expert de chaque paire, rendu par
    l aligneur simulé) ; vérifie size_m · top_k = paires, comme le lit le vrai noyau."""

    @staticmethod
    def charger(compiler=False):
        return None

    @staticmethod
    def choisir_block_size(M, top_k, E):
        return 8

    @staticmethod
    def aligner_blocs_capturable(eid_al, bloc, E, tampons):
        return eid_al, None, None

    @staticmethod
    def gemm_moe(a, w, s, g, sorted_ids, expert_ids, num_post, topk_weights, block_size, top_k, size_m, size_n,
                 size_k, workspace, mul_topk_weights=False, c=None):
        G = sorted_ids.numel()
        assert a.shape[0] == size_m and size_m * top_k == G, (a.shape, size_m, top_k, G)
        lignes = a[torch.arange(G) // top_k].float()
        c.copy_(torch.bmm(lignes.unsqueeze(1), w[sorted_ids.long()].float()).squeeze(1).to(torch.bfloat16))
        return c


def _act_simulee(g, u, m_gate, k_down, code, awq=None, e_sorted=None, *reste):
    v = (F.silu(g[:, :m_gate].float()) * u[:, :m_gate].float()).to(torch.bfloat16)
    if awq is not None:
        v = (v.float() / awq[e_sorted.long().clamp(min=0), :m_gate].float()).to(torch.bfloat16)
    return v


def _reference(x, eid, k, Wg, Wu, Wd, tg, td):
    xs = _xs_torch(x, eid, tg, k).float()
    e = eid.long().clamp(min=0)
    g = torch.bmm(xs.unsqueeze(1), Wg[e].float()).squeeze(1).to(torch.bfloat16)
    u = torch.bmm(xs.unsqueeze(1), Wu[e].float()).squeeze(1).to(torch.bfloat16)
    act = _act_simulee(g, u, Wg.shape[2], Wg.shape[2], 0, td, eid)
    return torch.bmm(act.float().unsqueeze(1), Wd[e].float()).squeeze(1).to(torch.bfloat16).float()


@pytest.mark.parametrize("w13", [False, True])
def test_temoin_glue_a4_a_sec(w13):
    from acvram.engine.moe import gemm_experts_tensor
    E, K, N, t, k = 16, 64, 32, 5, 4
    gen = torch.Generator().manual_seed(123)
    Wg, Wu = (torch.randn(E, K, N, generator=gen) / 8).to(torch.bfloat16), (torch.randn(E, K, N, generator=gen) / 8).to(torch.bfloat16)
    Wd = (torch.randn(E, N, K, generator=gen) / 8).to(torch.bfloat16)
    tg, td = _table(E, K + 16, 1), _table(E, N, 2)                  # table gate·up plus large que K (rembourrage)
    x = torch.randn(t, K, generator=gen).to(torch.bfloat16)
    x[-1] = 0
    eid = _eid(t, k, E, 1230, True)
    s3 = lambda n: torch.empty(E, 1, n)
    marlin = {"gate_proj": (Wg, s3(N), None, K, N), "up_proj": (Wu, s3(N), None, K, N), "down_proj": (Wd, s3(K), None, N, K)}
    if w13:
        marlin["w13"] = (torch.cat([Wg, Wu], 2), s3(2 * N), None, None, K, None, None, None)
    ext = SimpleNamespace(moe_act=_act_simulee)                     # sans moe_aligner_petit : glue A4 (témoin)
    args = (_PortSimule, ext, x, eid, marlin, k, N, N, 0, None, None)
    d = gemm_experts_tensor(*args, {}, {}, fusion=False, awq_gu=tg, awq_d=td)
    assert d.dtype == torch.float32 and d.shape == (t * k, K)
    assert torch.equal(d, _reference(x, eid, k, Wg, Wu, Wd, tg, td)), "témoin A4 ≠ référence x / s[e] par paire"
    sans = gemm_experts_tensor(*args, {}, {}, fusion=False)
    reelles = (eid >= 0) & (torch.arange(t * k) // k < t - 1)
    assert bool((d != sans).any(1)[reelles].all()), "tables AWQ sans effet sur une paire réelle"
    assert bool((d[t * k - k:] == 0).all()), "jeton fantôme (x nul) : ligne non nulle"


# ---------------------------------------------------------------- (a) et (b) : carte

def _ext_xs():
    from acvram import kernels
    ext = kernels.get_extension()
    if ext is None or not hasattr(ext, "moe_aligner_petit_xs"):
        pytest.skip("extension sans moe_aligner_petit_xs (recompiler)")
    return ext


@carte
@pytest.mark.parametrize("t,k", [(2, 4), (5, 8), (12, 8), (16, 8)])
def test_aligneur_xs_au_bit(t, k):
    ext = _ext_xs()
    from acvram.kernels import marlin_port as MP
    E, K = 128, 2048
    dev = torch.device("cuda", 0)
    eid = _eid(t, k, E, 1230 + t, True).to(dev)
    eid[1] = 0                                                      # l expert 0 servi en vrai, pas seulement par les fantômes
    x = torch.randn(t, K, dtype=torch.bfloat16, device=dev, generator=torch.Generator(dev).manual_seed(t))
    table = _table(E, K + 64, 7 + t, dev)
    bloc = MP.choisir_block_size(t, k, E)
    P = -(-(t * k + E * (bloc - 1)) // bloc) * bloc

    def tampons():
        return (torch.full((P,), -7, dtype=torch.int32, device=dev), torch.full((P // bloc,), -7, dtype=torch.int32, device=dev),
                torch.zeros(1, dtype=torch.int32, device=dev))
    ref, xs_t = tampons(), tampons()
    ext.moe_aligner_petit(eid, E, bloc, *ref)
    xs = torch.full((t * k, K), float("nan"), dtype=torch.bfloat16, device=dev)
    ext.moe_aligner_petit_xs(eid, E, bloc, *xs_t, x, table, k, xs)
    for a, b in zip(ref, xs_t):
        assert torch.equal(a, b), "alignement changé par l aligneur xs"
    attendu = _xs_torch(x, eid, table, k)
    assert torch.equal(xs, attendu), f"xs ≠ torch sur {int((xs != attendu).sum())} éléments"
    assert torch.equal(xs[2], (x[0] / table[0, :K]).to(torch.bfloat16)), "paire fantôme : expert 0 attendu"


@carte
@pytest.mark.parametrize("t,k", [(2, 4), (12, 8), (16, 8)])
def test_gemm_xs_noyau_egal_xs_torch(t, k):
    ext = _ext_xs()
    from acvram.engine.moe import gemm_experts_tensor
    from test_moe_tensor_decodage import _charger, _piles, E, K, I

    class _SansXs:                                                  # même extension, sans l aligneur xs
        def __getattr__(self, nom):
            if nom == "moe_aligner_petit_xs":
                raise AttributeError(nom)
            return getattr(ext, nom)
    dev = torch.device("cuda", 0)
    _, MP, _, banc = _charger()
    marlin = _piles(MP, banc, dev)
    eid = _eid(t, k, E, 1231 + t, True).to(dev)
    x = torch.randn(t, K, dtype=torch.bfloat16, device=dev, generator=torch.Generator(dev).manual_seed(t))
    x[-1] = 0
    tg, td = _table(E, K, 11, dev), _table(E, I, 12, dev)
    ws = MP.espace_travail(dev, 4); uns = torch.ones(t * k, 1, dtype=torch.float32, device=dev)

    def appel(e, **awq):
        return gemm_experts_tensor(MP, e, x, eid, marlin, k, I, I, 0, ws, uns, {}, {}, fusion=True, **awq).clone()
    d_noyau = appel(ext, awq_gu=tg, awq_d=td)
    d_torch = appel(_SansXs(), awq_gu=tg, awq_d=td)
    assert torch.equal(d_noyau, appel(ext, awq_gu=tg, awq_d=td)), "chemin xs non reproductible"
    assert torch.equal(d_noyau, d_torch), f"xs noyau ≠ xs torch : {int((d_noyau != d_torch).any(1).sum())} lignes"
    reelles = ((eid >= 0) & (torch.arange(t * k, device=dev) // k < t - 1))
    for sans in (appel(ext), appel(ext, awq_gu=tg), appel(ext, awq_d=td)):
        assert bool((d_noyau != sans).any(1)[reelles].all()), "une table AWQ sans effet sur une paire réelle"
    assert bool((d_noyau[t * k - k:] == 0).all()) and torch.isfinite(d_noyau.float()).all()


# ---------------------------------------------------------------- opt-in (24/09, verdict-123quater-relatif) : défaut inchangé

def test_awq_tensor_hors_defaut_sous_processus():
    """Le défaut ne doit jamais porter les tables AWQ sur le tensor : casse si ACVRAM_AWQ_TENSOR passe à "1" par défaut."""
    import os, subprocess
    env = {k: v for k, v in os.environ.items() if k != "ACVRAM_AWQ_TENSOR"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    r = subprocess.run([sys.executable, "-c", "import acvram.engine.moe as m; print(m._AWQ_TENSOR)"], env=env,
                       capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
    assert r.stdout.strip().splitlines()[-1] == "False", r.stdout + r.stderr


def test_raison_tensor_refuse_awq_sans_opt_in(monkeypatch):
    import acvram.engine.moe as moe
    from acvram import kernels
    from acvram.kernels import marlin_port as MP
    monkeypatch.setattr(MP, "charger", lambda *a, **k: object())
    monkeypatch.setattr(kernels, "get_extension", lambda: SimpleNamespace(moe_aligner_petit_xs=None))
    table = torch.ones(128, 2048, dtype=torch.bfloat16)
    bloc = SimpleNamespace(_stacks_marlin={"gate_proj": (0, 0, 0, 2048, 768), "down_proj": (0, 0, 0, 768, 2048)},
                           _stacks_awq={"gate_proj": table, "up_proj": table, "up_distinct": False}, experts=[0] * 128)
    monkeypatch.setattr(moe, "_AWQ_TENSOR", False)
    assert "ACVRAM_AWQ_TENSOR" in moe.MoEBlock._raison_tensor(bloc)          # défaut : repli GEMV nommé
    monkeypatch.setattr(moe, "_AWQ_TENSOR", True)
    assert moe.MoEBlock._raison_tensor(bloc) == ""                            # opt-in : chemin tensor accepté
