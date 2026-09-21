"""GEMM W4A16 dense à petit M (`kernels/gemm_dense_etroit.py`) contre la
déquantification de référence : le juge de B1 (2⁻⁷ × Σ|x·w|, `test_gemm_
grouped_w4a16`), formes q/k/v/o et gate/up/down de Qwen3.8 réduites, M = 2,
12, 16, 17, 32, entrée plus courte que `padded_in`, échelle globale par
ligne ; bras cassant de Sage : l'échelle de bloc décalée d'un rang → rouge.
Sans carte : interpréteur Triton, fp16."""
import pytest
import torch

from acvram.kernels import gemm_dense_etroit as GD
from acvram.quant.nvfp4 import dequantize_nvfp4, quantize_nvfp4

pytestmark = pytest.mark.skipif(not GD.disponible(), reason="Triton absent")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if torch.cuda.is_available() else torch.float16
TOL_REL = 2 ** -7


def _cas(M, K, N, seed=0, gsr=False):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(N, K, generator=g) * 0.05).to(torch.bfloat16)
    t = quantize_nvfp4(w)
    if gsr:
        t.global_scale_rows = (t.global_scale.float() * torch.linspace(0.5, 2.0, N)).contiguous()
    x = torch.randn(M, K, generator=g).to(DT)
    t.qweight, t.block_scale, t.global_scale = t.qweight.to(DEV), t.block_scale.to(DEV), t.global_scale.to(DEV)
    if gsr:
        t.global_scale_rows = t.global_scale_rows.to(DEV)
    return x.to(DEV), t


def _juger(y, x, t):
    wd = dequantize_nvfp4(t, torch.float32)[:, : x.shape[1]]
    attendu = x.float() @ wd.T
    borne = x.float().abs() @ wd.abs().T
    return int(((y.float() - attendu).abs() > TOL_REL * borne).sum())


@pytest.mark.parametrize("M", [2, 12, 16, 17, 32])
@pytest.mark.parametrize("K,N", [(256, 320), (512, 192)])       # q/k/v/o (N = K) et gate/up (N > K) réduits
def test_la_gemm_dense_etroite_egale_la_dequant(M, K, N):
    x, t = _cas(M, K, N, seed=M + N)
    y = GD.gemm_dense_etroit(x, t, bn=64, bk=64)
    assert y.shape == (M, N) and y.dtype == x.dtype
    assert _juger(y, x, t) == 0


def test_entree_plus_courte_que_padded_in_et_echelle_par_ligne():
    x, t = _cas(12, 200, 96, seed=5, gsr=True)
    assert t.padded_in > 200
    y = GD.gemm_dense_etroit(x, t, bn=32, bk=64)
    assert _juger(y, x, t) == 0


def test_plusieurs_tranches_k_se_somment(monkeypatch):
    x, t = _cas(12, 1024, 64, seed=9)
    monkeypatch.setattr(GD, "_PROGRAMMES_PAR_SM", 64)             # force plusieurs tranches K
    tranches, par = GD._tranches(64, 1024, x.device, 64, 64)
    assert tranches > 1 and par % 64 == 0
    y1 = GD.gemm_dense_etroit(x, t, bn=64, bk=64)
    assert _juger(y1, x, t) == 0
    # épilogue « dernier bloc » : les compteurs sont revenus à zéro (rejouable)
    # et un second appel rend le même bit (la somme suit l'ordre t = 0..T-1)
    assert int(GD._compteurs(x.device, 1).abs().sum()) == 0
    assert torch.equal(GD.gemm_dense_etroit(x, t, bn=64, bk=64), y1)


def test_bras_cassant_echelle_de_bloc_decalee_d_un_rang():
    x, t = _cas(12, 256, 128, seed=3)
    bs = t.block_scale.view(torch.uint8)
    t.block_scale = torch.roll(bs, 1, dims=1).view(torch.float8_e4m3fn)
    x2, t2 = _cas(12, 256, 128, seed=3)
    y = GD.gemm_dense_etroit(x, t, bn=64, bk=64)
    assert _juger(y, x2, t2) > 0, "l'échelle décalée d'un rang doit se voir"


def _projections(M, K, tailles, seed=11, scalers=True):
    """b projections NVFP4 de même entrée, chacune avec SON ChannelScaler
    (x / s, différents : ce que la fusion par empilement refuse)."""
    from acvram.engine.layers import QuantLinear
    from acvram.quant.calibrate import ChannelScaler
    g = torch.Generator().manual_seed(seed)
    lins = []
    for i, n in enumerate(tailles):
        w = (torch.randn(n, K, generator=g) * 0.05).to(torch.bfloat16)
        t = quantize_nvfp4(w)
        t.qweight, t.block_scale, t.global_scale = t.qweight.to(DEV), t.block_scale.to(DEV), t.global_scale.to(DEV)
        sc = ChannelScaler((0.5 + torch.rand(K, generator=g) * (1 + i)).to(torch.float16).to(DEV), 0) if scalers else None
        lins.append(QuantLinear(t, None, scaler=sc))
    x = torch.randn(M, K, generator=g).to(DT).to(DEV)
    return lins, x


def _attendu_separe(lins, x):
    """Le chemin actuel, projection par projection : x / s (arrondi bf16/fp16)
    puis produit sur la déquantification fp32."""
    ys, bornes = [], []
    for l in lins:
        xs = l.scaler.apply(x) if l.scaler is not None else x
        wd = dequantize_nvfp4(l.qweight, torch.float32)[:, : x.shape[1]]
        ys.append(xs.float() @ wd.T); bornes.append(xs.float().abs() @ wd.abs().T)
    return torch.cat(ys, 1), torch.cat(bornes, 1)


@pytest.mark.parametrize("M", [2, 12, 32])
def test_la_multi_projection_vaut_les_projections_separees(M):
    lins, x = _projections(M, 256, (192, 64, 64))
    mp = GD.MultiProjection(lins)
    y = mp(x, bn=64, bk=64)
    attendu, borne = _attendu_separe(lins, x)
    assert y.shape == (M, 320) and mp.tailles == (192, 64, 64)
    assert int(((y.float() - attendu).abs() > TOL_REL * borne).sum()) == 0


def test_la_multi_projection_sans_scaler_et_a_echelle_par_ligne():
    lins, x = _projections(12, 200, (96, 32), scalers=False)
    lins[0].qweight.global_scale_rows = (lins[0].qweight.global_scale.float() * torch.linspace(0.5, 2.0, 96).to(DEV)).contiguous()
    y = GD.MultiProjection(lins)(x, bn=32, bk=64)
    attendu, borne = _attendu_separe(lins, x)
    assert int(((y.float() - attendu).abs() > TOL_REL * borne).sum()) == 0


def test_bras_cassant_multi_scalers_echanges():
    lins, x = _projections(12, 256, (128, 128))
    attendu, borne = _attendu_separe(lins, x)
    lins[0].scaler, lins[1].scaler = lins[1].scaler, lins[0].scaler        # l'échelle de l'autre projection
    y = GD.MultiProjection(lins)(x, bn=64, bk=64)
    assert int(((y.float() - attendu).abs() > TOL_REL * borne).sum()) > 0, "l'échelle d'activation échangée doit se voir"


def test_le_gdn_projette_en_un_lancement_sous_dense_nvfp4_triton(monkeypatch):
    """Intégration : `GatedDeltaNet.fuse()` → multi-projection (qkv, gate,
    α, β) ; sortie de la couche contre la voie séparée (nvfp4 déquantifié)."""
    from acvram import kernels
    from acvram.engine.gdn import GatedDeltaNet
    from acvram.engine.layers import QuantLinear
    H, NK, NV, DK, DV, KER = 64, 2, 4, 16, 16, 4
    conv_dim = 2 * NK * DK + NV * DV
    torch.manual_seed(20260917)

    def qlin(o, i):
        t = quantize_nvfp4((torch.randn(o, i) * 0.2).to(torch.bfloat16))
        # sur carte, les poids doivent y être : `_multi_projection` refuse
        # un NVFP4 hôte (Laure, t0-test : fuse() → False sur le jouet)
        t.qweight, t.block_scale, t.global_scale = t.qweight.to(DEV), t.block_scale.to(DEV), t.global_scale.to(DEV)
        return QuantLinear(t, None, scaler=None)
    couche = GatedDeltaNet(qkv=qlin(conv_dim, H), gate=qlin(NV * DV, H), alpha=qlin(NV, H), beta=qlin(NV, H),
                           out=qlin(H, NV * DV), conv_weight=torch.randn(conv_dim, KER) * 0.3,
                           dt_bias=torch.rand(NV) - 0.5, a_log=torch.rand(NV) * 3 - 2,
                           norm_weight=torch.ones(DV), num_k_heads=NK, num_v_heads=NV, head_k_dim=DK, head_v_dim=DV).to(DEV)
    x = torch.randn(12, H).to(DT).to(DEV)
    from acvram.engine import model as MD
    monkeypatch.setattr(kernels, "_DENSE_NVFP4", "gemv")
    with torch.no_grad():
        y_sep, _ = couche(x, None)
        assert couche.fuse() and couche.multi.tailles == (conv_dim, NV * DV, NV, NV)
        monkeypatch.setattr(kernels, "_DENSE_NVFP4", "triton")
        monkeypatch.setattr(MD, "_MULTI_PROJ", True)                       # témoin, opt-in
        y_multi, _ = couche(x, None)
    ecart = ((y_multi.float() - y_sep.float()).abs().max() / y_sep.float().abs().max()).item()
    assert ecart < 2 ** -6, ecart


def test_les_logits_de_la_tete_en_fp32_egalent_la_gemv_fp32():
    """La tête (N = vocabulaire) par la GEMM dense étroite, sortie fp32 :
    les logits sont LE tenseur à comparer (sage-gemm-dense-palier2-non-
    ouvert § 2) — contre x fp32 @ déquant fp32 (le chemin GEMV fp32 actuel) :
    2⁻⁷ × borne ET même argmax sur chaque ligne."""
    x, t = _cas(12, 256, 1000, seed=21)
    y = GD.gemm_dense_etroit(x, t, bn=64, bk=64, sortie_fp32=True)
    assert y.dtype == torch.float32 and y.shape == (12, 1000)
    wd = dequantize_nvfp4(t, torch.float32)
    attendu = x.float() @ wd.T
    assert _juger(y, x, t) == 0
    assert torch.equal(y.argmax(-1), attendu.argmax(-1))
    assert ((y - attendu).abs().max() / attendu.abs().max()).item() < 1e-5
