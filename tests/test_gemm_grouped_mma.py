"""GEMM groupée NVFP4 sur la MMA FP4 native de sm_120 (W4A4) contre une
référence Python qui quantifie les activations exactement comme le noyau.

Trois vérités à établir, dans cet ordre :
1. la quantification des activations (E2M1 par bloc de 16, échelle UE4M3)
   est bit-identique entre le noyau et la référence Python — sinon le reste
   compare deux choses différentes ;
2. la GEMM, à partir de ces mêmes activations quantifiées, rend le produit
   exact en float64 à l'arrondi fp32/bf16 près, sur toutes les valeurs ;
3. les tables d'adresses (contrat bead pds) sont bien lues : des experts
   dispersés dans plusieurs piles rendent le même résultat.
Le cosinus contre le chemin bf16 (déquant + _grouped_mm) est rapporté mais
n'est pas la mesure de justesse : les activations changent de précision.
"""
import pytest
import torch

from acvram.quant.nvfp4 import quantize_nvfp4

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau CUDA requis")

_E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
_MID = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])


def _ext():
    from acvram.kernels import get_extension
    ext = get_extension()
    if not hasattr(ext, "nvfp4_gemm_grouped_mma") or not ext.nvfp4_gemm_grouped_mma_disponible():
        pytest.skip("noyau MMA FP4 indisponible (carte non sm_120 ou binaire sans cible famille)")
    return ext


def fwht_ref(xf: torch.Tensor, bloc: int) -> torch.Tensor:
    """FWHT fp32 par bloc de ``bloc`` colonnes, l'arithmétique du noyau : étages
    h = 1, 2, …, bloc/2, chaque sortie = a + c ou a − c (une seule somme,
    reproductible), puis × (1/√bloc) calculé en fp32 RN (√ puis division
    tenseur/tenseur), puis arrondi bf16 (comme ChannelScaler.apply)."""
    G, K = xf.shape
    y = xf.float().reshape(G, K // bloc, bloc).clone()
    h = 1
    while h < bloc:
        v = y.view(G, K // bloc, bloc // (2 * h), 2, h)
        a, c = v[..., 0, :], v[..., 1, :]
        y = torch.stack((a + c, a - c), dim=-2).reshape(G, K // bloc, bloc)
        h *= 2
    racine = torch.tensor(float(bloc), dtype=torch.float32, device=xf.device).sqrt()
    inv = torch.ones((), dtype=torch.float32, device=xf.device) / racine
    return (y.reshape(G, K) * inv).to(torch.bfloat16).float()


def quant_act_ref(x: torch.Tensor, awq=None, e_sorted=None, hadamard: int = 0):
    """Référence de nvfp4_quant_act (poste7-glm-pile-correctif § 7) : échelle
    globale PAR LIGNE g_r = amax_r / 2688 fp32 ; par bloc de 16,
    s = (amax_blk / amax_r) × 448 -> E4M3 au plus proche (448 exactement au bloc
    maximal, ≤ 448 ailleurs : pas de saturation),
    puis chaque valeur / (sdec · g_r) -> E2M1 au plus proche, égalités vers le
    code pair (cvt.rn.satfinite.e2m1x2). Le noyau divise en IEEE (__fdiv_rn) :
    sous --use_fast_math la division approchée basculait 5 égalités sur 5 376.
    ``awq`` [E, K] bf16 et ``e_sorted`` [G] : x/s[e] en bf16
    (bf16(bf16(v)/bf16(s))) avant tout. Rend (codes, échelles E4M3, g_r)."""
    G, K = x.shape
    xf = x.float()
    if hadamard:
        xf = fwht_ref(xf, hadamard)
    if awq is not None:
        xf = (xf / awq[e_sorted.long()].float()).to(torch.bfloat16).float()
    xb = xf.view(G, K // 16, 16)
    amax_b = xb.abs().amax(-1)                                    # [G, K/16]
    amax_r = amax_b.amax(-1)                                      # [G]
    # divisions tenseur / tenseur uniquement : torch divise par un scalaire
    # Python en multipliant par l'inverse (1 ulp d'écart, t-qa 065960a) ;
    # le noyau fait __fdiv_rn / __fmul_rn, RN explicites
    g = torch.where(amax_r > 0, amax_r / torch.full_like(amax_r, 2688.0), torch.zeros_like(amax_r))
    ok_r = (g > 0).unsqueeze(-1)
    s = torch.where(ok_r & (amax_b > 0), (amax_b / amax_r.clamp(min=1e-30).unsqueeze(-1)) * 448.0,
                    torch.zeros_like(amax_b))
    sbits = s.to(torch.float8_e4m3fn)
    sdec = sbits.float()
    ok = sdec > 0
    d = (sdec * g.unsqueeze(-1)).clamp(min=1e-30)
    v = torch.where(ok.unsqueeze(-1), xb / d.unsqueeze(-1), torch.zeros_like(xb))
    a = v.abs().clamp(max=6.0)
    mid = _MID.to(x.device)
    iu = torch.searchsorted(mid, a.reshape(-1).contiguous(), right=True).view_as(a)
    il = torch.searchsorted(mid, a.reshape(-1).contiguous(), right=False).view_as(a)
    code = torch.where(iu != il, torch.where(il % 2 == 0, il, iu), iu).to(torch.uint8)
    code = code | (torch.signbit(v).to(torch.uint8) << 3)
    code = torch.where(ok.unsqueeze(-1), code, torch.zeros_like(code))
    octets = code[..., 0::2] | (code[..., 1::2] << 4)
    return (octets.reshape(G, K // 2).contiguous(),
            sbits.view(torch.uint8).reshape(G, K // 16).contiguous(), g.contiguous())


def dequant_act_ref(xq, xsf, grow):
    """E2M1 [G, K/2] + UE4M3 [G, K/16] + g_r [G] -> float64 [G, K]."""
    G = xq.shape[0]
    xa = _dequant_nibbles(xq).view(G, -1, 16) * xsf.view(torch.float8_e4m3fn).double().unsqueeze(-1)
    return (xa * grow.double().view(G, 1, 1)).reshape(G, -1)


def _dequant_nibbles(q: torch.Tensor) -> torch.Tensor:
    """[..., K/2] uint8 -> [..., K] float64, nibble bas = indice pair."""
    lo, hi = q & 0xF, q >> 4
    codes = torch.stack([lo, hi], -1).reshape(*q.shape[:-1], -1)
    tab = _E2M1.to(q.device, torch.float64)
    val = tab[(codes & 7).long()]
    return torch.where(codes & 8 != 0, -val, val)


def _pile(E, M, K, amplitude, graine):
    g = torch.Generator(device="cuda").manual_seed(graine)
    ts = [quantize_nvfp4((torch.randn(M, K, device="cuda", generator=g) * amplitude).to(torch.bfloat16))
          for _ in range(E)]
    qw = torch.stack([t.qweight for t in ts]).contiguous()
    bs = torch.stack([t.block_scale.view(torch.uint8) for t in ts]).contiguous()
    gs = torch.stack([t.global_scale.reshape(()) for t in ts]).to(torch.float32).contiguous()
    return qw, bs, gs


def _w64(qw, bs):
    """Poids déquantifiés en float64 SANS la super-échelle : [E, M, K]."""
    w = _dequant_nibbles(qw)
    sc = bs.view(torch.float8_e4m3fn).double()
    return w.view(*w.shape[:-1], -1, 16) * sc.unsqueeze(-1)


def _tuiles(cnt, bt):
    from acvram.engine.model import MoEBlock
    return MoEBlock._tuiles(cnt, bt)


def _tables(qw, bs):
    """Tables d'adresses identité : chaque expert à sa tranche de la pile."""
    E = qw.shape[0]
    tq = qw.data_ptr() + torch.arange(E, dtype=torch.int64) * qw.stride(0)
    tb = bs.data_ptr() + torch.arange(E, dtype=torch.int64) * bs.stride(0)
    return tq.cuda(), tb.cuda()


def _x(G, K, graine):
    g = torch.Generator(device="cuda").manual_seed(graine)
    x = torch.randn(G, K, device="cuda", generator=g)
    # amplitudes variées par ligne, un bloc nul, un bloc minuscule, un bloc énorme
    x = x * torch.logspace(-3, 2, G, device="cuda").unsqueeze(1)
    if G > 2 and K >= 64:
        x[1, :16] = 0
        x[2, 16:32] *= 1e-4
        x[0, 32:48] *= 1e3
    return x.to(torch.bfloat16).contiguous()


@pytest.mark.parametrize("G,K", [(1, 64), (7, 1536), (33, 2048), (200, 4096)])
def test_quant_act_bit_identique(G, K):
    ext = _ext()
    x = _x(G, K, G * 31 + K)
    xq, xsf, gr = ext.nvfp4_quant_act(x)
    rq, rsf, rg = quant_act_ref(x)
    assert torch.equal(gr, rg), f"échelles de ligne : {int((gr != rg).sum())} différentes sur {G}"
    assert torch.equal(xsf, rsf), f"échelles : {int((xsf != rsf).sum())} octets différents sur {rsf.numel()}"
    assert torch.equal(xq, rq), f"codes : {int((xq != rq).sum())} octets différents sur {rq.numel()}"


@pytest.mark.parametrize("comptes", [
    [1, 0, 0, 2],
    [16, 16, 16, 16],
    [15, 17, 1, 33],
    [0, 0, 0, 5],
    [64, 65, 3, 130],
])
@pytest.mark.parametrize("M,K", [(1536, 2048), (2048, 1536), (1500, 1536)])
@pytest.mark.parametrize("bt", [16, 64, 128])
@pytest.mark.parametrize("etages", [0, 2, 3, 4])
def test_gemm_mma_contre_reference_w4a4(comptes, M, K, bt, etages):
    if bt == 128 and etages == 0:
        pytest.skip("bt=128 : variante a etages seulement")
    ext = _ext()
    E = len(comptes)
    qw, bs, gs = _pile(E, M, K, 0.05, sum(comptes) + M + bt)
    cnt = torch.tensor(comptes, device="cuda")
    G = int(cnt.sum())
    x = _x(G, K, G + K)
    xq, xsf, gr = ext.nvfp4_quant_act(x)
    rq, rsf, rg = quant_act_ref(x)
    assert torch.equal(xq, rq) and torch.equal(xsf, rsf) and torch.equal(gr, rg)
    te, t0, tn = _tuiles(cnt, bt)
    tq, tb = _tables(qw, bs)
    y = ext.nvfp4_gemm_grouped_mma(tq, tb, gs, xq, xsf, te, t0, tn, M, K, bt, etages, grow=gr)
    assert y.shape == (G, M) and y.dtype == torch.bfloat16
    # référence float64 à partir des MÊMES activations quantifiées (g_r comprise)
    xa = dequant_act_ref(xq, xsf, gr)
    w = _w64(qw, bs).reshape(E, M, K)
    attendu = torch.zeros(G, M, device="cuda", dtype=torch.float64)
    debut = 0
    for e, n in enumerate(comptes):
        if n:
            attendu[debut:debut + n] = (xa[debut:debut + n] @ w[e].T) * gs[e].double()
            debut += n
    ecart = (y.double() - attendu).abs()
    # accumulation fp32 (ordre différent) puis bf16 : 2^-8 relatif + un plancher
    tol = attendu.abs() * 2 ** -7 + 1e-3 * attendu.abs().max()
    hors = int((ecart > tol).sum())
    exact = (y.double() == attendu.to(torch.bfloat16).double()).float().mean().item()
    assert hors == 0, f"{hors} valeurs hors tolérance sur {ecart.numel()}, max {ecart.max().item():.3e}"
    assert exact > 0.9, f"seulement {exact:.3f} des sorties bit-identiques à bf16(référence)"


def test_bt32_et_etages_identiques():
    """bt=32 et les variantes a etages rendent la MEME sortie que la directe
    (memes fragments, meme ordre de somme) : bit-identique attendu."""
    ext = _ext()
    comptes = [15, 17, 1, 33]
    E, M, K = len(comptes), 1536, 2048
    qw, bs, gs = _pile(E, M, K, 0.05, 3)
    cnt = torch.tensor(comptes, device="cuda")
    G = int(cnt.sum())
    xq, xsf, gr = ext.nvfp4_quant_act(_x(G, K, 11))
    tq, tb = _tables(qw, bs)
    for bt in (16, 32, 64, 128):
        te, t0, tn = _tuiles(cnt, bt)
        ref = ext.nvfp4_gemm_grouped_mma(tq, tb, gs, xq, xsf, te, t0, tn, M, K, min(bt, 64), 0, grow=gr)
        for et in (2, 3, 4):
            for ks in (64, 128):
                y = ext.nvfp4_gemm_grouped_mma(tq, tb, gs, xq, xsf, te, t0, tn, M, K, bt, et, ks, grow=gr)
                assert torch.equal(y, ref), f"bt={bt} etages={et} ks={ks} differe de la variante directe"


def test_tables_adresses():
    """La table est lue : les experts dispersés dans deux piles distinctes et
    dans un ordre permuté rendent le même résultat ; une table fausse, non."""
    ext = _ext()
    comptes = [5, 20, 0, 40]
    E, M, K, bt = len(comptes), 1536, 2048, 32
    qw, bs, gs = _pile(E, M, K, 0.05, 99)
    cnt = torch.tensor(comptes, device="cuda")
    G = int(cnt.sum())
    xq, xsf, gr = ext.nvfp4_quant_act(_x(G, K, 5))
    te, t0, tn = _tuiles(cnt, bt)
    tq, tb = _tables(qw, bs)
    y = ext.nvfp4_gemm_grouped_mma(tq, tb, gs, xq, xsf, te, t0, tn, M, K, bt, grow=gr)
    # experts 0 et 2 dans une seconde pile, 1 et 3 permutés dans la première
    pile2 = torch.stack([qw[2], qw[0]]).contiguous(); bs2 = torch.stack([bs[2], bs[0]]).contiguous()
    pile1 = torch.stack([qw[3], qw[1]]).contiguous(); bs1 = torch.stack([bs[3], bs[1]]).contiguous()
    tq2 = torch.tensor([pile2.data_ptr() + pile2.stride(0), pile1.data_ptr() + pile1.stride(0),
                        pile2.data_ptr(), pile1.data_ptr()], dtype=torch.int64).cuda()
    tb2 = torch.tensor([bs2.data_ptr() + bs2.stride(0), bs1.data_ptr() + bs1.stride(0),
                        bs2.data_ptr(), bs1.data_ptr()], dtype=torch.int64).cuda()
    y2 = ext.nvfp4_gemm_grouped_mma(tq2, tb2, gs, xq, xsf, te, t0, tn, M, K, bt, grow=gr)
    assert torch.equal(y, y2)
    # table qui pointe l'expert 1 sur l'expert 3 : doit changer le résultat
    tq3 = tq2.clone(); tq3[1] = tq2[3]
    y3 = ext.nvfp4_gemm_grouped_mma(tq3, tb2, gs, xq, xsf, te, t0, tn, M, K, bt, grow=gr)
    assert not torch.equal(y, y3)


@pytest.mark.parametrize("comptes", [[16, 16, 16, 16], [15, 17, 1, 33]])
def test_cosinus_contre_chemin_bf16(comptes):
    """Information, pas verdict : W4A4 contre déquant bf16 + _grouped_mm."""
    if not hasattr(torch, "_grouped_mm"):
        pytest.skip("torch._grouped_mm absent")
    ext = _ext()
    E, M, K = len(comptes), 1536, 2048
    qw, bs, gs = _pile(E, M, K, 0.05, 7)
    from acvram.quant.nvfp4 import dequantize_nvfp4, quantize_nvfp4  # noqa
    cnt = torch.tensor(comptes, device="cuda")
    G = int(cnt.sum())
    g = torch.Generator(device="cuda").manual_seed(G)
    x = torch.randn(G, K, device="cuda", generator=g).to(torch.bfloat16)
    xq, xsf, gr = ext.nvfp4_quant_act(x)
    te, t0, tn = _tuiles(cnt, 64)
    tq, tb = _tables(qw, bs)
    y = ext.nvfp4_gemm_grouped_mma(tq, tb, gs, xq, xsf, te, t0, tn, M, K, 64, grow=gr)
    w = (_w64(qw, bs).reshape(E, M, K) * gs.double().view(E, 1, 1)).to(torch.bfloat16)
    offs = torch.cumsum(cnt, 0).to(torch.int32)
    y_ref = torch._grouped_mm(x, w.transpose(1, 2), offs=offs)
    cos = torch.nn.functional.cosine_similarity(y.float().reshape(-1), y_ref.float().reshape(-1), dim=0).item()
    print(f"\ncosinus W4A4 / bf16 : {cos:.6f}")
    assert cos > 0.99
