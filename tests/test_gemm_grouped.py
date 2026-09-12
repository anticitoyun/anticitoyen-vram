"""Le noyau de GEMM groupée NVFP4 du prefill MoE contre la référence.

Écrit pour le chantier 2 du plan et jamais validé numériquement jusqu'ici. Les
cas qui comptent sont les tuiles partielles : un expert qui reçoit 1, 2, 15 ou
17 jetons — c'est ce qui arrive au prefill d'une conversation, où quelques
jetons spéciaux partent vers des experts peu sollicités.
"""
import pytest
import torch

from acvram.quant.nvfp4 import quantize_nvfp4, dequantize_nvfp4

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau CUDA requis")


def _pile(E, M, K, amplitude, graine):
    g = torch.Generator(device="cuda").manual_seed(graine)
    ts = [quantize_nvfp4((torch.randn(M, K, device="cuda", generator=g) * amplitude).to(torch.bfloat16))
          for _ in range(E)]
    qw = torch.stack([t.qweight for t in ts]).contiguous()
    bs = torch.stack([t.block_scale.view(torch.uint8) for t in ts]).contiguous()
    gs = torch.stack([t.global_scale.reshape(()) for t in ts]).to(torch.float32).contiguous()
    ref = torch.stack([dequantize_nvfp4(t, torch.bfloat16) for t in ts])   # [E, M, K]
    return qw, bs, gs, ref


def _tuiles(cnt, bt=16):
    from acvram.engine.model import MoEBlock
    return MoEBlock._tuiles(cnt, bt)


@pytest.mark.parametrize("comptes", [
    [1, 0, 0, 2],            # deux experts presque vides, deux vides
    [16, 16, 16, 16],        # tuiles pleines
    [15, 17, 1, 33],         # partielles de toutes tailles
    [0, 0, 0, 5],            # un seul expert servi
])
@pytest.mark.parametrize("M,K", [(1536, 2048), (2048, 1536), (1500, 1536)])
def test_gemm_groupee_contre_reference(comptes, M, K):
    from acvram.kernels import get_extension
    ext = get_extension()
    if not hasattr(ext, "nvfp4_gemm_grouped"):
        pytest.skip("extension sans nvfp4_gemm_grouped")
    E = len(comptes)
    qw, bs, gs, ref = _pile(E, M, K, 0.05, sum(comptes) + M)
    cnt = torch.tensor(comptes, device="cuda")
    G = int(cnt.sum())
    g = torch.Generator(device="cuda").manual_seed(G)
    x = torch.randn(G, K, device="cuda", generator=g).to(torch.bfloat16)
    te, t0, tn = _tuiles(cnt)
    y = ext.nvfp4_gemm_grouped(qw, bs, gs, x, te, t0, tn, K)
    # référence : matmul par expert sur ses jetons, triés dans le même ordre
    attendu = torch.zeros(G, M, device="cuda", dtype=torch.float32)
    debut = 0
    for e, n in enumerate(comptes):
        if n:
            attendu[debut:debut + n] = x[debut:debut + n].float() @ ref[e].float().T
            debut += n
    ecart = (y.float() - attendu).abs()
    tol = attendu.abs() * 2 ** -6 + 1e-2       # bf16 par les tensor cores
    assert (ecart <= tol).float().mean().item() > 0.999, \
        f"{int((ecart > tol).sum())} valeurs hors tolérance, max {ecart.max().item():.3e}"


@pytest.mark.parametrize("comptes", [
    [1, 0, 0, 2],
    [16, 16, 16, 16],
    [15, 17, 1, 33],
])
@pytest.mark.parametrize("M,K", [(1536, 2048), (2048, 1536), (1500, 1536)])
def test_noyau_contre_grouped_mm(comptes, M, K):
    """nvfp4_gemm_grouped vs _pile_bf16 + torch._grouped_mm : cosinus > 0,999."""
    if not hasattr(torch, "_grouped_mm"):
        pytest.skip("torch._grouped_mm absent")
    from acvram.kernels import get_extension
    ext = get_extension()
    if not hasattr(ext, "nvfp4_gemm_grouped"):
        pytest.skip("extension sans nvfp4_gemm_grouped")
    E = len(comptes)
    qw, bs, gs, ref = _pile(E, M, K, 0.05, sum(comptes) + M + 7)
    cnt = torch.tensor(comptes, device="cuda")
    G = int(cnt.sum())
    if G == 0:
        return
    g = torch.Generator(device="cuda").manual_seed(G + 42)
    x = torch.randn(G, K, device="cuda", generator=g).to(torch.bfloat16)
    te, t0, tn = _tuiles(cnt)
    y_noyau = ext.nvfp4_gemm_grouped(qw, bs, gs, x, te, t0, tn, K)
    offs = torch.cumsum(cnt, 0).to(torch.int32)
    y_ref = torch._grouped_mm(x, ref.transpose(1, 2), offs=offs)
    cos = torch.nn.functional.cosine_similarity(
        y_noyau.float(), y_ref.float(), dim=-1)
    assert cos.min().item() > 0.999, \
        f"cosinus min {cos.min().item():.6f}, attendu > 0,999"


def test_tuiles_couvrent_exactement_les_jetons():
    cnt = torch.tensor([1, 0, 17, 16, 2], device="cuda")
    te, t0, tn = _tuiles(cnt)
    assert te.tolist() == [0, 2, 2, 3, 4]
    assert t0.tolist() == [0, 1, 17, 18, 34]
    assert tn.tolist() == [1, 16, 1, 16, 2]
