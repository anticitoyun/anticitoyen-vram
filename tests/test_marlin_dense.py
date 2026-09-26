"""Pièce 101 (23/09) : le Marlin DENSE porté de vLLM (`marlin_port.gemm_dense`) et sa modification acvram —
l'échelle globale NVFP4 PAR COLONNE dans l'épilogue (`gs_par_colonne`, marlin_template.h), pour q/k/v empilés
avec une échelle globale par segment (`global_scale_rows`).

Juge au bit : une même GEMM empilée, lancée une fois avec l'échelle par colonne et une fois par segment avec
l'échelle SCALAIRE de ce segment (même N, donc même découpage, même ordre de réduction) — les colonnes de chaque
segment doivent être identiques au bit. Ce test casse si l'épilogue revient à l'échelle scalaire (les segments k
et v prendraient l'échelle de q), ou si l'index de colonne est faux sur l'un des deux chemins d'écriture
(`m_block_size_8`, M ≤ 8 ; normal, M > 8). Carte requise ; skip sans port compilé avec `marlin_gemm`."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")

SEGMENTS = (512, 128, 128)          # q, k, v : multiples de 64, trois échelles globales distinctes
K = 1024


def _mp():
    from acvram.kernels import marlin_port as MP
    ops = MP.charger(compiler=False)
    if ops is None or not hasattr(ops, "marlin_gemm"):
        pytest.skip("port Marlin sans marlin_gemm (compiler à sec avant la prise, pièce 101)")
    return MP


def _empile(dev, graine=101):
    """Trois poids quantifiés SÉPARÉMENT (échelles globales différentes), empilés comme `engine/layers.py`
    (fusion q/k/v : qweight et block_scale concaténés, `global_scale_rows` par ligne)."""
    from acvram.quant.nvfp4 import NVFP4Tensor, quantize_nvfp4
    g = torch.Generator(device=dev).manual_seed(graine)
    ts = [quantize_nvfp4(torch.randn(n, K, device=dev, generator=g, dtype=torch.bfloat16) * a)
          for n, a in zip(SEGMENTS, (0.05, 0.004, 0.012))]
    gs = [t.global_scale_float() for t in ts]
    assert len({round(v, 12) for v in gs}) == 3, gs
    lignes = torch.cat([torch.full((t.qweight.shape[0],), t.global_scale_float(), dtype=torch.float32, device=dev)
                        for t in ts])
    pile = NVFP4Tensor(torch.cat([t.qweight for t in ts]).contiguous(), torch.cat([t.block_scale for t in ts]).contiguous(),
                       ts[0].global_scale, (sum(SEGMENTS), K), ts[0].padded_in, global_scale_rows=lignes.contiguous())
    return pile, ts


@pytest.mark.parametrize("M", [1, 4, 8, 12, 16])
def test_echelle_par_colonne_au_bit_des_segments(M):
    MP = _mp()
    dev = torch.device("cuda", 0)
    pile, ts = _empile(dev)
    N = sum(SEGMENTS)
    w, s, g_col = MP.preparer_dense(pile)
    assert g_col.numel() == N
    ws = MP.espace_travail(dev)
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16, generator=torch.Generator(device=dev).manual_seed(M))
    y_col = MP.gemm_dense(x, w, s, g_col, N, K, ws)
    debut = 0
    for n, t in zip(SEGMENTS, ts):
        g_seg = g_col[debut:debut + 1].clone()                 # l'échelle scalaire de CE segment, même traitement
        y_seg = MP.gemm_dense(x, w, s, g_seg, N, K, ws)
        assert torch.equal(y_col[:, debut:debut + n], y_seg[:, debut:debut + n]), \
            (M, debut, float((y_col[:, debut:debut + n].float() - y_seg[:, debut:debut + n].float()).abs().max()))
        debut += n


def test_echelle_par_colonne_uniforme_egale_la_scalaire():
    """Une échelle par colonne toute égale rend les bits de l'échelle scalaire (le drapeau ne change rien d'autre)."""
    MP = _mp()
    dev = torch.device("cuda", 0)
    pile, _ = _empile(dev, graine=7)
    N = sum(SEGMENTS)
    w, s, g_col = MP.preparer_dense(pile)
    ws = MP.espace_travail(dev)
    for M in (1, 12):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        g1 = g_col[:1].clone()
        assert torch.equal(MP.gemm_dense(x, w, s, g1.expand(N).contiguous(), N, K, ws),
                           MP.gemm_dense(x, w, s, g1, N, K, ws))


@pytest.mark.parametrize("M", [1, 12])
def test_proche_du_chemin_gsr_servi(M):
    """Même poids empilé : Marlin (échelle par colonne) contre le chemin servi d'acvram (`backends.matmul` sur le
    NVFP4Tensor à `global_scale_rows`) et contre fp64 — des noyaux différents, donc pas au bit : l'écart de Marlin
    à fp64 reste du même ordre que celui du chemin servi (≤ 2 × + 1e-3)."""
    MP = _mp()
    from acvram.kernels import backends
    from acvram.quant.nvfp4 import dequantize_nvfp4
    dev = torch.device("cuda", 0)
    pile, _ = _empile(dev, graine=11)
    N = sum(SEGMENTS)
    w, s, g_col = MP.preparer_dense(pile)
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    ref = x.double() @ dequantize_nvfp4(pile, torch.float32).double()[:, :K].t()

    def err(y):
        return float(((y.double() - ref).norm(dim=-1) / ref.norm(dim=-1)).max())
    e_marlin = err(MP.gemm_dense(x, w, s, g_col, N, K, MP.espace_travail(dev)))
    e_servi = err(backends.matmul(x, pile))
    assert e_marlin <= 2 * e_servi + 1e-3, (e_marlin, e_servi)
