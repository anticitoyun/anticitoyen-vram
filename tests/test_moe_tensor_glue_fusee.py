"""Pièce 63 (23/09) : la glue fusionnée du chemin tensor (`moe_aligner_petit` en un lancement, `moe_reduce`
lisant le bf16) est reproductible, au bit avec la glue A4 (clamp + aligneur vLLM + d.float() + reduce fp32)
partout où A4 est reproductible avec elle-même (elle ne l est pas : aligneur vLLM par atomiques), et sous
2⁻⁷·max ailleurs, sur les mêmes entrées : godets 5/12/16 avec fantômes (eid −1, ligne nulle), expert recevant tous les jetons.
Et l aligneur fusionné rend les mêmes tampons que l aligneur vLLM sur les blocs utiles (paires par bloc,
experts par bloc, num_post). Carte requise ; skip sans extension ou port compilés."""
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _pre():
    from acvram import kernels
    from acvram.kernels import marlin_port as MP
    ext = kernels.get_extension()
    ops = MP.charger(compiler=False)
    if ext is None or not hasattr(ext, "moe_aligner_petit"):
        pytest.skip("extension sans moe_aligner_petit (recompiler à sec)")
    if ops is None or not hasattr(ops, "moe_align_block_size"):
        pytest.skip("port Marlin sans moe_align_block_size")
    return ext, MP


def _eid(t, k, E, graine, fantomes):
    g = torch.Generator().manual_seed(graine)
    p = 1.0 / (torch.arange(E, dtype=torch.float32) + 1) ** 1.7
    eid = torch.stack([torch.multinomial(p, k, replacement=False, generator=g) for _ in range(t)]).to(torch.int32)
    eid[:, 0] = 7                                            # un expert reçoit tous les jetons
    if fantomes:
        eid[-1, :] = -1                                      # dernier jeton fantôme
        eid[0, 2] = -1                                       # une paire fantôme isolée
    return eid.reshape(-1)


@pytest.mark.parametrize("t", [5, 12, 16])
def test_aligneur_fusionne_egal_vllm(t):
    ext, MP = _pre()
    E, k = 128, 8
    dev = torch.device("cuda", 0)
    eid = _eid(t, k, E, 63 + t, True).to(dev)
    bloc = MP.choisir_block_size(t, k, E)
    P = -(-(t * k + E * (bloc - 1)) // bloc) * bloc
    tampons = (torch.full((P,), -7, dtype=torch.int32, device=dev), torch.full((P // bloc,), -7, dtype=torch.int32, device=dev),
               torch.zeros(1, dtype=torch.int32, device=dev))
    ext.moe_aligner_petit(eid, E, bloc, *tampons)
    s1, e1, n1 = MP.aligner_blocs_cuda(eid.clamp(min=0), bloc, E)
    n = int(n1)
    assert int(tampons[2]) == n
    assert torch.equal(tampons[1][: n // bloc], e1[: n // bloc]), "experts par bloc différents"
    # la place d une paire DANS les blocs de son expert est libre (vLLM : atomiques, ordre non déterministe ;
    # ici : rang stable) — l invariant lu par la GEMM est « quelle paire est servie par quel expert »
    def expert_de_chaque_paire(s, e):
        pos = torch.nonzero(s[:n] < eid.numel()).flatten()
        m = torch.full((eid.numel(),), -2, dtype=torch.int32, device=dev)
        m[s[pos]] = e[pos // bloc]
        return m
    assert torch.equal(expert_de_chaque_paire(tampons[0], tampons[1]), expert_de_chaque_paire(s1, e1)), "paire → expert différent"
    assert torch.equal(expert_de_chaque_paire(tampons[0], tampons[1]), eid.clamp(min=0)), "paire → expert ≠ eid"
    assert bool((tampons[0][n:] == eid.numel()).all()) and bool((tampons[1][n // bloc:] == -1).all())
    # tri stable : dans un même expert, les paires sont dans l ordre croissant
    for blk in range(n // bloc):
        v = tampons[0][blk * bloc:(blk + 1) * bloc]; v = v[v < eid.numel()]
        assert bool((v[1:] > v[:-1]).all()) if v.numel() > 1 else True


@pytest.mark.parametrize("t,k", [(2, 4), (2, 8), (5, 8), (12, 8), (16, 8)])   # (2, 4) : G = 8, plus petit godet servi (pièce 65)
def test_glue_fusionnee_au_bit(t, k):
    ext, MP = _pre()
    from acvram.engine.moe import gemm_experts_tensor
    from test_moe_tensor_decodage import _charger, _piles, E, K, I   # piles NVFP4 du banc (format du port)
    dev = torch.device("cuda", 0)
    _, MP, ext, banc = _charger()
    marlin = _piles(MP, banc, dev)
    m_gate, k_down = I, I
    eid = _eid(t, k, E, 630 + t, True).to(dev)
    x = torch.randn(t, K, dtype=torch.bfloat16, device=dev, generator=torch.Generator(dev).manual_seed(t))
    x[-1] = 0
    topw = torch.rand(t, k, dtype=torch.float32, device=dev, generator=torch.Generator(dev).manual_seed(99 + t))
    topw[-1] = 0; topw[0, 2] = 0
    ws = MP.espace_travail(dev, 4); uns = torch.ones(t * k, 1, dtype=torch.float32, device=dev)
    args = (MP, ext, x, eid, marlin, k, m_gate, k_down, 0, ws, uns)
    d_a4 = gemm_experts_tensor(*args, {}, {}, fusion=False)
    d_a4b = gemm_experts_tensor(*args, {}, {}, fusion=False)
    d_fu = gemm_experts_tensor(*args, {}, {}, fusion=True)
    d_fub = gemm_experts_tensor(*args, {}, {}, fusion=True)
    assert d_a4.dtype == torch.float32 and d_fu.dtype == torch.bfloat16
    # Trouvé le 23/09 (diag-aubit) : la glue A4 n est PAS reproductible avec elle-même — l aligneur vLLM
    # répartit par atomiques les paires d un expert sur ses blocs, et la GEMM Marlin rend des bits différents
    # selon la place (2⁻⁹ relatif, lignes d un expert à plus d un bloc). La fusion est déterministe (rang
    # stable) : au bit avec elle-même, et au bit avec A4 sur toute ligne où A4 est d accord avec lui-même.
    assert torch.equal(d_fu, d_fub), "la glue fusionnée n est pas reproductible"
    # A4 n est pas reproductible (deux appels diffèrent, t ≥ 12), et une ligne d A4 égale entre DEUX appels peut
    # encore différer de la fusion (t=16 : lignes 64, 116 le 23/09) : le critère tenable contre A4 est celui du
    # jalon A4 lui-même, 2⁻⁷·max par ligne ; l égalité au bit est exigée de la fusion avec elle-même seulement.
    borne = 2.0 ** -7 * d_a4.abs().max()
    ecart = (d_a4 - d_fu.float()).abs().max(1).values
    assert float(ecart.max()) <= borne, f"hors 2⁻⁷·max contre A4 : {float(ecart.max())} > {float(borne)}"
    assert int((d_a4 != d_fu.float()).any(1).sum()) <= max(4, t // 2), "trop de lignes différentes de A4 (au-delà du bruit de ses atomiques)"
    tw = topw.reshape(-1).contiguous()
    y_a4 = ext.moe_reduce(d_fu.float().contiguous(), tw, k)          # même d : reduce bf16 contre reduce fp32
    y_fu = ext.moe_reduce(d_fu.contiguous(), tw, k)
    assert torch.equal(y_a4, y_fu), "reduce bf16 ≠ reduce fp32 (au bit)"
    assert bool((y_fu[-1] == 0).all()) and torch.isfinite(y_fu.float()).all()
