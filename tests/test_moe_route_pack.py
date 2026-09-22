"""Frontend route+pack du MoE décodage-MMA en un noyau (`moe_route_pack`) :
(a) bit à bit égal au chemin torch (argsort stable, inv, poids, cnt, tuiles de
`_tuiles(cnt, bt, t_max)`, lignes rassemblées et rembourrées), fantômes
compris ; (b) la sortie du bloc est identique avec et sans le noyau ;
(c) compte de lancements par couche : casse si le routage torch revient."""
import os
import pytest

pytestmark = pytest.mark.pile_naturelle   # lit _stacks[nom][1], rendu (None) sous Marlin, défaut servi (T4 20/09)
import torch

from tests.test_moe_decode_mma_graphe import _bloc, _entree, CUDA, N_EXPERTS, TOP_K, T, CACHE


def _torch(bloc, x, topw, topi, bt, t_max):
    pg = bloc._stacks["gate_proj"]
    E = pg[1].shape[0]
    fantome = topi < 0
    flat_e = torch.where(fantome, torch.zeros_like(topi), topi).reshape(-1).to(torch.int64)
    tw = torch.where(fantome, torch.zeros_like(topw), topw).reshape(-1).to(torch.float32)
    flat_t = torch.arange(x.shape[0], device=x.device).repeat_interleave(topi.shape[1])
    ordre = torch.argsort(flat_e, stable=True)
    cnt = torch.zeros(E, dtype=torch.int64, device=x.device).scatter_add_(0, flat_e, torch.ones_like(flat_e))
    te, t0, tn = bloc._tuiles(cnt, bt, t_max=t_max)
    xs = x[flat_t[ordre]].to(torch.bfloat16)
    if xs.shape[1] != pg[4]:
        xs = torch.nn.functional.pad(xs, (0, pg[4] - xs.shape[1]))
    inv = torch.empty_like(ordre); inv[ordre] = torch.arange(ordre.numel(), device=x.device)
    return xs.contiguous(), ordre, inv, tw, cnt, te, t0, tn


@CUDA
@pytest.mark.parametrize("fantomes", [0, 5])
def test_route_pack_bit_a_bit(fantomes):
    from acvram.kernels import get_extension
    dev = torch.device("cuda:0")
    bloc = _bloc(dev)
    ext = get_extension()
    if not hasattr(ext, "moe_route_pack"):
        pytest.skip("extension sans moe_route_pack")
    x, topw, topi = _entree(dev)
    if fantomes:
        topi = topi.clone(); topi[T - fantomes:] = -1
        x = x.clone(); x[T - fantomes:] = 0
    # experts répétés entre jetons pour exercer le tri stable
    topi[1] = topi[0]; topi[3, :2] = topi[2, :2]
    bt = 16; t_max = -(-(T * TOP_K) // bt) + N_EXPERTS
    pg = bloc._stacks["gate_proj"]
    ref = _torch(bloc, x, topw, topi, bt, t_max)
    got = ext.moe_route_pack(topi.contiguous(), topw.contiguous(), x.contiguous(),
                             N_EXPERTS, bt, t_max, pg[4])
    noms = ("xs", "ordre", "inv", "tw", "cnt", "tile_e", "tile_t0", "tile_n")
    for nom, a, b in zip(noms, got, ref):
        assert torch.equal(a.to(b.dtype), b), f"{nom} : {int((a.to(b.dtype) != b).sum())} valeurs differentes"


@CUDA
def test_sortie_identique_avec_et_sans_noyau():
    from acvram.engine import model as M
    dev = torch.device("cuda:0")
    bloc = _bloc(dev)
    x, topw, topi = _entree(dev)
    topi = topi.clone(); topi[9:] = -1; x = x.clone(); x[9:] = 0
    ancien = M._MOE_ROUTE_PACK
    try:
        M._MOE_ROUTE_PACK = True; y1 = bloc._forward_grouped_mma(x, topw, topi)
        M._MOE_ROUTE_PACK = False; y0 = bloc._forward_grouped_mma(x, topw, topi)
    finally:
        M._MOE_ROUTE_PACK = ancien
    assert torch.equal(y0, y1)


@CUDA
def test_compte_de_lancements_par_couche():
    """3 GEMM + 2 quant_act + moe_act + moe_reduce_trie + route_pack = 8 noyaux
    par couche ; le chemin torch en lançait ~45. Casse si le routage torch revient."""
    from torch.profiler import profile, ProfilerActivity
    from acvram.engine import model as M
    dev = torch.device("cuda:0")
    bloc = _bloc(dev)
    x, topw, topi = _entree(dev)
    assert M._MOE_ROUTE_PACK, "le defaut doit etre le noyau route+pack"
    bloc._forward_grouped_mma(x, topw, topi); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        bloc._forward_grouped_mma(x, topw, topi); torch.cuda.synchronize()
    noyaux = [e for e in prof.key_averages() if e.self_device_time_total > 0
              and not e.key.startswith("aten::") and "Memcpy" not in e.key and "Memset" not in e.key]
    n = sum(e.count for e in noyaux)
    courts = [f"{e.count}x {e.key.split('(')[0].split('<')[0][-40:]}" for e in noyaux]
    assert n <= 8, f"{n} lancements par couche : {courts}"
