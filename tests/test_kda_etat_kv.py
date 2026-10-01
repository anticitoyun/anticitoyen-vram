"""61w (01/10) : l'état KDA en disposition [K, V] (celle de fla) partout — sorties et états AU BIT de l'ancien chemin
[V, K] (règle 9, `torch.equal`, jamais une tolérance).

* décodage du lot (`decode_static_batch`) : l'ancien chemin transposait S vers fla puis l'état final en retour ; le
  nouveau appelle `fused_recurrent_kda_fwd` en place (h0 = ht = le tampon statique). Témoin : l'ancien appel public
  (`fused_recurrent_kda`, état transposé, état final alloué puis retransposé), mêmes entrées. Joué sur carte ET sans
  carte (TRITON_INTERPRET=1 : les noyaux de fla interprétés).
* décodage b=1 (`kda_decode`, extension) : le noyau lit la colonne i de S [K, V] ; témoin `kda_decode_vk`, l'ancien
  noyau (ligne i de S [V, K]), 64 pas chaînés, sorties et état égaux au bit (l'état témoin transposé). Carte seulement.
* bras cassants : une transposition qui revient (état passé à fla sans [K, V], ou noyau b=1 lu par lignes) doit
  rendre rouge — sinon l'égalité ne prouve pas l'orientation."""
import pytest
import torch

from acvram.engine import kda as K
from acvram.engine.kda import KimiDeltaAttention

pytestmark = pytest.mark.skipif(K._recurrent_kda() is None, reason="flash-linear-attention absent")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def _sans_grad():
    with torch.no_grad():
        yield


def _lin(o, i, s=0.4):
    m = torch.nn.Linear(i, o, bias=False)
    m.weight.data.uniform_(-s, s)
    return m.float()


def _kda(H=64, NH=4, D=32, KER=4, R=8, graine=61):
    torch.manual_seed(graine)
    di = NH * D
    return KimiDeltaAttention(
        q_proj=_lin(di, H), k_proj=_lin(di, H), v_proj=_lin(di, H), out_proj=_lin(H, di),
        f_a=_lin(R, H), f_b=_lin(di, R), g_a=_lin(R, H), g_b=_lin(di, R), beta=_lin(NH, H),
        conv_q=torch.randn(di, KER) * 0.3, conv_k=torch.randn(di, KER) * 0.3, conv_v=torch.randn(di, KER) * 0.3,
        dt_bias=torch.rand(di) - 0.5, a=-(torch.rand(NH) * 2).exp(), norm_weight=torch.ones(D) + 0.1 * torch.randn(D),
        num_heads=NH, head_dim=D).to(DEV), H


def _ancien_lot(couche, h, conv, S_kv):
    """L'ancien `decode_static_batch` (avant 61w) sur un état [V, K] : transposition vers fla, appel public, état final
    alloué puis retransposé. Rend (y, S final [K, V]) pour comparer à la nouvelle disposition."""
    cq, ck, cv = (c.clone() for c in conv)
    S_vk = S_kv.transpose(-1, -2).contiguous()
    q, k, v, g1, beta, g2 = couche._lot_projete(h, cq, ck, cv)
    o, S_fin = K._recurrent_kda()(q, k, v, g=g1, beta=beta, initial_state=S_vk.transpose(-1, -2).contiguous(),
                                  output_final_state=True, use_qk_l2norm_in_kernel=True)
    S_vk_new = S_fin.transpose(-1, -2)
    return couche._lot_sortie(o, g2, h.dtype), S_vk_new.transpose(-1, -2).contiguous()


def _creneaux(couche, b, graine=7):
    g_ = torch.Generator().manual_seed(graine)
    st = [couche.new_static(torch.device(DEV)) for _ in range(b)]
    for s in st:
        for c in ("cq", "ck", "cv"):
            s[c].copy_(torch.randn(s[c].shape, generator=g_) * 0.5)
        s["S"].copy_(torch.randn(s["S"].shape, generator=g_) * 0.2)          # [nh, K, V]
    return st


@pytest.mark.parametrize("b", [1, 3, 12])
def test_lot_au_bit_de_l_ancien_chemin(b):
    couche, H = _kda()
    st = _creneaux(couche, b)
    g_ = torch.Generator().manual_seed(11)
    for pas in range(6):
        h = (torch.randn(b, H, generator=g_)).to(DEV)
        conv = [torch.stack([s[c] for s in st]) for c in ("cq", "ck", "cv")]
        S_kv = torch.stack([s["S"] for s in st])
        y_ancien, S_ancien = _ancien_lot(couche, h, conv, S_kv)
        y = couche.decode_static_batch(h, st)
        assert torch.equal(y, y_ancien), (b, pas)
        assert torch.equal(torch.stack([s["S"] for s in st]), S_ancien), (b, pas)


def test_bras_cassant_lot_etat_transpose(monkeypatch):
    """Une transposition qui revient (l'état [K, V] passé à fla comme s'il était [V, K]) doit casser l'égalité."""
    couche, H = _kda()
    st = _creneaux(couche, 3)
    vrai = K._recurrent_kda_fwd()

    def transpose(q, k, v, g, beta, initial_state, **kw):
        S_t = initial_state.transpose(-1, -2).contiguous()
        o, _ = vrai(q, k, v, g=g, beta=beta, initial_state=S_t, **kw)
        initial_state.copy_(S_t.transpose(-1, -2))
        return o, initial_state
    h = torch.randn(3, H, generator=torch.Generator().manual_seed(11)).to(DEV)
    conv = [torch.stack([s[c] for s in st]) for c in ("cq", "ck", "cv")]
    y_ancien, _ = _ancien_lot(couche, h, conv, torch.stack([s["S"] for s in st]))
    monkeypatch.setattr(K, "_recurrent_kda_fwd", lambda: transpose)
    y = couche.decode_static_batch(h, st)
    assert not torch.equal(y, y_ancien)


carte_ext = pytest.mark.skipif(not torch.cuda.is_available() or K._extension() is None
                               or not hasattr(K._extension(), "kda_decode_vk"),
                               reason="noyau CUDA kda_decode : carte et extension requises")


def _entrees_b1(NH, D, KER, graine):
    g_ = torch.Generator().manual_seed(graine)
    di = NH * D
    bf = lambda *s: (torch.randn(*s, generator=g_) * 0.7).to("cuda", torch.bfloat16)
    return bf(di), bf(di), bf(di), bf(di), bf(di), bf(NH)


@carte_ext
@pytest.mark.parametrize("D", [64, 128])
def test_noyau_b1_au_bit_du_temoin(D):
    NH, KER, ext = 32, 4, K._extension()
    di = NH * D
    g_ = torch.Generator().manual_seed(D)
    f = lambda *s, e=0.3: (torch.randn(*s, generator=g_) * e).to("cuda")
    wq, wk, wv = f(di, KER), f(di, KER), f(di, KER)
    dt, a, nw = f(di), -torch.rand(NH, generator=g_).to("cuda") - 0.5, 1 + f(D, e=0.1)
    conv_n = [f(di, KER - 1, e=0.5) for _ in range(3)]
    conv_t = [c.clone() for c in conv_n]
    S_kv = f(NH, D, D, e=0.2)                                   # [h, K, V]
    S_vk = S_kv.transpose(-1, -2).contiguous()                 # le témoin : [h, V, K]
    for pas in range(64):
        x = _entrees_b1(NH, D, KER, 1000 * D + pas)
        y = ext.kda_decode(*x, wq, wk, wv, *conv_n, dt, a, nw, S_kv, 1e-6)
        y_t = ext.kda_decode_vk(*x, wq, wk, wv, *conv_t, dt, a, nw, S_vk, 1e-6)
        if not (torch.equal(y, y_t) and torch.equal(S_kv, S_vk.transpose(-1, -2))):
            ulp_y = ((y.float() - y_t.float()).abs() / (y_t.float().abs() * 2.0 ** -7).clamp(min=2.0 ** -133)).max()
            dS = (S_kv - S_vk.transpose(-1, -2)).abs()
            ulp_S = (dS / (S_kv.abs() * 2.0 ** -23).clamp(min=2.0 ** -149)).max()
            raise AssertionError(f"D={D} pas {pas} : sortie bf16 {int((y != y_t).sum())} éléments ≠, max {float(ulp_y):.1f} "
                                 f"ulp ; état fp32 {int((dS > 0).sum())} éléments ≠, max {float(ulp_S):.1f} ulp")
        assert all(torch.equal(c, d) for c, d in zip(conv_n, conv_t)), pas


@carte_ext
def test_bras_cassant_noyau_b1_lu_par_lignes():
    """Le nouveau noyau nourri d'un état [V, K] (une transposition oubliée en amont) doit différer du témoin."""
    NH, D, KER, ext = 32, 128, 4, K._extension()
    di = NH * D
    g_ = torch.Generator().manual_seed(3)
    f = lambda *s, e=0.3: (torch.randn(*s, generator=g_) * e).to("cuda")
    w = [f(di, KER) for _ in range(3)]
    dt, a, nw = f(di), -torch.rand(NH, generator=g_).to("cuda") - 0.5, 1 + f(D, e=0.1)
    conv = [f(di, KER - 1, e=0.5) for _ in range(3)]
    S_vk = f(NH, D, D, e=0.2)
    x = _entrees_b1(NH, D, KER, 5)
    y_t = ext.kda_decode_vk(*x, *w, *[c.clone() for c in conv], dt, a, nw, S_vk.clone(), 1e-6)
    y = ext.kda_decode(*x, *w, *[c.clone() for c in conv], dt, a, nw, S_vk.clone(), 1e-6)
    assert not torch.equal(y, y_t)
