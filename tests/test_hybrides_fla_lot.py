"""KDA et Mamba2 : décodage du LOT en un lancement fla (`forward_batch`,
`decode_static_batch`) contre b appels de la voie torch séquence par séquence
(règle 9, poste7-priorite-apres-campagne-17-09 § 2). Bras cassant : l'état KDA
passé à fla sans la transposition [V, K] → [K, V] doit rendre rouge."""
import pytest
import torch

from acvram.engine import kda as K, mamba2 as M
from acvram.engine.kda import KimiDeltaAttention
from acvram.engine.mamba2 import Mamba2Mixer

pytestmark = pytest.mark.skipif(K._recurrent_kda() is None or M._recurrent_gla() is None,
                                reason="flash-linear-attention absent")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEUIL = 2 ** -7


@pytest.fixture(autouse=True)
def _sans_grad():
    with torch.no_grad():
        yield


def _lin(o, i, s=0.4):
    m = torch.nn.Linear(i, o, bias=False)
    m.weight.data.uniform_(-s, s)
    return m.float()


def _ecart(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()


def _kda(H=48, NH=3, D=16, KER=4, R=8):
    torch.manual_seed(20260917)
    di = NH * D
    return KimiDeltaAttention(
        q_proj=_lin(di, H), k_proj=_lin(di, H), v_proj=_lin(di, H), out_proj=_lin(H, di),
        f_a=_lin(R, H), f_b=_lin(di, R), g_a=_lin(R, H), g_b=_lin(di, R), beta=_lin(NH, H),
        conv_q=torch.randn(di, KER) * 0.3, conv_k=torch.randn(di, KER) * 0.3, conv_v=torch.randn(di, KER) * 0.3,
        dt_bias=torch.rand(di) - 0.5, a=-(torch.rand(NH) * 2).exp(), norm_weight=torch.ones(D) + 0.1 * torch.randn(D),
        num_heads=NH, head_dim=D).to(DEV), H


def _mamba(H=48, NHD=4, P=8, G=2, N=8, KER=4):
    torch.manual_seed(20260918)
    inner, conv_dim = NHD * P, NHD * P + 2 * G * N
    return Mamba2Mixer(
        in_proj=_lin(inner + conv_dim + NHD, H), out_proj=_lin(H, inner),
        conv_weight=torch.randn(conv_dim, KER) * 0.3, conv_bias=torch.randn(conv_dim) * 0.1,
        dt_bias=torch.rand(NHD) - 0.5, A=-(torch.rand(NHD) * 2).exp(), D=torch.randn(NHD),
        norm_weight=torch.ones(inner) + 0.1 * torch.randn(inner),
        num_heads=NHD, head_dim=P, n_groups=G, state_size=N).to(DEV), H


def _etats_et_jetons(couche, H, b=3):
    etats, xs = [], []
    for s in range(b):
        torch.manual_seed(100 + s)
        _, e = couche(torch.randn(10 + 5 * s, H, device=DEV), None)   # prefill torch (CPU : boucle)
        etats.append(e); xs.append(torch.randn(1, H, device=DEV))
    y_ref = torch.cat([couche(xs[s], etats[s])[0] for s in range(b)])
    e_ref = [couche(xs[s], etats[s])[1] for s in range(b)]
    return etats, torch.cat(xs), y_ref, e_ref


@pytest.mark.parametrize("fabrique", [_kda, _mamba], ids=["kda", "mamba2"])
def test_le_lot_en_un_lancement_vaut_b_appels(fabrique):
    couche, H = fabrique()
    etats, h, y_ref, e_ref = _etats_et_jetons(couche, H)
    b = h.shape[0]
    y, e_new = couche.forward_batch(h, etats)
    assert _ecart(y, y_ref) < SEUIL, _ecart(y, y_ref)
    for s in range(b):
        for j in range(len(e_ref[s])):
            assert _ecart(e_new[s][j], e_ref[s][j]) < max(SEUIL, 1e-5), (s, j)
    statics = [couche.new_static(torch.device(DEV)) for _ in range(b)]
    for s in range(b):
        couche.static_load(statics[s], etats[s])
    y2 = couche.decode_static_batch(h, statics)
    assert _ecart(y2, y_ref) < SEUIL
    cles = [k for k in statics[0] if k != "lot"]
    for s in range(b):
        for j, k in enumerate(cles):
            assert _ecart(statics[s][k], e_ref[s][j]) < max(SEUIL, 1e-5), (s, k)
    assert all(st["lot"] == (0, i) for i, st in enumerate(statics))


def test_bras_cassant_kda_sans_la_transposition_de_l_etat(monkeypatch):
    couche, H = _kda()
    etats, h, y_ref, _ = _etats_et_jetons(couche, H)
    vrai = K._recurrent_kda()

    def sans_transposition(q, k, v, g, beta, initial_state, **kw):
        return vrai(q, k, v, g=g, beta=beta, initial_state=initial_state.transpose(-1, -2).contiguous(), **kw)
    monkeypatch.setattr(K, "_recurrent_kda", lambda: sans_transposition)
    y, _ = couche.forward_batch(h, etats)
    assert _ecart(y, y_ref) > SEUIL, "l'axe [K, V] oublié doit se voir"
