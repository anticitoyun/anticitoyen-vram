"""cqy (29/09) : biais causal aligné en bas à droite (SDPA `causal_lower_right`) à la place du masque dense [q × kv] quand
les requêtes sont les dernières positions (préfixe en cache : kv = q_offset + q). Décision d'arithmétique : ÉQUIVALENCE AU
BIT contre le masque dense, sur processeur ici et sur carte (test GPU, sous carte.sh). Et le biais ne sert que dans ce cas."""
import pytest
import torch

from acvram.engine import layers

class _Espion:
    """`layers.F` seul : les appels internes de torch (dispatch de CausalBias) gardent le vrai SDPA."""
    def __init__(self):
        self.appels = []

    def __getattr__(self, nom):
        return getattr(torch.nn.functional, nom)

    def scaled_dot_product_attention(self, *a, **kw):
        self.appels.append(type(kw.get("attn_mask")).__name__)
        return torch.nn.functional.scaled_dot_product_attention(*a, **kw)


FORMES = [(1, 16, 17), (37, 61, 98), (5, 33, 38), (160, 512, 672), (16, 3000, 3016)]    # (q_offset, q, kv = q_offset + q)


def _paire(monkeypatch, q_offset, q_len, kv_len, dt, dev, n_rep):
    torch.manual_seed(q_len)
    q = torch.randn(q_len, 8, 64, dtype=dt, device=dev)
    k = torch.randn(kv_len, 8 // n_rep, 64, dtype=dt, device=dev)
    v = torch.randn(kv_len, 8 // n_rep, 64, dtype=dt, device=dev)
    monkeypatch.setattr(layers, "_BIAIS_BAS_DROITE", False)
    dense = layers.attention(q, k, v, True, 0.125, q_offset, n_rep=n_rep)
    monkeypatch.setattr(layers, "_BIAIS_BAS_DROITE", True)
    espion = _Espion()
    monkeypatch.setattr(layers, "F", espion)
    biais = layers.attention(q, k, v, True, 0.125, q_offset, n_rep=n_rep)
    return dense, biais, espion.appels


@pytest.mark.parametrize("n_rep", [1, 2])
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("forme", FORMES)
def test_au_bit_processeur(monkeypatch, forme, dt, n_rep):
    q_offset, q_len, kv_len = forme
    dense, biais, appels = _paire(monkeypatch, q_offset, q_len, kv_len, dt, "cpu", n_rep)
    assert appels == ["CausalBias"], appels                  # le biais sert, aucun masque dense
    assert torch.equal(dense, biais), (dense.float() - biais.float()).abs().max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="équivalence sur carte : sous carte.sh")
@pytest.mark.parametrize("n_rep", [1, 4])
@pytest.mark.parametrize("forme", FORMES + [(16, 8192, 8208), (1024, 7168, 8192)])   # préfixe d'un bloc (8fx), long préfixe
def test_au_bit_carte(monkeypatch, forme, n_rep):
    q_offset, q_len, kv_len = forme
    dense, biais, appels = _paire(monkeypatch, q_offset, q_len, kv_len, torch.bfloat16, "cuda", n_rep)
    assert appels == ["CausalBias"], appels
    assert torch.equal(dense, biais), (dense.float() - biais.float()).abs().max()


@pytest.mark.parametrize("kw", [dict(q_offset=0, q_len=40, kv_len=40),              # sans préfixe : is_causal, inchangé
                                dict(q_offset=3, q_len=20, kv_len=30),              # clés au-delà des requêtes
                                dict(q_offset=5, q_len=20, kv_len=25, window=7),    # fenêtre glissante
                                dict(q_offset=5, q_len=20, kv_len=25, images=[(8, 12)])])
def test_biais_seulement_quand_les_requetes_sont_les_dernieres(monkeypatch, kw):
    q = torch.randn(kw["q_len"], 2, 8)
    k = torch.randn(kw["kv_len"], 2, 8)
    monkeypatch.setattr(layers, "_BIAIS_BAS_DROITE", True)
    espion = _Espion()
    monkeypatch.setattr(layers, "F", espion)
    layers.attention(q, k, k.clone(), True, None, kw["q_offset"], kw.get("window", 0), 1, kw.get("images"))
    assert espion.appels and "CausalBias" not in espion.appels, espion.appels
