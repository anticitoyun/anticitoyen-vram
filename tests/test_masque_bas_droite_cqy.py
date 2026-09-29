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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="écart sur carte : sous carte.sh")
def test_ecart_carte_documente_et_biais_inactif(monkeypatch):
    """Mesuré le 29/09 (prise cqy, scratchpad/poste1-cqy-29-09/ecart.tsv) : sur carte, le biais bas-droite n'est PAS au bit
    du masque dense (10/14 cas, |Δ| 1e-3 à 3,9e-3 en bf16) — d'où `_BIAIS_BAS_DROITE = False`. Ce test fixe les deux faits :
    le biais reste inactif par défaut, et l'écart existe (borné). S'il disparaît (autre torch, autre noyau), le test casse :
    la décision cqy est alors à revoir, pas à ignorer."""
    assert layers._BIAIS_BAS_DROITE is False
    ecarts = []
    for n_rep in (1, 4):
        for q_offset, q_len, kv_len in FORMES + [(16, 8192, 8208), (1024, 7168, 8192)]:
            dense, biais, appels = _paire(monkeypatch, q_offset, q_len, kv_len, torch.bfloat16, "cuda", n_rep)
            assert appels == ["CausalBias"], appels
            ecarts.append((dense.float() - biais.float()).abs().max().item())
            monkeypatch.undo()
    assert layers._BIAIS_BAS_DROITE is False
    assert max(ecarts) > 0, "biais bas-droite devenu au bit sur carte : revoir la décision cqy (activer ?)"
    assert max(ecarts) <= 1e-2, f"écart hors de la borne mesurée (≤ 3,9e-3) : {max(ecarts):.2e}"


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
