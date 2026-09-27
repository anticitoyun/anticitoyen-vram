"""d19 (27/09) : cœur GDN par tranches (ACVRAM_GDN_MORCEAU), état porté d'une tranche à l'autre, AU BIT du cœur d'un seul
tenant (sortie et état final) — c'est ce qui l'autorise par défaut. Cassant : un état NON porté entre tranches doit rendre faux."""
import pytest
import torch

import acvram.engine.gdn as G
from tests.test_gdn_coeur_lot_245 import H, _couche

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="fla exige la carte")

LONGUEURS = [(130,), (1000,), (4097,), (64,), (63,)]


def _deux(couche, n, dtype, etat, monkeypatch, morceau):
    g = torch.Generator(device="cpu").manual_seed(n)
    h = torch.randn(n, H, generator=g).to("cuda", dtype)
    monkeypatch.setattr(G, "_GDN_MORCEAU", 0)
    y0, e0 = couche(h, etat)
    monkeypatch.setattr(G, "_GDN_MORCEAU", morceau)
    y1, e1 = couche(h, etat)
    return torch.equal(y0, y1) and torch.equal(e0[0], e1[0]) and torch.equal(e0[1], e1[1])


@CUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n", [130, 1000, 4097])
@pytest.mark.parametrize("morceau", [64, 128, 192])
@pytest.mark.parametrize("avec", [False, True])
def test_morceaux_au_bit(dtype, n, morceau, avec, monkeypatch):
    couche = _couche(dtype)
    etat = None
    if avec:
        g = torch.Generator(device="cpu").manual_seed(11)
        etat = couche(torch.randn(77, H, generator=g).to("cuda", dtype), None)[1]
    assert _deux(couche, n, dtype, etat, monkeypatch, morceau)


@CUDA
def test_cassant_etat_non_porte(monkeypatch):
    couche = _couche(torch.float32)
    vrai = couche._coeur_un
    monkeypatch.setattr(couche, "_coeur_un", lambda x, qkv, z, b, a, state: vrai(x, qkv, z, b, a, None))
    g = torch.Generator(device="cpu").manual_seed(3)
    h = torch.randn(1000, H, generator=g).to("cuda", torch.float32)
    monkeypatch.setattr(G, "_GDN_MORCEAU", 0)
    y0, _ = couche(h, None)
    monkeypatch.setattr(G, "_GDN_MORCEAU", 128)
    y1, _ = couche(h, None)
    assert not torch.equal(y0, y1)


def test_morceau_multiple_de_64(monkeypatch):
    import importlib
    monkeypatch.setenv("ACVRAM_GDN_MORCEAU", "100")
    with pytest.raises(ValueError):
        importlib.reload(G)
    monkeypatch.setenv("ACVRAM_GDN_MORCEAU", "4096")
    importlib.reload(G)
