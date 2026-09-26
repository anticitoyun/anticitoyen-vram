"""Pièce 194 b2 (poste1, 25/09) : β‖α sur un second flux pendant qkv‖gate (ACVRAM_GDN_AB_FLUX=1). Au bit du chemin série
par construction ; le test qui compte est celui qui CASSE quand la jointure manque : un retard posé sur le second flux fait
lire au consommateur une mémoire que β‖α n'a pas encore écrite."""
import pytest
import torch

from acvram.engine import gdn
from acvram.engine.layers import QuantLinear
from acvram.quant.formats import PlainTensor

NV, K, NK, DK, DV = 48, 5120, 2, 8, 8
carte = pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")
RETARD = 135_000_000          # ≈ 50 ms de cycles sur le second flux : le flux principal a fini bien avant


def _couche(monkeypatch, flux, device="cuda"):
    """Une vraie GatedDeltaNet avec β‖α fusionnés (fabrique de test_gdn_ab_175)."""
    monkeypatch.setattr(gdn, "_GDN_AB", "auto")
    monkeypatch.setattr(gdn, "_GDN_AB_FLUX", flux)
    torch.manual_seed(194)
    lin_bf16 = lambda w: QuantLinear(PlainTensor(w, tuple(w.shape), "bf16"), None, None, w.shape[0], w.shape[1])  # noqa: E731
    wa = (torch.randn(NV, K, device=device) * 0.02).to(torch.bfloat16)
    wb = (torch.randn(NV, K, device=device) * 0.02).to(torch.bfloat16)
    lin = lambda o, i: torch.nn.Linear(i, o, bias=False).to(device=device, dtype=torch.bfloat16)  # noqa: E731
    cd = 2 * NK * DK + NV * DV
    c = gdn.GatedDeltaNet(qkv=lin(cd, K), gate=lin(NV * DV, K), alpha=lin_bf16(wa), beta=lin_bf16(wb),
                          out=lin(K, NV * DV), conv_weight=torch.randn(cd, 4, device=device),
                          dt_bias=torch.rand(NV, device=device), a_log=torch.rand(NV, device=device),
                          norm_weight=torch.ones(DV, device=device),
                          num_k_heads=NK, num_v_heads=NV, head_k_dim=DK, head_v_dim=DV)
    c.ab = c._fusionner_ab()
    if c.ab is None and device == "cpu":           # la fusion refuse l'hôte (raison « hote ») : β‖α posé à la main
        w = torch.cat([wb, wa]).contiguous()
        c.ab = QuantLinear(PlainTensor(w, tuple(w.shape), "bf16"), None, None, w.shape[0], w.shape[1])
    assert c.ab is not None
    return c


def _retarder(monkeypatch, c):
    orig = c._ab
    monkeypatch.setattr(c, "_ab", lambda x, **kw: (torch.cuda._sleep(RETARD), orig(x, **kw))[1])


@carte
@pytest.mark.parametrize("m", [1, 2, 8, 16])
def test_au_bit_du_chemin_serie_meme_retarde(monkeypatch, m):
    x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16, generator=torch.Generator("cuda").manual_seed(m))
    temoin = _couche(monkeypatch, False)._projections(x)
    torch.cuda.synchronize()
    c = _couche(monkeypatch, True)
    _retarder(monkeypatch, c)
    sortie = [t.clone() for t in c._projections(x)]            # consommateur sur le flux principal, sans synchronisation
    torch.cuda.synchronize()
    for nom, a, b in zip(("qkv", "z", "b", "a"), temoin, sortie):
        assert torch.equal(a, b), (m, nom)


@carte
def test_sans_jointure_le_consommateur_lit_trop_tot(monkeypatch):
    """Le contrôle doit pouvoir rendre « faux » : jointure neutralisée → b et a lus avant d'être écrits."""
    x = torch.randn(8, K, device="cuda", dtype=torch.bfloat16, generator=torch.Generator("cuda").manual_seed(1940))
    temoin = _couche(monkeypatch, False)._projections(x)
    torch.cuda.synchronize()
    c = _couche(monkeypatch, True)
    _retarder(monkeypatch, c)
    monkeypatch.setattr(gdn, "_joindre", lambda courant, flux: None)
    _, _, b, a = c._projections(x)
    b_lu, a_lu = b.clone(), a.clone()
    torch.cuda.synchronize()
    assert not (torch.equal(b_lu, temoin[2]) and torch.equal(a_lu, temoin[3])), "la course n'a pas été vue"


@carte
def test_la_ligne_de_regime_le_dit(monkeypatch):
    monkeypatch.setattr(gdn, "_GDN_AB", "auto")
    monkeypatch.setattr(gdn, "_GDN_AB_FLUX", True)
    monkeypatch.setitem(gdn.AB_BILAN, "fusionnees", 0)
    assert "abflux" not in gdn._ab_texte(), "inerte sans β‖α fusionné : la ligne ne doit pas le prétendre"
    monkeypatch.setitem(gdn.AB_BILAN, "fusionnees", 48)
    assert gdn._ab_texte().endswith(" abflux")
    monkeypatch.setattr(gdn, "_GDN_AB_FLUX", False)
    assert "abflux" not in gdn._ab_texte()


def test_a_sec_aucun_flux(monkeypatch):
    monkeypatch.setattr(gdn, "_FLUX_AB", {})
    c = _couche(monkeypatch, True, device="cpu")
    temoin = _couche(monkeypatch, False, device="cpu")._projections(torch.ones(2, K, dtype=torch.bfloat16))
    sortie = c._projections(torch.ones(2, K, dtype=torch.bfloat16))
    assert gdn._FLUX_AB == {}
    for a, b in zip(temoin, sortie):
        assert torch.equal(a, b)


def test_le_defaut_est_le_second_flux():
    """Verdict 194 b2 (chef, 25/09) : 1 au défaut. Casse si le défaut revient à 0, dans le code OU dans la table."""
    import os
    import subprocess
    import sys
    from acvram import cli, regime
    v = next(v for v in regime.VARIABLES if v.nom == "GDN_AB_FLUX")
    assert v.defaut == "1" and "ACVRAM_GDN_AB_FLUX" in cli.VARIABLES_LUES
    env = {k: val for k, val in os.environ.items() if k != "ACVRAM_GDN_AB_FLUX"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    r = subprocess.run([sys.executable, "-c", "from acvram.engine import gdn; print(gdn._GDN_AB_FLUX)"],
                       env=env, capture_output=True, text=True, check=True)
    assert r.stdout.strip() == "True", r.stdout + r.stderr
