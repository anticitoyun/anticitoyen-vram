"""Gated DeltaNet par les noyaux de flash-linear-attention (`ACVRAM_GDN=fla`)
contre la voie torch (référence transformers, `ACVRAM_GDN=torch`) — règle 9 :
une optimisation qui change la sortie est un bogue (sage-priorite-apres-
campagne-17-09 § 2).

Sans carte, l'interpréteur Triton n'exécute que le noyau récurrent
(décodage) ; le noyau par blocs (prefill) est jugé sur carte. Bras cassant
exigé par Sage : l'état passé à fla avec l'axe [K, V] inversé doit rendre le
test rouge — il le fait parce que l'état vient d'un prefill à 40 jetons et
continue sur 24 : une transposition silencieuse (d_k = d_v) change la suite.
"""
import os

import pytest
import torch

from acvram import regime
from acvram.engine import gdn as G
from acvram.engine.gdn import GatedDeltaNet

pytestmark = pytest.mark.skipif(G._fla() is None, reason="flash-linear-attention absent")
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau par blocs : carte requise")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
H, NK, NV, DK, DV, KER = 64, 2, 4, 16, 16, 4
SEUIL = 2 ** -7                    # relatif, le critère de Sage (bf16) ; ici tout est fp32


@pytest.fixture(autouse=True)
def _sans_grad():
    with torch.no_grad():
        yield


def _couche(seed=20260917) -> GatedDeltaNet:
    torch.manual_seed(seed)

    def lin(o, i):
        m = torch.nn.Linear(i, o, bias=False)
        m.weight.data.uniform_(-0.4, 0.4)
        return m.float()
    conv_dim = 2 * NK * DK + NV * DV
    return GatedDeltaNet(
        qkv=lin(conv_dim, H), gate=lin(NV * DV, H), alpha=lin(NV, H), beta=lin(NV, H), out=lin(H, NV * DV),
        conv_weight=torch.randn(conv_dim, KER) * 0.3, dt_bias=torch.rand(NV) - 0.5,
        a_log=torch.rand(NV) * 3 - 2, norm_weight=torch.ones(DV) + 0.1 * torch.randn(DV),
        num_k_heads=NK, num_v_heads=NV, head_k_dim=DK, head_v_dim=DV).to(DEV)


def _ecart(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()


def _voie(monkeypatch, voie):
    monkeypatch.setattr(G, "_GDN_VOIE", voie)


def test_la_ligne_de_regime_porte_la_voie_gdn(monkeypatch):
    _voie(monkeypatch, "fla")
    assert "ACVRAM_GDN=fla" in regime.regime_ligne(), regime.regime_ligne()
    _voie(monkeypatch, "torch")
    assert "ACVRAM_GDN=torch" in regime.regime_ligne()


def test_decodage_fla_suit_torch_apres_un_prefill_torch(monkeypatch):
    couche = _couche()
    x = torch.randn(64, H, device=DEV)
    _voie(monkeypatch, "torch")
    y_ref, etat = couche(x[:40], None)                        # prefill torch (40 jetons)
    ys_t, e_t = [], etat
    for i in range(40, 64):
        y, e_t = couche(x[i:i + 1], e_t)
        ys_t.append(y)
    _voie(monkeypatch, "fla")
    ys_f, e_f = [], etat
    for i in range(40, 64):
        y, e_f = couche(x[i:i + 1], e_f)
        ys_f.append(y)
    yt, yf = torch.cat(ys_t), torch.cat(ys_f)
    assert _ecart(yf, yt) < SEUIL, _ecart(yf, yt)
    assert _ecart(e_f[1], e_t[1]) < SEUIL


def test_bras_cassant_l_axe_k_v_de_l_etat_inverse_rend_rouge(monkeypatch):
    couche = _couche()
    x = torch.randn(64, H, device=DEV)
    _voie(monkeypatch, "torch")
    _, etat = couche(x[:40], None)
    ys_t, e_t = [], etat
    for i in range(40, 64):
        y, e_t = couche(x[i:i + 1], e_t)
        ys_t.append(y)
    _voie(monkeypatch, "fla")
    ys_f, e_f = [], (etat[0], etat[1].transpose(-1, -2).contiguous())   # [V, K] au lieu de [K, V]
    for i in range(40, 64):
        y, e_f = couche(x[i:i + 1], e_f)
        ys_f.append(y)
    assert _ecart(torch.cat(ys_f), torch.cat(ys_t)) > SEUIL, "l'axe inversé doit se voir"


def test_le_lot_de_b_sequences_en_un_lancement_vaut_b_appels(monkeypatch):
    couche = _couche()
    b = 3
    _voie(monkeypatch, "torch")
    etats, xs = [], []
    for s in range(b):
        torch.manual_seed(100 + s)
        x = torch.randn(10 + 5 * s, H, device=DEV)
        _, e = couche(x, None)
        etats.append(e); xs.append(torch.randn(1, H, device=DEV))
    y_ref = torch.cat([couche(xs[s], etats[s])[0] for s in range(b)])
    e_ref = [couche(xs[s], etats[s])[1] for s in range(b)]
    _voie(monkeypatch, "fla")
    h = torch.cat(xs)
    y, e_new = couche.forward_batch(h, etats)                     # eager
    assert _ecart(y, y_ref) < SEUIL, _ecart(y, y_ref)
    for s in range(b):
        # l'état de convolution porte la projection : GEMM à M=b contre M=1,
        # ordre de réduction différent (5e-7), pas le même bit
        assert _ecart(e_new[s][1], e_ref[s][1]) < SEUIL and _ecart(e_new[s][0], e_ref[s][0]) < 1e-5
    statics = [couche.new_static(torch.device(DEV)) for _ in range(b)]   # formes fixes, tampon groupé
    for s in range(b):
        GatedDeltaNet.static_load(statics[s], etats[s])
    y2 = couche.decode_static_batch(h, statics)
    assert _ecart(y2, y_ref) < SEUIL
    for s in range(b):
        assert _ecart(statics[s]["S"], e_ref[s][1]) < SEUIL and _ecart(statics[s]["conv"], e_ref[s][0]) < 1e-5
    assert all(st["lot"] == (0, i) for i, st in enumerate(statics)), "créneaux 0..b-1 contigus dans le lot 0"


def test_un_seul_lot_sert_seize_creneaux_puis_un_second():
    couche = _couche()
    from acvram.engine.lot_etats import LOT
    sts = [couche.new_static(torch.device(DEV)) for _ in range(LOT + 1)]
    assert len(couche._lots) == 2 and sts[LOT]["lot"] == (1, 0)
    assert sts[0]["S"].data_ptr() == couche._lots[0]["S_"].data_ptr()


@CUDA
def test_prefill_fla_suit_torch_sur_carte(monkeypatch):
    couche = _couche()
    x = torch.randn(64, H, device=DEV)
    _voie(monkeypatch, "torch")
    y_t, e_t = couche(x, None)
    _voie(monkeypatch, "fla")
    y_f, e_f = couche(x, None)
    assert _ecart(y_f, y_t) < SEUIL, _ecart(y_f, y_t)
    assert _ecart(e_f[1], e_t[1]) < SEUIL
    # continuité prefill fla → décodage fla, contre tout-torch
    _voie(monkeypatch, "torch")
    y_suite_t, _ = couche(x[40:], couche(x[:40], None)[1])
    _voie(monkeypatch, "fla")
    e40 = couche(x[:40], None)[1]
    ys = []
    for i in range(40, 64):
        y, e40 = couche(x[i:i + 1], e40)
        ys.append(y)
    assert _ecart(torch.cat(ys), y_suite_t) < SEUIL


def test_un_moteur_charge_a_sec_ecrit_la_voie_gdn_dans_son_regime(converted, monkeypatch):
    """Le contrôle demandé par Jérôme/Sage : pas la lecture du code, la ligne
    de régime d'un chargement réel (celle que les JSON de certification
    copient dans `engine_regime`)."""
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    _voie(monkeypatch, "fla")
    engine = Engine(load_model(converted, dtype=torch.float32, device_override="cpu"), None,
                    max_batch_size=2, max_model_len=256)
    ligne = engine.regime_ligne()
    assert "ACVRAM_GDN=fla" in ligne and engine.regime()["gdn"] == "fla", ligne
    _voie(monkeypatch, "torch")
    assert "ACVRAM_GDN=torch" in engine.regime_ligne()
