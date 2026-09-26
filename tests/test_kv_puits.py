"""Repli 104 (1) — puits d'attention (scellé `scratchpad/poste1-p104s5-23-09/scelle-puits.md`) : sous
`ACVRAM_KV_FORMAT=k8v4`, les 16 premières positions de chaque séquence gardent V en int8 par jeton (K l'est
déjà), dans une réserve indexée comme K. À sec (jumeau torch, processeur) ; le noyau est jugé contre ce jumeau
par `test_kv_puits_carte.py`.

(a) le défaut int8 reste au bit : un cache int8 ignore la demande de puits, écrit et relit exactement pareil ;
    le k8v4 sans puits aussi (le quartet est écrit partout) ;
(b) CASSE si les 16 premières positions repassent en k8v4 : elles doivent valoir la déquantification int8
    du jumeau AU BIT, et rester dans la borne int8 que le k8v4 dépasse sur les mêmes valeurs.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from acvram.memory import kv_k8v4
from acvram.memory.kvcache import KVCacheConfig, PagedKVCache

HKV, D, BS = 4, 128, 16


def _cache(dtype="k8v4", puits=16, nb=8):
    return PagedKVCache(KVCacheConfig(num_layers=1, num_kv_heads=HKV, head_dim=D, num_blocks=nb,
                                      dtype=dtype, device="cpu", puits=puits))


def _donnees(n, graine):
    torch.manual_seed(graine)
    return (torch.randn(n, HKV, D).to(torch.bfloat16), torch.randn(n, HKV, D).to(torch.bfloat16))


def _slots(blocs, positions):
    return torch.tensor([blocs[p // BS] * BS + p % BS for p in positions])


def _ecrire(c, k, v, blocs, positions):
    c.write(_slots(blocs, positions), k, v, positions=torch.tensor(positions))


def test_valeurs_admises_de_la_variable():
    assert kv_k8v4.puits_demandes({}) == 0
    assert kv_k8v4.puits_demandes({"ACVRAM_KV_PUITS": "16"}) == 16
    for faux in ("8", "32", "oui"):
        with pytest.raises(ValueError):
            kv_k8v4.puits_demandes({"ACVRAM_KV_PUITS": faux})
    with pytest.raises(ValueError):
        KVCacheConfig(num_layers=1, num_kv_heads=HKV, head_dim=D, num_blocks=4, dtype="k8v4", puits=8)


def test_a_defaut_int8_au_bit_puits_ignores():
    avec, sans = _cache("int8", puits=16), _cache("int8", puits=0)
    assert avec.cfg.puits == 0 and avec.puits_v is None
    k, v = _donnees(40, 1)
    for c in (avec, sans):
        _ecrire(c, k, v, [5, 2, 7], list(range(40)))
    for a, b in ((avec.k, sans.k), (avec.v, sans.v), (avec.k_scale, sans.k_scale), (avec.v_scale, sans.v_scale)):
        assert torch.equal(a, b)
    t = torch.tensor([5, 2, 7])
    assert all(torch.equal(x, y) for x, y in zip(avec.gather(t, 40, torch.float32), sans.gather(t, 40, torch.float32)))


def test_a_k8v4_sans_puits_inchange_le_quartet_est_ecrit_partout():
    avec, sans = _cache(puits=16), _cache(puits=0)
    assert sans.puits_v is None and avec.puits_v is not None
    k, v = _donnees(40, 2)
    for c in (avec, sans):
        _ecrire(c, k, v, [5, 2, 7], list(range(40)))
    for a, b in ((avec.k, sans.k), (avec.v, sans.v), (avec.k_scale, sans.k_scale), (avec.v_scale, sans.v_scale)):
        assert torch.equal(a, b)


def _verifier_puits(vd, v, n_puits):
    q, s = kv_k8v4.quantifier_k(v[:n_puits])
    assert torch.equal(vd[:n_puits], q.to(torch.float32) * s.to(torch.float32).unsqueeze(-1)), \
        "positions puits : V n'est pas la déquantification int8 du jumeau"
    # rn(x/s)·s à s = amax/127 : ≤ s/2 = amax/254 ; plus l'échelle arrondie en half (≤ 127·s·2⁻¹¹ = amax·2⁻¹¹)
    borne = v[:n_puits].float().abs().amax(-1) * (1 / 254 + 1 / 1024) + 1e-6
    assert bool(((vd[:n_puits] - v[:n_puits].float()).abs().amax(-1) <= borne).all())
    # le témoin distingue bien les deux formats : k8v4 sort de la borne int8 sur les mêmes valeurs
    k8v4 = kv_k8v4.dequantifier_v(*kv_k8v4.quantifier_v(v[:n_puits]), torch.float32)
    assert bool(((k8v4 - v[:n_puits].float()).abs().amax(-1) > borne).any())


def test_b_seize_premieres_positions_en_int8_le_reste_en_k8v4():
    c = _cache()
    k, v = _donnees(40, 3)
    _ecrire(c, k, v, [5, 2, 7], list(range(40)))
    _, vd = c.gather(torch.tensor([5, 2, 7]), 40, torch.float32)
    _verifier_puits(vd, v, 16)
    vq, vs = kv_k8v4.quantifier_v(v[16:])
    assert torch.equal(vd[16:], kv_k8v4.dequantifier_v(vq, vs, torch.float32)), "hors puits : k8v4 attendu"


def test_b_sequence_courte_puis_decodage_jeton_par_jeton():
    """Invite de 5 jetons puis décodage : les positions 5..15 arrivent une à une et restent des puits."""
    c = _cache()
    k, v = _donnees(20, 4)
    _ecrire(c, k[:5], v[:5], [3, 6], list(range(5)))
    for p in range(5, 20):
        _ecrire(c, k[p:p + 1], v[p:p + 1], [3, 6], [p])
    _, vd = c.gather(torch.tensor([3, 6]), 20, torch.float32)
    _verifier_puits(vd, v, 16)


def test_gather_fixed_suit_gather():
    c = _cache()
    k, v = _donnees(40, 5)
    _ecrire(c, k, v, [5, 2, 7], list(range(40)))
    t = torch.tensor([5, 2, 7])
    _, vf = c.gather_fixed(t.view(1, -1), torch.float32)
    _, v1 = c.gather(t, 40, torch.float32)
    assert torch.equal(vf[0, :40], v1)


def test_ligne_de_regime_nomme_les_puits():
    from acvram.engine.runner import Engine
    faux = lambda c: SimpleNamespace(model=SimpleNamespace(caches={0: c}))
    assert Engine.kv_format_servi(faux(_cache(puits=16))) == "k8v4+puits16"
    assert Engine.kv_format_servi(faux(_cache(puits=0))) == "k8v4"
    assert Engine.kv_format_servi(faux(_cache("int8", puits=16))) == "int8"


def test_refus_nommes():
    c = _cache()
    k, v = _donnees(2, 6)
    with pytest.raises(RuntimeError, match="positions"):
        c.write(torch.tensor([0, 1]), k, v)
    with pytest.raises(RuntimeError, match="hôte"):
        c.export_block(0)
