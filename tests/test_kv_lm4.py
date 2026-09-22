"""Le format KV « lm4 » (kv_lm4.py, conception-kv-4bits-rotation-17-09), à sec.

Ce qui se prouve sans carte : les centroïdes de Lloyd-Max sont ceux de la
gaussienne (recalculés ici par itération de Lloyd), la rotation est
orthogonale et conserve les produits scalaires, l'erreur d'aller-retour est
celle prédite (9,7 % RMS à 4 bits) et CROÎT à 3 puis 2 bits — un instrument
qui ne casse pas à 2 bits ne prouve rien à 4 —, le cache paginé lit ce qu'il
a écrit, sur la moitié des octets de l'int8, et l'attention sur un cache lm4
reste proche de l'attention bf16 là où lm2 s'en écarte.
"""
import math

import pytest
import torch

from acvram.memory import kv_lm4
from acvram.memory.kvcache import KVCacheConfig, PagedKVCache

torch.manual_seed(0)


def _lloyd(b, iters=200, grille=400_000):
    """Itération de Lloyd sur N(0,1) discrétisée : centroïdes = moyennes
    conditionnelles de leur cellule, frontières = milieux."""
    x = torch.linspace(-6, 6, grille, dtype=torch.float64)
    p = torch.exp(-x * x / 2)
    c = torch.linspace(-2.5, 2.5, 2 ** b, dtype=torch.float64)
    for _ in range(iters):
        s = (c[1:] + c[:-1]) / 2
        cell = torch.bucketize(x, s)
        c = torch.stack([(p[cell == i] * x[cell == i]).sum() / p[cell == i].sum()
                         for i in range(2 ** b)])
    s = (c[1:] + c[:-1]) / 2
    err = (x - c[torch.bucketize(x, s)]) ** 2
    return c, float((p * err).sum() / p.sum())


@pytest.mark.parametrize("b", [4, 3, 2])
def test_les_centroides_sont_ceux_de_lloyd_max_gaussien(b):
    c, mse = _lloyd(b)
    assert torch.allclose(kv_lm4.table(b).double(), c, atol=5e-4), (kv_lm4.table(b), c)
    assert abs(mse / kv_lm4.MSE[b] - 1) < 0.01, (b, mse, kv_lm4.MSE[b])


@pytest.mark.parametrize("d", [32, 64, 128, 256, 512])
def test_la_rotation_est_orthogonale_et_conserve_le_produit_scalaire(d):
    r = kv_lm4.rotation(d)
    assert torch.allclose(r @ r.T, torch.eye(d), atol=1e-5)
    q, k = torch.randn(7, d), torch.randn(7, d)
    assert torch.allclose((q @ r) @ (k @ r).T, q @ k.T, atol=1e-4)
    with pytest.raises(ValueError):
        kv_lm4.rotation(96)


def test_la_rotation_etale_un_canal_aberrant():
    """Un puits d'attention : un canal à 200 σ. Après rotation, aucune
    coordonnée ne dépasse ~4 écarts-types de la norme partagée, contre √D
    avant (toute la norme dans un canal)."""
    x = torch.randn(64, 128)
    x[:, 5] = 200.0
    xr = x @ kv_lm4.rotation(128)
    sigma = xr.norm(dim=-1, keepdim=True) / math.sqrt(128)
    assert (xr.abs() / sigma).max() < 4.5
    assert (x.abs() / (x.norm(dim=-1, keepdim=True) / math.sqrt(128))).max() > 11   # √128 : tout dans un canal


def _rms(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def test_l_erreur_d_aller_retour_est_celle_predite_et_croit_quand_les_bits_baissent():
    x = torch.randn(512, 4, 128)
    err = {}
    for fmt in kv_lm4.FORMATS:
        q, s = kv_lm4.quantifier(x, fmt)
        assert q.dtype == torch.uint8 and q.shape == (512, 4, 64) and s.dtype == torch.float16
        err[fmt] = _rms(kv_lm4.dequantifier(q, s, fmt, torch.float32), x)
        attendu = math.sqrt(kv_lm4.MSE[kv_lm4.bits(fmt)])
        assert abs(err[fmt] - attendu) < 0.01, (fmt, err[fmt], attendu)
    assert err["lm4"] < err["lm3"] < err["lm2"], err
    assert err["lm2"] > 3 * err["lm4"], err


def test_quatre_bits_par_rotation_bat_int8_par_amax_sur_un_puits():
    """Là où int8 dépense son échelle sur un canal à 200 σ (pas de 1,6 σ pour
    les autres), lm4 garde 9,7 % : c'est l'argument du format, pas un
    détail."""
    x = torch.randn(256, 4, 128)
    x[..., 5] = 200.0
    q, s = kv_lm4.quantifier(x)
    lm4 = _rms(kv_lm4.dequantifier(q, s, "lm4", torch.float32), x)
    amax = x.abs().amax(dim=-1, keepdim=True)
    int8 = _rms((x / amax * 127).round() * amax / 127, x)
    assert lm4 < 0.11 and int8 > 0.02 and lm4 < 3 * int8, (lm4, int8)
    x[..., 5] = 1.0                     # sans puits, int8 est meilleur : dit aussi
    q, s = kv_lm4.quantifier(x)
    assert _rms(kv_lm4.dequantifier(q, s, "lm4", torch.float32), x) > 0.09


def test_les_formats_inconnus_sont_refuses():
    with pytest.raises(ValueError):
        kv_lm4.bits("lm5")
    with pytest.raises(KeyError):
        KVCacheConfig(num_layers=1, num_kv_heads=1, head_dim=128, num_blocks=1, dtype="lm5").torch_dtype


def _cache(fmt, nkv=2, d=128, blocs=4):
    return PagedKVCache(KVCacheConfig(num_layers=1, num_kv_heads=nkv, head_dim=d,
                                      num_blocks=blocs, dtype=fmt, device="cpu"))


def test_le_cache_pagine_stocke_la_moitie_des_octets_de_l_int8():
    lm4, int8, bf16 = _cache("lm4"), _cache("int8"), _cache("bf16")
    assert lm4.k.shape == (4, 16, 2, 64) and lm4.k.dtype == torch.uint8 and lm4.k_scale is not None
    assert lm4.cfg.bytes_per_block() == 2 * 16 * 2 * 128 // 2 + 2 * 16 * 2 * 2
    assert lm4.cfg.bytes_per_block() * 2 - int8.cfg.bytes_per_block() == 2 * 16 * 2 * 2
    assert lm4.nbytes * 2 - int8.nbytes == lm4.k_scale.numel() * 2 * 2
    assert lm4.nbytes < bf16.nbytes / 3.8


def test_le_cache_lit_ce_qu_il_a_ecrit_et_garde_la_sentinelle_negative():
    c = _cache("lm4")
    x_k, x_v = torch.randn(20, 2, 128), torch.randn(20, 2, 128)
    slots = torch.cat([torch.arange(16) + 16, torch.arange(4) + 48])   # blocs 1 et 3
    c.write(torch.cat([slots, torch.tensor([-1])]), torch.cat([x_k, torch.full((1, 2, 128), 9.0)]),
            torch.cat([x_v, torch.full((1, 2, 128), 9.0)]))
    assert not c.k[3, 15].any() and not c.k[0].any(), "la sentinelle −1 a rebouclé sur un bloc réel"
    k, v = c.gather(torch.tensor([1, 3]), 20, torch.float32)
    q, s = kv_lm4.quantifier(x_k)
    assert torch.equal(k, kv_lm4.dequantifier(q, s, "lm4", torch.float32))
    assert _rms(k, x_k) < 0.11 and _rms(v, x_v) < 0.11
    kf, vf = c.gather_fixed(torch.tensor([[1, 3]]), torch.float32)
    assert kf.shape == (1, 32, 2, 128) and torch.equal(kf[0, :20], k)
    bloc = c.export_block(1)
    assert bloc[0].shape == (16, 2, 64) and bloc[2].shape == (16, 2)
    c2 = _cache("lm4")
    c2.import_block(1, bloc)
    assert torch.equal(c2.k[1], c.k[1]) and torch.equal(c2.k_scale[1], c.k_scale[1])


def _attention(cache, q, n):
    k, v = cache.gather(torch.arange(4), n, torch.float32)          # [n, hkv, d]
    k = k.repeat_interleave(q.shape[0] // k.shape[1], dim=1)          # [n, h, d]
    v = v.repeat_interleave(q.shape[0] // v.shape[1], dim=1)
    s = torch.einsum("hd,nhd->hn", q.float(), k) / math.sqrt(q.shape[-1])
    return torch.einsum("hn,nhd->hd", s.softmax(-1), v)


def test_l_attention_sur_un_cache_lm4_suit_bf16_la_ou_lm2_decroche():
    """Sur 60 clés gaussiennes (scores d'écart-type 1), l'attention lue d'un
    cache lm4 s'écarte de bf16 de ~13 % RMS, int8 de ~1 % : le rapport (×14)
    est celui des pas de quantification (9,7 % contre 0,7 % par coordonnée),
    pas un défaut d'implémentation. Ce que le MODÈLE en fait (PPL) ne se
    lit pas à sec — c'est le scellé de la note, et ce jouet dit déjà que
    lm4 ≤ int8 + 0,003 sera serré. Bras qui doit casser : lm2 > 2 × lm4.
    Un puits (norme 30 × en position 0) rend le jouet chaotique — les
    bascules d'argmax dominent — et n'est pas un régime de jugement."""
    n, hkv, d = 60, 2, 128
    k, v = torch.randn(n, hkv, d), torch.randn(n, hkv, d)
    q = torch.randn(8, d)                       # 8 têtes de requête, n_rep 4
    outs = {}
    for fmt in ("bf16", "int8", "lm4", "lm3", "lm2"):
        c = _cache(fmt, hkv, d)
        c.write(torch.arange(n), k, v)
        outs[fmt] = _attention(c, q, n)
    ecarts = {f: _rms(outs[f], outs["bf16"]) for f in ("int8", "lm4", "lm3", "lm2")}
    assert 0.08 < ecarts["lm4"] < 0.18 and ecarts["int8"] < 0.02, ecarts
    assert ecarts["lm4"] < ecarts["lm3"] < ecarts["lm2"] and ecarts["lm2"] > 2 * ecarts["lm4"], ecarts


def _decode_logits(model, prompt, n_pas):
    """Prefill puis n_pas de décodage à travers le cache : ce sont les seuls
    logits qui LISENT le cache (un prefill à offset 0 ne le relit jamais —
    `acvram eval` est aveugle au format KV, evaluate.py:228-243)."""
    from acvram.engine.model import ForwardBatch
    from acvram.memory.kvcache import BLOCK_SIZE, BlockAllocator
    n = len(prompt)
    alloc = BlockAllocator(model.caches[0].cfg.num_blocks)
    blocks = alloc.allocate((n + n_pas + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)
    slot = lambda i: blocks[i // BLOCK_SIZE] * BLOCK_SIZE + i % BLOCK_SIZE
    table = [torch.tensor(blocks)]
    batch = ForwardBatch(torch.tensor(prompt), torch.arange(n), [n], [n], table,
                         torch.tensor([slot(i) for i in range(n)]), True)
    sorties = [model(batch)]
    tok = prompt[-1]
    for p in range(n_pas):
        batch = ForwardBatch(torch.tensor([tok]), torch.tensor([n + p]), [n + p + 1], [1], table,
                             torch.tensor([slot(n + p)]), False)
        sorties.append(model(batch))
        tok = (tok * 7 + 3) % 1000                 # teacher forcing : même entrée pour tous les bras
    return torch.cat(sorties).float()


def test_de_bout_en_bout_le_decodage_lit_un_cache_lm4(converted, monkeypatch):
    """Le mini-modèle du conftest, sur processeur, décode 6 jetons sur le
    cache par défaut du plan (int8 : le palier vient du manifeste), lm4 et
    lm2 (ACVRAM_KV_FORMAT lu par le chargeur) : lm4 doit rester près du
    défaut, lm2 doit s'en écarter davantage, et le prefill (offset 0, cache
    jamais relu) doit être IDENTIQUE dans les trois bras."""
    from acvram.engine.loader import load_model
    from acvram.memory import tiering
    prompt = [5, 42, 7, 99, 13, 250, 31, 8]
    logits = {}
    for fmt in ("", "lm4", "lm2"):
        monkeypatch.setattr(tiering, "_KV_FORMAT", fmt)
        loaded = load_model(converted, dtype=torch.float32, device_override="cpu")
        dtypes = {c.cfg.dtype for c in loaded.model.caches.values()}
        assert dtypes == ({fmt} if fmt else {"int8"}), (fmt, dtypes)
        logits[fmt or "defaut"] = _decode_logits(loaded.model, prompt, 6)
    assert torch.equal(logits["lm4"][0], logits["defaut"][0]), "le prefill ne lit pas le cache : il doit être identique"
    e4 = _rms(logits["lm4"][1:], logits["defaut"][1:])
    e2 = _rms(logits["lm2"][1:], logits["defaut"][1:])
    print(f"écart logits décodage : lm4 {e4:.4f}, lm2 {e2:.4f}")
    assert 0 < e4 < e2, (e4, e2)


# ---------------------------------------------------------------------------
# Deux interrupteurs de diagnostic (sage-kv-lm4-clos-17-09 § 1), pour la
# passe de cause de Laure : K seul lm4, V seul lm4, puits exempté. Rien
# n'y touche la carte ; `quantifier_diagnostic` fait juste une
# substitution numérique, testable au bit près.
# ---------------------------------------------------------------------------


def test_actif_par_defaut_les_deux_cotes(monkeypatch):
    monkeypatch.delenv("ACVRAM_KV_LM4_SEUL", raising=False)
    assert kv_lm4.actif("k") and kv_lm4.actif("v")


def test_seul_restreint_lm4_a_un_seul_cote(monkeypatch):
    monkeypatch.setenv("ACVRAM_KV_LM4_SEUL", "k")
    assert kv_lm4.actif("k") and not kv_lm4.actif("v")
    monkeypatch.setenv("ACVRAM_KV_LM4_SEUL", "v")
    assert kv_lm4.actif("v") and not kv_lm4.actif("k")


def test_seul_refuse_une_valeur_inconnue(monkeypatch):
    monkeypatch.setenv("ACVRAM_KV_LM4_SEUL", "q")
    with pytest.raises(ValueError):
        kv_lm4.actif("k")
    with pytest.raises(ValueError):
        kv_lm4.actif("q")           # coté lui-meme invalide, sans lire l'env


def test_puits_exempte_les_n_premieres_positions(monkeypatch):
    monkeypatch.delenv("ACVRAM_KV_LM4_PUITS", raising=False)
    positions = torch.arange(20)
    assert bool(kv_lm4.hors_puits(positions).all()), "sans puits, tout est quantifie lm4"
    monkeypatch.setenv("ACVRAM_KV_LM4_PUITS", "16")
    masque = kv_lm4.hors_puits(positions)
    assert not masque[:16].any() and masque[16:].all()


def test_puits_refuse_une_valeur_negative(monkeypatch):
    monkeypatch.setenv("ACVRAM_KV_LM4_PUITS", "-1")
    with pytest.raises(ValueError):
        kv_lm4.hors_puits(torch.arange(4))


def test_quantifier_diagnostic_bascule_en_int8_hors_cote(monkeypatch):
    """Témoin (REGLES § 5) : le côté exclu par ACVRAM_KV_LM4_SEUL doit
    RENDRE UN RÉSULTAT DIFFÉRENT du lm4 par défaut -- sinon l'interrupteur
    ne changerait rien et la mesure de Laure ne prouverait rien."""
    g = torch.Generator().manual_seed(3)
    x = torch.randn(64, 4, 128, generator=g)
    monkeypatch.delenv("ACVRAM_KV_LM4_SEUL", raising=False)
    lm4_defaut = kv_lm4.quantifier_diagnostic(x, "v")
    monkeypatch.setenv("ACVRAM_KV_LM4_SEUL", "k")               # v exclu
    int8_v = kv_lm4.quantifier_diagnostic(x, "v")
    assert not torch.allclose(int8_v, lm4_defaut), \
        "le temoin ne diverge pas : ACVRAM_KV_LM4_SEUL=k n'a rien change sur v"
    assert torch.allclose(int8_v, kv_lm4._int8_amax(x)), \
        "le cote exclu doit suivre exactement l'int8 par amax, pas un autre format"


def test_quantifier_diagnostic_garde_le_puits_en_int8():
    """Les positions du puits suivent l'int8 par amax au bit près, les
    autres suivent lm4 au bit près -- pas une approximation entre les deux."""
    g = torch.Generator().manual_seed(4)
    x = torch.randn(20, 4, 128, generator=g)
    positions = torch.arange(20)
    import os
    ancien = os.environ.pop("ACVRAM_KV_LM4_SEUL", None)
    try:
        os.environ["ACVRAM_KV_LM4_PUITS"] = "16"
        obtenu = kv_lm4.quantifier_diagnostic(x, "k", positions=positions)
        attendu_puits = kv_lm4._int8_amax(x[:16])
        q, s = kv_lm4.quantifier(x[16:])
        attendu_hors = kv_lm4.dequantifier(q, s, "lm4", x.dtype)
        assert torch.equal(obtenu[:16], attendu_puits)
        assert torch.equal(obtenu[16:], attendu_hors)
    finally:
        os.environ.pop("ACVRAM_KV_LM4_PUITS", None)
        if ancien is not None:
            os.environ["ACVRAM_KV_LM4_SEUL"] = ancien


def test_quantifier_diagnostic_sans_positions_ignore_le_puits(monkeypatch):
    """`positions=None` : ACVRAM_KV_LM4_PUITS ne doit avoir aucun effet --
    l'appelant qui ne mesure pas le puits sur cet appel ne doit pas en
    subir un silencieusement."""
    monkeypatch.delenv("ACVRAM_KV_LM4_SEUL", raising=False)
    monkeypatch.setenv("ACVRAM_KV_LM4_PUITS", "16")
    g = torch.Generator().manual_seed(5)
    x = torch.randn(8, 4, 128, generator=g)
    obtenu = kv_lm4.quantifier_diagnostic(x, "k", positions=None)
    q, s = kv_lm4.quantifier(x)
    assert torch.equal(obtenu, kv_lm4.dequantifier(q, s, "lm4", x.dtype))


def test_le_diagnostic_lm4_est_branche_a_l_ecriture_du_cache(monkeypatch):
    """Les interrupteurs de Manon (6a660fb) ne servent à rien s'ils ne sont
    appelés nulle part : `PagedKVCache.write` les applique AVANT le stockage
    quand l'un des deux est posé, sur un porteur int8. Bras qui doit casser :
    sans interrupteur, le cache int8 rend int8 ; avec `PUITS=0` (contrôle),
    il rend ≈ lm4 ; avec `SEUL=k`, V reste int8 ; avec `PUITS=4`, les
    positions < 4 restent int8 et les autres deviennent lm4."""
    from acvram.memory import kv_lm4
    monkeypatch.delenv("ACVRAM_KV_LM4_SEUL", raising=False)
    monkeypatch.delenv("ACVRAM_KV_LM4_PUITS", raising=False)
    x_k, x_v = torch.randn(8, 2, 128), torch.randn(8, 2, 128)
    pos = torch.arange(8)

    def lit(**env):
        for k_, v_ in env.items():
            monkeypatch.setenv(k_, v_)
        c = _cache("int8")
        c.write(torch.arange(8), x_k, x_v, positions=pos)
        for k_ in env:
            monkeypatch.delenv(k_)
        return c.gather(torch.tensor([0]), 8, torch.float32)

    q, s = kv_lm4.quantifier(x_k)
    lm4_k = kv_lm4.dequantifier(q, s, "lm4", torch.float32)
    k0, v0 = lit()
    assert _rms(k0, x_k) < 0.02 and _rms(v0, x_v) < 0.02
    k1, v1 = lit(ACVRAM_KV_LM4_PUITS="0")
    assert _rms(k1, lm4_k) < 0.02 and _rms(k1, x_k) > 0.08 and _rms(v1, x_v) > 0.08
    k2, v2 = lit(ACVRAM_KV_LM4_SEUL="k")
    assert _rms(k2, lm4_k) < 0.02 and _rms(v2, x_v) < 0.02
    k3, _ = lit(ACVRAM_KV_LM4_PUITS="4")
    assert _rms(k3[:4], x_k[:4]) < 0.02 and _rms(k3[4:], lm4_k[4:]) < 0.02
