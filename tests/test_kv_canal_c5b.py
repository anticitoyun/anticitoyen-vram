"""C5-b à sec (chantier-c5b-19-09) : jumeau torch des clés int8 à échelle par
canal sur chaque bloc de 16 (memory/kv_canal.py), sans carte.

(i)   l'erreur de K par canal est plus petite que par jeton, et le témoin
      CASSE : une clé à un canal aberrant (× 30) fait monter l'erreur des
      autres canaux par jeton, pas par canal (leçon lm4/KIVI, REGLES § 4) ;
(ii)  attention de référence fp32 sur (K par canal, V par jeton) contre
      l'attention bf16 exacte : ≤ 2⁻⁵ du max, et ≤ 0,8 × l'erreur par jeton ;
(iii) le bloc courant : 40 jetons écrits un par un donnent les mêmes blocs
      fermés que le préfill des 40 (au bit), le bloc courant (8 jetons) en
      bf16 exact ; préfill découpé idem ; réserve épuisée → par jeton, juste ;
(iv)  le régime : variable dans la table, garde du CLI, ligne kv=int8-canal16.
"""
from __future__ import annotations

import math

import pytest
import torch

from acvram.memory import kv_canal
from acvram.memory.kvcache import KVCacheConfig, PagedKVCache

HKV, D, BS = 4, 128, 16


def _cache(num_blocks=8, rangs=4):
    return PagedKVCache(KVCacheConfig(num_layers=1, num_kv_heads=HKV, head_dim=D,
                                      num_blocks=num_blocks, dtype="int8", device="cpu",
                                      canal=True, rangs=rangs))


def _par_jeton(x):
    q, s = kv_canal._par_jeton(x)
    return q.float() * s.float().unsqueeze(-1)


def _par_canal(x):
    codes, sc = kv_canal.quantifier_k_par_canal(x)
    return kv_canal.dequantifier_k_par_canal(codes, sc)


def _err(a, b, canaux=None):
    if canaux is not None:
        a, b = a[..., canaux], b[..., canaux]
    return float((a - b).norm() / b.norm())


# ---------------------------------------------------------------------------
# e4m3 vers le haut : la même arithmétique entière que le noyau
# ---------------------------------------------------------------------------
def _e4m3_haut_lent(x: float) -> int:
    """Référence par énumération : plus petit E4M3 ≥ x parmi les 127 valeurs."""
    valeurs = [(b, float(torch.tensor([b], dtype=torch.uint8).view(torch.float8_e4m3fn).float()))
               for b in range(0, 0x7F)]
    if x <= 0:
        return 1
    for b, v in valeurs:
        if v >= x:
            return b
    return 0x7E


def test_e4m3_haut_est_le_plus_petit_e4m3_au_dessus():
    torch.manual_seed(1)
    xs = torch.cat([torch.zeros(1), torch.logspace(-4, 2.7, 3000),
                    torch.tensor([0.001953125, 0.015625, 448.0, 1e4]),
                    torch.rand(500) * 0.02])
    octets = kv_canal.e4m3_haut(xs)
    lent = torch.tensor([_e4m3_haut_lent(float(x)) for x in xs], dtype=torch.uint8)
    assert torch.equal(octets, lent), (xs[octets != lent][:5], octets[octets != lent][:5], lent[octets != lent][:5])
    y = kv_canal.e4m3_vers_float(octets)
    assert (y[xs < 448] >= xs[xs < 448]).all()


def test_les_codes_ne_saturent_jamais_et_le_dequant_suit():
    torch.manual_seed(2)
    k = torch.randn(BS, HKV, D, dtype=torch.bfloat16) * 3
    codes, sc = kv_canal.quantifier_k_par_canal(k)
    assert codes.dtype == torch.int8 and sc.dtype == torch.uint8 and sc.shape == (HKV, D)
    assert int(codes.abs().max()) <= 127
    # sc arrondi vers le haut à 3 bits de mantisse : le code maximal d'un canal
    # vaut 127 / (1 + 2⁻³) = 113 au pire, jamais moins
    assert (codes.abs().amax(dim=0) >= 112).all(), "chaque canal doit atteindre le haut de sa plage"
    assert _err(_par_canal(k), k.float()) < 0.01


# ---------------------------------------------------------------------------
# (i) par canal < par jeton, et le témoin casse
# ---------------------------------------------------------------------------
def test_i_par_canal_bat_par_jeton_et_un_canal_aberrant_casse_par_jeton_seulement():
    torch.manual_seed(3)
    k = torch.randn(BS, HKV, D, dtype=torch.bfloat16)
    # amplitudes de canaux inégales, comme après RoPE (amax/rms par canal 5-7)
    k = (k.float() * torch.logspace(-1, 1, D)).to(torch.bfloat16)
    e_tok, e_can = _err(_par_jeton(k), k.float()), _err(_par_canal(k), k.float())
    assert e_can < 0.5 * e_tok, (e_can, e_tok)          # la porte : ≤ 0,5 (mesuré ×0,245 sur K réels)
    # témoin cassant : le plus grand canal × 30 (il porte l'amax de chaque jeton)
    k2 = k.clone()
    k2[..., D - 1] = (k2[..., D - 1].float() * 30).to(torch.bfloat16)
    autres = list(range(D - 1))
    e_tok2 = _err(_par_jeton(k2), k2.float(), autres)
    e_can2 = _err(_par_canal(k2), k2.float(), autres)
    e_tok1 = _err(_par_jeton(k), k.float(), autres)
    e_can1 = _err(_par_canal(k), k.float(), autres)
    assert e_tok2 > 3 * e_tok1, f"par jeton : l'erreur des autres canaux doit monter ({e_tok1:.4f} → {e_tok2:.4f})"
    assert e_can2 < 1.25 * e_can1, f"par canal : les autres canaux ne doivent pas bouger ({e_can1:.4f} → {e_can2:.4f})"
    assert e_can2 < 0.25 * e_tok2


# ---------------------------------------------------------------------------
# (ii) attention de référence
# ---------------------------------------------------------------------------
def test_ii_attention_canal_contre_bf16_exacte():
    torch.manual_seed(4)
    T, HQ = 300, 32
    # amplitudes de canaux inégales (× 10 entre extrêmes), scores d'écart-type ~1,9
    k = (torch.randn(T, HKV, D) * torch.logspace(-0.5, 0.5, D)).to(torch.bfloat16)
    # puits × 3 (un puits × 20 rend des scores de ±40 : 0,5 % d'erreur de K y
    # vaut 0,2 de score, 20 % de poids softmax — l'int8 par jeton y rend 0,27,
    # par canal 0,12 : hors de portée d'une borne serrée, quel que soit le format)
    k[0] = (k[0].float() * 3).to(torch.bfloat16)
    v = torch.randn(T, HKV, D, dtype=torch.bfloat16)
    q = torch.randn(HQ, D, dtype=torch.bfloat16)
    scale = 1 / math.sqrt(D)
    exact = kv_canal.attention_reference(q, k, v, scale)
    vq = torch.cat([_par_jeton(v[i:i + 1]) for i in range(T)])
    # blocs fermés par canal, les 12 derniers (bloc courant) exacts
    n_ferme = (T // BS) * BS
    kc = torch.cat([_par_canal(k[i:i + BS]) for i in range(0, n_ferme, BS)] + [k[n_ferme:].float()])
    kt = torch.cat([_par_jeton(k[i:i + 1]) for i in range(T)])
    out_can = kv_canal.attention_reference(q, kc, vq, scale)
    out_tok = kv_canal.attention_reference(q, kt, vq, scale)
    # mesuré à sec : max relatif canal 0,018, jeton 0,028, V seul 0,008 (le plancher) ;
    # Frobenius canal 0,009, jeton 0,0125
    borne = exact.abs().amax(-1, keepdim=True) * 2 ** -5
    assert ((out_can - exact).abs() <= borne).all(), float(((out_can - exact).abs() / borne).max())
    assert _err(out_can, exact) <= 0.8 * _err(out_tok, exact), (_err(out_can, exact), _err(out_tok, exact))
    # fenêtre glissante : change la sortie et reste dans la borne
    out_w = kv_canal.attention_reference(q, kc, vq, scale, window=64)
    exact_w = kv_canal.attention_reference(q, k, v, scale, window=64)
    assert not torch.allclose(out_w, out_can, atol=1e-3)
    assert ((out_w - exact_w).abs() <= exact_w.abs().amax(-1, keepdim=True) * 2 ** -5).all()


# ---------------------------------------------------------------------------
# (iii) le bloc courant, écrit un par un ou d'un coup
# ---------------------------------------------------------------------------
def _kv(n, graine=5):
    torch.manual_seed(graine)
    return (torch.randn(n, HKV, D, dtype=torch.bfloat16),
            torch.randn(n, HKV, D, dtype=torch.bfloat16))


def test_iii_un_par_un_egale_le_prefill_au_bit_et_le_bloc_courant_est_exact():
    n = 40
    k, v = _kv(n)
    slots = torch.arange(n)
    a = _cache(); a.write(slots, k, v)
    b = _cache()
    for i in range(n):
        b.write(slots[i:i + 1], k[i:i + 1], v[i:i + 1])
    c = _cache()                                     # préfill découpé : 0..9, 10..27, 28..39
    for lo, hi in ((0, 10), (10, 28), (28, 40)):
        c.write(slots[lo:hi], k[lo:hi], v[lo:hi])
    for autre in (b, c):
        assert torch.equal(autre.k[:2], a.k[:2]), "codes des blocs fermés"
        assert torch.equal(autre.k_scale_canal.view(torch.uint8)[:2], a.k_scale_canal.view(torch.uint8)[:2])
        assert torch.equal(autre.k_scale[:2], a.k_scale[:2]) and float(a.k_scale[0, 0, 0]) == kv_canal.KS_CANAL
        assert torch.equal(autre.v[:3], a.v[:3]) and torch.equal(autre.v_scale[:3], a.v_scale[:3])
        assert autre.tampon_de.tolist() == a.tampon_de.tolist() == [-1, -1, a.tampon_de[2].item()] + [-1] * 5
        assert int(autre.tampon_sommet[0]) == a.cfg.rangs - 1
    kg, vg = a.gather(torch.arange(8), n, torch.float32)
    assert torch.equal(kg[32:], k[32:].float()), "bloc courant : bf16 exact"
    assert _err(kg[:32], k[:32].float()) < 0.01
    assert _err(vg, v.float()) < 0.02
    # gather_fixed (formes fixes, torch.where) rend la même chose
    kf, vf = a.gather_fixed(torch.arange(8).view(1, 8), torch.float32)
    assert torch.equal(kf[0, :n], kg) and torch.equal(vf[0, :n], vg)
    # la fermeture du bloc 2 (jetons 40..47) rend sa ligne et quantifie par canal
    k2, v2 = _kv(8, graine=6)
    a.write(torch.arange(40, 48), k2, v2)
    assert a.tampon_de.tolist() == [-1] * 8 and int(a.tampon_sommet[0]) == a.cfg.rangs
    kk = torch.cat([k[32:], k2])
    codes, sc = kv_canal.quantifier_k_par_canal(kk)
    assert torch.equal(a.k[2], codes) and torch.equal(a.k_scale_canal.view(torch.uint8)[2], sc)


def test_iii_reserve_epuisee_repart_par_jeton_sans_faute():
    """4 lignes, 6 séquences ouvertes en même temps : les deux dernières
    repartent par jeton (sc = 1, ks = amax/127) et se relisent juste."""
    k, v = _kv(6)
    a = _cache(num_blocks=8, rangs=4)
    slots = torch.arange(6) * BS                     # décalage 0 de six blocs
    a.write(slots, k, v)
    td = a.tampon_de.tolist()
    assert sorted(td[:4]) == [0, 1, 2, 3] and td[4:6] == [-2, -2] and int(a.tampon_sommet[0]) == 0
    assert (a.k_scale_canal.view(torch.uint8)[4:6] == kv_canal.SC_PAR_JETON).all()
    for b in range(6):
        kg, _ = a.gather(torch.tensor([b]), 1, torch.float32)
        e = _err(kg, k[b:b + 1].float())
        assert e < (1e-6 if b < 4 else 0.02), (b, e)
    # le bloc 4 continue par jeton, le bloc 0 se ferme et rend sa ligne
    k2, v2 = _kv(16, graine=7)
    a.write(torch.arange(1, 16), k2[1:], v2[1:])
    a.write(torch.tensor([4 * BS + 1]), k2[:1], v2[:1])
    assert a.tampon_de[0] == -1 and a.tampon_de[4] == -2 and int(a.tampon_sommet[0]) == 1
    assert torch.equal(a.k[0], kv_canal.quantifier_k_par_canal(torch.cat([k[:1], k2[1:]]))[0])
    kg, _ = a.gather(torch.tensor([4]), 2, torch.float32)
    assert _err(kg, torch.cat([k[4:5], k2[:1]]).float()) < 0.02


def test_iii_la_sentinelle_de_rembourrage_n_ecrit_rien():
    k, v = _kv(3)
    a = _cache()
    avant = (a.k.clone(), a.tampon_de.clone(), int(a.tampon_sommet[0]))
    a.write(torch.tensor([-1, 16, -1]), k, v)             # seul le bloc 1 s'ouvre
    assert torch.equal(a.k, avant[0]) and int(a.tampon_sommet[0]) == avant[2] - 1
    assert a.tampon_de.tolist()[1] >= 0 and a.tampon_de.tolist()[:1] + a.tampon_de.tolist()[2:] == [-1] * 7
    kg, _ = a.gather(torch.tensor([1]), 1, torch.float32)
    assert torch.equal(kg[0], k[1].float())


def test_les_octets_par_bloc_sont_ceux_de_la_fiche():
    base = KVCacheConfig(num_layers=1, num_kv_heads=HKV, head_dim=D, num_blocks=1, dtype="int8",
                         device="cpu", canal=False)
    canal = KVCacheConfig(num_layers=1, num_kv_heads=HKV, head_dim=D, num_blocks=1, dtype="int8",
                          device="cpu", canal=True, rangs=64)
    assert base.bytes_per_block() == 16640
    assert canal.bytes_per_block() - base.bytes_per_block() == HKV * D + 4      # 128 o E4M3 par tête + tampon_de
    assert abs((canal.bytes_per_block() / base.bytes_per_block() - 1) - 0.031) < 0.002
    assert canal.octets_tampon() == 64 * BS * HKV * D * 2 + 64 * 4              # 1 Mio + pile
    assert canal.nom_format == "int8-canal16" and base.nom_format == "int8"
    assert not KVCacheConfig(num_layers=1, num_kv_heads=HKV, head_dim=D, num_blocks=1,
                             dtype="bf16", device="cpu", canal=True).canal, "canal n'existe qu'en int8"


# ---------------------------------------------------------------------------
# (iv) le régime
# ---------------------------------------------------------------------------
def test_iv_le_regime_nomme_le_format(monkeypatch):
    from acvram import regime, cli
    noms = {v.nom: v for v in regime.VARIABLES}
    assert noms["KV_INT8_CANAL"].defaut == "0" and noms["KV_INT8_CANAL"].lu_a == ("acvram.memory.kv_canal", "ACTIF")
    assert "KV_CANAL_RANGS" in noms
    assert {"ACVRAM_KV_INT8_CANAL", "ACVRAM_KV_CANAL_RANGS"} <= cli.VARIABLES_LUES
    assert kv_canal.ACTIF is False, "défaut 0 jusqu'au scellé"
    assert "kv=int8-canal16" not in regime.regime_ligne()
    monkeypatch.setattr(kv_canal, "ACTIF", True)
    ligne = regime.regime_ligne()
    assert "kv=int8-canal16" in ligne and "ACVRAM_KV_INT8_CANAL=1" in ligne
    cfg = KVCacheConfig(num_layers=1, num_kv_heads=HKV, head_dim=D, num_blocks=1, dtype="int8", device="cpu")
    assert cfg.canal is True
