"""C5-b sur carte (chantier-c5b-19-09) : les noyaux `kv_write_int8_canal` et
`paged_attention_canal` contre le jumeau torch (memory/kv_canal.py).
Sauté sans CUDA ou sans extension ; à lancer NU (jamais sous un verrou tenu par
un autre : scratchpad/c5b-carte-19-09/chaine.sh le refuse si ACVRAM_CARTE_TENUE).

* écriture : préfill, un par un, préfill découpé, réserve épuisée — codes de K,
  échelles E4M3, ks, lignes de la réserve, pile : AU BIT contre le jumeau ; V
  (chemin par jeton d'aujourd'hui, 1/sc en multiplication) à ± 1 code ;
* attention paginée canal contre la référence fp32 sur le cache déquantifié
  (K par canal, bloc courant bf16, V par jeton) : ≤ 2⁻⁸ du max de la ligne,
  comme test_attn_paginee ;
* canal contre le chemin par jeton, tous deux contre l'attention bf16 exacte :
  l'écart attendu est celui de la quantification — canal ≤ jeton (Frobenius)
  et ≤ 2⁻⁵ du max ;
* bloc courant : la ligne bf16 est k au bit, et la sortie l'utilise (l'effacer
  change la sortie) ;
* graphe CUDA : écriture + attention capturées puis rejouées = eager au bit.
"""
from __future__ import annotations

import math

import pytest
import torch

from acvram.memory import kv_canal
from acvram.memory.kvcache import KVCacheConfig, PagedKVCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau CUDA requis")

HKV, D, BS, N_REP = 4, 128, 16, 8
TOL = 2 ** -8


def _ext():
    from acvram.kernels import get_extension
    ext = get_extension()
    if ext is None or not hasattr(ext, "kv_write_int8_canal") or not hasattr(ext, "paged_attention_canal"):
        pytest.skip("extension absente ou sans les symboles C5-b")
    return ext


def _cache(num_blocks, rangs=8):
    return PagedKVCache(KVCacheConfig(num_layers=1, num_kv_heads=HKV, head_dim=D, num_blocks=num_blocks,
                                      dtype="int8", device="cuda", canal=True, rangs=rangs))


def _kv(n, graine, amplitude=True):
    torch.manual_seed(graine)
    k = torch.randn(n, HKV, D, device="cuda")
    if amplitude:                                   # canaux inégaux, comme après RoPE
        k = k * torch.logspace(-0.5, 0.5, D, device="cuda")
    return k.to(torch.bfloat16), torch.randn(n, HKV, D, device="cuda").to(torch.bfloat16)


def _etat(c):
    return dict(k=c.k.clone(), sc=c.k_scale_canal.view(torch.uint8).clone(), ks=c.k_scale.clone(),
                v=c.v.clone(), vs=c.v_scale.clone(), tampon=c.tampon.clone(), tampon_de=c.tampon_de.clone(),
                sommet=int(c.tampon_sommet[0]), libres=c.tampon_libres.clone())


def _memes(a, b, ou="", ecrites=None):
    """Tout au bit sauf V (± 1 code) ; le contenu des lignes de la réserve ne
    compte que pour les blocs ouverts, et seulement sur leurs lignes ÉCRITES
    (``ecrites`` : bloc → nombre de lignes) — une ligne reprise à la pile garde
    au-delà ce que son bloc précédent y avait laissé, et deux caches qui ont
    recyclé leurs lignes différemment (un par un : blocs 0 et 1 fermés par la
    réserve ; un seul appel : blocs 0 et 1 directs) n'ont pas le même vieux.
    Vu sur carte le 20/09 (« un par un ligne du bloc 2 »)."""
    for nom in ("k", "sc", "ks", "tampon_de"):
        assert torch.equal(a[nom], b[nom]), f"{ou} {nom} : {(a[nom] != b[nom]).sum().item()} cellules"
    assert a["sommet"] == b["sommet"], ou
    assert torch.equal(torch.sort(a["libres"][:a["sommet"]])[0], torch.sort(b["libres"][:b["sommet"]])[0]), ou
    assert (a["v"].int() - b["v"].int()).abs().max() <= 1, ou
    assert torch.equal(a["vs"], b["vs"]), ou
    for blk, r in enumerate(a["tampon_de"].tolist()):
        if r >= 0:
            n = BS if ecrites is None else ecrites[blk]
            assert torch.equal(a["tampon"][r, :n], b["tampon"][b["tampon_de"][blk], :n]), f"{ou} ligne du bloc {blk}"


def _jumeau_ecrire(c, slots, k, v):
    kv_canal.ecrire(c, slots, k, v)


# ---------------------------------------------------------------------------
# écriture
# ---------------------------------------------------------------------------
def test_ecriture_prefill_un_par_un_et_decoupe_au_bit_contre_le_jumeau():
    _ext()
    n = 40
    k, v = _kv(n, 10)
    slots = torch.arange(n, device="cuda")
    noyau, jumeau = _cache(8), _cache(8)
    noyau.write(slots, k, v)
    _jumeau_ecrire(jumeau, slots, k, v)
    torch.cuda.synchronize()
    _memes(_etat(noyau), _etat(jumeau), "préfill")
    assert noyau.tampon_de.tolist()[:3] == [-1, -1, noyau.tampon_de[2].item()] and noyau.tampon_de[2] >= 0
    assert torch.equal(noyau.tampon[int(noyau.tampon_de[2]), :8], k[32:]), "bloc courant : bf16 au bit"
    un = _cache(8)
    for i in range(n):
        un.write(slots[i:i + 1], k[i:i + 1], v[i:i + 1])
    dec = _cache(8)
    for lo, hi in ((0, 10), (10, 28), (28, 40)):
        dec.write(slots[lo:hi], k[lo:hi], v[lo:hi])
    torch.cuda.synchronize()
    _memes(_etat(un), _etat(noyau), "un par un", ecrites={2: 8})
    _memes(_etat(dec), _etat(noyau), "découpé", ecrites={2: 8})
    # fermeture du bloc 2 par le noyau (jetons 40..47) : la ligne rendue, codes par canal
    k2, v2 = _kv(8, 11)
    noyau.write(torch.arange(40, 48, device="cuda"), k2, v2)
    _jumeau_ecrire(jumeau, torch.arange(40, 48, device="cuda"), k2, v2)
    torch.cuda.synchronize()
    _memes(_etat(noyau), _etat(jumeau), "fermeture")
    assert noyau.tampon_de.tolist() == [-1] * 8 and int(noyau.tampon_sommet[0]) == 8
    codes, sc = kv_canal.quantifier_k_par_canal(torch.cat([k[32:], k2]))
    assert torch.equal(noyau.k[2], codes) and torch.equal(noyau.k_scale_canal.view(torch.uint8)[2], sc)


def test_ecriture_reserve_epuisee_et_rembourrage_comme_le_jumeau():
    _ext()
    k, v = _kv(6, 12)
    slots = torch.arange(6, device="cuda") * BS
    noyau, jumeau = _cache(8, rangs=4), _cache(8, rangs=4)
    noyau.write(slots, k, v)
    _jumeau_ecrire(jumeau, slots, k, v)
    torch.cuda.synchronize()
    _memes(_etat(noyau), _etat(jumeau), "épuisée")
    assert sorted(noyau.tampon_de.tolist()[:4]) == [0, 1, 2, 3] and noyau.tampon_de.tolist()[4:6] == [-2, -2]
    assert (noyau.k_scale_canal.view(torch.uint8)[4:6] == kv_canal.SC_PAR_JETON).all()
    # rembourrage (slot < 0) : rien n'est écrit, aucune ligne prise
    avant = _etat(noyau)
    k3, v3 = _kv(3, 13)
    noyau.write(torch.tensor([-1, -1, -1], device="cuda"), k3, v3)
    torch.cuda.synchronize()
    _memes(_etat(noyau), avant, "rembourrage")


# ---------------------------------------------------------------------------
# lecture
# ---------------------------------------------------------------------------
def _montage(lens, graine=20, rangs=8):
    """Un cache canal rempli par le noyau pour `lens` séquences (tables
    disjointes), q bf16 ; rend aussi les k/v bf16 d'origine par séquence."""
    B = len(lens)
    N = -(-max(max(lens), 1) // BS) + 1
    c = _cache(B * N, rangs=rangs)
    tables = torch.arange(B * N, device="cuda").view(B, N)
    kv = {}
    for b, n in enumerate(lens):
        if n:
            slots = torch.tensor([int(tables[b, i // BS]) * BS + i % BS for i in range(n)], device="cuda")
            k, v = _kv(n, graine + b)
            k[0] = (k[0].float() * 3).to(torch.bfloat16)               # puits modéré
            c.write(slots, k, v)
            kv[b] = (k, v)
    torch.manual_seed(graine + 100)
    q = torch.randn(B, HKV * N_REP, D, device="cuda").to(torch.bfloat16)
    return c, tables, torch.tensor(lens, device="cuda"), q, kv


def _attention(ext, c, q, tables, lens, scale, window=0, q_len=1):
    return ext.paged_attention_canal(q.contiguous(), c.k, c.k_scale, c.v, c.v_scale,
                                     c.k_scale_canal.view(torch.uint8), c.tampon, c.tampon_de,
                                     tables.contiguous(), lens.contiguous(), HKV, float(scale),
                                     int(q_len), int(window))


def _reference(c, tables, lens, q, scale, window=0):
    B, HQ, _ = q.shape
    ref = torch.zeros(B, HQ, D, dtype=torch.float32, device="cuda")
    for b in range(B):
        n = int(lens[b])
        if n:
            k, v = c.gather(tables[b], n, torch.float32)
            ref[b] = kv_canal.attention_reference(q[b], k, v, scale, window)
    return ref


def _hors(out, ref, tol=TOL):
    ecart = (out.float() - ref).abs().amax(-1)
    borne = ref.abs().amax(-1).clamp(min=1e-6) * tol
    return int((ecart > borne).sum()), float((ecart / borne).max())


@pytest.mark.parametrize("lens", [[37, 0, 300], [1], [2048, 17], [16, 32, 33, 1]])
def test_attention_canal_suit_la_reference_fp32_a_2_moins_8(lens):
    ext = _ext()
    c, tables, L, q, _ = _montage(lens)
    scale = 1 / math.sqrt(D)
    for window in (0, 100):
        out = _attention(ext, c, q, tables, L, scale, window)
        assert torch.isfinite(out.float()).all()
        hors, pire = _hors(out, _reference(c, tables, L, q, scale, window))
        assert hors == 0, f"fenêtre {window} : {hors} lignes hors 2^-8, pire {pire:.2f} × la tolérance"
        for b, n in enumerate(lens):
            if n == 0:
                assert not out[b].float().any(), "un fantôme du godet doit rendre zéro"


def test_canal_contre_par_jeton_l_ecart_est_celui_de_la_quantification():
    ext = _ext()
    lens = [300, 45]
    c, tables, L, q, kv = _montage(lens)
    scale = 1 / math.sqrt(D)
    # le même contenu dans un cache par jeton (le témoin d'aujourd'hui)
    B, N = tables.shape
    t = PagedKVCache(KVCacheConfig(num_layers=1, num_kv_heads=HKV, head_dim=D, num_blocks=B * N,
                                   dtype="int8", device="cuda", canal=False))
    for b, (k, v) in kv.items():
        n = k.shape[0]
        slots = torch.tensor([int(tables[b, i // BS]) * BS + i % BS for i in range(n)], device="cuda")
        t.write(slots, k, v)
    out_c = _attention(ext, c, q, tables, L, scale)
    out_t = ext.paged_attention(q.contiguous(), t.k, t.k_scale, t.v, t.v_scale, tables.contiguous(),
                                L.contiguous(), HKV, float(scale), 1, 0)
    assert not torch.equal(out_c, out_t), "les deux formats doivent différer"
    exact = torch.stack([kv_canal.attention_reference(q[b], kv[b][0], kv[b][1], scale) for b in range(B)])
    e_c = float((out_c.float() - exact).norm() / exact.norm())
    e_t = float((out_t.float() - exact).norm() / exact.norm())
    assert e_c <= e_t, f"canal {e_c:.4f} doit faire mieux que par jeton {e_t:.4f} contre bf16 exact"
    assert _hors(out_c, exact, 2 ** -5)[0] == 0, _hors(out_c, exact, 2 ** -5)


def test_le_bloc_courant_est_lu_en_bf16():
    ext = _ext()
    c, tables, L, q, kv = _montage([40])
    scale = 1 / math.sqrt(D)
    r = int(c.tampon_de[int(tables[0, 2])])
    assert r >= 0 and torch.equal(c.tampon[r, :8], kv[0][0][32:40])
    out = _attention(ext, c, q, tables, L, scale)
    # effacer la ligne (K du bloc courant = 0) change la sortie : le noyau la lit bien
    sauve = c.tampon[r].clone(); c.tampon[r].zero_()
    out0 = _attention(ext, c, q, tables, L, scale)
    c.tampon[r].copy_(sauve)
    assert not torch.allclose(out.float(), out0.float(), atol=1e-3)
    # fermer le bloc (jetons 40..47) puis relire : la ligne n'est plus consultée
    k2, v2 = _kv(8, 30)
    c.write(torch.tensor([int(tables[0, 2]) * BS + j for j in range(8, 16)], device="cuda"), k2, v2)
    assert int(c.tampon_de[int(tables[0, 2])]) == -1
    out48 = _attention(ext, c, q, tables, torch.tensor([48], device="cuda"), scale)
    hors, pire = _hors(out48, _reference(c, tables, torch.tensor([48], device="cuda"), q, scale))
    assert hors == 0, (hors, pire)


def test_graphe_cuda_ecriture_et_attention_rejouees_egalent_l_eager():
    """Le pas de décodage (écriture d'un jeton + attention) capturé dans un
    graphe et rejoué, jetons 40..55 : au bit contre le même pas en eager sur un
    second cache — le plan par jeton, la pile et la fermeture vivent sur
    l'appareil, rien n'est figé à la capture (leçon F3a, REGLES § 7)."""
    ext = _ext()
    c_g, tables, L, q, kv = _montage([40], graine=40)
    c_e, _, _, _, _ = _montage([40], graine=40)
    scale = 1 / math.sqrt(D)
    slot_s = torch.zeros(1, dtype=torch.long, device="cuda")
    k_s, v_s = torch.zeros(1, HKV, D, dtype=torch.bfloat16, device="cuda"), torch.zeros(1, HKV, D, dtype=torch.bfloat16, device="cuda")
    len_s = torch.zeros(1, dtype=torch.long, device="cuda")

    def pas(c):
        c.write(slot_s, k_s, v_s)
        return _attention(ext, c, q, tables, len_s, scale)

    ks, vs = _kv(16, 41)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):                    # chauffe hors capture sur un 3e cache
        c_w, _, _, _, _ = _montage([40], graine=40)
        slot_s.fill_(int(tables[0, 2]) * BS + 8); k_s.copy_(ks[:1]); v_s.copy_(vs[:1]); len_s.fill_(41)
        pas(c_w)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_s = pas(c_g)
    for j in range(16):
        pos = 40 + j
        slot_s.fill_(int(tables[0, pos // BS]) * BS + pos % BS); k_s.copy_(ks[j:j + 1]); v_s.copy_(vs[j:j + 1]); len_s.fill_(pos + 1)
        g.replay()
        out_e = pas(c_e)
        torch.cuda.synchronize()
        assert torch.equal(out_s, out_e), f"pas {pos}"
    _memes(_etat(c_g), _etat(c_e), "graphe")
    assert int(c_g.tampon_de[int(tables[0, 2])]) == -1 and int(c_g.tampon_de[int(tables[0, 3])]) >= 0
