"""Décodage MLA batché (bead 6wa) contre la boucle par créneau : bit-identique.

Deux niveaux : le noyau seul (caches non contigus, ``len`` hétérogènes,
créneaux de rembourrage à ``len`` 0), puis le module ``MLAttention`` complet
(projections, RoPE absent, norme, écriture du latent, o_proj) — la boucle
``decode_static`` contre ``decode_static_batch`` sur les mêmes états.
"""
import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau CUDA requis")


def _ext():
    from acvram.kernels import get_extension
    ext = get_extension()
    if not hasattr(ext, "mla_decode_batch"):
        pytest.skip("extension sans mla_decode_batch")
    return ext


@pytest.mark.parametrize("B,H,rank,rope,L,lens", [
    (12, 20, 512, 64, 2048, [5, 300, 2047, 0, 1023, 17, 900, 1500, 2, 64, 777, 1999]),
    (3, 20, 512, 64, 384, [10, 0, 0]),                # b_reel=1 + deux rembourrages
    (12, 8, 128, 32, 4096, [4000] * 6 + [0] * 6),     # L > 48 Ko de shared
])
def test_noyau_batch_bit_identique(B, H, rank, rope, L, lens):
    ext = _ext()
    W = rank + rope
    g = torch.Generator(device="cuda").manual_seed(B * 7 + L)
    caches = [torch.randn(L + 64, W, device="cuda", generator=g).to(torch.bfloat16) for _ in range(B)]
    q = torch.randn(B, H, W, device="cuda", generator=g)
    lens_t = [torch.tensor(n, dtype=torch.long, device="cuda") for n in lens]
    scale = (W - rank + 64) ** -0.5
    # boucle : un créneau à la fois, avec son propre tampon de scores
    attendu = []
    for i in range(B):
        sc = torch.zeros(H, L + 64, device="cuda")
        attendu.append(ext.mla_decode(q[i].contiguous(), caches[i], lens_t[i], sc, L, rank, scale))
    attendu = torch.stack(attendu)
    ptrs = torch.tensor([c.data_ptr() for c in caches], dtype=torch.int64, device="cuda")
    scores = torch.zeros(B, H, L + 64, device="cuda")
    y = ext.mla_decode_batch(q.contiguous(), ptrs, torch.stack(lens_t), scores, L, rank, scale)
    assert y.shape == (B, H, rank)
    assert torch.equal(y, attendu)
    # contrôle qui peut rendre faux : une table qui permute deux caches change la sortie
    ptrs2 = ptrs.clone(); ptrs2[0], ptrs2[1] = ptrs[1].item(), ptrs[0].item()
    y2 = ext.mla_decode_batch(q.contiguous(), ptrs2, torch.stack(lens_t), scores, L, rank, scale)
    assert not torch.equal(y2, attendu)


def _module(nh, nope, rope, rank, dv, hidden, seed):
    from acvram.engine.mla import MLAttention
    torch.manual_seed(seed)
    dt = torch.bfloat16
    q_proj = nn.Linear(hidden, nh * (nope + rope), bias=False).to("cuda", dt)
    kv_a = nn.Linear(hidden, rank + rope, bias=False).to("cuda", dt)
    o_proj = nn.Linear(nh * dv, hidden, bias=False).to("cuda", dt)
    kv_norm = torch.ones(rank, device="cuda", dtype=dt)
    k_b = (torch.randn(nh, rank, nope, device="cuda") * 0.05).to(dt)
    v_b = (torch.randn(nh, dv, rank, device="cuda") * 0.05).to(dt)
    return MLAttention(q_proj, kv_a, o_proj, kv_norm, k_b, v_b, nh, nope, rope, rank, dv)


@pytest.mark.parametrize("B,lens", [(12, [0, 3, 40, 127, 5, 5, 60, 99, 1, 0, 120, 33]), (4, [10, 0, 0, 0])])
def test_module_batch_bit_identique(B, lens):
    _ext()
    nh, nope, rope, rank, dv, hidden, L = 4, 32, 16, 64, 32, 128, 128
    bucket = L
    la = _module(nh, nope, rope, rank, dv, hidden, B)
    if la.rope_emb is not None:
        pytest.skip("test écrit sans RoPE")
    torch.manual_seed(B + 1)
    x = torch.randn(B, hidden, device="cuda").to(torch.bfloat16)
    # deux jeux d'états identiques (caches remplis jusqu'à len, non triviaux)
    def etats():
        sts = []
        for n in lens:
            st = la.new_static(torch.device("cuda"), L + 16, torch.bfloat16)
            st["cache"][:n] = (torch.randn(n, rank + rope, device="cuda") * 0.3).to(torch.bfloat16)
            st["len"].fill_(n)
            sts.append(st)
        return sts
    torch.manual_seed(7); sts_a = etats()
    torch.manual_seed(7); sts_b = etats()
    with torch.inference_mode():
        ya = torch.cat([la.decode_static(x[i:i + 1], sts_a[i], bucket) for i in range(B)], dim=0)
        ptrs = torch.tensor([st["cache"].data_ptr() for st in sts_b], dtype=torch.int64, device="cuda")
        scores = torch.zeros(B, nh, L + 16, device="cuda")
        yb = la.decode_static_batch(x, sts_b, bucket, ptrs, scores)
    assert torch.equal(ya, yb)
    for a, b in zip(sts_a, sts_b):
        assert torch.equal(a["cache"], b["cache"]) and torch.equal(a["len"], b["len"])


@pytest.mark.parametrize("B,lens", [(12, [0, 3, 40, 127, 5, 5, 60, 99, 1, 0, 120, 33]), (4, [10, 0, 0, 0]),
                                    (1, [40])])   # C15 niveau 2 : b=1 par le chemin complet
def test_module_batch_complet(B, lens):
    """``decode_static_batch_complet`` (projections batchées) : bit-identique à
    ``forward_batch`` (même numérique de lot), et à un ulp bf16 de la boucle
    par créneau (les GEMV à M=B et M=1 n'arrondissent pas forcément pareil —
    l'écart est mesuré et borné, pas supposé nul)."""
    _ext()
    nh, nope, rope, rank, dv, hidden, L = 4, 32, 16, 64, 32, 128, 128
    bucket = L
    la = _module(nh, nope, rope, rank, dv, hidden, B + 100)
    torch.manual_seed(B + 2)
    x = torch.randn(B, hidden, device="cuda").to(torch.bfloat16)

    def etats(seed):
        torch.manual_seed(seed)
        sts = []
        for n in lens:
            st = la.new_static(torch.device("cuda"), L + 16, torch.bfloat16)
            st["cache"][:n] = (torch.randn(n, rank + rope, device="cuda") * 0.3).to(torch.bfloat16)
            st["len"].fill_(n)
            sts.append(st)
        return sts
    sts_a, sts_b, sts_c = etats(7), etats(7), etats(7)
    with torch.inference_mode():
        y_boucle = torch.cat([la.decode_static(x[i:i + 1], sts_a[i], bucket) for i in range(B)], dim=0)
        ptrs = torch.tensor([st["cache"].data_ptr() for st in sts_b], dtype=torch.int64, device="cuda")
        scores = torch.zeros(B, nh, L + 16, device="cuda")
        len_ptrs = torch.tensor([st["len"].data_ptr() for st in sts_b], dtype=torch.int64, device="cuda")
        import acvram.engine.mla as mla_mod
        noyau = mla_mod._MLA_PREP_NOYAU
        mla_mod._MLA_PREP_NOYAU = False                 # témoin torch : bit-identique à forward_batch
        try:
            y_complet = la.decode_static_batch_complet(x, sts_b, bucket, ptrs, scores, len_ptrs)
        finally:
            mla_mod._MLA_PREP_NOYAU = noyau
        sts_d = etats(7)
        ptrs_d = torch.tensor([st["cache"].data_ptr() for st in sts_d], dtype=torch.int64, device="cuda")
        len_ptrs_d = torch.tensor([st["len"].data_ptr() for st in sts_d], dtype=torch.int64, device="cuda")
        y_noyau = la.decode_static_batch_complet(x, sts_d, bucket, ptrs_d, scores, len_ptrs_d)
        # forward_batch : même lot, caches passés comme tenseurs [len, W] (None si vides)
        caches = [None if n == 0 else sts_c[i]["cache"][:n].clone() for i, n in enumerate(lens)]
        y_fb, _ = la.forward_batch(x, caches)
    # les états avancent pareil (la norme kv_a du noyau est bit-identique : caches égaux)
    for a, b in zip(sts_a, sts_b):
        assert torch.equal(a["cache"], b["cache"]) and torch.equal(a["len"], b["len"])
    for a, d in zip(sts_a, sts_d):
        assert torch.equal(a["cache"], d["cache"]) and torch.equal(a["len"], d["len"])
    assert torch.equal(y_complet, y_fb), "le chemin complet doit reproduire forward_batch bit a bit"
    a, b = y_complet.float(), y_boucle.float()
    tol = b.abs() * 2 ** -7 + 1e-3 * b.abs().max()
    hors = int(((a - b).abs() > tol).sum())
    ident = (a == b).float().mean().item()
    assert hors == 0, f"{hors} valeurs hors tolerance, max {(a - b).abs().max().item():.3e}"
    print(f"\ncomplet vs boucle : {ident:.4f} identiques, ecart max {(a - b).abs().max().item():.3e}")
    # préparation en un noyau (mla_prep_batch) : seul l'einsum k_b change d'ordre de sommes
    if hasattr(_ext(), "mla_prep_batch"):
        c = y_noyau.float()
        hors_n = int(((c - b).abs() > tol).sum())
        assert hors_n == 0, f"prep noyau : {hors_n} valeurs hors tolerance vs boucle, max {(c - b).abs().max().item():.3e}"
        print(f"prep noyau vs témoin torch : {(c == a).float().mean().item():.4f} identiques")


def test_ecrit_latent_un_lancement():
    """``mla_ecrit_latent`` : k_new[b] à la ligne len_b de chaque cache, puis
    len_b += 1 — identique à la boucle index_copy_ / add_, et le nombre de
    lancements du chemin complet ne dépend plus de B (12 créneaux coûtaient
    12 index_copy_ + 12 add_ par couche, poste7-duel-verdict-16-09 § 6)."""
    ext = _ext()
    if not hasattr(ext, "mla_ecrit_latent"):
        pytest.skip("extension sans mla_ecrit_latent")
    from torch.profiler import profile, ProfilerActivity
    nh, nope, rope, rank, dv, hidden, L = 4, 32, 16, 64, 32, 128, 128
    W = rank + rope
    torch.manual_seed(3)
    for B in (4, 12):
        lens = [(7 * i) % 100 for i in range(B)]
        sts_a, sts_b = [], []
        for n in lens:
            for sts in (sts_a, sts_b):
                st = {"cache": torch.zeros(L + 16, W, dtype=torch.bfloat16, device="cuda"),
                      "len": torch.tensor(n, dtype=torch.long, device="cuda")}
                sts.append(st)
        k_new = torch.randn(B, W, device="cuda").to(torch.bfloat16)
        for i, st in enumerate(sts_a):
            st["cache"].index_copy_(0, st["len"].view(1), k_new[i:i + 1]); st["len"].add_(1)
        ptrs = torch.tensor([st["cache"].data_ptr() for st in sts_b], dtype=torch.int64, device="cuda")
        lptrs = torch.tensor([st["len"].data_ptr() for st in sts_b], dtype=torch.int64, device="cuda")
        ext.mla_ecrit_latent(k_new, ptrs, lptrs)
        torch.cuda.synchronize()
        for a, b in zip(sts_a, sts_b):
            assert torch.equal(a["cache"], b["cache"]) and torch.equal(a["len"], b["len"])
    # lancements du chemin complet : indépendants de B
    la = _module(nh, nope, rope, rank, dv, hidden, 200)
    compte = {}
    for B in (4, 12):
        sts = []
        for i in range(B):
            st = la.new_static(torch.device("cuda"), L + 16, torch.bfloat16); st["len"].fill_(5 * i); sts.append(st)
        x = torch.randn(B, hidden, device="cuda").to(torch.bfloat16)
        ptrs = torch.tensor([st["cache"].data_ptr() for st in sts], dtype=torch.int64, device="cuda")
        lptrs = torch.tensor([st["len"].data_ptr() for st in sts], dtype=torch.int64, device="cuda")
        scores = torch.zeros(B, nh, L + 16, device="cuda")
        with torch.inference_mode():
            la.decode_static_batch_complet(x, sts, L, ptrs, scores, lptrs); torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                la.decode_static_batch_complet(x, sts, L, ptrs, scores, lptrs); torch.cuda.synchronize()
        compte[B] = sum(e.count for e in prof.key_averages() if e.self_device_time_total > 0
                        and "Memcpy" not in e.key and "Memset" not in e.key)
    assert compte[4] == compte[12], f"lancements par appel : B=4 -> {compte[4]}, B=12 -> {compte[12]}"
    print(f"\nlancements par couche MLA (chemin complet) : {compte[12]}")


def test_prep_batch_avec_rope():
    """``mla_prep_batch`` avec RoPE : q_pe et k_pe tournés comme le chemin torch
    (cos/sin bf16, produits et somme arrondis en bf16) — bit-identique sur k_new
    (norme + RoPE) et sur la partie RoPE de q_eff ; q_abs à ≤ 1 ulp bf16."""
    ext = _ext()
    if not hasattr(ext, "mla_prep_batch"):
        pytest.skip("extension sans mla_prep_batch")
    from acvram.engine.layers import RotaryEmbedding
    import acvram.engine.mla as mla_mod
    nh, nope, rope, rank, dv, hidden, L = 4, 32, 16, 64, 32, 128, 128
    B = 6
    la = _module(nh, nope, rope, rank, dv, hidden, 11)
    la.rope_emb = RotaryEmbedding(rope, 4096, 10000.0, None, torch.device("cuda"), torch.bfloat16)
    torch.manual_seed(5)
    x = torch.randn(B, hidden, device="cuda").to(torch.bfloat16)
    lens = [0, 3, 40, 127, 5, 60]
    res = {}
    for noyau in (False, True):
        sts = []
        for n in lens:
            st = la.new_static(torch.device("cuda"), L + 16, torch.bfloat16)
            st["len"].fill_(n); sts.append(st)
        ptrs = torch.tensor([st["cache"].data_ptr() for st in sts], dtype=torch.int64, device="cuda")
        lptrs = torch.tensor([st["len"].data_ptr() for st in sts], dtype=torch.int64, device="cuda")
        scores = torch.zeros(B, nh, L + 16, device="cuda")
        garde = mla_mod._MLA_PREP_NOYAU
        mla_mod._MLA_PREP_NOYAU = noyau
        try:
            with torch.inference_mode():
                y = la.decode_static_batch_complet(x, sts, L, ptrs, scores, lptrs)
        finally:
            mla_mod._MLA_PREP_NOYAU = garde
        torch.cuda.synchronize()
        res[noyau] = (y, [st["cache"][n].clone() for st, n in zip(sts, lens)])
    for kt, kn in zip(res[False][1], res[True][1]):
        assert torch.equal(kt, kn), "k_new (norme + RoPE) doit être bit-identique au chemin torch"
    a, b = res[True][0].float(), res[False][0].float()
    tol = b.abs() * 2 ** -7 + 1e-3 * b.abs().max()
    hors = int(((a - b).abs() > tol).sum())
    assert hors == 0, f"{hors} valeurs hors tolerance, max {(a - b).abs().max().item():.3e}"
