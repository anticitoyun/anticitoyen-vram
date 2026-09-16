"""MLA absorbée à UNE PASSE (poste7, revue/poste7-duel-verdict-16-09.md § 3) :
`mla_decode_1p` lit le cache latent une fois pour les H têtes (softmax en
ligne, tranches de L recombinées) là où `mla_scores` + `mla_reduce` le
lisaient 2 × H fois. Contrats : même résultat que les deux noyaux d'avant au
bruit fp32 près (cos ≥ 0,9999 par tête, erreur relative ≤ 1e-4 contre une
référence float64), longueurs quelconques (0, milieu de tuile, godet plein),
B = 1 sans table d'adresses, H = 20 (GLM) et H = 32 (seconde variante),
capturable dans un graphe CUDA (aucune synchronisation hôte)."""
import pytest
import torch

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="pas de GPU")
W, R = 576, 512


def _ext():
    from acvram.kernels import get_extension
    ext = get_extension()
    if ext is None or not hasattr(ext, "mla_decode_1p"):
        pytest.skip("extension sans mla_decode_1p")
    return ext


def _cas(dev, B, H, L, lens, graine=1):
    g = torch.Generator(device=dev).manual_seed(graine)
    q = torch.randn(B, H, W, device=dev, generator=g) * 0.3
    caches = [torch.randn(L, W, device=dev, generator=g).to(torch.bfloat16).contiguous() for _ in range(B)]
    ptrs = torch.tensor([c.data_ptr() for c in caches], dtype=torch.int64, device=dev)
    lens_t = torch.tensor(lens, dtype=torch.long, device=dev)
    return q.contiguous(), caches, ptrs, lens_t


def _ref64(q, caches, lens, scale):
    B, H, _ = q.shape
    out = torch.zeros(B, H, R, dtype=torch.float64, device=q.device)
    for b in range(B):
        n = int(lens[b]) + 1                                  # r <= len, comme mla_scores
        k = caches[b][:n].double()                            # bf16 ou float64 (fp8 déquantifié exact)
        s = (q[b].double() @ k.T) * scale                     # [H, n]
        p = torch.softmax(s, dim=-1)
        out[b] = p @ k[:, :R]
    return out


def _compare(y, ref, y_old=None):
    B, H, _ = y.shape
    yf = y.double()
    cos = torch.nn.functional.cosine_similarity(yf.reshape(B * H, R), ref.reshape(B * H, R), dim=-1)
    assert cos.min().item() >= 0.9999, f"cos min {cos.min().item():.6f}"
    rel = ((yf - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max().item()
    assert rel <= 1e-4, f"erreur relative max {rel:.2e} contre float64"
    if y_old is not None:
        rel_o = ((yf - y_old.double()).norm(dim=-1) / y_old.double().norm(dim=-1).clamp(min=1e-30)).max().item()
        assert rel_o <= 1e-4, f"erreur relative max {rel_o:.2e} contre mla_decode_batch"


@CUDA
@pytest.mark.parametrize("H", [20, 32])
@pytest.mark.parametrize("L,lens", [
    (128, [0, 1, 17, 63, 64, 100, 126, 127]),
    (2048, [0, 31, 32, 33, 500, 1023, 2000, 2047]),
])
def test_une_passe_egale_deux_noyaux(H, L, lens):
    ext = _ext(); dev = torch.device("cuda:0")
    B = len(lens)
    q, caches, ptrs, lens_t = _cas(dev, B, H, L, lens)
    scale = 1.0 / (W ** 0.5)
    scores = torch.zeros(B, H, L, device=dev)
    y_old = ext.mla_decode_batch(q, ptrs, lens_t, scores, L, R, scale)
    y = ext.mla_decode_1p(q, ptrs, None, lens_t, L, R, scale)
    torch.cuda.synchronize()
    assert y.shape == (B, H, R) and torch.isfinite(y).all()
    _compare(y, _ref64(q, caches, lens, scale), y_old)


@CUDA
def test_une_sequence_sans_table():
    ext = _ext(); dev = torch.device("cuda:0")
    q, caches, ptrs, lens_t = _cas(dev, 1, 20, 256, [200], graine=5)
    scale = 1.0 / (W ** 0.5)
    y_tab = ext.mla_decode_1p(q, ptrs, None, lens_t, 256, R, scale)
    y_dir = ext.mla_decode_1p(q, None, caches[0], lens_t, 256, R, scale)
    assert torch.equal(y_tab, y_dir)
    scores = torch.zeros(20, 256, device=dev)
    y_old = ext.mla_decode(q[0].contiguous(), caches[0], lens_t[0], scores, 256, R, scale)
    _compare(y_dir, _ref64(q, caches, [200], scale), y_old.unsqueeze(0))


@CUDA
def test_scores_dominants_et_masque():
    """Une ligne du cache alignée sur q (score ≫ les autres) : le softmax doit
    la sélectionner ; la même ligne placée AU-DELÀ de len doit être ignorée —
    le contrôle rend faux si le masque r ≤ len ou le rescalage en ligne
    (m qui saute entre deux tuiles) est cassé."""
    ext = _ext(); dev = torch.device("cuda:0")
    B, H, L = 2, 20, 512
    q, caches, ptrs, lens_t = _cas(dev, B, H, L, [300, 300], graine=7)
    scale = 1.0 / (W ** 0.5)
    # créneau 0 : ligne 290 (dans la 10e tuile) = 30 × q[0, 3] ; créneau 1 : ligne 400 (> len) idem
    caches[0][290] = (30 * q[0, 3]).to(torch.bfloat16)
    caches[1][400] = (30 * q[1, 3]).to(torch.bfloat16)
    y = ext.mla_decode_1p(q, ptrs, None, lens_t, L, R, scale)
    ref = _ref64(q, caches, [300, 300], scale)
    _compare(y, ref)
    cos0 = torch.nn.functional.cosine_similarity(y[0, 3].double(), caches[0][290, :R].double(), dim=0).item()
    cos1 = torch.nn.functional.cosine_similarity(y[1, 3].double(), caches[1][400, :R].double(), dim=0).item()
    assert cos0 > 0.99, f"la ligne dominante n'est pas sélectionnée : cos {cos0:.3f}"
    assert cos1 < 0.5, f"une ligne au-delà de len a été lue : cos {cos1:.3f}"


@CUDA
def test_capturable_dans_un_graphe():
    ext = _ext(); dev = torch.device("cuda:0")
    q, caches, ptrs, lens_t = _cas(dev, 4, 20, 512, [10, 100, 300, 511], graine=3)
    scale = 1.0 / (W ** 0.5)
    attendu = ext.mla_decode_1p(q, ptrs, None, lens_t, 512, R, scale).clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            ext.mla_decode_1p(q, ptrs, None, lens_t, 512, R, scale)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = ext.mla_decode_1p(q, ptrs, None, lens_t, 512, R, scale)
    y.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, attendu)
    # les longueurs changent sans recapture (elles sont lues sur la carte)
    lens_t.fill_(5)
    g.replay(); torch.cuda.synchronize()
    assert torch.equal(y, ext.mla_decode_1p(q, ptrs, None, lens_t, 512, R, scale))


@CUDA
def test_cache_fp8_meme_arithmetique_sur_le_dequantifie():
    """Cache latent fp8 (commit 3) : mla_decode_1p en fp8 sur les codes ==
    mla_decode_1p bf16 sur le cache DÉQUANTIFIÉ (mêmes nombres, même
    arithmétique fp32 ensuite) ; mla_ecrit_latent fp8 == _fp8_quant_rows au
    bit ; et l'écart fp8 / bf16 d'origine est mesuré (information, pas
    scellé ici : c'est la PPL 3 tranches ± 0,004 qui tranche)."""
    ext = _ext(); dev = torch.device("cuda:0")
    from acvram.engine.mla import _fp8_quant_rows, _fp8_dequant_rows, FP8_PAD
    B, H, L = 4, 20, 512
    lens = [0, 100, 300, 511]
    q, caches, ptrs, lens_t = _cas(dev, B, H, L, lens, graine=11)
    scale = 1.0 / (W ** 0.5)
    c8 = [_fp8_quant_rows(c).contiguous() for c in caches]
    assert c8[0].shape == (L, W + FP8_PAD)
    ptrs8 = torch.tensor([c.data_ptr() for c in c8], dtype=torch.int64, device=dev)
    y8 = ext.mla_decode_1p(q, ptrs8, None, lens_t, L, R, scale, True)
    # référence float64 sur les valeurs EXACTES code × échelle (un cache bf16
    # déquantifié arrondirait code × s à 8 bits : 1,7e-3 d'écart, t-qa 3a1d2fd)
    deq64 = [_fp8_dequant_rows(c, W, torch.float64) for c in c8]
    ref = _ref64(q, deq64, lens, scale)
    torch.cuda.synchronize()
    rel = ((y8.double() - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max().item()
    assert rel <= 1e-4, f"fp8 sur codes vs float64 sur code × échelle : {rel:.2e}"
    y_bf = ext.mla_decode_1p(q, ptrs, None, lens_t, L, R, scale, False)
    cos = torch.nn.functional.cosine_similarity(y8.double().reshape(B * H, R), y_bf.double().reshape(B * H, R), dim=-1)
    print(f"\nfp8 vs bf16 d'origine : cos min {cos.min().item():.6f}")
    assert cos.min().item() > 0.99
    # écriture d'une ligne en fp8 par le noyau == référence torch
    k_new = torch.randn(B, W, device=dev).to(torch.bfloat16).contiguous()
    lens_w = [torch.tensor(n, dtype=torch.long, device=dev) for n in lens]
    lptrs = torch.tensor([t.data_ptr() for t in lens_w], dtype=torch.int64, device=dev)
    ext.mla_ecrit_latent(k_new, ptrs8, lptrs, True)
    torch.cuda.synchronize()
    ref = _fp8_quant_rows(k_new)
    for b, n in enumerate(lens):
        assert torch.equal(c8[b][n], ref[b]), f"créneau {b} : ligne fp8 ≠ référence"
        assert int(lens_w[b]) == n + 1


@CUDA
@pytest.mark.parametrize("H", [20])
def test_puits_d_attention_en_position_0(H):
    """poste7 § 12 : GLM porte un puits d'attention en position 0 ([gMASK]<sop>) —
    la ligne 0 du cache attire presque toute la masse du softmax de toutes les
    têtes. Le noyau à une passe (softmax en ligne, tranches recombinées : la
    ligne 0 est dans la première tuile de la première tranche, son maximum
    domine M) doit rendre la référence float64 et les deux noyaux d'avant
    comme pour un cache ordinaire — un puits qui casserait ici nommerait le
    noyau ; s'il passe, le défaut est ailleurs (prefill, créneaux, blocs)."""
    ext = _ext(); dev = torch.device("cuda:0")
    B, L = 4, 2048
    lens = [1, 300, 1500, 2047]
    q, caches, ptrs, lens_t = _cas(dev, B, H, L, lens, graine=21)
    scale = 1.0 / (W ** 0.5)
    for b in range(B):
        # ligne 0 alignée sur TOUTES les têtes : direction moyenne des q, × 40
        puits = q[b].mean(0)
        caches[b][0] = (40 * puits / puits.norm() * q[b].norm(dim=-1).mean()).to(torch.bfloat16)
    y = ext.mla_decode_1p(q, ptrs, None, lens_t, L, R, scale)
    scores = torch.zeros(B, H, L, device=dev)
    y_old = ext.mla_decode_batch(q, ptrs, lens_t, scores, L, R, scale)
    torch.cuda.synchronize()
    ref = _ref64(q, caches, lens, scale)
    # le puits domine bien : p_0 > 0,9 en moyenne sur les têtes du créneau 3
    k = caches[3][:2048].double(); s = (q[3].double() @ k.T) * scale
    p0 = torch.softmax(s, -1)[:, 0].mean().item()
    assert p0 > 0.9, f"le puits n'attire que {p0:.2f}"
    _compare(y, ref, y_old)
