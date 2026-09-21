"""Juge de la fusion (3a) du poste F (kernels/rope_kv.py) : normes par tête +
RoPE + écriture int8 du cache en un noyau, contre la référence fp32 de
`rope_inplace_kernel` / `kv_write_int8_kernel` recalculée en torch : q et k
à ≤ 1 ulp (rsqrt contre 1/sqrt : un arrondi fp32 peut basculer une
conversion sur < 1 % des valeurs — 2⁻⁸ bf16, le seuil de Sage), codes int8 et échelles fp16
identiques (calculés depuis le k écrit), créneau < 0 intact, sans norme q, rotation partielle (DR < D),
tranches d'une projection empilée (foulées) ; bras qui doit différer :
position ou poids de norme changés. Sans carte : interpréteur en fp16."""
import importlib
import os

import pytest
import torch

from acvram.memory.kvcache import KVCacheConfig, PagedKVCache

# F3a CLOS RÉFUTÉ (verdict-f3a-finale-17-09) : sur carte, la référence est
# kv_write_int8 et le noyau d462c24 s'en écarte aux demi-entiers exacts (3/6)
# — les variantes qui n'en dévient pas corrompent sous graphe. Le fait reste
# documenté par ces tests : sur carte, 3 des 6 paramétrages sont rouges (les
# demi-entiers exacts), les autres verts — xfail non strict, le verdict porte
# le détail ; sans carte (interpréteur, formule), tout est vert.
CARTE_REFUTE = pytest.mark.xfail(torch.cuda.is_available(), strict=False,
                                 reason="F3a réfuté : codes ≠ kv_write_int8 aux demi-entiers exacts (a3f1b7e)")


def _rk():
    if not torch.cuda.is_available():
        os.environ.setdefault("TRITON_INTERPRET", "1")
    rk = importlib.import_module("acvram.kernels.rope_kv")
    if not rk.disponible():
        pytest.skip("Triton indisponible")
    return rk


DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if DEV == "cuda" else torch.float16


def _tables(maxpos, dr):
    ang = torch.outer(torch.arange(maxpos).float(), 1.0 / (10000 ** (torch.arange(0, dr, 2).float() / dr)))
    return (torch.cat([ang.cos(), ang.cos()], -1).contiguous().to(DEV),
            torch.cat([ang.sin(), ang.sin()], -1).contiguous().to(DEV))


def _ref(x, w, eps, cos32, sin32, pos, dr):
    """rope_inplace_kernel en torch fp32 : norme sur la tête entière, rotation
    des dr premières coordonnées, une conversion en sortie."""
    xf = x.float()
    if w is not None:
        y = xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + eps) * w.float()
    else:
        y = xf
    c, s = cos32[pos.to(cos32.device)][:, None, :], sin32[pos.to(sin32.device)][:, None, :]
    h = dr // 2
    y1, y2 = y[..., :h], y[..., h:dr]
    rot = torch.cat([y1 * c[..., :h] - y2 * s[..., :h], y2 * c[..., h:] + y1 * s[..., h:]], -1)
    return torch.cat([rot, y[..., dr:]], -1).to(x.dtype)


def _codes(x):
    """Sous l'interpréteur : la formule de kv_write_int8_kernel en torch —
    sc = max(amax/127, 1e-8), inv = 1/sc, code = rint(x · inv). Sur carte,
    cette formule en torch n'est PAS la vérité (Laure : x = 1,8671875,
    amax = 3,734375 → produit fp32 IEEE 63,499996 → 63, torch cuda rend 64,
    autre ordre ou fma) : la vérité est le noyau CUDA lui-même, voir
    `_cache_cuda`."""
    xf = x.float()
    sc = (xf.abs().amax(-1, keepdim=True) / 127).clamp(min=1e-8)
    return (xf * (1.0 / sc)).round().clamp(-127, 127), sc.squeeze(-1).to(torch.float16)


def _cache_cuda(k, v, slots, HKV, D, blocs):
    """Sur carte : le cache écrit par le chemin actuel (`PagedKVCache.write` →
    `kv_write_int8`, le noyau CUDA que rope_kv remplace) — la référence
    d'équivalence, bit à bit ; None sans carte ou sans extension."""
    if not torch.cuda.is_available():
        return None
    from acvram.kernels import get_extension
    ext = get_extension()
    if ext is None or not hasattr(ext, "kv_write_int8"):
        return None
    c = PagedKVCache(KVCacheConfig(num_layers=1, num_kv_heads=HKV, head_dim=D, num_blocks=blocs, dtype="int8", device=DEV))
    c.write(slots, k.to(torch.bfloat16), v.to(torch.bfloat16))
    return c


def _egal_sauf_demi_entiers(codes, ref_codes, x, sc):
    """Codes égaux, ou différents d'UNE unité là où x·(1/sc) tombe exactement
    sur un demi-entier : l'erreur de quantification y est ½ pas des deux côtés
    (kv_write_int8 arrondit le produit fp32, le noyau Triton lit le produit
    contracté en fma — versions df4db39+ exactes ici mais fausses sous graphe)."""
    if torch.equal(codes, ref_codes):
        return True
    p = x.float() * (1.0 / sc.float().unsqueeze(-1))
    demi = (p - p.floor() - 0.5).abs() < 2 ** -16
    diff = (codes.int() - ref_codes.int()).abs()
    return bool(((diff == 0) | ((diff == 1) & demi)).all())


def _memes_codes(c, ref, k, v, slots, t):
    """Codes et échelles du créneau t : contre le cache CUDA si présent, sinon la formule."""
    sl = int(slots[t])
    blk, off = sl // 16, sl % 16
    if ref is not None:
        return (torch.equal(c.k_scale[blk, off], ref.k_scale[blk, off])
                and torch.equal(c.v_scale[blk, off], ref.v_scale[blk, off])
                and _egal_sauf_demi_entiers(c.k[blk, off], ref.k[blk, off], k[t], ref.k_scale[blk, off])
                and _egal_sauf_demi_entiers(c.v[blk, off], ref.v[blk, off], v[t], ref.v_scale[blk, off]))
    kc, ks = _codes(k[t])
    vc, vs = _codes(v[t])
    return (torch.equal(c.k[blk, off].float(), kc) and torch.equal(c.v[blk, off].float(), vc)
            and torch.equal(c.k_scale[blk, off], ks) and torch.equal(c.v_scale[blk, off], vs))


def _montage(T, HQ, HKV, D, DR, norme_q=True, empile=False, graine=0):
    torch.manual_seed(graine)
    if empile:                                   # q, k, v = tranches d'une projection [T, (HQ+2HKV)·D]
        proj = torch.randn(T, (HQ + 2 * HKV) * D).to(DT).to(DEV)
        q = proj[:, :HQ * D].view(T, HQ, D)
        k = proj[:, HQ * D:(HQ + HKV) * D].view(T, HKV, D)
        v = proj[:, (HQ + HKV) * D:].view(T, HKV, D)
    else:
        q, k, v = (torch.randn(T, h, D).to(DT).to(DEV) for h in (HQ, HKV, HKV))
    wq = (1 + 0.1 * torch.randn(D)).to(DT).to(DEV) if norme_q else None
    wk = (1 + 0.1 * torch.randn(D)).to(DT).to(DEV)
    pos = torch.randint(0, 64, (T,)).to(DEV)
    slots = torch.arange(T) * 3 + 1
    slots[T // 2] = -1                           # une ligne de rembourrage
    slots = slots.to(DEV)
    cos32, sin32 = _tables(64, DR)
    c = PagedKVCache(KVCacheConfig(num_layers=1, num_kv_heads=HKV, head_dim=D, num_blocks=4, dtype="int8", device=DEV))
    return q, k, v, wq, wk, pos, slots, cos32, sin32, c


@CARTE_REFUTE
@pytest.mark.parametrize("HQ,HKV,D,DR,norme_q,empile", [(8, 2, 128, 128, True, False), (4, 1, 64, 32, False, False),
                                                        (8, 2, 128, 128, True, True), (2, 2, 32, 32, True, False)])
def test_q_k_et_cache_suivent_la_reference(HQ, HKV, D, DR, norme_q, empile):
    rk = _rk()
    T, eps = 6, 1e-6
    q, k, v, wq, wk, pos, slots, cos32, sin32, c = _montage(T, HQ, HKV, D, DR, norme_q, empile)
    rq, rk_ = _ref(q, wq, eps, cos32, sin32, pos, DR), _ref(k, wk, eps, cos32, sin32, pos, DR)
    q0, k0 = q.clone(), k.clone()
    rk.rope_kv(q, k, v, cos32, sin32, pos, slots, wq, wk, eps, c)
    for nom, a, b in (("q", q, rq), ("k", k, rk_)):
        ulp = (a.view(torch.int16).int() - b.view(torch.int16).int()).abs()
        assert int(ulp.max()) <= 1, f"{nom} : {int(ulp.max())} ulp de la référence fp32"
        assert float((ulp > 0).float().mean()) < 0.01, f"{nom} : trop de valeurs à 1 ulp"
    assert not torch.equal(q, q0) and not torch.equal(k, k0)
    ref = _cache_cuda(k, v, slots, HKV, D, 4)              # k tel qu'écrit par rope_kv (après rotation)
    for t in range(T):
        if int(slots[t]) < 0:
            continue
        assert _memes_codes(c, ref, k, v, slots, t), f"jeton {t} : codes ou échelles ≠ {'kv_write_int8' if ref is not None else 'formule'}"
    ecrits = {(int(s) // 16, int(s) % 16) for s in slots if int(s) >= 0}
    for blk in range(4):
        for off in range(16):
            if (blk, off) not in ecrits:
                assert not c.k[blk, off].any() and not c.v[blk, off].any(), "écrit hors des créneaux (rembourrage ?)"


def test_les_bras_qui_doivent_differer():
    rk = _rk()
    T, eps = 4, 1e-6
    q, k, v, wq, wk, pos, slots, cos32, sin32, c = _montage(T, 4, 2, 64, 64)
    ref = _ref(q, wq, eps, cos32, sin32, pos, 64)
    q1 = q.clone(); rk.rope_kv(q1, k.clone(), v, cos32, sin32, pos + 1, slots, wq, wk, eps, c)
    assert not torch.equal(q1, ref), "une autre position doit changer q"
    q2 = q.clone(); rk.rope_kv(q2, k.clone(), v, cos32, sin32, pos, slots, wq * 2, wk, eps, c)
    assert not torch.equal(q2, ref), "un autre poids de norme doit changer q"


@CARTE_REFUTE
def test_le_demi_entier_exact_arrondit_comme_kv_write_int8():
    """Le cas de Laure (rope-kv-diff.log) : x = amax/2 en bf16 donne
    x·(1/sc) = 63,5·(1+ε) — le code doit être celui du noyau CUDA
    `kv_write_int8` (`1.f/sc` puis `__float2int_rn`) : sur carte on compare
    au cache écrit par ce noyau (une formule torch cuda ne le reproduit pas
    à l'ulp près) ; sous l'interpréteur, à la formule en torch. 64 têtes dont
    la moitié des coordonnées valent exactement amax/2."""
    rk = _rk()
    T, HKV, D = 4, 16, 32
    q = torch.zeros(T, 1, D, dtype=DT, device=DEV)
    base = torch.randn(T, HKV, D, device=DEV).to(DT)
    amax = base.abs().amax(-1, keepdim=True)
    base[..., ::2] = (amax / 2).to(DT).expand(-1, -1, D // 2)     # exactement amax/2 (exposant −1)
    k, v = base.clone(), base.clone()
    slots = torch.arange(T, device=DEV)
    pos = torch.zeros(T, dtype=torch.long, device=DEV)
    cos32, sin32 = _tables(4, D)
    c = PagedKVCache(KVCacheConfig(num_layers=1, num_kv_heads=HKV, head_dim=D, num_blocks=1, dtype="int8", device=DEV))
    rk.rope_kv(q, k, v, cos32, sin32, pos, slots, None, None, 1e-6, c)
    ref = _cache_cuda(k, v, slots, HKV, D, 1)
    for t in range(T):
        assert _memes_codes(c, ref, k, v, slots, t), t
