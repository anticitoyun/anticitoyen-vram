"""C15 niveau 3 (revue/chantier-c15-niveau3-coder-20-09), ACVRAM_GLUE_COMPACT :
les trois retraits de glue AU BIT du pas GQA (item 2 de la fiche) —

* `kv_write_int8` lit k et v par leur pas de jeton (tranches de la projection
  q/k/v empilée) au lieu de deux copies contiguës par couche
  (acvram_kernels.cu, `contigu_par_tete`) ; le témoin les recopie en Python
  (kvcache.write) ;
* l'attention paginée Triton lit q par ses pas (`_partiel_kernel`, stride_qb /
  stride_qh) : plus de `q.contiguous()` dans kernels.paged_attention ;
* `valid = slots >= 0` calculé une fois par pas (ACVRamModel.decode_fixed)
  au lieu d'une fois par couche (DecoderLayer.decode_fixed_res).

Sans carte : l'attention Triton sous ``TRITON_INTERPRET=1`` (q fp16) et la
plomberie ; le noyau CUDA (pas de jeton) n'a de preuve que sur carte.
"""
import importlib
import math
import os

import pytest
import torch

from acvram.memory.kvcache import KVCacheConfig, PagedKVCache

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _ap():
    if not torch.cuda.is_available():
        os.environ.setdefault("TRITON_INTERPRET", "1")
    ap = importlib.import_module("acvram.kernels.attn_paginee")
    if not ap.disponible():
        pytest.skip("Triton indisponible")
    return ap


def _cache(lens, hkv=2, d=128, graine=0):
    torch.manual_seed(graine)
    B = len(lens)
    N = -(-max(lens) // 16) + 1
    c = PagedKVCache(KVCacheConfig(num_layers=1, num_kv_heads=hkv, head_dim=d,
                                   num_blocks=B * N, dtype="int8", device=DEV))
    tables = torch.arange(B * N, device=DEV).view(B, N)
    for b, n in enumerate(lens):
        slots = torch.tensor([int(tables[b, i // 16]) * 16 + i % 16 for i in range(n)], device=DEV)
        c.write(slots, torch.randn(n, hkv, d, device=DEV), torch.randn(n, hkv, d, device=DEV))
    return c, tables, torch.tensor(lens, device=DEV)


def test_attention_triton_lit_q_par_ses_pas_au_bit():
    """q tranche d'une projection empilée [B, (HQ + 2·HKV)·D] → même sortie,
    au bit, que la copie contiguë ; et le bras qui casse : la tranche voisine
    (k) donnée pour q rend autre chose."""
    ap = _ap()
    c, tables, L = _cache([37, 300])
    B, hkv, n_rep, d = 2, 2, 8, 128
    dt = torch.bfloat16 if DEV == "cuda" else torch.float16
    qkv = torch.randn(B, (hkv * n_rep + 2 * hkv) * d, device=DEV).to(dt)
    q_vue = qkv[:, : hkv * n_rep * d].view(B, hkv * n_rep, d)          # pas (…, 128, 1) : non contigu
    assert not q_vue.is_contiguous() and q_vue.stride(1) == d
    scale = 1 / math.sqrt(d)
    a = ap.paged_attention(q_vue, c.k, c.k_scale, c.v, c.v_scale, tables, L, hkv, scale)
    b = ap.paged_attention(q_vue.contiguous(), c.k, c.k_scale, c.v, c.v_scale, tables, L, hkv, scale)
    assert torch.equal(a, b)
    k_vue = qkv[:, d: d + hkv * n_rep * d].view(B, hkv * n_rep, d)  # décalée d'une tête
    autre = ap.paged_attention(k_vue, c.k, c.k_scale, c.v, c.v_scale, tables, L, hkv, scale)
    assert not torch.equal(a, autre)


def _ulp_max(a, b):
    """Écart max en ulp du dtype 16 bits de la référence a (b comparé)."""
    a32, b32 = a.float(), b.float()
    mant = 7 if a.dtype == torch.bfloat16 else 10
    ulp = torch.where(a32 != 0, 2.0 ** (torch.floor(torch.log2(a32.abs().clamp_min(1e-30))) - mant),
                      torch.full_like(a32, 2.0 ** -(126 + mant)))
    return float(((a32 - b32).abs() / ulp).max())


@pytest.mark.parametrize("lens", [[37, 0, 300], [1], [2048, 17], [16, 32, 33, 1]])
def test_attention_un_noyau_egale_deux_noyaux(lens):
    """Item 4 : `_partiel_reduit_kernel` (réduction par le dernier programme,
    ordre 0..C−1) contre `_partiel` + `_reduce` : ≤ 1 ulp 16 bits par valeur
    (seul l'ordre de la somme de `lg` peut différer), fantômes nuls, et les
    compteurs revenus à zéro après le lancement."""
    ap = _ap()
    c, tables, L = _cache([max(n, 1) for n in lens])
    L = torch.tensor(lens, device=DEV)
    hkv, n_rep, d = 2, 8, 128
    dt = torch.bfloat16 if DEV == "cuda" else torch.float16
    torch.manual_seed(3)
    q = torch.randn(len(lens), hkv * n_rep, d, device=DEV).to(dt)
    scale = 1 / math.sqrt(d)
    deux = ap.paged_attention(q, c.k, c.k_scale, c.v, c.v_scale, tables, L, hkv, scale, compact=False)
    un = ap.paged_attention(q, c.k, c.k_scale, c.v, c.v_scale, tables, L, hkv, scale, compact=True)
    assert torch.isfinite(un.float()).all()
    assert _ulp_max(deux, un) <= 1.0, _ulp_max(deux, un)
    if not torch.cuda.is_available():
        assert torch.equal(deux, un), "à sec (une seule disposition) : au bit, 8 warps compris"
    for b, n in enumerate(lens):
        if n == 0:
            assert not un[b].float().any()
    cnt = ap._COMPTEURS[(len(lens) * hkv, str(q.device))]
    assert not cnt.any(), cnt


@pytest.mark.a_sec
def test_attention_un_noyau_le_compteur_porte_la_reduction():
    """Le bras qui doit casser : un compteur qui ne repart pas de zéro (la
    faute que l'auto-remise à zéro évite) fait réduire un programme qui n'est
    pas le dernier — la sortie du groupe est fausse ; remis à zéro, elle
    redevient celle du témoin."""
    # Graine fixée : sans elle, q dépend de l'état du générateur laissé par les tests précédents
    # et le bras « faux > 1 ulp » pouvait tomber dans la CI publique selon l'ordre (24/09).
    torch.manual_seed(0)
    ap = _ap()
    c, tables, L = _cache([300, 300])
    hkv, n_rep, d = 2, 8, 128
    C, _ = ap._tranches(tables.shape[1], 2, hkv, tables.device)
    if C < 2:
        pytest.skip("une seule tranche : pas de réduction à fausser")
    dt = torch.bfloat16 if DEV == "cuda" else torch.float16
    q = torch.randn(2, hkv * n_rep, d, device=DEV).to(dt)
    scale = 1 / math.sqrt(d)
    ref = ap.paged_attention(q, c.k, c.k_scale, c.v, c.v_scale, tables, L, hkv, scale, compact=False)
    cnt = ap._compteur(2 * hkv, q.device)
    cnt[1] = 1                                              # groupe (b=0, hkv=1) : compteur faussé
    faux = ap.paged_attention(q, c.k, c.k_scale, c.v, c.v_scale, tables, L, hkv, scale, compact=True)
    # « faux » doit s'écarter : un écart NaN (−inf lu dans un tampon non réduit, selon l'état de
    # l'allocateur laissé par les tests précédents) est un écart, pas une égalité — d'où `not <=`.
    assert not (_ulp_max(ref[0, n_rep:2 * n_rep], faux[0, n_rep:2 * n_rep]) <= 1.0)
    assert _ulp_max(ref[1], faux[1]) <= 1.0                 # les autres groupes, intacts
    cnt.zero_()
    bon = ap.paged_attention(q, c.k, c.k_scale, c.v, c.v_scale, tables, L, hkv, scale, compact=True)
    assert _ulp_max(ref, bon) <= 1.0


def test_lanceur_ne_recopie_q_que_sous_le_temoin(monkeypatch):
    """kernels.paged_attention : sous GLUE_COMPACT=1 la vue passe telle
    quelle ; sous 0 (témoin) une copie contiguë — les nœuds d'avant."""
    from acvram import kernels
    ap = _ap()
    vus = []
    monkeypatch.setattr(ap, "paged_attention", lambda q, *a, **k: vus.append(q) or torch.zeros_like(q))
    monkeypatch.setattr(kernels, "get_extension", lambda: object())
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    c, tables, L = _cache([16, 16])
    q = torch.randn(2, 16 * 128 + 256).to(torch.bfloat16)[:, :16 * 128].view(2, 16, 128)
    monkeypatch.setattr(kernels, "_GLUE_COMPACT", 1)
    kernels.paged_attention(q, c, tables, L, 8, 0.1)
    monkeypatch.setattr(kernels, "_GLUE_COMPACT", 0)
    kernels.paged_attention(q, c, tables, L, 8, 0.1)
    assert vus[0].data_ptr() == q.data_ptr() and not vus[0].is_contiguous()
    assert vus[1].data_ptr() != q.data_ptr() and vus[1].is_contiguous()


def test_kv_write_temoin_recopie_compact_non(monkeypatch):
    """kvcache.write : le témoin donne au noyau deux copies contiguës, le
    compact les tranches telles quelles (le .cu lit `stride(0)`)."""
    from acvram import kernels
    vus = []

    class Ext:
        def kv_write_int8(self, k, v, *a):
            vus.append((k, v))
    monkeypatch.setattr("acvram.kernels.get_extension", lambda: Ext())
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    c = PagedKVCache(KVCacheConfig(num_layers=1, num_kv_heads=2, head_dim=64, num_blocks=2,
                                   dtype="int8", device="cpu"))
    qkv = torch.randn(3, 8 * 64 + 4 * 64).to(torch.bfloat16)
    k = qkv[:, 8 * 64: 10 * 64].view(3, 2, 64)
    v = qkv[:, 10 * 64:].view(3, 2, 64)
    slots = torch.arange(3)
    monkeypatch.setattr(kernels, "_GLUE_COMPACT", 1)
    c.write(slots, k, v)
    monkeypatch.setattr(kernels, "_GLUE_COMPACT", 0)
    c.write(slots, k, v)
    assert vus[0][0].data_ptr() == k.data_ptr() and vus[0][1].data_ptr() == v.data_ptr()
    assert vus[1][0].is_contiguous() and vus[1][1].is_contiguous() and vus[1][0].data_ptr() != k.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="noyau CUDA : carte seulement")
def test_kv_write_int8_par_tranches_au_bit_sur_carte():
    """Sur carte : le cache écrit depuis les tranches (pas de jeton 12·64)
    est identique au bit à celui écrit depuis les copies contiguës, codes et
    échelles ; le bras qui casse : les tranches k et v échangées."""
    from acvram import kernels
    ext = kernels.get_extension()
    if ext is None or not hasattr(ext, "kv_write_int8"):
        pytest.skip("extension absente")
    torch.manual_seed(1)
    T, H, D = 12, 4, 64
    qkv = torch.randn(T, (8 + 2 * H) * D, device="cuda").to(torch.bfloat16)
    k = qkv[:, 8 * D: (8 + H) * D].view(T, H, D)
    v = qkv[:, (8 + H) * D:].view(T, H, D)
    slots = torch.arange(T, device="cuda")

    def ecrit(kk, vv):
        c = PagedKVCache(KVCacheConfig(num_layers=1, num_kv_heads=H, head_dim=D, num_blocks=2,
                                       dtype="int8", device="cuda"))
        ext.kv_write_int8(kk, vv, slots, c.k.view(-1, *c.k.shape[2:]), c.v.view(-1, *c.v.shape[2:]),
                          c.k_scale.view(-1, c.k_scale.shape[-1]), c.v_scale.view(-1, c.v_scale.shape[-1]), 16)
        return c
    a, b = ecrit(k, v), ecrit(k.contiguous(), v.contiguous())
    assert all(torch.equal(getattr(a, n), getattr(b, n)) for n in ("k", "v", "k_scale", "v_scale"))
    faux = ecrit(v, k)
    assert not torch.equal(faux.k, a.k)


@pytest.mark.parametrize("compact", [1, 0])
@pytest.mark.a_sec
def test_c15_3b_le_routeur_triton_remplace_cublas_il_ne_s_ajoute_pas(monkeypatch, compact):
    """verdict-c15-niveau3-coder-19-09 (a) : sous GLUE_COMPACT=1 le nsys B
    montrait `_route_logits_fusee` ET cuBLAS + moe_route + route_prep par
    couche — la branche compacte retombait dans `_route` (model.py, forward).
    Ici : par couche, COMPACT=1 → 1 route_logits_fusee (= 1 F.linear, le même
    que le témoin, + 1 route_fusee à num_warps=1), 0 `_route`, et les experts
    reçoivent ses poids ; COMPACT=0 → 1 F.linear + 1 route_fusee (4 warps),
    0 route_logits_fusee, 0 `_route`. Le routeur ne tourne qu'UNE fois."""
    import torch.nn.functional as F
    from acvram import kernels
    from acvram.engine import model as M
    from acvram.engine import moe as MOE
    from acvram.kernels import route_prep as rp
    if not torch.cuda.is_available():
        os.environ.setdefault("TRITON_INTERPRET", "1")
    if not rp.disponible():
        pytest.skip("Triton indisponible")
    from acvram.engine.layers import QuantLinear
    from acvram.quant.formats import PlainTensor
    E, H = 4, 64

    def plain(n, k):
        t = torch.randn(n, k, dtype=torch.bfloat16) * 0.02
        return QuantLinear(PlainTensor(t, tuple(t.shape), "bf16"), out_features=n, in_features=k)
    bloc = MOE.MoEBlock(plain(E, H), [M.MLP(plain(32, H), plain(32, H), plain(H, 32)) for _ in range(E)],
                      top_k=2, scoring="softmax", score_bias=None)
    bloc._usage_routage = torch.zeros(E, dtype=torch.int64)
    bloc._stack_state = "oui"
    appels = {"linear": 0, "fusee": 0, "logits_fusee": 0, "_route": 0, "grouped": [], "warps": []}
    vrai_linear, vrai_fusee, vrai_lf = F.linear, rp.route_fusee, rp.route_logits_fusee
    monkeypatch.setattr(F, "linear", lambda *a, **k: appels.__setitem__("linear", appels["linear"] + 1) or vrai_linear(*a, **k))
    monkeypatch.setattr(rp, "route_fusee", lambda *a, **k: (appels.__setitem__("fusee", appels["fusee"] + 1),
                                                            appels["warps"].append(k.get("num_warps", 4))) and vrai_fusee(*a, **k))
    monkeypatch.setattr(rp, "route_logits_fusee", lambda *a, **k: appels.__setitem__("logits_fusee", appels["logits_fusee"] + 1) or vrai_lf(*a, **k))
    monkeypatch.setattr(MOE.MoEBlock, "_route", lambda self, x: appels.__setitem__("_route", appels["_route"] + 1) or (None, None))
    monkeypatch.setattr(MOE.MoEBlock, "_forward_grouped_mma", lambda self, x, tw, ti: None)
    monkeypatch.setattr(MOE.MoEBlock, "_forward_grouped", lambda self, x, tw, ti, eid=None: appels["grouped"].append((tw, ti, eid)) or x)
    monkeypatch.setattr(MOE, "_ROUTE_PREP", 2)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(kernels, "_GLUE_COMPACT", compact)
    x = torch.randn(12, H).to(torch.bfloat16)
    y = bloc.forward(x, valid=torch.ones(12, dtype=torch.bool))
    assert y is x and len(appels["grouped"]) == 1 and appels["_route"] == 0
    tw, ti, eid = appels["grouped"][0]
    assert tw.shape == (12, 2) and ti.shape == (12, 2) and eid.shape == (24,) and tw.dtype == torch.float32
    assert (appels["fusee"], appels["linear"]) == (1, 1), appels            # le routeur une seule fois
    assert (appels["logits_fusee"], appels["warps"]) == ((1, [1]) if compact else (0, [4])), appels


def test_c15_3b_items_debranche_chaque_fusion_seule(monkeypatch):
    """ACVRAM_GLUE_COMPACT_ITEMS (bissection de poste2) : sous COMPACT=1, seule
    la fusion nommée est prise, les autres suivent le témoin ; vide = toutes ;
    sous COMPACT=0, rien quoi qu'on nomme ; un nom inconnu est refusé."""
    from acvram import kernels
    monkeypatch.setattr(kernels, "_GLUE_COMPACT", 1)
    monkeypatch.setattr(kernels, "_GLUE_COMPACT_ITEMS", "")
    assert all(kernels.glue_compact(f) for f in kernels.GLUE_COMPACT_FUSIONS) and kernels.glue_compact()
    monkeypatch.setattr(kernels, "_GLUE_COMPACT_ITEMS", "attn")
    assert kernels.glue_compact("attn") and not any(kernels.glue_compact(f) for f in ("routeur", "etroit", "kv"))
    monkeypatch.setattr(kernels, "_GLUE_COMPACT_ITEMS", "routeur,kv")
    assert kernels.glue_compact("routeur") and kernels.glue_compact("kv") and not kernels.glue_compact("etroit")
    monkeypatch.setattr(kernels, "_GLUE_COMPACT", 0)
    assert not any(kernels.glue_compact(f) for f in kernels.GLUE_COMPACT_FUSIONS)
    monkeypatch.setattr(kernels, "_GLUE_COMPACT", 1)
    with pytest.raises(AssertionError):
        kernels.glue_compact("rope")


def test_valid_une_fois_par_pas_est_transmis(monkeypatch):
    """DecoderLayer.decode_fixed_res : `valid` reçu est donné tel quel au
    MoE ; sans lui, la couche calcule `slots >= 0` elle-même (témoin)."""
    from acvram.engine.model import DecoderLayer, MoEBlock
    from acvram.engine.layers import RMSNorm
    vus = []

    class Attn:
        def decode_fixed(self, h, *a, **k):
            return h

    class Moe(MoEBlock):
        def __init__(self):
            torch.nn.Module.__init__(self)

        def forward(self, h, valid=None):
            vus.append(valid)
            return h

    n1, n2 = RMSNorm(torch.ones(8, dtype=torch.bfloat16)), RMSNorm(torch.ones(8, dtype=torch.bfloat16))
    couche = DecoderLayer(0, Attn(), Moe(), n1, n2, torch.device("cpu"))
    x = torch.randn(4, 8).to(torch.bfloat16)
    slots = torch.tensor([0, 1, -1, -1])
    valid = slots >= 0
    couche.decode_fixed_res(x, None, None, slots, None, None, 4, None, valid=valid)
    couche.decode_fixed_res(x, None, None, slots, None, None, 4, None)
    assert vus[0] is valid and vus[1] is not valid and torch.equal(vus[1], valid)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")
@pytest.mark.parametrize("b,ctx", [(12, 320), (12, 576), (12, 1088), (1, 2048), (3, 40)])
def test_reduction_deroulee_au_bit(b, ctx, monkeypatch):
    """Pièce 92 : la réduction des tranches déroulée (lectures en vol ensemble) rend les MÊMES bits que la boucle
    série — mêmes sommes, même ordre. Casse si l'ordre change ou si une tranche au-delà de C est sommée."""
    from acvram.kernels import attn_paginee as ap
    from acvram.memory.kvcache import KVCacheConfig, PagedKVCache, bucket_blocks
    g = torch.Generator(device="cuda").manual_seed(ctx + b)
    nblk = -(-ctx // 16)
    c = PagedKVCache(KVCacheConfig(num_layers=1, num_kv_heads=4, head_dim=128, num_blocks=b * nblk + 1,
                                   dtype="int8", device="cuda"))
    c.k.random_(-127, 128, generator=g); c.v.random_(-127, 128, generator=g)
    c.k_scale.uniform_(0.01, 0.05, generator=g); c.v_scale.uniform_(0.01, 0.05, generator=g)
    t = torch.zeros(b, bucket_blocks(nblk), dtype=torch.long, device="cuda")
    t[:, :nblk] = torch.randperm(b * nblk, device="cuda", generator=g).view(b, nblk) + 1
    lens = torch.tensor([max(1, ctx - 7 * i) for i in range(b)], dtype=torch.long, device="cuda")
    q = torch.randn(b, 32, 128, device="cuda", generator=g).to(torch.bfloat16)
    f = lambda: ap.paged_attention(q, c.k, c.k_scale, c.v, c.v_scale, t, lens, 4, 0.088, 0, compact=True)
    monkeypatch.setattr(ap, "REDUC_DEROULEE", False)
    y0 = f()
    monkeypatch.setattr(ap, "REDUC_DEROULEE", True)
    y1 = f()
    assert torch.equal(y0, y1), (b, ctx, (y0.float() - y1.float()).abs().max().item())
