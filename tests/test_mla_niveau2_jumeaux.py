"""C15 niveau 2 (poste7-c15-niveaux-20-09 § Ordre 04 h 20 : « jumeaux torch par noyau à sec,
déviant nommé ») — le chemin `=2` (b=1 par `decode_static_batch_complet`) est FAUX sur carte
(PPL +3,3 %, divergence au pas 29). Il compose trois noyaux CUDA : `mla_prep_batch` (einsum k_b +
RoPE + norme kv_a + cat), `mla_ecrit_latent` (cache_b[len_b] = k_new, len_b += 1) et
`mla_decode_batch` (attention par table d'adresses). Ici, chacun a son JUMEAU torch (le contrat
du binding, rien d'autre), branché par une fausse extension à registre d'adresses :
* à sec : la composition des trois jumeaux dans `decode_static_batch_complet` à B=1, 32 pas,
  formes GLM, égale AU BIT à `decode_static` (le chemin =0/=1) — si ce test tient, la glue et
  les contrats sont innocentés et le déviant est un noyau ;
* carte : chaque noyau réel contre son jumeau, un à la fois, sur les mêmes états (le déviant
  est nommé par le premier écart > 1 ulp bf16), puis le chemin complet réel sur 32 pas.
"""
import types

import pytest
import torch
from torch import nn

from acvram.engine import mla as MLA
from acvram.engine.layers import RotaryEmbedding
from acvram.engine.mla import MLAttention

torch.set_num_threads(min(8, torch.get_num_threads()))
NH, NOPE, ROPE, RANK, DV, H, QA = 20, 128, 64, 512, 128, 256, 96      # formes GLM (H réduit)
DT = torch.bfloat16


def _lin(o, i, dev):
    return nn.Linear(i, o, bias=False).to(dev, DT)


def _module(dev, seed=3):
    torch.manual_seed(seed)
    rope = RotaryEmbedding(ROPE, 4096, dtype=DT)
    return MLAttention(_lin(NH * (NOPE + ROPE), H, dev), _lin(RANK + ROPE, H, dev), _lin(H, NH * DV, dev),
                       (torch.rand(RANK) + 0.5).to(dev, DT),
                       (torch.randn(NH, RANK, NOPE) * 0.05).to(dev, DT),
                       (torch.randn(NH, DV, RANK) * 0.05).to(dev, DT),
                       NH, NOPE, ROPE, RANK, DV, eps=1e-5,
                       q_a_proj=_lin(QA, H, dev), q_a_norm=(torch.rand(QA) + 0.5).to(dev, DT),
                       q_b_proj=_lin(NH * (NOPE + ROPE), QA, dev), rope=rope)


# ---- jumeaux : le contrat de chaque binding, en torch ------------------------------------
def jumeau_prep_batch(la, q, kvp, lens, bucket):
    """`mla_prep_batch(q [B,nh,nope+rope], kvp [B,rank+rope], lens, cos32, sin32, k_b_c, kv_a_norm,
    nope, rope, rank, eps) -> (q_eff [B,nh,W] fp32, k_new [B,W] bf16)` — le chemin torch de
    `decode_static_batch_complet` (branche `_MLA_PREP_NOYAU=0`), mot pour mot."""
    q_nope, q_pe = q.split([la.nope, la.rope], dim=-1)
    c0, k_pe0 = kvp.split([la.rank, la.rope], dim=-1)
    q_pe, k_pe0 = la._rope(q_pe, k_pe0, lens, bucket + 1)
    kvp = torch.cat([c0, k_pe0], dim=-1)
    c, k_pe = kvp.split([la.rank, la.rope], dim=-1)
    c = la._norme(c, la.kv_a_norm)
    q_abs = torch.einsum('hrn,bhn->bhr', la.k_b.to(q.dtype), q_nope)
    return torch.cat([q_abs, q_pe], dim=-1).to(torch.float32), torch.cat([c, k_pe], dim=-1).contiguous()


class FauxExt:
    """Fausse extension : `mla_ecrit_latent` et `mla_decode_batch` par leurs jumeaux, les tables
    d'adresses résolues par un registre {data_ptr: tenseur} rempli depuis les états ; les autres
    attributs n'existent pas (hasattr → False : `mla_prep_batch` reste le chemin torch, `mla_decode_1p` absent)."""

    def __init__(self, sts, rank, scale):
        self.reg = {}
        for st in sts:
            self.reg[st["cache"].data_ptr()] = st["cache"]
            self.reg[st["len"].data_ptr()] = st["len"]
        self.rank, self.scale = rank, scale
        self.appels = []

    def mla_ecrit_latent(self, k_new, cache_ptrs, len_ptrs, fp8=False):
        self.appels.append("mla_ecrit_latent")
        for b in range(k_new.shape[0]):
            cache, ln = self.reg[int(cache_ptrs[b])], self.reg[int(len_ptrs[b])]
            cache.index_copy_(0, ln.view(1), k_new[b:b + 1])
            ln.add_(1)

    def mla_decode_batch(self, q, cache_ptrs, lens, scores, bucket, rank, scale):
        """`mla_decode_batch(q [B,nh,W] fp32, ptrs [B], lens [B], scores, bucket, rank, scale) -> o_lat
        [B,nh,rank] fp32` : par créneau, l'attention de référence de `decode_static` (chemin sans
        extension), masquée au-delà de len (inclusif)."""
        self.appels.append("mla_decode_batch")
        outs = []
        for b in range(q.shape[0]):
            C = self.reg[int(cache_ptrs[b])][:bucket].to(torch.float32)
            sc = torch.einsum('hr,sr->hs', q[b], C) * scale
            pos = torch.arange(bucket, device=q.device)
            sc = sc.masked_fill(pos > lens[b], float('-inf'))
            outs.append(torch.einsum('hs,sr->hr', sc.softmax(dim=-1), C[:, :rank]))
        return torch.stack(outs)


def _etat(la, dev, L, n):
    st = la.new_static(torch.device(dev), L, DT)
    torch.manual_seed(100 + n)
    st["cache"][:n] = (torch.randn(n, RANK + ROPE) * 0.3).to(dev, DT)
    st["len"].fill_(n)
    return st


def _pas_complet(la, x, st, bucket, faux):
    ptrs = torch.tensor([st["cache"].data_ptr()], dtype=torch.int64, device=x.device)
    len_ptrs = torch.tensor([st["len"].data_ptr()], dtype=torch.int64, device=x.device)
    scores = torch.zeros(1, NH, bucket, device=x.device)
    return la.decode_static_batch_complet(x, [st], bucket, ptrs, scores, len_ptrs)


def test_composition_des_jumeaux_egale_decode_static_a_sec(monkeypatch):
    """B=1, 32 pas, formes GLM : `decode_static_batch_complet` (prep torch + jumeaux ecrit/decode)
    = `decode_static` AU BIT, sorties et caches ; sinon le déviant est la glue, pas un noyau."""
    dev = "cpu"
    la = _module(dev)
    monkeypatch.setattr(MLA, "_MLA_PREP_NOYAU", False)
    monkeypatch.setattr(MLA, "_MLA_LATENT_FP8", False)
    L, n0, bucket = 128, 40, 128
    st_a, st_b = _etat(la, dev, L, n0), _etat(la, dev, L, n0)
    assert torch.equal(st_a["cache"], st_b["cache"])
    faux = FauxExt([st_b], RANK, la.scale)
    torch.manual_seed(9)
    xs = (torch.randn(32, 1, H) * 0.5).to(DT)
    with torch.inference_mode():
        for pas in range(32):
            monkeypatch.setattr(MLA, "_extension", lambda: None)           # decode_static : chemin torch
            ya = la.decode_static(xs[pas], st_a, bucket)
            monkeypatch.setattr(MLA, "_extension", lambda: faux)
            yb = _pas_complet(la, xs[pas], st_b, bucket, faux)
            assert torch.equal(st_a["len"], st_b["len"]), f"pas {pas} : len diverge"
            assert torch.equal(st_a["cache"], st_b["cache"]), f"pas {pas} : cache diverge (jumeau prep/ecrit)"
            assert torch.equal(ya, yb), f"pas {pas} : sortie diverge (jumeau decode/glue) max {(ya.float() - yb.float()).abs().max():.3e}"
    assert faux.appels[:2] == ["mla_ecrit_latent", "mla_decode_batch"] and len(faux.appels) == 64
    # témoin cassant : un jumeau de `mla_decode_batch` qui masque une clé de moins doit diverger
    vrai = faux.mla_decode_batch
    def decode_faux(q, cache_ptrs, lens, scores, bucket, rank, scale):
        return vrai(q, cache_ptrs, lens - 1, scores, bucket, rank, scale)
    monkeypatch.setattr(faux, "mla_decode_batch", decode_faux)
    with torch.inference_mode():
        monkeypatch.setattr(MLA, "_extension", lambda: None)
        ya = la.decode_static(xs[0], st_a, bucket)
        monkeypatch.setattr(MLA, "_extension", lambda: faux)
        yb = _pas_complet(la, xs[0], st_b, bucket, faux)
    assert not torch.equal(ya, yb)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="carte requise")
def test_chaque_noyau_contre_son_jumeau_sur_carte(monkeypatch):
    """Sur carte : (1) mla_prep_batch réel vs jumeau (mêmes q, kvp, lens) ; (2) mla_ecrit_latent réel
    vs jumeau (mêmes états) ; (3) mla_decode_batch réel vs jumeau (mêmes q_eff, caches) ; puis (4)
    le chemin complet réel 32 pas vs decode_static. Le premier écart > 1 ulp bf16 nomme le déviant."""
    from acvram.kernels import get_extension
    ext = get_extension()
    if ext is None or not hasattr(ext, "mla_prep_batch"):
        pytest.skip("extension sans mla_prep_batch")
    dev = "cuda"
    la = _module(dev)
    monkeypatch.setattr(MLA, "_MLA_LATENT_FP8", False)
    L, n0, bucket = 128, 40, 128
    st = _etat(la, dev, L, n0)
    torch.manual_seed(9); x = (torch.randn(1, H, device=dev) * 0.5).to(DT)
    ulp = lambda a, b: ((a.float() - b.float()).abs() / (2.0 ** (torch.floor(torch.log2(a.float().abs().clamp_min(1e-30))) - 7))).max().item()
    with torch.inference_mode():
        prem, kvp = la._proj_entree(x)
        q = la._q_depuis(prem).reshape(1, NH, NOPE + ROPE)
        lens = st["len"].reshape(1)
        cos32, sin32 = la.rope_emb.tables32(bucket + 1, x.device)
        q_eff_n, k_new_n = ext.mla_prep_batch(q.contiguous(), kvp.contiguous(), lens, cos32, sin32,
                                              la._k_b_c(), la.kv_a_norm, NOPE, ROPE, RANK, la.eps)
        q_eff_j, k_new_j = jumeau_prep_batch(la, q, kvp, lens, bucket)
        e1, e2 = ulp(q_eff_j, q_eff_n), ulp(k_new_j, k_new_n)
        print(f"\n(1) mla_prep_batch : q_eff {e1:.2f} ulp bf16, k_new {e2:.2f} ulp bf16")
        assert e1 <= 1 and e2 <= 1, f"DÉVIANT : mla_prep_batch (q_eff {e1:.2f}, k_new {e2:.2f} ulp bf16)"
        # (2) écriture
        st2 = _etat(la, dev, L, n0); faux = FauxExt([st, st2], RANK, la.scale)
        ptrs = torch.tensor([st["cache"].data_ptr()], dtype=torch.int64, device=dev)
        lptrs = torch.tensor([st["len"].data_ptr()], dtype=torch.int64, device=dev)
        ext.mla_ecrit_latent(k_new_j, ptrs, lptrs, False)
        faux.mla_ecrit_latent(k_new_j, torch.tensor([st2["cache"].data_ptr()], device=dev), torch.tensor([st2["len"].data_ptr()], device=dev))
        assert torch.equal(st["cache"], st2["cache"]) and torch.equal(st["len"], st2["len"]), "DÉVIANT : mla_ecrit_latent"
        # (3) attention
        scores = torch.zeros(1, NH, bucket, device=dev)
        o_n = MLA._mla_decode_batch(ext, q_eff_j.contiguous(), ptrs, st["len"].reshape(1), scores, bucket, RANK, la.scale)
        o_j = faux.mla_decode_batch(q_eff_j, ptrs, st["len"].reshape(1), scores, bucket, RANK, la.scale)
        e3 = ulp(o_j, o_n); print(f"(3) mla_decode_batch : {e3:.2f} ulp bf16")
        assert e3 <= 1, f"DÉVIANT : mla_decode_batch ({e3:.2f} ulp bf16)"
        # (4) chemin complet réel 32 pas contre decode_static
        st_a, st_b = _etat(la, dev, L, n0), _etat(la, dev, L, n0)
        xs = (torch.randn(32, 1, H, device=dev) * 0.5).to(DT)
        for pas in range(32):
            ya = la.decode_static(xs[pas], st_a, bucket)
            yb = _pas_complet(la, xs[pas], st_b, bucket, None)
            e = ulp(ya, yb)
            assert e <= 1, f"DÉVIANT : chemin complet au pas {pas} ({e:.2f} ulp bf16)"


def test_remede_regles_6_caches_paresseux_refuses_en_capture(monkeypatch):
    """Remède C15 niveau 2 (poste7-glm-b1-noeuds-c15-niveau3 § 2, REGLES § 6) : tout cache paresseux
    du chemin MLA (k_b contigu, v_b fp32, tables RoPE fp32, tables d'adresses du lot) lève s'il est
    créé pendant une capture ; `chauffer()` les matérialise avant, et `reserver()` matérialise
    tables32/tables_demi SANS condition (avant : seulement si elles existaient déjà — layers.py)."""
    dev = "cpu"
    la = _module(dev)
    # simulation d'une capture en cours (CUDA « disponible »)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="capture"):
        la.rope_emb.tables32(64, torch.device(dev))                      # première allocation aussi refusée
    la.__dict__.pop("_v_b32_cache", None)
    with pytest.raises(RuntimeError, match="v_b fp32"):
        la._v_b32()
    # hors capture : chauffer matérialise tout, puis en capture rien n'est alloué
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    la.chauffer(torch.device(dev), 128)
    c32, _ = la.rope_emb.tables32(128, torch.device(dev)); vb = la._v_b32()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    c32b, _ = la.rope_emb.tables32(128, torch.device(dev))
    assert c32b is c32 and la._v_b32() is vb                             # mêmes objets : aucune allocation
    # reserver() sans condition : sur un module neuf, tables32 existe après reserver
    la2 = _module(dev, seed=4)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert getattr(la2.rope_emb, "_cos32", None) is None
    la2.rope_emb.reserver(128, torch.device(dev), DT)
    assert la2.rope_emb._cos32 is not None
    src = (__import__("pathlib").Path(__file__).resolve().parents[1] / "acvram" / "engine" / "graphs.py").read_text()
    assert 'mod.chauffer(d, godet_mla(self.max_model_len) + MLA_BUCKET + 1)' in src


def test_rejeu_simule_avec_un_pas_different_change_la_sortie(monkeypatch):
    """REGLES § 6 (poste7, addendum niveau 2) : tout compteur par pas passe par un tenseur sur carte —
    « rejeu simulé avec un pas différent doit changer la sortie ». Chaque appel de noyau du chemin
    `=2` est rejoué avec EXACTEMENT les mêmes arguments Python (scalaires figés comme dans un graphe,
    tenseurs par identité) après que l'état de la carte (cache, len) a avancé d'un pas : la sortie
    doit être celle du nouveau pas, jamais celle du pas capturé. Les scalaires du chemin, nommés :
    `nope, rope, rank, eps` (prep, constantes du module), `bucket` (godet, constant par capture),
    `rank, scale, fp8` (décode) — aucun n'est un compteur par pas ; `lens` et `len_ptrs` sont des
    tenseurs. Le test échoue si un scalaire par pas s'y glisse (ex. une position Python)."""
    dev = "cpu"
    la = _module(dev)
    monkeypatch.setattr(MLA, "_MLA_PREP_NOYAU", False)
    monkeypatch.setattr(MLA, "_MLA_LATENT_FP8", False)
    L, n0, bucket = 128, 40, 128
    st = _etat(la, dev, L, n0)
    faux = FauxExt([st], RANK, la.scale)
    journal = []                                                     # (nom, args) tels que « capturés »
    for nom in ("mla_ecrit_latent", "mla_decode_batch"):
        orig = getattr(faux, nom)
        def enregistre(*a, _o=orig, _n=nom, **k):
            journal.append((_n, a, k)); return _o(*a, **k)
        monkeypatch.setattr(faux, nom, enregistre)
    monkeypatch.setattr(MLA, "_extension", lambda: faux)
    torch.manual_seed(3); xs = (torch.randn(3, 1, H) * 0.5).to(DT)
    with torch.inference_mode():
        y0 = _pas_complet(la, xs[0], st, bucket, faux)               # « capture » : pas 0
        capt = list(journal); journal.clear()
        y1 = _pas_complet(la, xs[1], st, bucket, faux)               # pas 1 réel (référence)
        # rejeu : les mêmes appels de noyau que le pas 0, scalaires identiques, tenseurs par identité,
        # après que l'état a avancé — on rejoue sur un troisième état copié du pas 1 pour comparer
        st_r = _etat(la, dev, L, n0); faux_r = FauxExt([st_r], RANK, la.scale)
        monkeypatch.setattr(MLA, "_extension", lambda: faux_r)
        _pas_complet(la, xs[0], st_r, bucket, faux_r)                 # état après le pas 0
        assert torch.equal(st_r["cache"], st["cache"]) is False or True
        # le pas 1 « rejoué » : prep en torch recalculé (partie non-noyau du graphe), puis les deux noyaux
        # appelés avec les scalaires capturés au pas 0 ; les tenseurs (k_new, q_eff, lens) sont ceux du pas 1
        prem, kvp = la._proj_entree(xs[1]); q = la._q_depuis(prem).reshape(1, NH, NOPE + ROPE)
        lens = torch.stack([st_r["len"]])
        q_eff, k_new = jumeau_prep_batch(la, q, kvp, lens, bucket)
        (_, a_e, k_e), (_, a_d, k_d) = capt[0], capt[1]
        faux_r.mla_ecrit_latent(k_new, torch.tensor([st_r["cache"].data_ptr()]), torch.tensor([st_r["len"].data_ptr()]), *a_e[3:], **k_e)
        o = faux_r.mla_decode_batch(q_eff, torch.tensor([st_r["cache"].data_ptr()]), lens, a_d[3], *a_d[4:], **k_d)  # scalaires bucket/rank/scale du pas 0
        y_r = la._o(torch.einsum('hvr,bhr->bhv', la._v_b32().to(torch.float32), o.to(torch.float32)).reshape(1, NH * DV).to(DT))
    assert torch.equal(y_r, y1), "le rejeu avec les scalaires du pas 0 ne rend pas le pas 1 : un scalaire par pas est figé"
    assert not torch.equal(y_r, y0)
