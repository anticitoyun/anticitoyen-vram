"""C14-b (revue/chantier-c14b-19-09, sage-fiches-c5b-c13c-c14b-20-09 § 3) : au décodage par lot de
GLM (b=12), (1) v_b·o_lat dans le combine de `mla_decode_1p` (`mla_1p_combine_vb_kernel`, sortie
bf16 [B, nh, dv]) sous ACVRAM_MLA_BATCH_FUSION=1, à la place de l'einsum fp32 'hvr,bhr->bhv' que
cuBLAS sert à M=12 par gemmSN_TN. (Le geste (2), `mla_prep_batch` regrillé au bit, est jugé dans
tests/test_mla_prep_regrille_c14b.py.)

À sec : la glue (le chemin de lot passe v_b au noyau seulement quand tout s'y prête, et retombe
sur l'einsum sinon), la variable de régime nommée sur la ligne, la source CUDA. Sur carte : le combine
fusionné contre l'einsum (≤ 1 ulp bf16 de la sortie ; juge fp64 par ligne RELATIF : d(noyau) ≤
d(einsum) + 6 ulp bf16 — le fp32 n'est jamais matérialisé par le noyau) sur les formes réelles (b=12, nh=20,
rank=512, dv=128, S=8 → SL=1 ; b=3, L=4 096 → SL=4).
"""
import math
import os
import re

import pytest
import torch

from acvram import regime
from acvram.engine import mla as MLA

NH, NOPE, ROPE, RANK, DV = 20, 128, 64, 512, 128
W = RANK + ROPE
DT = torch.bfloat16
CU = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "acvram", "kernels", "acvram_kernels.cu")


def _ulp_bf16(a, b):
    """Écart max en ulp bf16 entre deux tenseurs (l'ulp lu sur a)."""
    a32, b32 = a.float(), b.float()
    e = torch.floor(torch.log2(a32.abs().clamp_min(1e-30))) - 7
    return ((a32 - b32).abs() / (2.0 ** e)).max().item()


# ---- à sec : la glue --------------------------------------------------------------------------
class _Ext:
    """Fausse extension : `mla_decode_1p` rend o_lat fp32 sans v_b, y bf16 = v_b·o_lat avec, et
    note ce qu'on lui a passé ; `mla_ecrit_latent` avance les longueurs (registre d'adresses)."""

    def __init__(self, sts):
        self.reg = {st["len"].data_ptr(): st["len"] for st in sts}
        self.reg.update({st["cache"].data_ptr(): st["cache"] for st in sts})
        self.appels = []

    def mla_ecrit_latent(self, k_new, cache_ptrs, len_ptrs, fp8=False):
        for b in range(k_new.shape[0]):
            cache, ln = self.reg[int(cache_ptrs[b])], self.reg[int(len_ptrs[b])]
            cache[int(ln)] = k_new[b]
            ln.add_(1)

    def mla_decode_1p(self, q, cache_ptrs, cache, lens, L, rank, scale, fp8=False, v_b=None):
        self.appels.append(("mla_decode_1p", v_b is not None))
        B, H = q.shape[0], q.shape[1]
        torch.manual_seed(int(lens[0]) + 11)
        o_lat = torch.randn(B, H, rank) * 0.1                   # même o_lat pour un même pas
        if v_b is None:
            return o_lat
        return torch.einsum('hvr,bhr->bhv', v_b.float(), o_lat).to(DT)


def _module():
    import importlib.util, pathlib
    _spec = importlib.util.spec_from_file_location("test_mla_niveau2_jumeaux", pathlib.Path(__file__).with_name("test_mla_niveau2_jumeaux.py"))
    _m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m); _module = _m._module
    return _module("cpu")


def _etats(la, B, n=40, L=128):
    sts = []
    for i in range(B):
        st = la.new_static(torch.device("cpu"), L, DT)
        torch.manual_seed(100 + i)
        st["cache"][:n] = (torch.randn(n, W) * 0.3).to(DT)
        st["len"].fill_(n)
        sts.append(st)
    return sts


def _pas(la, x, sts, ext, bucket=128):
    ptrs = torch.tensor([st["cache"].data_ptr() for st in sts], dtype=torch.int64)
    lptrs = torch.tensor([st["len"].data_ptr() for st in sts], dtype=torch.int64)
    scores = torch.zeros(len(sts), NH, bucket)
    return la.decode_static_batch_complet(x, sts, bucket, ptrs, scores, lptrs)


@pytest.mark.parametrize("fusion", [False, True])
def test_glue_passe_v_b_au_noyau_seulement_sous_fusion(monkeypatch, fusion):
    """B=3 à sec (prep torch, attention par la fausse extension) : sous FUSION=1 le chemin de lot
    appelle `mla_decode_1p` AVEC v_b et rend o_proj(y) ; sous 0 il l'appelle SANS et fait l'einsum.
    Les deux sorties sont égales au bit ici (la fausse extension fait le même einsum) : le test
    juge la GLUE (quel noyau, quels arguments, longueurs avancées), pas l'arithmétique du noyau."""
    la = _module()
    monkeypatch.setattr(MLA, "_MLA_PREP_NOYAU", False)
    monkeypatch.setattr(MLA, "_MLA_UNE_PASSE", True)
    monkeypatch.setattr(MLA, "_MLA_BATCH_FUSION", fusion)
    monkeypatch.setattr(MLA, "_MLA_CORE_DECODE", "fp32")
    sts_a, sts_b = _etats(la, 3), _etats(la, 3)
    ext_a, ext_b = _Ext(sts_a), _Ext(sts_b)
    torch.manual_seed(5)
    x = (torch.randn(3, 256) * 0.5).to(DT)
    with torch.inference_mode():
        monkeypatch.setattr(MLA, "_extension", lambda: ext_a)
        ya = _pas(la, x, sts_a, ext_a)
        monkeypatch.setattr(MLA, "_MLA_BATCH_FUSION", False)
        monkeypatch.setattr(MLA, "_extension", lambda: ext_b)
        yb = _pas(la, x, sts_b, ext_b)
    assert ext_a.appels == [("mla_decode_1p", fusion)], ext_a.appels
    assert ext_b.appels == [("mla_decode_1p", False)]
    assert ya.dtype == DT and ya.shape == (3, 256)
    assert all(int(st["len"]) == 41 for st in sts_a + sts_b), "longueurs non avancées"
    assert torch.equal(ya, yb)


def test_fusion_refusee_hors_regime(monkeypatch):
    """`_fusion_vb_possible` rend faux dès qu'une condition manque : variable à 0, noyau à une passe
    absent ou non pris, v_b non bf16, sortie non bf16, MLA_CORE_DECODE ≠ fp32 — et vrai sinon."""
    ext = _Ext([])
    vb = torch.zeros(NH, DV, RANK, dtype=DT)
    x = torch.zeros(2, 256, dtype=DT)
    monkeypatch.setattr(MLA, "_MLA_BATCH_FUSION", True)
    monkeypatch.setattr(MLA, "_MLA_UNE_PASSE", True)
    monkeypatch.setattr(MLA, "_MLA_CORE_DECODE", "fp32")
    assert MLA._fusion_vb_possible(ext, x, vb, False)
    monkeypatch.setattr(MLA, "_MLA_BATCH_FUSION", False)
    assert not MLA._fusion_vb_possible(ext, x, vb, False)
    monkeypatch.setattr(MLA, "_MLA_BATCH_FUSION", True)
    monkeypatch.setattr(MLA, "_MLA_UNE_PASSE", False)
    assert not MLA._fusion_vb_possible(ext, x, vb, False)
    assert MLA._fusion_vb_possible(ext, x, vb, True)          # fp8 : le noyau 1p est le chemin
    monkeypatch.setattr(MLA, "_MLA_UNE_PASSE", True)
    assert not MLA._fusion_vb_possible(ext, x, vb.float(), False)
    assert not MLA._fusion_vb_possible(ext, x.float(), vb, False)
    assert not MLA._fusion_vb_possible(None, x, vb, False)
    assert not MLA._fusion_vb_possible(object(), x, vb, False)
    monkeypatch.setattr(MLA, "_MLA_CORE_DECODE", "tf32")
    assert not MLA._fusion_vb_possible(ext, x, vb, False)


def test_variable_de_regime_nommee_sur_la_ligne(monkeypatch):
    """ACVRAM_MLA_BATCH_FUSION : défaut 0, lue à l'import dans mla._MLA_BATCH_FUSION, hors défaut
    (donc sur la ligne de régime) quand le module la porte à 1."""
    v = {x.nom: x for x in regime.VARIABLES}["MLA_BATCH_FUSION"]
    assert v.defaut == "0" and v.lu_a == ("acvram.engine.mla", "_MLA_BATCH_FUSION") and v.torch == "0"
    assert "chantier-c14b" in v.note
    monkeypatch.setattr(MLA, "_MLA_BATCH_FUSION", False)
    assert "ACVRAM_MLA_BATCH_FUSION" not in regime.regime_noyaux()["hors_defaut"]
    monkeypatch.setattr(MLA, "_MLA_BATCH_FUSION", True)
    assert regime.regime_noyaux()["hors_defaut"]["ACVRAM_MLA_BATCH_FUSION"] == "1"
    assert "ACVRAM_MLA_BATCH_FUSION=1" in regime.regime_ligne()


def test_source_cu_porte_le_combine_fusionne():
    """La source CUDA : `mla_decode_1p` prend v_b (py::none() par défaut) et lance
    `mla_1p_combine_vb_kernel` sous `fusion`, un bloc par (b, h), les quatre SL ; le noyau fusionné
    convertit v_b bf16 en fp32, combine toutes les colonnes dans le bloc et arrondit une fois en bf16."""
    src = open(CU, encoding="utf-8").read()
    assert 'py::arg("fp8") = false, py::arg("v_b") = py::none()' in src
    lanceur = src.split("torch::Tensor mla_decode_1p(")[1].split("\n}\n")[0]
    assert "const bool fusion = v_b.has_value() && v_b->defined();" in lanceur
    assert "mla_1p_combine_vb_kernel<1><<<B * H, MLA1P_CMB_FILS, 0, stream>>>" in lanceur
    assert lanceur.count("mla_1p_combine_vb_kernel<") == 4          # SL = 8, 4, 2, 1
    noyau = src.split("mla_1p_combine_vb_kernel(")[1].split("\n}\n")[0]
    assert "__bfloat1622float2(p2[" in noyau and "__float2bfloat16(p)" in noyau
    assert "for (int c0 = 0; c0 < R; c0 += CH)" in noyau             # toutes les colonnes dans le bloc


# ---- à sec : tranche vide et annulation (verdict Manon 09 h 00 : 37 lignes / 191 760 à > 6 ulp de |y64|) ----------
def _combine_jumeau(ws, garde=True):
    """Arithmétique du combine (témoin .cu:6040 = fusionné .cu:6125, colonne par colonne) : ws [B, S, H, R+2]
    (o_s, m_s, l_s par tranche) → o [B, H, R] ; `garde=False` = le témoin cassant qui oublie w = 0 sur une
    tranche vide (m = −inf : exp(−inf − M) vaut 0 mais 0·o_s avec o_s = 0 est 0… sauf que M = −inf si TOUTES les
    tranches sont vides ; ici la faute cassante est l'absence du test l > 0 : inv = 1/0)."""
    o_s, m_s, l_s = ws[..., :-2], ws[..., -2], ws[..., -1]                    # [B,S,H,R], [B,S,H], [B,S,H]
    M = m_s.amax(dim=1, keepdim=True)                                         # [B,1,H]
    w = torch.where(m_s != float("-inf"), torch.exp(m_s - M), torch.zeros_like(m_s)) if garde else torch.exp(m_s - M)
    Lsum = (l_s * w).sum(dim=1)                                               # [B,H]
    inv = torch.where(Lsum > 0, 1.0 / Lsum, torch.zeros_like(Lsum)) if garde else 1.0 / Lsum
    return (o_s * w[..., None]).sum(dim=1) * inv[..., None]


def _ws_avec_tranches_vides(B=3, S=4, H=2, R=8, seed=0):
    torch.manual_seed(seed)
    ws = torch.randn(B, S, H, R + 2)
    ws[..., -2] = torch.rand(B, S, H) * 2 - 1                                  # m_s
    ws[..., -1] = torch.rand(B, S, H) + 0.5                                    # l_s
    vides = [(1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]                          # (b, s) : b=1 court, b=2 très court
    for b, s_ in vides:
        ws[b, s_, :, :-2] = 0; ws[b, s_, :, -2] = float("-inf"); ws[b, s_, :, -1] = 0
    return ws, vides


def test_tranche_vide_combinee_a_poids_nul_a_sec():
    """La prédiction de Sage (09 h 00) : les 37 lignes viendraient d'une tranche VIDE mal combinée. Par lecture,
    témoin et fusionné écrivent la même garde (`w = m != -INFINITY ? exp(m − M) : 0` .cu:6068 / :6154, `inv = Lsum
    > 0 ? 1/Lsum : 0` :6079 / :6165) sur les tranches que mla_1p_kernel marque vides (m = −inf, l = 0, o = 0,
    .cu:5867). Ici : le jumeau de cette arithmétique sur des (b, h) à tranches vides rend un o fini, égal à la
    combinaison des seules tranches pleines ; le témoin cassant sans garde rend NaN — la garde est bien ce qui
    compte, et elle est dans les deux noyaux. Le test carte (lens inégaux) exerce les vrais noyaux."""
    ws, vides = _ws_avec_tranches_vides()
    o = _combine_jumeau(ws)
    assert torch.isfinite(o).all()
    # même résultat que la combinaison restreinte aux tranches pleines
    for b in range(ws.shape[0]):
        pleines = [s_ for s_ in range(ws.shape[1]) if (b, s_) not in vides]
        o_pleines = _combine_jumeau(ws[b:b + 1, pleines])
        assert torch.allclose(o[b:b + 1], o_pleines, rtol=1e-6, atol=1e-7), b
    # témoin cassant : sans la garde, une tranche vide fait exp(−inf − M) = 0 (inoffensif) mais un b dont TOUTES les
    # tranches sauf une sont vides garde Lsum > 0… la faute qui casse vraiment : b entièrement vide → 1/0
    ws2 = ws.clone(); ws2[2, :, :, -2] = float("-inf"); ws2[2, :, :, -1] = 0; ws2[2, :, :, :-2] = 0
    assert torch.isfinite(_combine_jumeau(ws2)).all()
    assert not torch.isfinite(_combine_jumeau(ws2, garde=False)).all()


def test_annulation_fait_des_milliers_d_ulp_de_y64_sans_faute():
    """Ce que valent « 2 296 ulp bf16 de |y64| » : sur une ligne où Σ_r v·o s'annule (|y64| ≪ Σ|v·o|), deux ordres
    de somme fp32 EXACTS À LEUR BORNE diffèrent de milliers d'ulp de |y64| — l'ulp du résultat n'est pas une
    échelle là où le résultat s'annule (leçon « un rapport ne classe pas deux objets »). Ici : 512 produits
    fp32, deux ordres (séquentiel = l'einsum ; par 8 voies puis arbre = le noyau), ligne d'annulation forcée :
    d_k et d_e en ulp bf16 de |y64| dépassent 1 000 et se classent au hasard, alors que le juge à l'échelle des
    termes (1 ulp bf16 + 32·eps32·Σ|termes|) est tenu par les deux. Le juge carte est donc celui des termes."""
    torch.manual_seed(1)
    n, essais = 512, 200
    pires = []
    for _ in range(essais):
        v = torch.randn(n, dtype=torch.float32) * 0.05
        o = torch.randn(n, dtype=torch.float32) * 30
        # annulation : le dernier terme compense la somme des autres à 1e-6 près
        o[-1] = -(v[:-1].double() * o[:-1].double()).sum().item() / v[-1].item() * (1 + 1e-6)
        prod64 = v.double() * o.double(); y64 = prod64.sum()
        y_seq = torch.zeros((), dtype=torch.float32)
        for k in range(n): y_seq = y_seq + v[k] * o[k]                         # ordre séquentiel (cuBLAS SIMT)
        voies = [torch.zeros((), dtype=torch.float32) for _ in range(32)]
        for k in range(n): voies[(k // 8) % 32] = voies[(k // 8) % 32] + v[k] * o[k]   # 32 voies de 8 (le noyau)
        while len(voies) > 1: voies = [voies[i] + voies[i + 1] for i in range(0, len(voies), 2)]
        y_noy = voies[0]
        u = 2.0 ** (math.floor(math.log2(abs(y64.item()))) - 7)
        d_k, d_e = abs(y_noy.to(DT).double() - y64).item() / u, abs(y_seq.to(DT).double() - y64).item() / u
        borne = u + 32 * 2.0 ** -23 * prod64.abs().sum().item()
        assert abs(y_noy.to(DT).double() - y64).item() <= borne and abs(y_seq.to(DT).double() - y64).item() <= borne
        pires.append((d_k, d_e))
    plus_de_6 = sum(1 for k, e in pires if k > e + 6); inverse = sum(1 for k, e in pires if e > k + 6)
    print(f"\n{essais} lignes d'annulation : max d_k {max(k for k, _ in pires):.0f}, max d_e {max(e for _, e in pires):.0f} ulp bf16 de |y64| ; "
          f"d_k > d_e + 6 : {plus_de_6}, d_e > d_k + 6 : {inverse} — juge des termes tenu partout")
    assert max(max(k, e) for k, e in pires) > 1000 and plus_de_6 > 0 and inverse > 0


def _juge_termes_jumeau(y, y_64, vb, o_lat):
    """Le juge des termes sur un y quelconque [B, nh, dv] bf16 : (lignes fautives, ratio max)."""
    somme_abs = torch.einsum('hvr,bhr->bhv', vb.double().abs(), o_lat.double().abs())
    borne = 2.0 ** (torch.floor(torch.log2(y_64.abs().clamp_min(1e-30))) - 7) + 32 * 2.0 ** -23 * somme_abs
    r = ((y.double() - y_64).abs() / borne).amax(dim=-1)
    return (r > 1).sum().item(), r.max().item(), r


def test_le_juge_des_termes_casse_sur_une_faute_construite_a_sec():
    """Sage 09 h 25 : un juge qui ne casse que sur NaN n'est pas un contrôle. Jumeau torch (fp32, ordre
    de l'einsum) : v_b avec UNE tête permutée (h ↔ h+1) ou UNE colonne de rank permutée (r ↔ r+1) →
    les lignes touchées passent à ratio ≫ 1, verdict FAUX ; le jumeau sain reste ≤ 1 sur toutes les
    lignes (et son ratio max est publié : ici ≪ 0,1)."""
    torch.manual_seed(4)
    B, nh, dv, rank = 4, 6, 32, 128
    vb = (torch.randn(nh, dv, rank) * 0.05).to(DT)
    o = torch.randn(B, nh, rank) * 3
    y64 = torch.einsum('hvr,bhr->bhv', vb.double(), o.double())
    y_sain = torch.einsum('hvr,bhr->bhv', vb.float(), o).to(DT)
    f0, r0, _ = _juge_termes_jumeau(y_sain, y64, vb, o)
    assert f0 == 0 and r0 <= 1, (f0, r0)
    # faute 1 : une tête permutée dans v_b → les lignes (b, 0) et (b, 1) fautives, les autres non
    vb_h = vb.clone(); vb_h[[0, 1]] = vb[[1, 0]]
    y_h = torch.einsum('hvr,bhr->bhv', vb_h.float(), o).to(DT)
    f1, r1, rr = _juge_termes_jumeau(y_h, y64, vb, o)
    assert f1 == 2 * B and r1 > 10 and bool((rr[:, 2:] <= 1).all()), (f1, r1)
    # faute 2 : une colonne de rank permutée (r ↔ r+1) → toutes les lignes fautives, ratio > 1
    vb_r = vb.clone(); vb_r[..., [5, 6]] = vb[..., [6, 5]]
    y_r = torch.einsum('hvr,bhr->bhv', vb_r.float(), o).to(DT)
    f2, r2, _ = _juge_termes_jumeau(y_r, y64, vb, o)
    assert f2 == B * nh and r2 > 1, (f2, r2)
    print(f"\nsain : ratio max {r0:.4f} ; tête permutée : {f1} lignes, ratio max {r1:.1f} ; colonne permutée : {f2} lignes, ratio max {r2:.1f}")


# ---- carte ------------------------------------------------------------------------------------
def _ext_carte():
    if not torch.cuda.is_available():
        pytest.skip("carte requise")
    from acvram.kernels import get_extension
    ext = get_extension()
    if ext is None or not hasattr(ext, "mla_decode_1p") or not hasattr(ext, "mla_prep_batch"):
        pytest.skip("extension sans mla_decode_1p / mla_prep_batch")
    return ext


def _lot(B, L, n, seed=1):
    torch.manual_seed(seed)
    caches = [(torch.randn(L, W, device="cuda") * 0.3).to(DT) for _ in range(B)]
    lens = torch.tensor(n if isinstance(n, (list, tuple)) else [n] * B, dtype=torch.int64, device="cuda")
    ptrs = torch.tensor([c.data_ptr() for c in caches], dtype=torch.int64, device="cuda")
    q = torch.randn(B, NH, W, device="cuda") * 0.5
    return caches, lens, ptrs, q


def _juge_termes(y, y_e, y_64, vb, o_lat):
    """Juge fp64 à l'ÉCHELLE DES TERMES (Manon 09 h 00, 37 lignes à > 6 ulp de |y64| sur 191 760 : là où la
    somme s'annule, l'ulp de |y64| n'est pas une échelle — « un rapport ne classe pas deux objets ») :
    |y − y64| ≤ 1 ulp bf16(|y64|) (l'arrondi de sortie, exposant compris) + 32·eps32·Σ_r |v_b[h,v,r]·o[b,h,r]|
    (l'accumulation fp32 de 512 produits, quel que soit l'ordre : borne (n−1)·eps, typique √n·eps). Rend (lignes fautives noyau, lignes fautives einsum, max ratio noyau,
    max ratio einsum) — ratio = |y − y64| / borne, par ligne (b, h)."""
    somme_abs = torch.einsum('hvr,bhr->bhv', vb.double().abs(), o_lat.double().abs())
    borne = 2.0 ** (torch.floor(torch.log2(y_64.abs().clamp_min(1e-30))) - 7) + 32 * 2.0 ** -23 * somme_abs
    r_k = ((y.double() - y_64).abs() / borne).amax(dim=-1)
    r_e = ((y_e.double() - y_64).abs() / borne).amax(dim=-1)
    return (r_k > 1).sum().item(), (r_e > 1).sum().item(), r_k.max().item(), r_e.max().item()


@pytest.mark.parametrize("B,L,n", [(12, 512, 300), (12, 128, 128), (3, 4096, 3900), (1, 4096, 4000),
                                   (3, 4096, [3900, 40, 700]), (12, 512, [300, 3, 511, 64, 1, 200, 300, 17, 400, 9, 128, 256])])
def test_combine_fusionne_contre_einsum_sur_carte(B, L, n):
    """b=12, L=512 (S=8, SL=1) ; b=3, L=4 096 (S≥32, SL≥2 : le morceau de colonnes en boucle) ;
    b=1 (régime FIN) ; lens INÉGAUX (b=3 : [3900, 40, 700] à L=4 096 → S ≥ 32 tranches dont la plupart VIDES pour
    b=1 et b=2 ; b=12 : lens de 1 à 511 à S=8) : la tranche vide (m = −inf, l = 0, o = 0 écrits par mla_1p_kernel
    .cu:5867) est combinée à poids 0 par le témoin (.cu:6068) comme par le fusionné (.cu:6154), à l'identique.
    Juge (d) (Sage 07 h 50, forme précisée par Manon 08 h 10) : sortie bf16 du
    noyau contre bf16(einsum fp32 de v_b32 · o_lat) ≤ 1 ulp bf16 (absolu) ET juge fp64 PAR LIGNE,
    RELATIF à la référence (REGLES § 7) : d(noyau, fp64) ≤ d(einsum, fp64) + 6 ulp bf16 — pas
    « ≤ 1 absolu » : sur la carte le noyau est à 2,0 ulp de fp64 là où l'einsum l'est aussi (c'est
    la résolution bf16 de la sortie, pas le combine). Les trois écarts (noyau/einsum, noyau/fp64,
    einsum/fp64) sont imprimés AVANT l'assertion ; positions ≠ comptées (frontières d'arrondi)."""
    ext = _ext_carte()
    caches, lens, ptrs, q = _lot(B, L, n)
    scale = 1.0 / (NOPE + ROPE) ** 0.5
    vb = (torch.randn(NH, DV, RANK, device="cuda") * 0.05).to(DT).contiguous()
    o_lat = ext.mla_decode_1p(q, ptrs, None, lens, L, RANK, scale, False)
    y = ext.mla_decode_1p(q, ptrs, None, lens, L, RANK, scale, False, vb)
    assert y.dtype == DT and y.shape == (B, NH, DV)
    y_e = torch.einsum('hvr,bhr->bhv', vb.float(), o_lat).to(DT)
    y_64 = torch.einsum('hvr,bhr->bhv', vb.double(), o_lat.double())
    u = 2.0 ** (torch.floor(torch.log2(y_64.abs().clamp_min(1e-30))) - 7)         # ulp bf16 de la référence
    d_k = ((y.double() - y_64).abs() / u).amax(dim=-1)                                # [B, nh] par ligne (b, h)
    d_e = ((y_e.double() - y_64).abs() / u).amax(dim=-1)
    e_ke, e_k64, e_e64 = _ulp_bf16(y_e, y), d_k.max().item(), d_e.max().item()
    diff = (y != y_e).sum().item()
    print(f"\nB={B} L={L} : noyau vs einsum {e_ke:.2f} ulp bf16 ({diff}/{y.numel()} positions ≠) ; "
          f"noyau vs f64 {e_k64:.2f}, einsum vs f64 {e_e64:.2f} ulp bf16 (max par ligne) ; "
          f"pire d(noyau) − d(einsum) par ligne {(d_k - d_e).max().item():+.2f}")
    f_k, f_e, r_k, r_e = _juge_termes(y, y_e, y_64, vb, o_lat)
    # part d'accumulation seule (fp32 de l'einsum avant l'arrondi bf16, contre 32·eps32·Σ|termes|) : le ratio
    # complet vaut ≈ 0,5 au plancher par le seul arrondi de sortie (½ ulp sur 1 ulp) ; c'est CE ratio-ci qui dit
    # si le 32 masque (> 0,1 : à dire, Sage 09 h 25)
    y32_e = torch.einsum('hvr,bhr->bhv', vb.float(), o_lat)
    somme_abs = torch.einsum('hvr,bhr->bhv', vb.double().abs(), o_lat.double().abs())
    r_acc = ((y32_e.double() - y_64).abs() / (32 * 2.0 ** -23 * somme_abs)).max().item()
    print(f"  RESULTAT juge des termes : lignes fautives noyau {f_k}, einsum {f_e} ; ratio_max_noyau {r_k:.4f}, ratio_max_einsum {r_e:.4f} "
          f"(plancher ≈ 0,5 = arrondi bf16 de sortie) ; accumulation seule de l'einsum fp32 : {r_acc:.4f} du 32·eps32"
          + (" — > 0,1 : le 32 masque, à dire" if r_acc > 0.1 else ""))
    # Les deux juges en ulp de |y| (noyau vs einsum ≤ 1 ulp ; d(noyau) ≤ d(einsum) + 6) sont RETIRÉS du scellé
    # (Sage 09 h 25 : l'ulp du résultat n'est pas une échelle là où Σ v·o s'annule ; M1 09 h 47 : 3,00 ulp sur
    # 8/30 720 positions, noyau/f64 1,30 < einsum/f64 1,70) — publiés à titre d'information, le scellé est le
    # juge des termes ci-dessous.
    print(f"  information (hors scellé) : noyau vs einsum {e_ke:.2f} ulp bf16 ; d(noyau) > d(einsum) + 6 ulp sur "
          f"{(d_k > d_e + 6).sum().item()} ligne(s)")
    assert f_k == 0 and f_e == 0, (f_k, f_e)
    # faute construite (Sage 09 h 25) : v_b avec une tête permutée (0 ↔ 1) passé au noyau, jugé contre
    # la référence saine → les lignes (b, 0) et (b, 1) à ratio > 1, verdict FAUX ; les autres têtes ≤ 1
    vb2 = vb.clone(); vb2[[0, 1]] = vb[[1, 0]]; vb2 = vb2.contiguous()
    y2 = ext.mla_decode_1p(q, ptrs, None, lens, L, RANK, scale, False, vb2)
    f2, _, r2, _ = _juge_termes(y2, y_e, y_64, vb, o_lat)
    somme_abs = torch.einsum('hvr,bhr->bhv', vb.double().abs(), o_lat.double().abs())
    borne = 2.0 ** (torch.floor(torch.log2(y_64.abs().clamp_min(1e-30))) - 7) + 32 * 2.0 ** -23 * somme_abs
    rr = ((y2.double() - y_64).abs() / borne).amax(dim=-1)
    print(f"  faute construite (tête 0 ↔ 1) : {f2} lignes fautives (attendu {2 * B}), ratio max {r2:.1f}, autres têtes max {rr[:, 2:].max().item():.3f}")
    assert f2 == 2 * B and r2 > 1 and bool((rr[:, 2:] <= 1).all()), (f2, r2)
