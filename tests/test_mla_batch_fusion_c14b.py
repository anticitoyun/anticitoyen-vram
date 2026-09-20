"""C14-b (revue/chantier-c14b-19-09, poste7-fiches-c5b-c13c-c14b-20-09 § 3) : au décodage par lot de
GLM (b=12), (1) v_b·o_lat dans le combine de `mla_decode_1p` (`mla_1p_combine_vb_kernel`, sortie
bf16 [B, nh, dv]) sous ACVRAM_MLA_BATCH_FUSION=1, à la place de l'einsum fp32 'hvr,bhr->bhv' que
cuBLAS sert à M=12 par gemmSN_TN. (Le geste (2), `mla_prep_batch` regrillé au bit, est jugé dans
tests/test_mla_prep_regrille_c14b.py.)

À sec : la glue (le chemin de lot passe v_b au noyau seulement quand tout s'y prête, et retombe
sur l'einsum sinon), la variable de régime nommée sur la ligne, la source CUDA. Sur carte : le combine
fusionné contre l'einsum (≤ 1 ulp bf16 de la sortie, les deux à ≤ 1 ulp bf16 d'une référence
float64 — le fp32 n'est jamais matérialisé par le noyau) sur les formes réelles (b=12, nh=20,
rank=512, dv=128, S=8 → SL=1 ; b=3, L=4 096 → SL=4).
"""
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
    from tests.test_mla_niveau2_jumeaux import _module
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
    lens = torch.full((B,), n, dtype=torch.int64, device="cuda")
    ptrs = torch.tensor([c.data_ptr() for c in caches], dtype=torch.int64, device="cuda")
    q = torch.randn(B, NH, W, device="cuda") * 0.5
    return caches, lens, ptrs, q


@pytest.mark.parametrize("B,L,n", [(12, 512, 300), (12, 128, 128), (3, 4096, 3900), (1, 4096, 4000)])
def test_combine_fusionne_contre_einsum_sur_carte(B, L, n):
    """b=12, L=512 (S=8, SL=1) ; b=3, L=4 096 (S≥32, SL≥2 : le morceau de colonnes en boucle) ;
    b=1 (régime FIN). Sortie bf16 du noyau contre bf16(einsum fp32 de v_b32 · o_lat) : ≤ 1 ulp
    bf16, et les deux à ≤ 1 ulp bf16 d'une référence float64 de v_b·o_lat (o_lat = celui du
    noyau) ; le nombre de positions ≠ est affiché (elles sont aux frontières d'arrondi bf16)."""
    ext = _ext_carte()
    caches, lens, ptrs, q = _lot(B, L, n)
    scale = 1.0 / (NOPE + ROPE) ** 0.5
    vb = (torch.randn(NH, DV, RANK, device="cuda") * 0.05).to(DT).contiguous()
    o_lat = ext.mla_decode_1p(q, ptrs, None, lens, L, RANK, scale, False)
    y = ext.mla_decode_1p(q, ptrs, None, lens, L, RANK, scale, False, vb)
    assert y.dtype == DT and y.shape == (B, NH, DV)
    y_e = torch.einsum('hvr,bhr->bhv', vb.float(), o_lat).to(DT)
    y_64 = torch.einsum('hvr,bhr->bhv', vb.double(), o_lat.double())
    e_ke, e_k64, e_e64 = _ulp_bf16(y_e, y), _ulp_bf16(y_64.to(DT), y), _ulp_bf16(y_64.to(DT), y_e)
    diff = (y != y_e).sum().item()
    print(f"\nB={B} L={L} : noyau vs einsum {e_ke:.2f} ulp bf16 ({diff}/{y.numel()} positions ≠), "
          f"noyau vs f64 {e_k64:.2f}, einsum vs f64 {e_e64:.2f}")
    assert e_ke <= 1 and e_k64 <= 1 and e_e64 <= 1
    # témoin cassant : un v_b décalé d'une tête doit se voir
    vb2 = torch.roll(vb, 1, dims=0).contiguous()
    y2 = ext.mla_decode_1p(q, ptrs, None, lens, L, RANK, scale, False, vb2)
    assert not torch.equal(y, y2)
