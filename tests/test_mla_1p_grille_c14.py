"""Chantier C14 (poste7-glm-decode-budget-c14-c15-19-09) : grille de mla_1p_kernel.

À sec, sans carte ni compilation du .cu. Deux choses sont prouvées ici :

1. la règle de découpage `mla_1p_tranches(L, TL, B)` (acvram_kernels.cu),
   rejouée à l'identique en Python : le lot tient en une vague (B·S ≤ SM,
   un CTA par SM à 88 Ko de shared), au plus une tranche par tuile, aucune
   tranche vide, et le même arrondi de `rows_par_cta` que le lanceur ;
2. la recombinaison flash-decoding de `mla_1p_combine_kernel` — partiels
   (o_s, m_s, l_s) par tranche, o = Σ o_s·e^{m_s−M} / Σ l_s·e^{m_s−M} —
   rend la même sortie quel que soit S, à ≤ 8 ulp de l'AMPLITUDE de la tête
   (2^-23 · max_c |o_h|) — mesuré 4,7 au pire sur 8 graines × 5 grilles ;
   un témoin cassant (m non rescalé) sort de la borne de quatre ordres de
   grandeur, un second (tranche vide comptée l = 1) d'un ordre.

Pourquoi pas « 1 ulp » élément par élément : o_h est une somme de 1 024
termes de signes mêlés ; ses plus petits coefficients (~1e-5 pour une
amplitude ~0,3) portent l'erreur absolue de la somme (~1e-7), soit des
milliers d'ulp à eux seuls — même l'ancien chemin à S=1 est à 3,2 ulp
d'amplitude du softmax exact en float64. Et la sortie n'est PAS identique au
bit entre deux S : quand le maximum d'une tranche est sous celui de la
précédente, l'ancien chemin (tuiles enchaînées, p = e^{s−m_courant}) et le
nouveau (p = e^{s−m_tranche} puis × e^{m_tranche−M} au combine) arrondissent
différemment. La borne mesurable est celle-ci, pas « ± 1 ulp » posé.
"""
import math
import os
import re

import pytest
import torch

torch.set_num_threads(min(8, torch.get_num_threads()))

_CU = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "acvram", "kernels", "acvram_kernels.cu")
SM_5090 = 170
S_MAX = 1024


# ---------------------------------------------------------------------------
# 1. règle de grille, miroir de mla_1p_tranches (acvram_kernels.cu)
# ---------------------------------------------------------------------------
def rows_par_cta(L: int, S: int, TL: int) -> int:
    """Même arrondi que le lanceur mla_decode_1p : ceil(ceil(L/S)/TL)·TL."""
    return -(-(-(-L // S)) // TL) * TL


def tranches_avant(L: int, TL: int) -> int:
    """Règle d'avant C14 : ≈ 64 lignes par CTA, S ≤ 32."""
    tuiles = -(-L // TL)
    return max(1, min(32, (tuiles + 1) // 2))


def tranches_c14(L: int, TL: int, B: int, sm: int = SM_5090) -> int:
    tuiles = -(-L // TL)
    S = max(1, min(tuiles, sm // max(1, B)))
    S = min(S, S_MAX)
    rows = rows_par_cta(L, S, TL)
    return max(1, -(-L // rows))


def demi_tuile(L: int, B: int, H: int = 20, sm: int = SM_5090) -> bool:
    """Condition du lanceur : TL=16 si H ≤ 20, B ≤ 2 et ceil(L/16) ≤ SM/B."""
    return H <= 20 and B <= 2 and -(-L // 16) <= sm // B


def grille(L: int, B: int, H: int = 20, sm: int = SM_5090):
    TL = 16 if H > 20 or demi_tuile(L, B, H, sm) else 32
    S = tranches_c14(L, TL, B, sm)
    return TL, S, rows_par_cta(L, S, TL)


@pytest.mark.parametrize("L, B, attendu", [
    # (TL, S, rows) — chiffres de la fiche chantier-c14-19-09
    (512, 1, (16, 32, 16)),      # godet mesuré par nsys : 8 CTA avant → 32
    (1024, 1, (16, 64, 16)),
    (2048, 1, (16, 128, 16)),
    (4096, 1, (32, 128, 32)),    # ceil(4096/16) = 256 > 170 : demi-tuile inutile
    (512, 2, (16, 32, 16)),
    (512, 12, (32, 8, 64)),      # inchangé : 96 CTA, une vague
    (2048, 12, (32, 13, 160)),   # 384 CTA en 2,3 vagues avant → 156 en une
    (1024, 8, (32, 16, 64)),
    (2048, 16, (32, 10, 224)),
    (128, 1, (16, 8, 16)),
])
def test_grille_par_lot_et_contexte(L, B, attendu):
    assert grille(L, B) == attendu


@pytest.mark.parametrize("B", [1, 2, 4, 8, 12, 16, 32, 256])
@pytest.mark.parametrize("L", [128, 256, 512, 1024, 2048, 4096, 8192, 32768])
def test_invariants_de_la_regle(L, B):
    TL, S, rows = grille(L, B)
    tuiles = -(-L // TL)
    assert 1 <= S <= tuiles                          # au plus une tranche par tuile
    assert S * rows >= L                              # tout le contexte est couvert
    assert (S - 1) * rows < L                         # aucune tranche vide
    assert rows % TL == 0
    assert rows_par_cta(L, S, TL) == rows             # point fixe : le lanceur retrouve rows
    assert B * S <= max(SM_5090, B)                   # une vague (1 CTA/SM) dès que le lot le permet
    if B == 1:
        assert S >= tranches_avant(L, TL)             # jamais moins de CTA qu'avant à b=1


def test_temoin_regle_avant_sous_occupee():
    """Ce que la règle d'avant faisait à b=1 : 8 CTA sur 170 SM à L=512."""
    assert tranches_avant(512, 32) == 8
    assert tranches_avant(1024, 32) == 16
    assert tranches_avant(2048, 32) == 32
    assert tranches_avant(4096, 32) == 32             # plafond : 4 tuiles par CTA


def test_source_cu_porte_la_regle():
    src = open(_CU, encoding="utf-8").read()
    assert "static int mla_1p_tranches(int L, int TL, int B)" in src
    assert re.search(r"mla_1p_tranches\(\(int\)L, TL, B\)", src)
    corps = src.split("static int mla_1p_tranches(int L, int TL, int B)")[1].split("\n}\n")[0]
    assert "min(32," not in corps                      # l'ancien plafond a disparu
    assert "MLA1P_S_MAX" in corps
    assert "MLA1P_LANCE(20, 16, 2, false)" in src       # la demi-tuile est instanciée
    assert "__shared__ float w_s[MLA1P_S_MAX]" in src   # combine : poids en shared


# ---------------------------------------------------------------------------
# 2. recombinaison flash-decoding : même sortie quel que soit S
# ---------------------------------------------------------------------------
def _cle_ulp(x: torch.Tensor) -> torch.Tensor:
    """Entier monotone en la valeur fp32 (distance = nombre d'ulp)."""
    i = x.contiguous().view(torch.int32).to(torch.int64)
    return torch.where(i < 0, -2147483648 - i, i)


def ulp_max(a: torch.Tensor, b: torch.Tensor) -> int:
    """Distance élément par élément (diagnostic : dominée par l'annulation)."""
    return int((_cle_ulp(a) - _cle_ulp(b)).abs().max())


def ulp_amplitude(a: torch.Tensor, ref: torch.Tensor) -> float:
    """max |a − ref| en unités de 2^-23 · max_c |ref_h| (par tête)."""
    amp = ref.abs().amax(dim=1, keepdim=True)
    return float(((a - ref).abs() / (amp * 2.0 ** -23)).max())


def partiels(scores: torch.Tensor, v: torch.Tensor, r0: int, r1: int, TL: int, valide: int):
    """Ce que fait un CTA de mla_1p_kernel sur les lignes [r0, r1) : tuiles de
    TL lignes, softmax en ligne par tête, (o, m, l) fp32. scores [H, L] déjà
    × scale (le produit q·k n'est pas l'objet du test), v [L, R] fp32."""
    H, R = scores.shape[0], v.shape[1]
    r1 = min(r1, valide)
    m = torch.full((H,), -math.inf)
    l = torch.zeros(H)
    o = torch.zeros(H, R)
    if r0 >= r1:                                       # tranche vide : m = −inf, l = 0, o = 0
        return o, m, l
    for t0 in range(r0, r1, TL):
        n = min(TL, r1 - t0)
        Sh = scores[:, t0:t0 + n]
        m_new = torch.maximum(m, Sh.max(dim=1).values)
        a = torch.where(m == -math.inf, torch.zeros(H), torch.exp(m - m_new))
        P = torch.exp(Sh - m_new[:, None])
        l = l * a + P.sum(dim=1)
        m = m_new
        o = o * a[:, None] + P @ v[t0:t0 + n]
    return o, m, l


def combine(parts, rescale: bool = True):
    """mla_1p_combine_kernel : o = Σ o_s·e^{m_s−M} / Σ l_s·e^{m_s−M}.
    rescale=False est le témoin cassant : Σ o_s / Σ l_s sans les poids."""
    o_s = torch.stack([p[0] for p in parts])            # [S, H, R]
    m_s = torch.stack([p[1] for p in parts])            # [S, H]
    l_s = torch.stack([p[2] for p in parts])            # [S, H]
    M = m_s.max(dim=0).values
    w = torch.where(m_s == -math.inf, torch.zeros_like(m_s), torch.exp(m_s - M)) if rescale \
        else torch.where(m_s == -math.inf, torch.zeros_like(m_s), torch.ones_like(m_s))
    Lsum = (l_s * w).sum(dim=0)
    acc = (o_s * w[:, :, None]).sum(dim=0)
    return acc / Lsum[:, None]


def sortie_par_tranches(scores, v, S: int, TL: int, valide: int, rescale: bool = True):
    L = scores.shape[1]
    rows = rows_par_cta(L, S, TL)
    parts = [partiels(scores, v, s * rows, (s + 1) * rows, TL, valide) for s in range(S)]
    return combine(parts, rescale)


def _jeu(seed: int, H=20, W=576, R=512, L=1024, scale=1.0 / math.sqrt(192)):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(H, W, generator=g)
    k = torch.randn(L, W, generator=g).to(torch.bfloat16).float()   # cache latent bf16
    scores = (q @ k.T) * scale                                         # [H, L] fp32, calculé UNE fois
    return scores, k[:, :R]


BORNE_ULP = 8      # ulp d'amplitude ; mesuré à sec : 4,7 au pire sur 8 graines × 5 grilles


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("S, TL", [(16, 32), (32, 32), (64, 16), (64, 32)])
def test_recombinaison_independante_de_S(seed, S, TL):
    scores, v = _jeu(seed)
    L = scores.shape[1]
    ref = sortie_par_tranches(scores, v, 1, 32, L)      # un seul CTA, tuiles de 32 : l'ancien chemin à S=1
    out = sortie_par_tranches(scores, v, S, TL, L)
    assert torch.isfinite(out).all()
    assert ulp_amplitude(out, ref) <= BORNE_ULP


def test_recombinaison_contexte_partiel_et_tranches_vides():
    """valide < L : les tranches au-delà écrivent m = −inf, l = 0 et ne
    pèsent rien ; la sortie est celle du contexte tronqué."""
    scores, v = _jeu(7)
    valide = 300                                         # 1024 lignes de godet, 300 écrites
    ref = sortie_par_tranches(scores[:, :valide], v[:valide], 1, 32, valide)
    for S, TL in [(16, 32), (64, 16), (64, 32)]:
        out = sortie_par_tranches(scores, v, S, TL, valide)
        assert ulp_amplitude(out, ref) <= BORNE_ULP


def test_contre_reference_float64():
    """Tout S (l'ancien S=1 compris) reste à ≤ 8 ulp d'amplitude du softmax
    exact en float64 ; l'ancien chemin lui-même est à ~3 : le nouveau
    découpage n'est pas plus loin de l'exact que l'ancien."""
    scores, v = _jeu(11)
    exact = (torch.softmax(scores.double(), dim=1) @ v.double()).float()
    for S, TL in [(1, 32), (16, 32), (64, 16)]:
        out = sortie_par_tranches(scores, v, S, TL, scores.shape[1])
        assert ulp_amplitude(out, exact) <= BORNE_ULP


def test_temoin_cassant_m_non_rescale():
    """Sans les poids e^{m_s−M} la recombinaison est fausse de plusieurs
    ordres de grandeur : la borne en ulp rend « faux »."""
    scores, v = _jeu(0)
    L = scores.shape[1]
    ref = sortie_par_tranches(scores, v, 1, 32, L)
    casse = sortie_par_tranches(scores, v, 16, 32, L, rescale=False)
    assert ulp_amplitude(casse, ref) > 1e4 * BORNE_ULP


def test_temoin_cassant_tranche_vide_comptee():
    """Une tranche vide comptée avec l = 1 (au lieu de 0) fausse le
    dénominateur : le test le voit."""
    scores, v = _jeu(0)
    L = scores.shape[1]
    ref = sortie_par_tranches(scores, v, 1, 32, L)
    rows = rows_par_cta(L, 64, 32)                       # 32 tranches pleines + 32 vides
    parts = [partiels(scores, v, s * rows, (s + 1) * rows, 32, L) for s in range(64)]
    o, m, l = parts[40]
    assert m.eq(-math.inf).all() and l.eq(0).all()
    parts[40] = (o, torch.zeros_like(m), torch.ones_like(l))
    assert ulp_amplitude(combine(parts), ref) > 10 * BORNE_ULP
