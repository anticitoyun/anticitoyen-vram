"""Chantiers C14 et C14-c : grille de mla_1p_kernel et recombinaison.

À sec, sans carte ni compilation du .cu. Trois choses sont prouvées ici :

1. la règle de découpage — `mla_1p_tranches(L, TL, B)` pour b ≥ 3 (C14,
   inchangée) et `mla_1p_tranches_fin(L, B, G)` pour b ≤ 2 (C14-c : tranches
   de 8 lignes, G = ⌈H/5⌉ groupes de têtes, grille B × S × G ≤ 2·SM blocs
   résidents) — rejouée à l'identique en Python : ≥ 170 blocs à b=1 dès
   L=512 (256), aucune tranche vide, même arrondi de `rows_par_cta` que le
   lanceur, b=12 mot pour mot comme avant ;
2. la recombinaison flash-decoding — partiels (o_s, m_s, l_s) par tranche,
   o = Σ o_s·e^{m_s−M} / Σ l_s·e^{m_s−M} — rend la même sortie quel que
   soit S, quel que soit le découpage des têtes en groupes (C14-c : le bloc
   (b, s, g) écrit ses Hb têtes aux mêmes places de ws, le combine ne voit
   pas la répartition) et quel que soit le nombre SL de voies du combine
   (C14-c : sommes partielles par voie puis réduction), à ≤ 8 ulp de
   l'AMPLITUDE de la tête (2^-23 · max_c |o_h|) ; en float64 la répartition
   des têtes et les voies ne changent rien du tout (≤ 1e-12 relatif) ;
3. des témoins cassants : m non rescalé (> 8·10⁴ ulp), tranche vide comptée
   l = 1 (> 80), groupe de têtes écrit au mauvais décalage h0, voie du
   combine qui oublie son reste (s ≥ 8·SL·k).

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
# C14-c : régime fin (b ≤ 2, H ≤ 20) — constantes MLA1P_FIN_* du .cu
FIN_HMAX, FIN_TL, FIN_BLOCS_SM = 5, 8, 2


# ---------------------------------------------------------------------------
# 1. règle de grille, miroir de mla_1p_tranches / mla_1p_tranches_fin (.cu)
# ---------------------------------------------------------------------------
def rows_par_cta(L: int, S: int, TL: int) -> int:
    """Même arrondi que le lanceur mla_decode_1p : ceil(ceil(L/S)/TL)·TL."""
    return -(-(-(-L // S)) // TL) * TL


def tranches_avant(L: int, TL: int) -> int:
    """Règle d'avant C14 : ≈ 64 lignes par CTA, S ≤ 32."""
    tuiles = -(-L // TL)
    return max(1, min(32, (tuiles + 1) // 2))


def tranches_c14(L: int, TL: int, B: int, sm: int = SM_5090) -> int:
    """mla_1p_tranches : une vague à 1 CTA/SM (b ≥ 3)."""
    tuiles = -(-L // TL)
    S = max(1, min(tuiles, sm // max(1, B)))
    S = min(S, S_MAX)
    rows = rows_par_cta(L, S, TL)
    return max(1, -(-L // rows))


def groupes_fin(H: int) -> int:
    return -(-H // FIN_HMAX)


def tranches_fin(L: int, B: int, G: int, sm: int = SM_5090) -> int:
    """mla_1p_tranches_fin : S = min(⌈L/8⌉, ⌊2·SM/(B·G)⌋), tranches non vides."""
    tuiles = -(-L // FIN_TL)
    cap = FIN_BLOCS_SM * sm // max(1, B * G)
    S = max(1, min(tuiles, cap))
    S = min(S, S_MAX)
    rows = rows_par_cta(L, S, FIN_TL)
    return max(1, -(-L // rows))


def grille(L: int, B: int, H: int = 20, sm: int = SM_5090):
    """(TL, S, rows, G) du lanceur mla_decode_1p."""
    if H <= 20 and B <= 2:
        G = groupes_fin(H)
        S = tranches_fin(L, B, G, sm)
        return FIN_TL, S, rows_par_cta(L, S, FIN_TL), G
    TL = 32 if H <= 20 else 16
    S = tranches_c14(L, TL, B, sm)
    return TL, S, rows_par_cta(L, S, TL), 1


def blocs(L: int, B: int, H: int = 20, sm: int = SM_5090) -> int:
    _, S, _, G = grille(L, B, H, sm)
    return B * S * G


@pytest.mark.parametrize("L, B, attendu", [
    # (TL, S, rows, G) — chiffres du commentaire de mla_1p_tranches_fin (.cu)
    (512, 1, (8, 64, 8, 4)),       # godet mesuré par nsys : 32 blocs (C14) → 256
    (1024, 1, (8, 64, 16, 4)),
    (2048, 1, (8, 64, 32, 4)),
    (4096, 1, (8, 74, 56, 4)),
    (128, 1, (8, 16, 8, 4)),
    (512, 2, (8, 32, 16, 4)),
    (2048, 2, (8, 37, 56, 4)),
    (512, 12, (32, 8, 64, 1)),     # inchangé depuis C14 : 96 CTA, une vague
    (2048, 12, (32, 13, 160, 1)),
    (1024, 8, (32, 16, 64, 1)),
    (2048, 16, (32, 10, 224, 1)),
    (512, 3, (32, 16, 32, 1)),     # premier lot du régime b ≥ 3
])
def test_grille_par_lot_et_contexte(L, B, attendu):
    assert grille(L, B) == attendu


@pytest.mark.parametrize("L, B, n", [
    (512, 1, 256), (1024, 1, 256), (2048, 1, 256), (512, 2, 256), (2048, 2, 296),
    (512, 12, 96), (2048, 12, 156),
])
def test_nombre_de_blocs(L, B, n):
    """Formule : B·S·G ; scellé C14-c : ≥ 170 blocs à b=1 (L=512 : 256)."""
    assert blocs(L, B) == n


@pytest.mark.parametrize("B", [1, 2, 4, 8, 12, 16, 32, 256])
@pytest.mark.parametrize("L", [128, 256, 512, 1024, 2048, 4096, 8192, 32768])
def test_invariants_de_la_regle(L, B):
    TL, S, rows, G = grille(L, B)
    tuiles = -(-L // TL)
    assert 1 <= S <= tuiles                          # au plus une tranche par tuile
    assert S * rows >= L                              # tout le contexte est couvert
    assert (S - 1) * rows < L                         # aucune tranche vide
    assert rows % TL == 0
    assert rows_par_cta(L, S, TL) == rows             # point fixe : le lanceur retrouve rows
    assert S <= S_MAX
    if B <= 2:
        assert (TL, G) == (FIN_TL, 4)
        assert B * S * G <= FIN_BLOCS_SM * SM_5090    # une vague à 2 blocs résidents par SM
        if L >= 512:
            assert B * S * G >= SM_5090               # scellé : ≥ 170 blocs (b=1 et b=2)
        assert S >= tranches_avant(L, TL)             # jamais moins de tranches qu'avant
    else:
        assert G == 1
        assert B * S <= max(SM_5090, B)               # une vague (1 CTA/SM) dès que le lot le permet
        assert S == tranches_c14(L, TL, B)            # C14 mot pour mot


def test_temoin_regle_avant_sous_occupee():
    """Ce que les règles d'avant faisaient à b=1 : 8 CTA (avant C14) puis 32
    (C14, demi-tuile) sur 170 SM à L=512."""
    assert tranches_avant(512, 32) == 8
    assert tranches_avant(4096, 32) == 32             # plafond : 4 tuiles par CTA
    assert tranches_c14(512, 16, 1) == 32             # C14 : 32 blocs, verdict-c14-bis


def test_source_cu_porte_la_regle():
    src = open(_CU, encoding="utf-8").read()
    assert "static int mla_1p_tranches(int L, int TL, int B)" in src
    assert "static int mla_1p_tranches_fin(int L, int B, int G)" in src
    assert re.search(r"constexpr int MLA1P_FIN_HMAX = 5, MLA1P_FIN_TL = 8, MLA1P_FIN_BLOCS_SM = 2;", src)
    corps = src.split("static int mla_1p_tranches(int L, int TL, int B)")[1].split("\n}\n")[0]
    assert "min(32," not in corps                      # l'ancien plafond a disparu
    assert "MLA1P_S_MAX" in corps
    fin = src.split("static int mla_1p_tranches_fin(int L, int B, int G)")[1].split("\n}\n")[0]
    assert "MLA1P_FIN_BLOCS_SM * mla_1p_sm_count() / max(1, B * G)" in fin
    lanceur = src.split("torch::Tensor mla_decode_1p(")[1].split("\n}\n")[0]
    assert "const bool fin = H <= 20 && B <= 2;" in lanceur
    assert "dim3 grid(B, S, G);" in lanceur
    assert "MLA1P_LANCE_FIN(5, 8, 1, false)" in lanceur    # le régime fin est instancié (MINB=2)
    assert "MLA1P_LANCE(20, 32, 4, false)" in lanceur  # b ≥ 3 inchangé
    assert "MLA1P_LANCE(20, 16, 2" not in src          # la demi-tuile C14 a disparu
    assert "dim3 g2(B * H, (R + CH - 1) / CH);" in lanceur
    # noyau : groupe de têtes h0 = g·HMAX, partiels aux places h0.. de ws
    assert "constexpr bool GROUPES = HMAX <= MLA1P_FIN_HMAX;" in src
    assert "const int h0 = GROUPES ? (int)blockIdx.z * HMAX : 0;" in src
    assert "const int Hb = GROUPES ? min(HMAX, H - h0) : H;" in src    # b ≥ 3 : SASS d'avant
    assert "float *out = ws + ((size_t)(b * S + s) * H + h0) * (R + 2);" in src
    assert "template <int HMAX, int TL, int RW, bool FP8, int MINB>\n__global__ void __launch_bounds__(MLA1P_FILS, MINB) mla_1p_kernel(" in src
    assert "#define MLA1P_LANCE(HM, T, RWW, F8) MLA1P_LANCE_K(HM, T, RWW, F8, 0)" in lanceur          # b ≥ 3 : bornes d'avant
    assert "#define MLA1P_LANCE_FIN(HM, T, RWW, F8) MLA1P_LANCE_K(HM, T, RWW, F8, MLA1P_FIN_BLOCS_SM)" in lanceur
    # combine : un modèle par nombre de voies, poids en shared, lectures en vol
    assert "template <int SL>\n__global__ void __launch_bounds__(MLA1P_CMB_FILS) mla_1p_combine_kernel(" in src
    assert "__shared__ float w_s[MLA1P_S_MAX]" in src
    assert "for (; s + 7 * SL < S; s += 8 * SL)" in src


# ---------------------------------------------------------------------------
# 2. recombinaison flash-decoding : même sortie quel que soit S, G, SL
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
    """Ce que fait un bloc de mla_1p_kernel sur les lignes [r0, r1) pour les
    têtes de `scores` : tuiles de TL lignes, softmax en ligne par tête,
    (o, m, l) fp32. scores [Hb, L] déjà × scale (le produit q·k n'est pas
    l'objet du test), v [L, R] fp32."""
    H, R = scores.shape[0], v.shape[1]
    r1 = min(r1, valide)
    m = torch.full((H,), -math.inf, dtype=scores.dtype)
    l = torch.zeros(H, dtype=scores.dtype)
    o = torch.zeros(H, R, dtype=scores.dtype)
    if r0 >= r1:                                       # tranche vide : m = −inf, l = 0, o = 0
        return o, m, l
    for t0 in range(r0, r1, TL):
        n = min(TL, r1 - t0)
        Sh = scores[:, t0:t0 + n]
        m_new = torch.maximum(m, Sh.max(dim=1).values)
        a = torch.where(m == -math.inf, torch.zeros_like(m), torch.exp(m - m_new))
        P = torch.exp(Sh - m_new[:, None])
        l = l * a + P.sum(dim=1)
        m = m_new
        o = o * a[:, None] + P @ v[t0:t0 + n]
    return o, m, l


def partiels_par_groupes(scores, v, r0, r1, TL, valide, G: int, h0_faux: bool = False):
    """C14-c : la tranche [r0, r1) traitée par G blocs de ⌈H/G⌉ têtes ; chaque
    bloc écrit ses têtes aux places h0 = g·Hb de ws. h0_faux est le témoin
    cassant : tous les groupes écrits au décalage 0 (le dernier gagne)."""
    H = scores.shape[0]
    Hb = -(-H // G)
    o = torch.zeros(H, v.shape[1], dtype=scores.dtype)
    m = torch.zeros(H, dtype=scores.dtype)
    l = torch.zeros(H, dtype=scores.dtype)
    for g in range(G):
        h0, h1 = g * Hb, min(H, (g + 1) * Hb)
        og, mg, lg = partiels(scores[h0:h1], v, r0, r1, TL, valide)
        d = 0 if h0_faux else h0
        o[d:d + h1 - h0], m[d:d + h1 - h0], l[d:d + h1 - h0] = og, mg, lg
    return o, m, l


def combine(parts, rescale: bool = True, SL: int = 1, reste: bool = True):
    """mla_1p_combine_kernel : o = Σ o_s·e^{m_s−M} / Σ l_s·e^{m_s−M}.
    SL voies (C14-c) : la voie k somme les tranches s ≡ k (mod SL) par
    groupes de 8 puis son reste, les voies sont réduites ensuite.
    rescale=False : témoin cassant, Σ o_s / Σ l_s sans les poids ;
    reste=False : témoin cassant, la voie oublie les tranches après son
    dernier groupe de 8."""
    o_s = torch.stack([p[0] for p in parts])            # [S, H, R]
    m_s = torch.stack([p[1] for p in parts])            # [S, H]
    l_s = torch.stack([p[2] for p in parts])            # [S, H]
    S = o_s.shape[0]
    M = m_s.max(dim=0).values
    w = torch.where(m_s == -math.inf, torch.zeros_like(m_s), torch.exp(m_s - M)) if rescale \
        else torch.where(m_s == -math.inf, torch.zeros_like(m_s), torch.ones_like(m_s))
    Lsum = (l_s * w).sum(dim=0)
    acc = torch.zeros_like(o_s[0])
    for k in range(SL):
        voie = torch.zeros_like(acc)
        s = k
        while s + 7 * SL < S:
            for j in range(8):
                voie = voie + o_s[s + j * SL] * w[s + j * SL][:, None]
            s += 8 * SL
        if reste:
            while s < S:
                voie = voie + o_s[s] * w[s][:, None]
                s += SL
        acc = acc + voie
    return acc / Lsum[:, None]


def sortie_par_tranches(scores, v, S: int, TL: int, valide: int, rescale: bool = True,
                        G: int = 1, SL: int = 1, h0_faux: bool = False, reste: bool = True):
    L = scores.shape[1]
    rows = rows_par_cta(L, S, TL)
    parts = [partiels_par_groupes(scores, v, s * rows, (s + 1) * rows, TL, valide, G, h0_faux)
             for s in range(S)]
    return combine(parts, rescale, SL, reste)


def _jeu(seed: int, H=20, W=576, R=512, L=1024, scale=1.0 / math.sqrt(192)):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(H, W, generator=g)
    k = torch.randn(L, W, generator=g).to(torch.bfloat16).float()   # cache latent bf16
    scores = (q @ k.T) * scale                                         # [H, L] fp32, calculé UNE fois
    return scores, k[:, :R]


BORNE_ULP = 8      # ulp d'amplitude ; mesuré à sec : 4,7 au pire sur 8 graines × 5 grilles


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("S, TL", [(16, 32), (32, 32), (64, 16), (64, 32), (64, 8), (128, 8)])
def test_recombinaison_independante_de_S(seed, S, TL):
    scores, v = _jeu(seed)
    L = scores.shape[1]
    ref = sortie_par_tranches(scores, v, 1, 32, L)      # un seul CTA, tuiles de 32 : l'ancien chemin à S=1
    out = sortie_par_tranches(scores, v, S, TL, L)
    assert torch.isfinite(out).all()
    assert ulp_amplitude(out, ref) <= BORNE_ULP


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("G", [1, 2, 4, 5, 20])
@pytest.mark.parametrize("SL", [1, 2, 4, 8])
def test_recombinaison_independante_des_groupes_et_des_voies(seed, G, SL):
    """C14-c, la grille du régime fin (S=64, TL=8, G=4) et le combine à SL
    voies : ≤ 8 ulp d'amplitude de l'ancien chemin, pour tout G et tout SL."""
    scores, v = _jeu(seed)
    L = scores.shape[1]
    ref = sortie_par_tranches(scores, v, 1, 32, L)
    out = sortie_par_tranches(scores, v, 64, 8, L, G=G, SL=SL)
    assert torch.isfinite(out).all()
    assert ulp_amplitude(out, ref) <= BORNE_ULP


def test_groupes_et_voies_exacts_en_float64():
    """En float64 la répartition des têtes et le nombre de voies ne changent
    rien : ≤ 1e-12 relatif entre (G, SL) et (1, 1), à S=64, TL=8 — ce sont
    les mêmes termes, sommés dans un autre ordre."""
    scores, v = _jeu(5)
    scores, v = scores.double(), v.double()
    L = scores.shape[1]
    ref = sortie_par_tranches(scores, v, 64, 8, L)
    for G, SL in [(4, 8), (4, 1), (1, 8), (2, 4), (5, 2), (20, 8)]:
        out = sortie_par_tranches(scores, v, 64, 8, L, G=G, SL=SL)
        assert float(((out - ref).abs() / ref.abs().amax(dim=1, keepdim=True)).max()) <= 1e-12


def test_recombinaison_contexte_partiel_et_tranches_vides():
    """valide < L : les tranches au-delà écrivent m = −inf, l = 0 et ne
    pèsent rien ; la sortie est celle du contexte tronqué — aussi en groupes
    de têtes et à 8 voies."""
    scores, v = _jeu(7)
    valide = 300                                         # 1024 lignes de godet, 300 écrites
    ref = sortie_par_tranches(scores[:, :valide], v[:valide], 1, 32, valide)
    for S, TL, G, SL in [(16, 32, 1, 1), (64, 16, 1, 1), (64, 32, 1, 1), (64, 8, 4, 8), (128, 8, 4, 8)]:
        out = sortie_par_tranches(scores, v, S, TL, valide, G=G, SL=SL)
        assert ulp_amplitude(out, ref) <= BORNE_ULP


def test_contre_reference_float64():
    """Tout S (l'ancien S=1 compris) reste à ≤ 8 ulp d'amplitude du softmax
    exact en float64 ; l'ancien chemin lui-même est à ~3 : le nouveau
    découpage n'est pas plus loin de l'exact que l'ancien."""
    scores, v = _jeu(11)
    exact = (torch.softmax(scores.double(), dim=1) @ v.double()).float()
    for S, TL, G, SL in [(1, 32, 1, 1), (16, 32, 1, 1), (64, 16, 1, 1), (64, 8, 4, 8)]:
        out = sortie_par_tranches(scores, v, S, TL, scores.shape[1], G=G, SL=SL)
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


def test_temoin_cassant_groupe_au_mauvais_decalage():
    """C14-c : un bloc de têtes qui écrit au décalage 0 au lieu de h0 (le
    `+ h0` de `out` oublié) laisse les têtes 0..4 à celles du dernier groupe
    et les têtes 5..19 jamais écrites (l = 0 : NaN ici, mémoire non
    initialisée sur la carte) : le critère « fini et ≤ 8 ulp » rend faux."""
    scores, v = _jeu(0)
    L = scores.shape[1]
    ref = sortie_par_tranches(scores, v, 1, 32, L)
    casse = sortie_par_tranches(scores, v, 64, 8, L, G=4, SL=8, h0_faux=True)
    assert not torch.isfinite(casse).all()
    assert ulp_amplitude(casse[:5], ref[:5]) > 1e4 * BORNE_ULP    # têtes 0..4 = têtes 15..19


def test_temoin_cassant_voie_sans_reste():
    """C14-c : à S=64 et SL=8 chaque voie fait exactement un groupe de 8 —
    le reste est vide et l'oubli passerait inaperçu ; à S=74 tranches
    pleines de 8 lignes (L=592) ou S=37 (L=296, SL=4) il ne passe pas :
    > 10³ ulp d'amplitude."""
    for L, S, SL in [(592, 74, 8), (296, 37, 4)]:
        scores, v = _jeu(3, L=L)
        ref = sortie_par_tranches(scores, v, 1, 32, L)
        assert rows_par_cta(L, S, 8) == 8                # S tranches, toutes non vides
        assert ulp_amplitude(sortie_par_tranches(scores, v, S, 8, L, G=4, SL=SL), ref) <= BORNE_ULP
        casse = sortie_par_tranches(scores, v, S, 8, L, G=4, SL=SL, reste=False)
        assert ulp_amplitude(casse, ref) > 1e3 * BORNE_ULP
