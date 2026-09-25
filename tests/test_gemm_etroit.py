"""Juge du poste C (kernels/gemm_etroit.py) : le GEMM étroit W8A16 Triton
contre la référence float64 (déquant exacte de `_dequantize_int8`), à
2⁻⁸ × Σ|x·w| par valeur ; bras qui doivent casser : zéro ignoré, échelle
décalée d'un groupe, entrée plus courte que k_pad ; tête en fp32 ;
plusieurs tranches K (b = 1) et une seule. Sans carte :
``TRITON_INTERPRET=1`` en fp16.
"""
import importlib
import os

import pytest
import torch

from acvram.quant.formats import INT8Tensor, _dequantize_int8, _quantize_int8

TOL = 2 ** -8


def _ge():
    if not torch.cuda.is_available():
        os.environ.setdefault("TRITON_INTERPRET", "1")
    ge = importlib.import_module("acvram.kernels.gemm_etroit")
    if not ge.disponible():
        pytest.skip("Triton indisponible")
    return ge


def _montage(m, n, k, graine=0, group=128):
    torch.manual_seed(graine)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w = torch.randn(n, k) * 0.05 + torch.randn(n, 1) * 0.02
    t = _quantize_int8(w, group_size=group)
    t.qweight, t.scales, t.zeros = t.qweight.to(dev), t.scales.to(dev), t.zeros.to(dev)
    x = (torch.randn(m, k) * torch.logspace(-1, 1, m).unsqueeze(1)).to(dev)
    x = x.to(torch.bfloat16 if dev == "cuda" else torch.float16)
    return x, t


def _juger(y, x, t):
    w = _dequantize_int8(t, torch.float64)
    ref = x.double() @ w.T
    borne = x.double().abs() @ w.abs().T
    hors = ((y.double() - ref).abs() > TOL * borne).sum()
    return int(hors), ref


@pytest.mark.parametrize("m,n,k", [(12, 2048, 2048), (1, 512, 4096), (16, 5120, 1024), (3, 100, 200)])
def test_la_sortie_suit_la_reference(m, n, k):
    ge = _ge()
    x, t = _montage(m, n, k)
    y = ge.gemm_etroit(x, t)
    assert y.shape == (m, n) and y.dtype == x.dtype
    hors, _ = _juger(y, x, t)
    assert hors == 0, hors


def test_la_tete_rend_du_fp32():
    ge = _ge()
    x, t = _montage(12, 3000, 512)
    y = ge.gemm_etroit(x, t, sortie_fp32=True)
    assert y.dtype == torch.float32 and _juger(y, x, t)[0] == 0


def test_les_bras_qui_doivent_casser():
    ge = _ge()
    x, t = _montage(12, 1024, 1024)
    _, ref = _juger(ge.gemm_etroit(x, t), x, t)
    z0 = INT8Tensor(t.qweight, t.scales, torch.zeros_like(t.zeros), t.group_size, t.shape)
    assert _juger(ge.gemm_etroit(x, z0), x, t)[0] > 0, "zéro ignoré invisible"
    sd = INT8Tensor(t.qweight, t.scales.roll(1, dims=1), t.zeros, t.group_size, t.shape)
    assert _juger(ge.gemm_etroit(x, sd), x, t)[0] > 0, "échelle décalée d'un groupe invisible"


def test_l_entree_plus_courte_que_k_pad():
    ge = _ge()
    x, t = _montage(8, 256, 200)
    assert t.qweight.shape[1] > 200
    y = ge.gemm_etroit(x, t)
    assert _juger(y, x, t)[0] == 0


def test_la_bascule_triton_porte_sa_mesure_et_le_regime_l_imprime(monkeypatch):
    """Condition (b) de poste7 (poste7-e-c-verdict-17-09 § 2) : la constante de
    bascule est le plus petit lot où Triton bat CUDA dans la mesure de poste3
    (rejeu de graphe, pas dense Coder + tête) — pas « 8 » par intuition ;
    (c) `regime_ligne()` l'imprime."""
    from acvram import kernels
    mesure = kernels.MESURE_BASCULE_DENSE
    assert sorted(mesure) == [1, 2, 4, 8, 12]
    premier = min(b for b, (cuda, triton) in mesure.items() if triton < cuda)
    assert kernels._NARROW_TRITON_MIN_B == premier == 2, (kernels._NARROW_TRITON_MIN_B, premier)
    assert all(triton < cuda for b, (cuda, triton) in mesure.items() if b >= premier)
    assert mesure[1][1] > mesure[1][0], "à b = 1 CUDA gagne : c'est ce qui justifie le mixte"
    monkeypatch.setattr(kernels, "_NARROW_KERNEL", "mixte")
    assert kernels.narrow_choix(1) == "cuda" and kernels.narrow_choix(2) == "triton" and kernels.narrow_choix(12) == "triton"
    assert kernels.narrow_regime() == "triton≥2|cuda"
    monkeypatch.setattr(kernels, "_NARROW_KERNEL", "tete")
    assert kernels.narrow_choix(1, sortie_fp32=True) == "triton" and kernels.narrow_choix(12) == "cuda"
    monkeypatch.setattr(kernels, "_NARROW_KERNEL", "cuda")
    assert kernels.narrow_choix(12) == "cuda" and kernels.narrow_regime() == "cuda"


def test_la_derniere_tranche_ne_lit_pas_au_dela_des_echelles():
    """poste3 b34a1bb : accès mémoire illégal dans le moteur (b = 12, capture
    de graphe) — NG groupes non multiple de la taille de tranche, la dernière
    tranche itérait g ≥ NG et lisait échelle et zéro au-delà du tenseur.
    L'interpréteur ne borne pas les lectures : on place les échelles et les
    zéros en TÊTE d'un tampon dont la suite est une SENTINELLE (NaN / 255),
    on force une découpe qui déborde (NG = 16, 6 tranches de 3) — une lecture
    au-delà rend une sortie non finie ou fausse (vérifié : la version d'avant
    le correctif échoue ici). Puis la garde sur la forme des échelles."""
    ge = _ge()
    x, t = _montage(12, 256, 2048)
    N, ng = t.scales.shape
    assert ng == 16
    base_s = torch.full((N * ng + 4096,), float("nan"), dtype=t.scales.dtype, device=t.scales.device)
    base_s[:N * ng] = t.scales.reshape(-1)
    base_z = torch.full((N * ng + 4096,), 255, dtype=t.zeros.dtype, device=t.zeros.device)
    base_z[:N * ng] = t.zeros.reshape(-1)
    piege = INT8Tensor(t.qweight, base_s[:N * ng].view(N, ng), base_z[:N * ng].view(N, ng), t.group_size, t.shape)
    orig = ge._programmes
    ge._programmes = lambda device: 12          # voulu = ceil(24 / 4) = 6 tranches, gpt 3 : 6 × 3 = 18 > 16
    try:
        y = ge.gemm_etroit(x, piege)
    finally:
        ge._programmes = orig
    assert torch.isfinite(y.float()).all(), "la dernière tranche a lu la sentinelle au-delà des échelles"
    assert _juger(y, x, t)[0] == 0
    mauvais = INT8Tensor(t.qweight, t.scales[:, :-1].contiguous(), t.zeros, t.group_size, t.shape)
    with pytest.raises(AssertionError):
        ge.gemm_etroit(x, mauvais)


def _dans_un_piege(t_src, remplissage, marge=4096):
    """Copie ``t_src`` en tête d'un tampon plat dont la suite vaut
    ``remplissage`` : toute lecture au-delà du tenseur ramène la sentinelle."""
    base = torch.full((t_src.numel() + marge,), remplissage, dtype=t_src.dtype, device=t_src.device)
    base[:t_src.numel()] = t_src.reshape(-1)
    return base[:t_src.numel()].view(t_src.shape)


@pytest.mark.parametrize("godet", [1, 2, 8, 16])
def test_le_noyau_ne_lit_rien_au_dela_du_godet_ni_des_poids(godet):
    """Condition (1) de poste7 (poste7-e-c-verdict, remesure b34a1bb, REGLES § 3) :
    le noyau se valide au RÉGIME DU MOTEUR — le lot est le godet
    (bucket_batch : 1, 2, 8, 16) avec ses lignes fantômes, pas le lot exact
    d'un banc. Ici x (godet entier, fantômes = lignes nulles), qw, échelles et
    zéros sont chacun en tête d'un tampon à sentinelle (NaN ou 255), N = 200
    (pas multiple de la tuile 64), K < k_pad, NG = 16 découpé en tranches qui
    débordent : toute lecture hors bornes rend une sortie non finie ou fausse."""
    ge = _ge()
    x, t = _montage(godet, 200, 2000)               # k_pad 2048 > K 2000, ng 16
    reel = max(1, godet - godet // 4)                # les dernières lignes du godet sont des fantômes
    x[reel:] = 0
    x = _dans_un_piege(x, float("nan"))
    piege = INT8Tensor(_dans_un_piege(t.qweight, 255), _dans_un_piege(t.scales, float("nan")),
                       _dans_un_piege(t.zeros, 255), t.group_size, t.shape)
    orig = ge._programmes
    ge._programmes = lambda device: 12
    try:
        y = ge.gemm_etroit(x, piege)
        y32 = ge.gemm_etroit(x, piege, sortie_fp32=True)
    finally:
        ge._programmes = orig
    assert y.shape == (godet, 200) and torch.isfinite(y.float()).all() and torch.isfinite(y32).all()
    assert _juger(y, x, t)[0] == 0 and _juger(y32, x, t)[0] == 0
    assert not y[reel:].float().any(), "une ligne fantôme (x nulle) doit rendre zéro"


# --- C15 niveau 3, item 3 : un seul nœud (`_etroit_reduit_kernel`) ----------

def _ulp_max(a, b):
    """Écart max de b en ulp du dtype de a (bf16 : 2⁻⁷ relatif, fp16 : 2⁻¹⁰, fp32 : 2⁻²³)."""
    a32, b32 = a.float(), b.float()
    mant = {torch.bfloat16: 7, torch.float16: 10, torch.float32: 23}[a.dtype]
    ulp = torch.where(a32 != 0, 2.0 ** (torch.floor(torch.log2(a32.abs().clamp_min(1e-30))) - mant),
                      torch.full_like(a32, 2.0 ** -(126 + mant)))
    return float(((a32 - b32).abs() / ulp).max())


@pytest.mark.parametrize("m,n,k,programmes", [(12, 6144, 2048, 170), (12, 2048, 4096, 170), (1, 512, 4096, 170),
                                              (12, 3000, 512, 170), (12, 3000, 512, 4), (5, 200, 2000, 12)])
def test_c15_un_noeud_egale_zeros_noyau_somme_cast(m, n, k, programmes):
    """`compact=True` contre le témoin (torch.zeros + noyau + y.sum(0) + cast) :
    seul l'ordre de la somme fp32 des tranches peut différer — sortie 16 bits
    à ≤ 1 ulp par valeur (au bit sur ces formes), sortie fp32 à 2⁻²⁰ × Σ|x·w|
    (le bruit d'accumulation fp32, qui devient des dizaines d'ulp fp32 là où
    la somme s'annule — un ulp fp32 n'est pas un critère à ces valeurs) ;
    au bit quand une seule tranche ; formes de Coder b=12 (q/k/v N=6144
    K=2048 ; o N=2048 K=4096), b=1, tête fp32 (1 tranche), et la découpe qui
    déborde (NG 16, 6 × 3)."""
    ge = _ge()
    x, t = _montage(m, n, k)
    orig = ge._programmes
    ge._programmes = lambda device: programmes
    try:
        for fp32 in (False, True):
            temoin = ge.gemm_etroit(x, t, sortie_fp32=fp32)
            un = ge.gemm_etroit(x, t, sortie_fp32=fp32, compact=True)
            assert un.dtype == temoin.dtype and un.shape == temoin.shape
            if fp32:
                w = _dequantize_int8(t, torch.float64)
                borne = (x.double().abs() @ w.abs().T).clamp_min(1e-30)
                assert float(((temoin.double() - un.double()).abs() / borne).max()) <= 2 ** -20
            else:
                assert _ulp_max(temoin, un) <= 1.0, _ulp_max(temoin, un)
            assert _juger(un, x, t)[0] == 0
    finally:
        ge._programmes = orig
    ng = t.qweight.shape[1] // t.group_size
    tuiles_n = -(-n // ge.BN)
    if min(ng, -(-2 * programmes // tuiles_n)) == 1:
        assert torch.equal(temoin, un), "une tranche : au bit"
    cnt = ge._COMPTEURS[(tuiles_n, str(x.device))]
    assert not cnt.any(), "compteurs remis à zéro par le dernier programme"


@pytest.mark.a_sec
def test_c15_un_noeud_le_compteur_porte_la_reduction():
    """Le bras qui doit casser : un compteur qui ne repart pas de zéro fait
    réduire un programme qui n'est pas le dernier — la tuile est fausse
    (hors 2⁻⁸ de la référence) ; remis à zéro, elle redevient juste."""
    ge = _ge()
    x, t = _montage(12, 256, 2048)                      # 4 tuiles N, 16 groupes
    orig = ge._programmes
    ge._programmes = lambda device: 170                 # voulu = 85 → 16 tranches de 1 groupe
    try:
        ref = ge.gemm_etroit(x, t)
        cnt = ge._compteur(4, x.device)
        cnt[2] = 1                                      # tuile 2 faussée
        faux = ge.gemm_etroit(x, t, compact=True)
        assert _juger(faux[:, 128:192], x, INT8Tensor(t.qweight[128:192], t.scales[128:192], t.zeros[128:192],
                                                      t.group_size, (64, t.shape[1])))[0] > 0
        assert _ulp_max(ref[:, :128], faux[:, :128]) <= 1.0
        cnt.zero_()
        bon = ge.gemm_etroit(x, t, compact=True)
        assert _ulp_max(ref, bon) <= 1.0
    finally:
        ge._programmes = orig


# --- pièce 195 : noyau K entier par canal, opt-in ACVRAM_ETROIT_CANAL (hors bit, jamais au défaut) ---

def _montage_canal(m, n, k, graine=0):
    """INT8 symétrique par canal (échelle [N, 1], zéro 128, groupe = K), comme les tenseurs de l'alias mixte-i8c."""
    torch.manual_seed(graine)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.randint(0, 256, (n, k), dtype=torch.uint8, device=dev)
    s = (torch.rand(n, 1) * 0.01 + 0.005).to(torch.float16).to(dev)
    t = INT8Tensor(q, s, torch.full((n, 1), 128, dtype=torch.uint8, device=dev), k, (n, k))
    x = (torch.randn(m, k) * torch.logspace(-1, 1, m).unsqueeze(1)).to(dev)
    return x.to(torch.bfloat16 if dev == "cuda" else torch.float16), t


@pytest.mark.parametrize("m,n,k,geo", [(8, 5120, 6144, (32, 512, 8, 3)), (16, 1000, 640, (16, 128, 4, 2)), (2, 100, 200, (64, 256, 4, 4))])
def test_195_le_canal_k_entier_suit_la_reference(m, n, k, geo):
    ge = _ge()
    x, t = _montage_canal(m, n, k)
    assert ge.canal_eligible(t)
    y = ge.gemm_canal(x, t, geo)
    assert y.shape == (m, n) and y.dtype == x.dtype
    w = (t.qweight.double() - 128) * t.scales.double()
    ref = x.double() @ w.T
    borne = x.double().abs() @ w.abs().T
    assert int(((y.double() - ref).abs() > TOL * borne).sum()) == 0


def test_195_le_canal_refuse_ce_qui_n_est_pas_par_canal():
    ge = _ge()
    _, t = _montage(4, 256, 512)                      # groupes de 128 : pas par canal
    assert not ge.canal_eligible(t)
    _, t = _montage_canal(4, 256, 512)
    t.zeros[3, 0] = 127                               # un zéro ≠ 128 : refusé aussi
    assert not ge.canal_eligible(t)


def test_195_le_canal_est_le_defaut_et_0_le_temoin_nomme(monkeypatch):
    """25/09, décision déléguée par l'utilisateur : le K entier par canal est SERVI PAR DÉFAUT ; ce test casse si le
    défaut revient à 0 (code, table des variables, ligne de régime). 0 reste le témoin nommé, imprimé `+canal(temoin)`."""
    ge = _ge()
    monkeypatch.delenv("ACVRAM_ETROIT_CANAL", raising=False)
    monkeypatch.delenv("ACVRAM_ETROITES_FORME", raising=False)
    ge.regler_forme(None)
    assert ge.CANAL_DEFAUT == "1"
    assert ge.canal_actif() and ge.etroites_texte() == "serie+canal(table)"
    assert ge.geometrie_canal(5120, 6144) == ge.GEOMETRIE_CANAL[(5120, 6144)]
    assert ge.geometrie_canal(1, 2) is None                              # forme non mesurée : reste sur le noyau d'avant
    from acvram import regime
    assert any(v.nom == "ETROIT_CANAL" and v.defaut == "1" for v in regime.VARIABLES)
    monkeypatch.setenv("ACVRAM_ETROIT_CANAL", "0")
    assert not ge.canal_actif() and ge.etroites_texte() == "serie+canal(temoin)"
    monkeypatch.setenv("ACVRAM_ETROIT_CANAL", "32,512,8,3")
    assert ge.geometrie_canal(5120, 6144) == (32, 512, 8, 3) and ge.etroites_texte() == "serie+canal(32x512x8x3)"


@pytest.mark.parametrize("m,n,k,geo", [(8, 300, 640, (32, 128, 4, 2)), (2, 100, 200, (16, 128, 4, 2))])
def test_195_le_zero_point_et_k_entier_sont_juges(m, n, k, geo):
    """Pièce 198 (poste2) : la tolérance de `_juger` laissait passer un zéro à ±1 (Σx ≈ 0 sur un x centré) et un K tronqué
    sur la plus petite forme. Ici x a une moyenne non nulle et le juge est serré ; les deux bras cassants (zéro 127, K/2)
    doivent être REFUSÉS par le même juge, sinon le test ne juge rien."""
    ge = _ge()
    x, t = _montage_canal(m, n, k)
    x = (x.float().abs() * 0.5 + 0.5).to(x.dtype)                 # Σx ≫ 0 : un zéro faux pèse s·Σx par colonne
    y = ge.gemm_canal(x, t, geo).double()
    q, s = t.qweight.double(), t.scales.double()
    ref = x.double() @ ((q - 128) * s).T
    tol = 2 ** -7 * ref.abs().max()
    assert (y - ref).abs().max() <= tol
    faux_zero = x.double() @ ((q - 127) * s).T
    assert (y - faux_zero).abs().max() > tol, "le juge ne voit pas un zéro à ±1"
    k_tronque = x[:, : k // 2].double() @ ((q[:, : k // 2] - 128) * s).T
    assert (y - k_tronque).abs().max() > tol, "le juge ne voit pas K tronqué de moitié"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="chemin servi : extension CUDA")
def test_195_le_defaut_est_au_bit_du_chemin_servi_et_l_opt_in_prend(monkeypatch):
    """(a) de chef (198), retourné au défaut 1 (25/09) : à 0 (témoin nommé), `int8_matmul` rend EXACTEMENT la sortie
    du noyau d'avant (vue g128, `gemm_etroit` compact) et ne compte aucun `etroit_canal` ; sans variable (défaut) et à 1,
    le chemin canal est pris (compteur) sur une forme de la table et sa sortie reste dans ± 2⁻⁷ de la référence fp64."""
    from acvram import kernels
    ge = _ge()
    if kernels.get_extension() is None:
        pytest.skip("extension CUDA absente")
    x, t = _montage_canal(8, 5120, 6144, graine=3)                # forme de GEOMETRIE_CANAL (o_proj / out GDN)
    temoin = ge.gemm_etroit(x, kernels.vue_g128(t), compact=kernels.glue_compact("etroit"))
    ref = x.double() @ ((t.qweight.double() - 128) * t.scales.double()).T
    for valeur in (None, "1"):
        if valeur is None:
            monkeypatch.delenv("ACVRAM_ETROIT_CANAL", raising=False)
        else:
            monkeypatch.setenv("ACVRAM_ETROIT_CANAL", valeur)
        kernels.CHEMINS_INT8.clear()
        y1 = kernels.int8_matmul(x, t)
        assert kernels.CHEMINS_INT8["etroit_canal"] == 1 and kernels.CHEMINS_INT8["etroit_triton"] == 0, valeur
        assert (y1.double() - ref).abs().max() <= 2 ** -7 * ref.abs().max()
    monkeypatch.setenv("ACVRAM_ETROIT_CANAL", "0")
    kernels.CHEMINS_INT8.clear()
    assert torch.equal(kernels.int8_matmul(x, t), temoin)         # le témoin nommé : au bit du noyau d'avant
    assert kernels.CHEMINS_INT8["etroit_canal"] == 0 and kernels.CHEMINS_INT8["etroit_triton"] == 1
