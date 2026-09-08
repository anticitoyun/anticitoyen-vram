# SPDX-FileCopyrightText: 2026 Anticitoyen
# SPDX-License-Identifier: Apache-2.0
"""Équivalence du format q3n : référence processeur, puis noyau fusionné.

Ce fichier est écrit **avant** le noyau GEMV et ne doit pas bouger quand il
arrivera. Deux parties, deux régimes :

- Partie A, la référence. Elle tourne partout et doit passer aujourd'hui.
  Elle fixe la numérique que le noyau devra égaler, et elle contient les deux
  tests qui auraient attrapé les défauts déjà commis sur ce format : une table
  asymétrique, et un rapport signal/bruit qui s'améliore quand le bloc grandit.

- Partie B, le noyau. Elle **échoue** tant que `acvram.kernels.q3n_gemm`
  n'existe pas ou ne rend pas la bonne valeur. Elle ne se saute que s'il n'y a
  aucune carte : un noyau absent sur une machine qui a un GPU est un échec,
  pas une dispense.

La cible d'équivalence est ``linear(x, dequantize_q3n(t, float32))`` en fp32,
jamais ``q3n_gemv`` : ce dernier arrondit sa sortie en bf16, et comparer un
noyau à une référence plus bruitée que lui ne prouve rien.
"""

from __future__ import annotations

import math

import pytest
import torch

from acvram.quant.q3n import (
    TABLE_Q3N, Q3NTensor, quantize_q3n, dequantize_q3n, q3n_gemv,
    empaqueter_q3, depaqueter_q3,
)


# --------------------------------------------------------------------------
# Outils communs
# --------------------------------------------------------------------------

def _snr_db(reference: torch.Tensor, obtenu: torch.Tensor) -> float:
    """Rapport signal sur bruit en décibels, sur la reconstruction."""
    r = reference.to(torch.float64)
    e = obtenu.to(torch.float64) - r
    return 10.0 * math.log10(float(r.pow(2).sum() / e.pow(2).sum().clamp(min=1e-30)))


def _reference_gemv(x: torch.Tensor, t: Q3NTensor) -> torch.Tensor:
    """La seule cible d'équivalence légitime : poids déquantifiés, tout en fp32."""
    w = dequantize_q3n(t, torch.float32)
    return torch.nn.functional.linear(x.to(torch.float32), w)


def _poids(sortie: int, entree: int, graine: int = 0,
           loi: str = "normal") -> torch.Tensor:
    g = torch.Generator().manual_seed(graine)
    if loi == "normal":
        return torch.randn(sortie, entree, generator=g)
    if loi.startswith("queue"):
        # Queue lourde : gaussienne modulée par une lognormale de paramètre
        # sigma. La spécification a été écrite sur sigma = 0,5, une queue
        # MODÉRÉE — le document le dit maintenant explicitement, parce que le
        # rapport signal/bruit s'effondre quand sigma monte : 13,6 dB à 0,5,
        # 9,5 à 1,0, 6,7 à 1,5.
        sigma = float(loi.split(":")[1]) if ":" in loi else 0.5
        w = torch.randn(sortie, entree, generator=g)
        return w * torch.exp(torch.randn(sortie, entree, generator=g) * sigma)
    raise ValueError(loi)


def _quantifier_entier3(w: torch.Tensor, block: int = 32) -> torch.Tensor:
    """Entiers 3 bits, niveaux également espacés, même échelle par bloc.

    Sert de témoin : c'est le format que les quantiles doivent battre, et
    c'est cette comparaison — non un seuil en décibels — qui justifie d'avoir
    choisi une table non uniforme.
    """
    sortie, entree = w.shape
    blocs = w.to(torch.float32).reshape(sortie, entree // block, block)
    echelle = blocs.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    q = torch.round(blocs / echelle * 3.5 + 3.5).clamp(0, 7)
    return ((q - 3.5) / 3.5 * echelle).reshape(sortie, entree)


# ==========================================================================
# Partie A — la référence processeur
# ==========================================================================

def test_table_symetrique():
    """Quatre niveaux positifs et leurs opposés, maximum à 1,0.

    Une table à zéro exact — la forme de NF4 — a quatre niveaux négatifs pour
    trois positifs, le plus grand valant 0,675. Avec une échelle prise sur
    max(|w|), elle écrête d'un tiers tout poids positif proche du maximum du
    bloc. Ce test refuse cette forme par construction.
    """
    assert len(TABLE_Q3N) == 8
    assert TABLE_Q3N == tuple(sorted(TABLE_Q3N)), "table non triée"
    for i in range(4):
        assert TABLE_Q3N[i] == pytest.approx(-TABLE_Q3N[7 - i], abs=1e-9), \
            f"niveaux {i} et {7 - i} non opposés"
    assert max(TABLE_Q3N) == pytest.approx(1.0), \
        "le plus grand niveau doit valoir l'échelle, sinon le maximum est écrêté"
    # L'hypothèse « pas de zéro exact » de la spécification d'origine a été
    # réfutée par la mesure du 8/09 (+1,73 dB pour sept niveaux symétriques
    # avec zéro) : la table du manifeste PEUT porter un zéro. La contrainte
    # qui reste est structurelle : huit entrées, symétrie, bornes ±1 — et
    # c'est valider_table_q3n qui la fait respecter sur toute table lue.
    from acvram.quant.q3n import (TABLE_Q3N_LLOYD_CODER_NEXT,
                                  valider_table_q3n)
    assert valider_table_q3n(TABLE_Q3N) == TABLE_Q3N
    assert valider_table_q3n(TABLE_Q3N_LLOYD_CODER_NEXT) \
        == TABLE_Q3N_LLOYD_CODER_NEXT
    with pytest.raises(ValueError):
        valider_table_q3n((-1, -0.5, -0.3, 0, 0.3, 0.5, 1))       # 7 entrées
    with pytest.raises(ValueError):
        valider_table_q3n((-1, -0.5, 0.3, -0.3, 0, 0.5, 1, 1))    # non triée
    with pytest.raises(ValueError):
        valider_table_q3n((-0.9, -0.5, -0.3, 0, 0.3, 0.5, 0.9, 0.9))  # bornes
    # L'asymétrie est ADMISE depuis l'arbitrage du 8/09 au soir : bornes ±1
    # aux deux extrémités, c'est elles qui interdisent l'écrêtage.
    assert valider_table_q3n((-1, -0.724, -0.4796, -0.2379,
                              0, 0.2488, 0.569, 1)) is not None
    with pytest.raises(ValueError):
        valider_table_q3n((-1, -0.5, -0.5, 0, 0.2, 0.5, 0.9, 1))  # plateau interne
    with pytest.raises(ValueError):
        valider_table_q3n((-1, -0.5, 0, 0, 0.2, 0.5, 0.9, 1))     # deux zéros


def test_erreur_symetrique_entre_signes():
    """L'erreur ne doit pas dépendre du signe du poids.

    Le test qui manquait. Une table asymétrique donne un bruit nettement plus
    fort sur un signe que sur l'autre, et ce déséquilibre se voit directement,
    sans avoir besoin d'une anomalie pour le révéler.
    """
    w = _poids(64, 512, graine=1)
    r = dequantize_q3n(quantize_q3n(w), torch.float32)
    err = (r - w).pow(2)
    pos = err[w > 0].mean().item()
    neg = err[w < 0].mean().item()
    assert pos == pytest.approx(neg, rel=0.15), \
        f"erreur asymétrique : {pos:.3e} sur les positifs, {neg:.3e} sur les négatifs"


@pytest.mark.parametrize("loi", ["normal", "queue:0.5", "queue:1.0", "queue:1.5"])
def test_bloc_petit_toujours_meilleur(loi):
    """Un bloc plus petit ne peut pas perdre : il partage l'échelle entre moins
    de poids. L'inverse est impossible, et c'est cette impossibilité qui a
    révélé la table asymétrique — le bloc 32 battait le bloc 16.
    """
    w = _poids(32, 2048, graine=2, loi=loi)
    snr16 = _snr_db(w, dequantize_q3n(quantize_q3n(w, block=16), torch.float32))
    snr32 = _snr_db(w, dequantize_q3n(quantize_q3n(w, block=32), torch.float32))
    assert snr16 >= snr32 - 1e-6, \
        f"impossible : bloc 16 {snr16:.2f} dB < bloc 32 {snr32:.2f} dB"


@pytest.mark.parametrize("loi,plancher", [("normal", 14.5), ("queue:0.5", 13.0)])
def test_snr_conforme_a_la_specification(loi, plancher):
    """Les deux chiffres annoncés dans docs/FORMAT-3BITS.md, avec de la marge.

    Spécification, bloc de 32 : 15,00 dB en gaussien, 13,57 dB sur la queue
    lourde — cette dernière étant une lognormale de sigma 0,5, ce que le
    document nomme désormais. Le plancher absorbe le tirage, pas une
    régression : il est posé un demi-décibel sous la valeur annoncée.
    """
    w = _poids(64, 4096, graine=3, loi=loi)
    snr = _snr_db(w, dequantize_q3n(quantize_q3n(w, block=32), torch.float32))
    assert snr >= plancher, f"{loi} : {snr:.2f} dB, plancher {plancher}"


@pytest.mark.parametrize("loi", ["normal", "queue:0.5", "queue:1.0", "queue:1.5"])
@pytest.mark.parametrize("block", [16, 32])
def test_quantiles_battent_les_entiers(loi, block):
    """La raison d'être du format, sous une forme qui ne se règle pas.

    Un seuil en décibels dépend de la loi tirée ; cette comparaison-ci non.
    Si des niveaux également espacés faisaient aussi bien, la table de
    quantiles et sa lecture indirecte ne se justifieraient pas.
    """
    w = _poids(32, 2048, graine=18, loi=loi)
    q = _snr_db(w, dequantize_q3n(quantize_q3n(w, block=block), torch.float32))
    e = _snr_db(w, _quantifier_entier3(w, block=block))
    assert q > e, f"{loi} bloc {block} : quantiles {q:.2f} dB, entiers {e:.2f} dB"


@pytest.mark.parametrize("n", [8, 16, 32, 64, 4096])
def test_empaquetage_aller_retour_exact(n):
    """Trois octets pour huit valeurs, sans perte d'un seul bit."""
    g = torch.Generator().manual_seed(4)
    q = torch.randint(0, 8, (7, n), generator=g, dtype=torch.uint8)
    assert torch.equal(depaqueter_q3(empaqueter_q3(q), n).to(torch.uint8), q)


@pytest.mark.parametrize("block,attendu", [(32, 3.25), (16, 3.50)])
def test_bits_par_poids(block, attendu):
    """Le compte doit tomber juste, l'échelle globale amortie sur un vrai tenseur."""
    t = quantize_q3n(_poids(256, 4096, graine=5), block=block)
    assert t.bits_per_weight == pytest.approx(attendu, abs=0.01)


def test_bloc_de_tres_faible_amplitude():
    """Un bloc mille fois plus faible que le maximum global ne doit pas exploser.

    L'échelle de bloc est stockée en FP8 e4m3, dont le plus petit dénormal vaut
    2⁻⁹. Un bloc dont l'amplitude est sous g/512 voit son échelle sous-floter à
    zéro. Une matrice réelle en contient — un canal mort, une ligne peu
    sollicitée — et le résultat ne doit être ni NaN ni infini.
    """
    w = torch.full((4, 64), 1e-6)
    w[0, 0] = 1.0                      # impose un maximum global mille fois plus grand
    r = dequantize_q3n(quantize_q3n(w, block=32), torch.float32)
    assert torch.isfinite(r).all(), "NaN ou infini sur un bloc de faible amplitude"
    assert r.abs().max() <= w.abs().max() * 1.5, \
        "un bloc écrasé par l'échelle globale a produit une valeur hors bornes"


def test_poids_tous_nuls():
    """Zéro en entrée, zéro en sortie, sans division par une échelle nulle."""
    w = torch.zeros(4, 64)
    r = dequantize_q3n(quantize_q3n(w), torch.float32)
    assert torch.isfinite(r).all()
    assert r.abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_gemv_reference_egale_la_dequantification():
    """Le chemin de référence ne doit rien ajouter à la déquantification."""
    t = quantize_q3n(_poids(48, 256, graine=6))
    x = torch.randn(1, 256, generator=torch.Generator().manual_seed(7))
    attendu = _reference_gemv(x, t)
    obtenu = q3n_gemv(x, t).to(torch.float32)
    assert _snr_db(attendu, obtenu) > 30.0


@pytest.mark.parametrize("forme", [(1, 31), (1, 100), (3, 2, 64)])
def test_gemv_refuse_proprement_une_forme_inadaptee(forme):
    """Contrat de repli, identique aux autres chemins : ``None``, pas d'exception.

    Le planificateur essaie un chemin puis retombe sur le suivant ; une
    exception à cet endroit fait tomber la requête entière au lieu de replier.
    """
    t = quantize_q3n(_poids(16, 64, graine=8))
    x = torch.randn(*forme)
    assert q3n_gemv(x, t) is None


# ==========================================================================
# Partie B — le noyau fusionné
# ==========================================================================
#
# Contrat attendu, en miroir de acvram/kernels/fp4_gemm.py :
#
#     from acvram.kernels.q3n_gemm import q3n_gemv_available, q3n_gemv_fused
#     q3n_gemv_available(device=None) -> bool
#     q3n_gemv_fused(x, t) -> torch.Tensor | None
#
# q3n ne décode que des bits et accumule en fp32 : il n'a **aucun** plancher de
# capacité, contrairement à NVFP4 qui exige sm_100 pour ses tensor cores. Un
# garde recopié depuis fp4_gemm éteindrait le noyau sur sm_86 sans raison — et
# c'est la carte où llama.cpp nous bat.

SANS_CARTE = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="aucune carte : il n'y a pas de noyau à comparer")


def _noyau():
    """Importe le noyau. Absent, c'est un échec, jamais un saut."""
    try:
        from acvram.kernels import q3n_gemm
    except ImportError as e:                              # noqa: BLE001
        pytest.fail(f"acvram.kernels.q3n_gemm absent : {e}")
    for nom in ("q3n_gemv_available", "q3n_gemv_fused"):
        assert hasattr(q3n_gemm, nom), f"q3n_gemm n'expose pas {nom}"
    return q3n_gemm


def _sur_carte(w: torch.Tensor, block: int, dev) -> Q3NTensor:
    return quantize_q3n(w, block=block).to(dev)


def _compare(noyau, x, t, tol_db: float = 40.0):
    obtenu = noyau.q3n_gemv_fused(x, t)
    assert obtenu is not None, "le noyau a refusé une forme qu'il doit accepter"
    attendu = _reference_gemv(x, t)
    snr = _snr_db(attendu, obtenu.to(torch.float32))
    assert snr >= tol_db, f"noyau à {snr:.1f} dB de la référence (seuil {tol_db})"
    return obtenu


@SANS_CARTE
def test_noyau_existe_et_s_annonce():
    noyau = _noyau()
    assert noyau.q3n_gemv_available(torch.device("cuda:0")) is True


@SANS_CARTE
def test_noyau_disponible_sur_toutes_les_cartes():
    """Pas de plancher de capacité : q3n décode des bits, il n'a pas besoin de
    tensor cores FP4. Un garde `capacite < (10, 0)` recopié de fp4_gemm est un
    défaut, pas une prudence.
    """
    noyau = _noyau()
    for i in range(torch.cuda.device_count()):
        d = torch.device(f"cuda:{i}")
        cc = torch.cuda.get_device_capability(d)
        assert noyau.q3n_gemv_available(d) is True, \
            f"noyau éteint sur {torch.cuda.get_device_name(d)} (sm_{cc[0]}{cc[1]})"


@SANS_CARTE
@pytest.mark.parametrize("sortie", [1, 7, 33, 127, 256])
@pytest.mark.parametrize("entree", [32, 64, 96, 160, 4096])
def test_noyau_equivaut_a_la_reference(sortie, entree):
    """Queues de tuile dans les deux dimensions.

    96 et 160 sont multiples de 32 mais pas de 64 ni de 128 : ils prennent en
    défaut une tuile dimensionnée sur une puissance de deux. 1, 7, 33 et 127
    laissent une tuile de sortie partiellement vide.
    """
    dev = torch.device("cuda:0")
    noyau = _noyau()
    t = _sur_carte(_poids(sortie, entree, graine=sortie * 31 + entree), 32, dev)
    x = torch.randn(1, entree, device=dev,
                    generator=torch.Generator(device=dev).manual_seed(9))
    _compare(noyau, x, t)


@SANS_CARTE
@pytest.mark.parametrize("block", [16, 32])
def test_noyau_gere_les_deux_tailles_de_bloc(block):
    """La taille de bloc est une constante, jamais une valeur en dur.

    Le choix entre 16 et 32 n'est pas tranché : 1,48 dB sur la queue lourde
    contre 0,25 bit. Un noyau qui ne sait faire que 32 ferme ce choix.
    """
    dev = torch.device("cuda:0")
    noyau = _noyau()
    t = _sur_carte(_poids(64, 512, graine=10), block, dev)
    x = torch.randn(1, 512, device=dev,
                    generator=torch.Generator(device=dev).manual_seed(11))
    _compare(noyau, x, t)


@SANS_CARTE
def test_noyau_accepte_un_vecteur_a_une_dimension():
    dev = torch.device("cuda:0")
    noyau = _noyau()
    t = _sur_carte(_poids(32, 256, graine=12), 32, dev)
    x = torch.randn(256, device=dev)
    obtenu = noyau.q3n_gemv_fused(x, t)
    assert obtenu is not None
    assert obtenu.reshape(-1).shape == (32,)


@SANS_CARTE
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_noyau_accepte_les_trois_types_d_entree(dtype):
    """Le seuil se relâche avec le type d'entrée, pas avec le noyau : bf16 n'a
    que huit bits de mantisse, et la référence lit le même x.
    """
    dev = torch.device("cuda:0")
    noyau = _noyau()
    t = _sur_carte(_poids(64, 1024, graine=13), 32, dev)
    x = torch.randn(1, 1024, device=dev).to(dtype)
    _compare(noyau, x, t, tol_db=25.0 if dtype is torch.bfloat16 else 35.0)


@SANS_CARTE
def test_noyau_sur_une_entree_non_contigue():
    """Une tranche d'un tenseur plus large ; le noyau doit lire les bons octets
    ou refuser, jamais lire à côté.
    """
    dev = torch.device("cuda:0")
    noyau = _noyau()
    t = _sur_carte(_poids(32, 512, graine=14), 32, dev)
    large = torch.randn(1, 1024, device=dev)
    x = large[:, ::2]
    assert not x.is_contiguous()
    obtenu = noyau.q3n_gemv_fused(x, t)
    if obtenu is None:
        pytest.skip("le noyau refuse une entrée non contiguë : contrat tenu")
    snr = _snr_db(_reference_gemv(x, t), obtenu.to(torch.float32))
    assert snr >= 35.0, f"entrée non contiguë lue de travers : {snr:.1f} dB"


@SANS_CARTE
def test_noyau_deterministe():
    """Deux appels identiques, même résultat bit à bit. Sans quoi une mesure
    d'énergie ou de perplexité n'est pas reproductible.
    """
    dev = torch.device("cuda:0")
    noyau = _noyau()
    t = _sur_carte(_poids(128, 1024, graine=15), 32, dev)
    x = torch.randn(1, 1024, device=dev)
    a = noyau.q3n_gemv_fused(x, t)
    b = noyau.q3n_gemv_fused(x, t)
    assert torch.equal(a, b)


@SANS_CARTE
def test_noyau_refuse_ou_traite_un_lot():
    """GEMV est écrit pour un jeton. Pour plusieurs, deux réponses sont
    acceptables — le bon résultat, ou ``None`` pour laisser replier. La
    troisième, un résultat faux, est celle que ce test interdit.
    """
    dev = torch.device("cuda:0")
    noyau = _noyau()
    t = _sur_carte(_poids(64, 512, graine=16), 32, dev)
    x = torch.randn(8, 512, device=dev)
    obtenu = noyau.q3n_gemv_fused(x, t)
    if obtenu is None:
        return
    assert obtenu.shape == (8, 64)
    assert _snr_db(_reference_gemv(x, t), obtenu.to(torch.float32)) >= 35.0


@SANS_CARTE
@pytest.mark.parametrize("cas", ["nuls", "pic", "faible"])
def test_noyau_sur_des_poids_degeneres(cas):
    """Trois matrices qu'un modèle réel produit et qu'un banc synthétique rate."""
    dev = torch.device("cuda:0")
    noyau = _noyau()
    if cas == "nuls":
        w = torch.zeros(32, 256)
    elif cas == "pic":
        w = torch.randn(32, 256, generator=torch.Generator().manual_seed(17))
        w[0, 0] = 1e4                       # une valeur écrase l'échelle globale
    else:
        w = torch.full((32, 256), 1e-6)
        w[0, 0] = 1.0                       # blocs sous le dénormal e4m3
    t = _sur_carte(w, 32, dev)
    x = torch.randn(1, 256, device=dev)
    obtenu = noyau.q3n_gemv_fused(x, t)
    assert obtenu is not None
    assert torch.isfinite(obtenu).all(), f"{cas} : NaN ou infini en sortie"
    attendu = _reference_gemv(x, t)
    ecart = (obtenu.to(torch.float32) - attendu).abs().max().item()
    echelle = max(attendu.abs().max().item(), 1e-6)
    assert ecart <= 1e-2 * echelle, f"{cas} : écart {ecart:.3e} pour {echelle:.3e}"


# --------------------------------------------------------------------------
# Le défaut que la partie B ne pouvait pas voir
# --------------------------------------------------------------------------
#
# Un noyau peut passer toute la partie B et rester inutilisable : il suffit
# qu'il synchronise. Les tests d'équivalence appellent le noyau seul, hors
# capture, et une synchronisation y est invisible. Ces deux tests-ci la
# rendent visible.

def test_echelle_globale_lue_une_seule_fois():
    """``.item()`` sur un tenseur CUDA synchronise le flux à chaque GEMV.

    Ce n'est pas une hypothèse : NVFP4 a rencontré le même piège sur ce dépôt,
    et ce seul appel pesait 62 % du temps de décodage au profil. Q3NTensor
    doit exposer le même accès mémorisé côté hôte.
    """
    t = quantize_q3n(_poids(16, 64, graine=19))
    assert hasattr(t, "global_scale_float"), \
        "Q3NTensor doit exposer global_scale_float(), comme NVFP4Tensor"
    v = t.global_scale_float()
    assert isinstance(v, float)
    assert v == pytest.approx(float(t.global_scale.item()))

    # Le second appel ne doit plus toucher au tenseur : on le remplace par un
    # objet qui hurle si on le lit.
    class Piege:
        def item(self):
            raise AssertionError("global_scale relu : la valeur n'est pas mémorisée")
    t.global_scale = Piege()
    assert t.global_scale_float() == pytest.approx(v)


@SANS_CARTE
def test_noyau_capturable_dans_un_graphe_cuda():
    """acvram capture ses graphes ; un noyau qui synchronise n'y entre pas.

    C'est le test qui manquait à la partie B : l'équivalence numérique ne dit
    rien de la capturabilité, et un noyau juste mais non capturable retire au
    décodage le gain pour lequel les graphes existent.
    """
    dev = torch.device("cuda:0")
    noyau = _noyau()
    t = _sur_carte(_poids(32, 256, graine=20), 32, dev)
    x = torch.randn(1, 256, device=dev)

    noyau.q3n_gemv_fused(x, t)          # réchauffe : allocations hors capture
    torch.cuda.synchronize()
    flux = torch.cuda.Stream()
    flux.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(flux):
        for _ in range(3):
            noyau.q3n_gemv_fused(x, t)
    torch.cuda.current_stream().wait_stream(flux)

    graphe = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graphe):
        sortie = noyau.q3n_gemv_fused(x, t)
    graphe.replay()
    torch.cuda.synchronize()
    snr = _snr_db(_reference_gemv(x, t), sortie.to(torch.float32))
    assert snr >= 35.0, f"graphe rejoué : {snr:.1f} dB"


def test_table_du_manifeste_traverse_tout():
    """La table Lloyd d'un modèle doit produire EXACTEMENT ses niveaux à la
    déquantification, et l'aller-retour disque (manifeste sans table = repli
    spécification) doit rester intact."""
    from acvram.quant.q3n import (TABLE_Q3N, TABLE_Q3N_LLOYD_CODER_NEXT,
                                  dequantize_q3n, quantize_q3n)
    torch.manual_seed(3)
    w = torch.randn(16, 64) * 0.02
    t = quantize_q3n(w, table=TABLE_Q3N_LLOYD_CODER_NEXT)
    assert t.table == TABLE_Q3N_LLOYD_CODER_NEXT
    deq = dequantize_q3n(t, torch.float32)
    echelle = t.block_scale.to(torch.float32) * float(t.global_scale)
    reduit = (deq.reshape(16, -1, t.block)
              / echelle.clamp(min=1e-12).unsqueeze(-1)).reshape(-1)
    niveaux = torch.tensor(sorted(set(TABLE_Q3N_LLOYD_CODER_NEXT)))
    dmin = (reduit.unsqueeze(1) - niveaux.unsqueeze(0)).abs().min(dim=1).values
    assert float(dmin[echelle.reshape(-1).repeat_interleave(t.block) > 1e-12]
                 .max()) < 1e-5, "une valeur reconstruite hors des niveaux"
    # zéro exact reconstruit exactement
    assert (deq == 0).float().mean() > 0.05
    # repli : un tenseur sans table garde la spécification
    t2 = quantize_q3n(w)
    assert t2.table == TABLE_Q3N


def test_rehydrate_transporte_la_table():
    from acvram.engine.layers import _rehydrate
    from acvram.quant.q3n import TABLE_Q3N_LLOYD_CODER_NEXT, quantize_q3n
    torch.manual_seed(4)
    t = quantize_q3n(torch.randn(8, 32), table=TABLE_Q3N_LLOYD_CODER_NEXT)
    r = _rehydrate(t, {"qweight": t.qweight.clone(),
                       "block_scale": t.block_scale.clone().view(torch.uint8),
                       "global_scale": t.global_scale.clone()})
    assert r.table == TABLE_Q3N_LLOYD_CODER_NEXT


def test_q3n_est_promouvable_par_le_filet():
    """8/09 : la reconversion « à filet égal » (snr_floor 25) a rendu un
    manifeste identique au sans-filet — q3n manquait à PROMOTE et le filet
    l'ignorait en silence. Tout format quantifié servi par le convertisseur
    doit avoir une issue de promotion."""
    from acvram.quant.convert import PROMOTE
    from acvram.quant.formats import FORMATS
    for fmt in FORMATS:
        if fmt in ("bf16", "fp16"):
            continue
        assert fmt in PROMOTE, f"{fmt} sans issue de promotion : le filet snr_floor l'ignore en silence"
    assert PROMOTE["q3n"] == "int8"
