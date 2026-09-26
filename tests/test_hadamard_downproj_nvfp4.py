"""Piste 2 de la campagne duck.ai (revue/duck-poste2-12-09.md) REFUTEE : la
rotation de Hadamard aveugle sur down_proj ne se justifie pas en NVFP4.

Bridging Gap (arXiv:2509.23202) documente des outliers massifs a l'entree
de down_proj (valeurs >1000-1400 sur Llama-2-7B) et la campagne duck.ai du
12/09 en concluait qu'une rotation de Hadamard avant quantification bloc
E2M1 devrait aider — comme elle aide deja l'INT4-AWQ (groupes de 128) en
mode `auto` de `TensorRouter.wants_hadamard`.

Mesure ici, a sec, sur le MEME motif d'outlier que le temoin existant
(`tests/test_quant.py::test_hadamard_helps_int4_on_outlier_channels`,
colonnes d'entree x8 tous les 64 canaux) : la rotation AIDE toujours
l'INT4-AWQ (+7,3 dB, confirme) mais DEGRADE legerement et de facon stable
le NVFP4 (-0,38 dB, mesure sur six graines, ecart-type < 0,02 dB). Pas
d'implementation livree : ajouter une option `hadamard_downproj_nvfp4` sur
la foi de la piste 2 aurait expedie une fonctionnalite que ce test refute
lui-meme.

A un motif plus extreme (colonnes x1000 au lieu de x8), la rotation
devient franchement destructrice pour LES DEUX formats (NVFP4 -2,2 dB,
INT4-AWQ passe de 38,9 a 0,4 dB) — un outlier isole reste local a un seul
groupe de quantification sans rotation, la rotation l'etale sur TOUS les
groupes et corrompt l'ensemble au lieu d'un seul. Ce regime n'est donc pas
retenu comme motif de test (l'ecart-type entre formats y devient
lui-meme un artefact de magnitude, pas de la structure blocs 16 contre
128 qui interesse cette piste).

Explication qualitative pour le regime modere retenu : le bloc de 16 du
NVFP4 porte deja une echelle E4M3 propre a CHAQUE groupe de 16 colonnes —
un outlier confine a 16 colonnes ne penalise QUE cette echelle locale.
Le groupe de 128 de l'INT4-AWQ est assez large pour qu'etaler un outlier
sur ses 128 colonnes reste une amelioration nette ; le bloc de 16 est deja
assez fin pour qu'etaler l'outlier sur TOUT le tenseur (la rotation
Hadamard n'est pas bloc-locale ici, `largest_pow2_divisor` prend le plus
grand diviseur de la dimension entiere) fasse PERDRE la localisation que
le format offrait deja gratuitement.

Un chemin different (DuQuant, arXiv:2406.01721 : rotation ET permutation
CIBLEES, pas une rotation aveugle sur tout le tenseur) resterait a tester
separement — ce test ne le couvre pas et ne pretend refuter que la
rotation aveugle mesuree ici.
"""
import torch

from acvram.quant.calibrate import quantize_with_calibration


# MEME MOTIF que test_quant.py::test_hadamard_helps_int4_on_outlier_channels,
# pour que les deux formats soient compares sur un regime deja valide comme
# raisonnable (pas choisi pour faire mentir la piste dans un sens ou l'autre).
def _tenseur_a_outliers(graine: int) -> torch.Tensor:
    torch.manual_seed(graine)
    w = torch.randn(256, 1024) * 0.02
    w[:, ::64] *= 8.0
    return w


def test_hadamard_degrade_legerement_le_snr_nvfp4_sur_ce_motif():
    """Contrairement a l'INT4-AWQ (voir le temoin ci-dessous), la rotation
    NUIT au NVFP4 sur le motif d'outlier deja valide."""
    w = _tenseur_a_outliers(0)
    _, _, plain = quantize_with_calibration(w, "nvfp4", None, use_hadamard=False,
                                            use_awq=False)
    _, _, rotated = quantize_with_calibration(w, "nvfp4", None, use_hadamard=True,
                                              use_awq=False)
    assert rotated["out_snr_db"] < plain["out_snr_db"] - 0.1, (
        f"rotated={rotated['out_snr_db']:.3f} plain={plain['out_snr_db']:.3f} : "
        f"la degradation mesuree precedemment (~-0,38 dB) a disparu")


def test_la_degradation_nvfp4_est_stable_sur_plusieurs_graines():
    """Un delta negatif sur UNE graine pourrait etre du bruit d'arrondi.
    Sur six graines, l'ecart reste dans une fenetre etroite : c'est un
    effet structurel du bloc 16, pas un artefact d'une graine particuliere."""
    deltas = []
    for graine in range(6):
        w = _tenseur_a_outliers(graine)
        _, _, plain = quantize_with_calibration(w, "nvfp4", None,
                                                use_hadamard=False, use_awq=False)
        _, _, rotated = quantize_with_calibration(w, "nvfp4", None,
                                                  use_hadamard=True, use_awq=False)
        deltas.append(rotated["out_snr_db"] - plain["out_snr_db"])
    assert all(d < 0 for d in deltas), deltas
    ecart = max(deltas) - min(deltas)
    assert ecart < 0.1, (
        f"dispersion {ecart:.3f} dB sur six graines : trop large pour "
        f"conclure a un effet stable, {deltas}")


def test_hadamard_aide_toujours_l_int4_awq_sur_le_meme_motif():
    """Temoin : le mecanisme `auto` existant (INT4-AWQ, groupes de 128)
    n'est pas remis en cause par ce qui precede. Reprend exactement le
    motif et le seuil de test_quant.py::test_hadamard_helps_int4_on_outlier_channels
    (ecart de +2 dB minimum) pour rester sur un regime deja valide."""
    w = _tenseur_a_outliers(0)
    _, _, plain = quantize_with_calibration(w, "int4_awq", None,
                                            use_hadamard=False, use_awq=False)
    _, _, rotated = quantize_with_calibration(w, "int4_awq", None,
                                              use_hadamard=True, use_awq=False)
    assert rotated["out_snr_db"] > plain["out_snr_db"] + 2.0, (
        f"rotated={rotated['out_snr_db']:.3f} plain={plain['out_snr_db']:.3f} : "
        f"l'INT4-AWQ ne beneficie plus de la rotation sur ce motif")


def test_a_magnitude_extreme_la_rotation_nuit_aux_deux_formats():
    """Hors du regime retenu ci-dessus (colonnes a x1000 au lieu de x8) :
    un outlier isole reste local a un seul groupe SANS rotation, et la
    rotation (non bloc-locale : `largest_pow2_divisor` prend le plus grand
    diviseur de la dimension entiere) l'etale sur tout le tenseur — les
    DEUX formats en souffrent alors, meme l'INT4-AWQ dont le groupe de 128
    est cense mieux absorber un outlier etale."""
    torch.manual_seed(0)
    w = torch.randn(256, 1024) * 0.02
    w[:, ::128] *= 1000.0
    for fmt in ("nvfp4", "int4_awq"):
        _, _, plain = quantize_with_calibration(w, fmt, None,
                                                use_hadamard=False, use_awq=False)
        _, _, rotated = quantize_with_calibration(w, fmt, None,
                                                  use_hadamard=True, use_awq=False)
        assert rotated["out_snr_db"] < plain["out_snr_db"], (
            f"{fmt}: rotated={rotated['out_snr_db']:.2f} >= "
            f"plain={plain['out_snr_db']:.2f} — le regime extreme ne "
            f"degrade plus les deux formats comme mesure precedemment")
