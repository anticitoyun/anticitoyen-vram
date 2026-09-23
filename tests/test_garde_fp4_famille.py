"""Le seuil qui décide du FP4 matériel est une valeur qui décide : elle se garde.

Sans la forme *family-specific* (`sm_120f`), `__nv_cvt_fp4x2_to_halfraw2`
retombe sur vingt-cinq instructions par paire de poids au lieu de l'unique
`cvt.rn.f16x2.e2m1x2` du matériel. La marge est d'UNE version : le seuil est
12.9 et le toolkit du virtualenv fournit 13.0. Qu'il disparaisse, et la perte
revient — c'est pourquoi elle doit être bruyante.
"""
import warnings

from acvram.kernels import _MIN_CUDA_FOR_FAMILY, _arch_flags


def test_le_seuil_de_la_forme_famille_reste_a_12_9():
    """12.9 est la première version de nvcc qui accepte `compute_120f`."""
    assert _MIN_CUDA_FOR_FAMILY == (12, 9)


def test_un_nvcc_trop_ancien_avertit_au_lieu_de_perdre_le_fp4_en_silence():
    with warnings.catch_warnings(record=True) as vus:
        warnings.simplefilter("always")
        flags = _arch_flags((12, 8))
    assert not any("120f" in f for f in flags), \
        "sm_120f ne doit pas être demandé à un nvcc qui le refuse"
    assert any("FP4" in str(v.message) for v in vus), \
        "la perte du FP4 matériel doit être annoncée, pas subie"


def test_un_nvcc_capable_demande_la_forme_famille_sans_rien_dire():
    with warnings.catch_warnings(record=True) as vus:
        warnings.simplefilter("always")
        flags = _arch_flags(_MIN_CUDA_FOR_FAMILY)
    assert any("sm_120f" in f for f in flags), \
        "à partir du seuil, la forme famille doit être demandée"
    assert not any("FP4" in str(v.message) for v in vus), \
        "un détecteur qui crie quand tout va bien finit désactivé"
