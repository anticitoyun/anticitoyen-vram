"""Pièce 243 : sous une portée `depaquetage_partage` (boucle par séquence GDN au préfill), le seuil GEMV → GEMM int8
vaut 16 par défaut (0.7.0, hors bit, KL tenue, service mixte b=8 +9,70 %) ; hors portée il reste INT8_GEMV_MAX = 80,
car la déquant NON partagée régresse (733,7 µs contre 636,4 de GEMV à n = 78). Cassant : remettre le défaut à 80, ou
laisser la portée sans effet sur le seuil, rend ce fichier rouge."""

import os

import pytest

from acvram import kernels


@pytest.mark.skipif(bool(os.environ.get("ACVRAM_INT8_GEMV_MAX_PARTAGE") or os.environ.get("ACVRAM_INT8_GEMV_MAX")),
                    reason="seuils posés par l'environnement")
def test_defaut_16_sous_portee_80_hors_portee(monkeypatch):
    assert kernels._INT8_GEMV_MAX_PARTAGE == 16
    assert kernels._INT8_GEMV_MAX == 80
    monkeypatch.setattr(kernels, "_DEPAQ_PARTAGE", True)
    assert kernels.seuil_gemv_int8() == 80
    with kernels.depaquetage_partage():
        assert kernels.seuil_gemv_int8() == 16
        with kernels.depaquetage_partage():                       # imbriqué : même portée
            assert kernels.seuil_gemv_int8() == 16
    assert kernels.seuil_gemv_int8() == 80


def test_temoin_80_rend_la_sortie_d_avant(monkeypatch):
    monkeypatch.setattr(kernels, "_DEPAQ_PARTAGE", True)
    monkeypatch.setattr(kernels, "_INT8_GEMV_MAX_PARTAGE", 80)
    with kernels.depaquetage_partage():
        assert kernels.seuil_gemv_int8() == kernels._INT8_GEMV_MAX
