"""La garde de disposition du bras FP8 (`banc-etroites-noyaux.py`) : elle doit
refuser la disposition qui a fait planter le banc, et accepter la bonne.

`torch._scaled_mm(A [M, K], B [K, N])` veut A rangée-majeure, B
COLONNE-majeure (stride(0) == 1) et des échelles fp32 [M, 1] / [1, N]. Le banc
passait `qw.t().contiguous().t()` puis `.t()` : B redevenait rangée-majeur et
l appel mourait sur un stride, avant toute mesure (Manon d964e2cc). Rien de
tout cela ne s exécute sur processeur — le test vérifie la GARDE, avec des
tenseurs int8 de la bonne géométrie, et les trois refus qu elle doit rendre.
"""
from __future__ import annotations

import importlib.util
import os

import pytest
import torch

CHEMIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outils", "gpu", "mesure", "banc-etroites-noyaux.py")


def _garde():
    spec = importlib.util.spec_from_file_location("banc_etroites_noyaux", CHEMIN)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m._garde_scaled_mm


def _jeu(M=12, K=2048, N=5120):
    f8 = torch.float8_e4m3fn
    a = torch.zeros(M, K).to(f8)
    qw = torch.zeros(N, K).to(f8)             # poids [N, K] rangée-majeur
    b = qw.t()                                # [K, N] colonne-majeur : LA bonne disposition
    sa = torch.ones(M, 1, dtype=torch.float32)
    sb = torch.ones(1, N, dtype=torch.float32)
    return a, b, sa, sb


def test_la_bonne_disposition_passe():
    _garde()(*_jeu())


def test_b_rangee_majeure_est_refusee():
    a, b, sa, sb = _jeu()
    mauvais = b.contiguous()                  # [K, N] rangée-majeur : l ancien bogue
    with pytest.raises(RuntimeError, match="COLONNE-majeure"):
        _garde()(a, mauvais, sa, sb)


def test_echelles_mal_formees_sont_refusees():
    a, b, sa, sb = _jeu()
    with pytest.raises(RuntimeError, match="scale_b"):
        _garde()(a, b, sa, sb.reshape(-1, 1))          # [N, 1] au lieu de [1, N]
    with pytest.raises(RuntimeError, match="scale_a"):
        _garde()(a, b, sa.to(torch.bfloat16), sb)      # pas fp32


def test_k_non_multiple_de_16_est_refuse():
    a, b, sa, sb = _jeu(K=2040)
    with pytest.raises(RuntimeError, match="multiple de 16"):
        _garde()(a, b, sa, sb)
