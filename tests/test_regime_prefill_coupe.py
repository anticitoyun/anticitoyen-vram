"""Ligne de régime : `prefill=…(coupé@<pas>)` sur un hybride sous cache de préfixe ON — le régime servi
coupe le prefill à la frontière d'instantané (`_frontiere_insta`) et un prefill en deux morceaux n'a pas
la numérique d'un morceau (MECANISMES 20/09, contrôle 3.2 : 14,9613 contre 14,7888). Dense, ou cache
OFF → aucune mention ; le pas affiché est `ACVRAM_INSTA_PAS`, pas une constante (Sage 08 h 25)."""
from __future__ import annotations

import torch

from acvram.engine.loader import load_model
from acvram.engine.runner import Engine


def _engine(converted, cache=True, insta=None, monkeypatch=None):
    if insta is not None:
        monkeypatch.setenv("ACVRAM_INSTA_PAS", str(insta))
    loaded = load_model(converted, dtype=torch.bfloat16, max_model_len=512, device_override="cpu")
    return Engine(loaded, None, max_batch_size=1, max_model_len=512, enable_cuda_graphs=False,
                  enable_prefix_cache=cache)


def test_dense_pas_de_mention(converted):
    e = _engine(converted)
    assert not e.est_hybride and "coupé@" not in e.regime_ligne()


def test_hybride_cache_on_nomme_le_pas_lu(converted, monkeypatch):
    e = _engine(converted, insta=512, monkeypatch=monkeypatch)
    e.est_hybride = True                       # le jouet est dense : on simule un hybride (layer_types non vide)
    assert "(coupé@512)" in e.regime()["prefill"] and "prefill=" in e.regime_ligne() and "(coupé@512)" in e.regime_ligne()


def test_hybride_cache_off_pas_de_mention(converted):
    e = _engine(converted, cache=False)
    e.est_hybride = True
    assert "coupé@" not in e.regime_ligne()
