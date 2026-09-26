"""Correctif gemma-4-26B-A4B, capture du godet 1 en OOM
(`chantier-gemma-capture-godet1-20-09`) : après CHAQUE pile d'experts
construite (`MoEBlock._try_build_stacks`, paresseux, dans
`GraphRunner._eligible` du premier moteur), les blocs libérés (sources +
pile naturelle) sont rendus au pilote (`torch.cuda.empty_cache`). Sans
cela l'allocateur les garde en cache, la première capture — qui exige des
segments neufs pour le bassin privé du graphe, et pendant laquelle
l'allocateur ne rend jamais son cache — tombe en OOM ; Triton (Ornith)
alloue hors de l'allocateur et tombe de même.

À sec : la carte n'existe pas, `empty_cache` est remplacé par un compteur
et `is_available` forcé à vrai — le test vérifie le GESTE (rendu après une
pile construite, pas après une pile refusée), ce qu'une relecture du code
ne prouve pas. Témoin : `_rendre_le_cache_apres_la_pile` sans CUDA ne
touche pas au compteur.
"""
import torch

from acvram.engine import model as M
from acvram.engine import moe as MOE
from acvram.engine.loader import load_model


def _compteur(monkeypatch, disponible=True):
    appels = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: disponible)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: appels.append(1))
    return appels


def test_une_pile_construite_rend_le_cache(tiny_moe, monkeypatch):
    loaded = load_model(tiny_moe, dtype=torch.float32, device_override="cpu")
    bloc = next(m for m in loaded.model.modules() if isinstance(m, MOE.MoEBlock))
    appels = _compteur(monkeypatch)
    assert bloc._try_build_stacks(), bloc._raison_repli
    assert len(appels) == 1, "la pile construite doit rendre le cache une fois"


def test_une_pile_refusee_ne_rend_pas_par_ce_chemin(tiny_moe, monkeypatch):
    """Le rendu suit une CONSTRUCTION ; la branche OOM de `_try_build_stacks`
    a le sien, ce test ne le compte pas : un refus (piles hétérogènes) sort
    avant le repack, sans rien à rendre."""
    loaded = load_model(tiny_moe, dtype=torch.float32, device_override="cpu")
    bloc = next(m for m in loaded.model.modules() if isinstance(m, MOE.MoEBlock))
    appels = _compteur(monkeypatch)
    monkeypatch.setattr(bloc, "_noms_experts", lambda: ["gate_proj", "inexistant"])
    try:
        ok = bloc._try_build_stacks()
    except AttributeError:
        ok = False
    assert not ok and appels == []


def test_sans_cuda_rien_n_est_rendu(monkeypatch):
    appels = _compteur(monkeypatch, disponible=False)
    MOE.MoEBlock._rendre_le_cache_apres_la_pile()
    assert appels == []


def test_temoin_sans_rendu_est_nomme_et_ne_rend_pas(monkeypatch, capsys):
    """Bras A de la chaîne carte : `ACVRAM_PILE_SANS_RENDU=1` coupe le rendu
    et le DIT (une fois) — un témoin muet serait un défaut déguisé."""
    appels = _compteur(monkeypatch)
    monkeypatch.setenv("ACVRAM_PILE_SANS_RENDU", "1")
    monkeypatch.setattr(MOE.MoEBlock, "_sans_rendu_dit", False, raising=False)
    MOE.MoEBlock._rendre_le_cache_apres_la_pile()
    MOE.MoEBlock._rendre_le_cache_apres_la_pile()
    assert appels == []
    assert capsys.readouterr().out.count("TÉMOIN ACVRAM_PILE_SANS_RENDU=1") == 1


from test_moe_grouped import tiny_moe  # noqa: E402,F401  (fixture de session, réutilisée)
