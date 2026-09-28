"""276 j : graphe CUDA de la tour de vision à forme fixe — cassants à sec :
(1) sans CUDA la tour reste eager (aucun graphe, résultat identique) ; (2) la clé de forme porte les valeurs d'une petite
annexe et refuse une grande (Gemma 4) ; (3) un rejeu copie l'entrée dans le tampon statique et CLONE les sorties (le rejeu
suivant ne réécrit pas le résultat rendu), une capture par forme ; (4) le plafond de formes rend eager, compté ;
(5) les précalculs Qwen3-VL rendent la grille et cu_seqlens sur CPU et refusent une autre annexe ; (6) régime déclaré."""
from __future__ import annotations

import pytest
import torch

from acvram.engine import vision
from acvram.engine.vision import GrapheTour, SortieTour, TourVision


def _tour():
    return TourVision(lambda pv, **k: SortieTour((pv * 2,), []), torch.device("cpu"), nom="factice", hidden=3)


def test_sans_cuda_eager_et_aucun_graphe(monkeypatch):
    monkeypatch.setattr(vision, "_TOUR_GRAPHE", 1)
    t = _tour()
    t.precalculs = lambda pv, supp: {}
    assert t.graphes_actifs() is False               # device cpu
    pv = torch.ones(4, 3)
    out, nv = t.traits_niveaux(pv, 4, {"image_grid_thw": torch.tensor([[1, 2, 2]])})
    assert torch.equal(out, (pv * 2).to(torch.bfloat16)) and nv is None and t._graphes == {}


def test_cle_de_forme_petite_annexe_par_valeurs_grande_refusee():
    pv = torch.zeros(4, 3)
    c = TourVision.cle_graphe(pv, {"image_grid_thw": torch.tensor([[1, 2, 2]]), "n": 3})
    assert c == ((4, 3), "torch.float32", ("image_grid_thw", (1, 3), "torch.int64", (1, 2, 2)), ("n", 3))
    assert TourVision.cle_graphe(pv, {"image_grid_thw": torch.tensor([[1, 2, 4]])}) != c        # autre grille
    assert TourVision.cle_graphe(torch.zeros(8, 3), {"image_grid_thw": torch.tensor([[1, 2, 2]])}) != c
    assert TourVision.cle_graphe(pv, {"image_position_ids": torch.zeros(2520)}) is None       # Gemma 4 : eager
    assert TourVision.cle_graphe(pv, {"x": object()}) is None
    assert TourVision.cle_graphe([1, 2], {}) is None


class _GrapheFactice:
    def __init__(self, g):
        self.g, self.rejeux = g, 0

    def replay(self):
        self.rejeux += 1
        self.g.pooler[0].copy_(self.g.entree * 2)


def test_rejeu_copie_l_entree_et_clone_les_sorties_une_capture_par_forme(monkeypatch):
    monkeypatch.setattr(vision, "_TOUR_GRAPHE", 1)
    t = _tour()
    t.precalculs = lambda pv, supp: {}
    monkeypatch.setattr(t, "graphes_actifs", lambda: True)
    captures = []

    def capturer(pv, supp, cle):
        g = GrapheTour(None, pv.clone(), (torch.empty_like(pv),), [])
        g.graphe = _GrapheFactice(g)
        captures.append(cle)
        return g
    monkeypatch.setattr(t, "_capturer", capturer)
    supp = {"image_grid_thw": torch.tensor([[1, 2, 2]])}
    a = torch.full((4, 3), 1.0)
    b = torch.full((4, 3), 5.0)
    oa, _ = t.traits_niveaux(a, 4, supp)
    ob, _ = t.traits_niveaux(b, 4, supp)
    assert torch.equal(oa, (a * 2).to(torch.bfloat16)) and torch.equal(ob, (b * 2).to(torch.bfloat16))
    assert len(captures) == 1 and t.graphes_rejeux == 2                # même forme : un graphe, deux rejeux
    oc, _ = t.traits_niveaux(torch.full((8, 3), 1.0), 8, supp)          # autre forme : seconde capture
    assert len(captures) == 2 and oc.shape == (8, 3)


def test_plafond_de_formes_rend_eager_et_compte(monkeypatch):
    monkeypatch.setattr(vision, "_TOUR_GRAPHE", 1)
    monkeypatch.setattr(vision, "_TOUR_GRAPHE_MAX", 0)
    t = _tour()
    t.precalculs = lambda pv, supp: {}
    monkeypatch.setattr(t, "graphes_actifs", lambda: True)
    pv = torch.ones(4, 3)
    out, _ = t.traits_niveaux(pv, 4, {"image_grid_thw": torch.tensor([[1, 2, 2]])})
    assert torch.equal(out, (pv * 2).to(torch.bfloat16))
    assert t.graphes_refuses == 1 and t.graphes_captures == 0 and list(t._graphes.values()) == [None]


def test_precalculs_qwen3vl_grille_et_cu_seqlens_sur_cpu():
    pytest.importorskip("transformers.vision_utils")

    class Cfg:
        spatial_merge_size = 2
        _attn_implementation = "sdpa"

    class Visual:
        config = Cfg()
        num_grid_per_side = 48
        interpolation_mode = "bilinear"
        interpolation_align_corners = True
        spatial_merge_size = 2
    pre = vision.precalculs_qwen3vl(Visual())
    grid = torch.tensor([[1, 4, 4]])
    d = pre(torch.zeros(16, 1536), {"image_grid_thw": grid})
    assert set(d) >= {"image_grid_thw", "interp_indices", "interp_weights", "position_ids", "cu_seqlens"}
    assert d["image_grid_thw"].device.type == "cpu" and d["cu_seqlens"].device.type == "cpu"
    assert d["cu_seqlens"].tolist() == [0, 16] and tuple(d["position_ids"].shape) == (16, 2)
    assert tuple(d["interp_indices"].shape) == (16, 4) and d["interp_weights"].shape == d["interp_indices"].shape
    assert pre(torch.zeros(16, 1536), {"image_position_ids": torch.zeros(16)}) is None
    assert pre(torch.zeros(16, 1536), {"image_grid_thw": grid, "autre": 1}) is None


def test_variables_de_regime_declarees():
    from acvram import regime
    noms = {v.nom for v in regime.VARIABLES}
    assert {"TOUR_GRAPHE", "TOUR_GRAPHE_MAX"} <= noms


def test_le_graphe_garde_ses_annexes_precalculees():
    """Les annexes précalculées sont des entrées du graphe (adresses lues à chaque rejeu) : GrapheTour les retient —
    sinon libérées, réattribuées, gather hors bornes au 2e rejeu en service (28/09)."""
    import inspect
    assert "annexes" in GrapheTour.__dataclass_fields__
    src = inspect.getsource(TourVision._capturer)
    assert "annexes=kw" in src


def test_defaut_opt_in_decision_chef_28_09(monkeypatch):
    """Défaut 0 (opt-in) tant que le regroupement des préfills n'est pas neutralisé (276 k) ; 1 = graphe."""
    import importlib
    monkeypatch.delenv("ACVRAM_TOUR_GRAPHE", raising=False)
    assert importlib.reload(vision)._TOUR_GRAPHE == 0
    monkeypatch.setenv("ACVRAM_TOUR_GRAPHE", "1")
    assert importlib.reload(vision)._TOUR_GRAPHE == 1
    monkeypatch.delenv("ACVRAM_TOUR_GRAPHE", raising=False)
    importlib.reload(vision)
