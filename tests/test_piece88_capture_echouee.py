"""Pièce 88 (23/09) : une capture de graphe qui échoue sur un modèle HYBRIDE.

`GraphRunner._capture` photographie les états récurrents (GDN), joue deux pas
d'échauffement qui les font avancer, capture, puis restaure — mais la
restauration n'était que sur le chemin du succès : une capture qui échoue
laissait l'état avancé de deux pas, et le pas eager de repli calculait sur un
état faux (sortie fausse, pas seulement lente).

À sec : primitives CUDA remplacées par des doublures ; la « capture » n'exécute
pas le pas (comme une vraie capture), et son échec est injecté à la sortie du
contexte, là où `capture_end` lève « operation failed due to a previous error
during capture »."""
import contextlib
from types import SimpleNamespace

import pytest
import torch

from acvram.engine import graphs as G

INVALIDEE = "CUDA error: operation failed due to a previous error during capture"


class _Flux:
    def wait_stream(self, autre):
        pass


class _Graphe:
    def pool(self):
        return object()

    def replay(self):
        pass


class _Couche:
    """Couche hybride minimale : un état récurrent par créneau."""
    def __init__(self, valeur):
        self.statics = [{"s": torch.tensor([float(valeur)])}]
        self.linear_attn = SimpleNamespace(
            static_export=lambda st: st["s"].clone(),
            static_load=lambda st, e: st["s"].copy_(e))


def _doublures(monkeypatch, echec):
    etat = {"capture": False}

    @contextlib.contextmanager
    def graphe(g, pool=None):
        etat["capture"] = True
        try:
            yield
        finally:
            etat["capture"] = False
        if echec:
            raise RuntimeError(INVALIDEE)

    monkeypatch.setattr(torch.cuda, "synchronize", lambda d=None: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda d=None: _Flux())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda d=None: _Flux())
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "CUDAGraph", _Graphe)
    monkeypatch.setattr(torch.cuda, "graph", graphe)
    return etat


def _runner(couche, etat):
    def pas(x, positions, slots, tables, seq_lens, max_pos, q_len=1):
        if not etat["capture"]:          # une capture enregistre, n'exécute pas
            couche.statics[0]["s"] += 1.0
        return couche.statics[0]["s"].clone()

    modele = SimpleNamespace(spec=SimpleNamespace(hidden_size=4), dtype=torch.float32,
                             modules=lambda: [], decode_fixed=pas, mtp=None)
    gr = G.GraphRunner.__new__(G.GraphRunner)
    gr.model, gr.device, gr.max_model_len = modele, "cpu", 2304
    gr.sampler_graphe, gr._last_key, gr._pool, gr.captures = False, None, None, 0
    gr.hybrid_layers = [couche]
    gr._fill = lambda entry, batch: None
    return gr, pas


def test_capture_echouee_rend_l_etat_recurrent_intact(monkeypatch):
    couche = _Couche(5.0)
    etat = _doublures(monkeypatch, echec=True)
    gr, pas = _runner(couche, etat)
    with pytest.raises(RuntimeError, match="previous error during capture"):
        gr._capture(1, 1, 8, batch=None)
    assert couche.statics[0]["s"].item() == 5.0, "état avancé par l'échauffement, non restauré"
    # le pas eager de repli doit rendre ce qu'il aurait rendu sans la capture ratée
    temoin = _Couche(5.0)
    _, pas_temoin = _runner(temoin, {"capture": False})
    assert pas(None, None, None, None, None, 0).item() == pas_temoin(None, None, None, None, None, 0).item() == 6.0


def test_capture_reussie_restaure_comme_avant(monkeypatch):
    couche = _Couche(5.0)
    etat = _doublures(monkeypatch, echec=False)
    gr, _ = _runner(couche, etat)
    entry = gr._capture(1, 1, 8, batch=None)
    assert "graph" in entry and gr.captures == 1
    assert couche.statics[0]["s"].item() == 5.0
