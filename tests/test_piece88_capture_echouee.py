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


# ---- tri des échecs de capture, par `preparer` (à sec) -----------------------

def _gr_tri(monkeypatch, erreurs, sync_leve=False):
    """`erreurs` : exceptions levées par les captures successives (None = succès)."""
    def sync(d=None):
        if sync_leve:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
    monkeypatch.setattr(torch.cuda, "synchronize", sync)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    gr = G.GraphRunner.__new__(G.GraphRunner)
    gr.enabled, gr.paged_ok, gr.hybrid_layers = True, True, []
    gr.max_model_len, gr.device, gr.replays = 2304, "cpu", 0
    gr._raisons_eager_vues, gr.replis_eager = set(), 0
    gr.abandon_capture, gr._photo_memoire = None, lambda: None
    gr._avant_premiere_capture = lambda: None
    gr._apres_echec_capture = lambda: None
    gr._surveiller_capture = lambda cle: SimpleNamespace(cancel=lambda: None)
    gr._pool = "pool-ancien"
    gr.graphs = {("saine",): {}}
    file = list(erreurs)
    gr.tentatives = 0

    def capture(b, ql, nblk, batch):
        gr.tentatives += 1
        err = file.pop(0) if file else None
        if err is not None:
            raise err
        return {"graph": "g"}
    gr._capture = capture
    return gr


def _lot(b=12):
    return SimpleNamespace(is_prefill=False, query_lens=[1] * b, batch_size=b,
                           block_tables=[torch.zeros(20, dtype=torch.int32)] * b)


def test_transitoire_reprise_bornee_sur_pool_neuf(monkeypatch):
    gr = _gr_tri(monkeypatch, [RuntimeError(INVALIDEE)] * 3)
    assert gr.preparer(_lot()) is False
    assert gr.enabled and ("saine",) in gr.graphs and gr._pool is None   # rien de sain perdu, pool abandonné
    assert gr.tentatives == 1
    for _ in range(G.GraphRunner.DELAI_REPRISE_CAPTURE - 1):             # attente : pas de nouvel essai
        assert gr.preparer(_lot()) is False
    assert gr.tentatives == 1
    assert gr.preparer(_lot()) is False and gr.tentatives == 2          # reprise, échoue encore
    for _ in range(G.GraphRunner.DELAI_REPRISE_CAPTURE):
        gr.preparer(_lot())
    assert gr.tentatives == 3                                          # 3e essai → refus définitif
    for _ in range(3 * G.GraphRunner.DELAI_REPRISE_CAPTURE):
        gr.preparer(_lot())
    assert gr.tentatives == 3 and gr.enabled
    assert any("transitoire, 3 essais" in r for r in gr._raisons_eager_vues)
    D = G.GraphRunner.DELAI_REPRISE_CAPTURE
    assert gr.replis_eager == 1 + (D - 1) + 1 + D + 3 * D                 # chaque pas hors graphe compte


def test_transitoire_puis_succes_capture_la_cle(monkeypatch):
    gr = _gr_tri(monkeypatch, [RuntimeError(INVALIDEE), None])
    gr.preparer(_lot())
    for _ in range(G.GraphRunner.DELAI_REPRISE_CAPTURE - 1):
        gr.preparer(_lot())
    assert gr.preparer(_lot()) is True and gr.tentatives == 2 and len(gr.graphs) == 2


def test_deterministe_refuse_la_seule_cle(monkeypatch):
    gr = _gr_tri(monkeypatch, [RuntimeError("CUDA error: out of memory"), None])
    assert gr.preparer(_lot(12)) is False and gr.enabled
    for _ in range(200):
        assert gr.preparer(_lot(12)) is False
    assert gr.tentatives == 1                                          # jamais retentée
    assert any("déterministe" in r and "out of memory" in r for r in gr._raisons_eager_vues)
    assert gr.preparer(_lot(1)) is True and gr.tentatives == 2          # une autre forme se capture
    assert gr.replis_eager == 201


def test_contexte_en_erreur_coupe_tout_comme_avant(monkeypatch, capsys):
    gr = _gr_tri(monkeypatch, [RuntimeError("CUDA error: an illegal memory access was encountered")],
                 sync_leve=True)
    assert gr.preparer(_lot()) is False
    assert gr.enabled is False and gr.graphs == {}
    assert "graphes CUDA desactives" in capsys.readouterr().out
