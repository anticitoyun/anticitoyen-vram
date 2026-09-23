"""Pièce 86 (23/09) : à b=2 en dense, des propositions n-gramme de longueurs
différentes faisaient tomber la vérification en eager (longueurs de requête
mêlées), ~40 ms le pas, sans compteur. Sous graphe, un tel lot décode désormais
sans spéculer, compté ; à longueurs égales la spéculation reste. À sec."""
from types import SimpleNamespace

from acvram.engine.runner import Engine, EngineStats
from acvram.engine.speculative import Proposal


def _moteur(longueurs, graphes=True):
    e = Engine.__new__(Engine)
    e.est_hybride, e.spec_k, e.max_model_len = False, 4, 2304
    e.stats = EngineStats()
    e.graphs = SimpleNamespace(enabled=graphes) if graphes else None
    props = iter(longueurs)
    e.speculator = SimpleNamespace(propose=lambda seq, k: Proposal([7] * next(props)))
    e.simple, e.grandis = [], []
    e._plain_decode = lambda dec: e.simple.append(len(dec)) or ["simple"]
    e._grow = lambda seq, extra=0: e.grandis.append(extra) or True
    e._build_spec_batch = lambda *a: (_ for _ in ()).throw(RuntimeError("vérification lancée"))
    seqs = [SimpleNamespace(id=i, params=SimpleNamespace(max_tokens=1024), output_ids=[1] * 10,
                            length=300, finished=False) for i in range(len(longueurs))]
    return e, seqs


def test_longueurs_melees_sous_graphe_decodent_sans_speculer():
    e, seqs = _moteur([4, 1])
    assert e._speculative_decode(seqs) == ["simple"]
    assert e.simple == [2] and e.grandis == []          # aucun bloc réservé pour rien
    assert e.stats.spec_longueurs_melees == 1
    assert e.stats.to_dict()["spec_longueurs_melees"] == 1


def test_longueurs_egales_speculent_encore():
    e, seqs = _moteur([3, 3])
    try:
        e._speculative_decode(seqs)
    except RuntimeError as err:
        assert "vérification lancée" in str(err)
    assert e.simple == [] and e.grandis == [3, 3] and e.stats.spec_longueurs_melees == 0


def test_sans_graphe_la_verification_melee_reste_permise():
    e, seqs = _moteur([4, 1], graphes=False)
    try:
        e._speculative_decode(seqs)
    except RuntimeError:
        pass
    assert e.simple == [] and e.stats.spec_longueurs_melees == 0
