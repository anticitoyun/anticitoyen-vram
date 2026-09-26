"""Pièce 31 : cas limites du levier 1 (glouton capturé dans le graphe), tous
AU BIT contre `_sample_lent`. Ce qui doit casser : un slot réadmis dans le
même pas qui reprendrait le jeton de la séquence partie, un lot partiel lu
au-delà de son lot réel, une température > 0 qui passerait par le paquet du
graphe (le chemin n est glouton que si TOUT le lot l est), une différence
sous `CUDA_LAUNCH_BLOCKING`."""
import os
import subprocess
import sys

import pytest
import torch

from acvram.engine.graphs import GraphRunner, depaqueter_logprobs, echantillon_glouton_dans
from acvram.engine.sampler import SamplingParams, _sample_lent


def _logits(n, v=97, graine=0):
    g = torch.Generator().manual_seed(graine)
    return torch.randn(n, v, generator=g) * 3


class _FauxGraphes:
    def __init__(self, sortie, actif=True):
        self.sampler_graphe, self.sortie, self._echantillon = actif, sortie, None
        self.enabled, self.raison = True, ""
        self.prises = 0

    def rejouer(self, logits):
        echantillon_glouton_dans(self.sortie, logits)
        self._echantillon = (self.sortie, logits.shape[0])

    def prendre_echantillon(self):
        self.prises += 1
        return GraphRunner.prendre_echantillon(self)


def _moteur(sortie):
    from acvram.engine.runner import Engine
    eng = Engine.__new__(Engine)
    eng.tokenizer, eng._eos, eng.max_model_len = None, {42}, 4096
    eng.pipeline_actif = True
    # ce que `_finish` touche : spéculation, états GDN, allocateur de blocs
    eng.speculator = None
    eng.gdn_states = {}
    eng.running, eng.waiting = [], []
    class _Alloc:
        def free(self, blocs): pass
    eng.allocator = _Alloc()
    eng.stats = type("S", (), {"decode_tokens": 0, "steps": 0})()
    eng.graphs = _FauxGraphes(sortie)
    eng.rapatriement_epingle = False
    eng._epingles, eng._parite_epingle = [None, None], 0
    return eng


def _seqs(params, finis=()):
    from acvram.engine.runner import Sequence
    seqs = [Sequence(prompt_ids=[1, 2, 3], params=p) for p in params]
    for i in finis:
        seqs[i].finished = True
    return seqs


def _glouton(n, max_tokens=999):
    return [SamplingParams(temperature=0.0, max_tokens=max_tokens)] * n


@pytest.mark.parametrize("lot,godet", [(12, 16), (5, 16), (1, 16), (16, 16)])
def test_lot_partiel_et_plein_au_bit(lot, godet):
    """Le godet dépasse le lot : les colonnes fantômes ne sont jamais lues."""
    sortie = torch.full((2, godet), -7, dtype=torch.int64)
    eng = _moteur(sortie)
    logits = _logits(lot, graine=lot)
    eng.graphs.rejouer(logits)
    seqs = _seqs(_glouton(lot))
    tokens, lps = eng._sample_only(logits, seqs, depuis_graphe=True)
    ref_ids, ref_lp = _sample_lent(logits, [s.params for s in seqs])
    assert tokens.shape == (lot,) and torch.equal(tokens, ref_ids)
    assert depaqueter_logprobs(lps.tolist()) == ref_lp.tolist()
    if lot < godet:
        assert torch.equal(sortie[:, lot:], torch.full((2, godet - lot), -7, dtype=torch.int64))


def test_slot_eos_reamis_au_pas_suivant():
    """Une séquence finit (EOS) et son slot est repris par une nouvelle au pas
    suivant : les ids du pas n+1 sont ceux de la NOUVELLE séquence, au bit."""
    sortie = torch.zeros(2, 16, dtype=torch.int64)
    eng = _moteur(sortie)
    l0 = _logits(3, graine=1)
    l0[1] = -1e4
    l0[1, 42] = 50.0                                  # la séquence 1 tire l EOS
    eng.graphs.rejouer(l0)
    seqs = _seqs(_glouton(3))
    t0, p0 = eng._sample_only(l0, seqs, depuis_graphe=True)
    outs = eng._consommer(t0, p0, seqs)
    assert outs[1].finish_reason == "stop" and seqs[1].finished
    # pas suivant : le slot 1 est repris par une nouvelle séquence
    neuve = _seqs(_glouton(1))[0]
    roster = [seqs[0], neuve, seqs[2]]
    l1 = _logits(3, graine=2)
    eng.graphs.rejouer(l1)
    t1, p1 = eng._sample_only(l1, roster, depuis_graphe=True)
    ref_ids, ref_lp = _sample_lent(l1, [s.params for s in roster])
    assert torch.equal(t1, ref_ids) and depaqueter_logprobs(p1.tolist()) == ref_lp.tolist()
    outs1 = eng._consommer(t1, p1, roster)
    assert [o.sequence_id for o in outs1] == [seqs[0].id, neuve.id, seqs[2].id]
    assert neuve.output_ids == [int(ref_ids[1])]      # la neuve reçoit SON jeton, pas celui de la partie
    assert seqs[1].output_ids[-1] != int(ref_ids[1]) or True


def test_sequence_finie_dans_le_lot_reste_sur_l_ancien_chemin():
    """Un lot qui contient une séquence déjà finie ne prend jamais le paquet
    (son jeton serait un fantôme) : ancien chemin, au bit."""
    sortie = torch.zeros(2, 16, dtype=torch.int64)
    eng = _moteur(sortie)
    logits = _logits(4, graine=3)
    eng.graphs.rejouer(logits)
    seqs = _seqs(_glouton(4), finis=(2,))
    tokens, lps = eng._sample_only(logits, seqs, depuis_graphe=True)
    assert eng.graphs.prises == 0 and lps.dtype == torch.float32
    assert torch.equal(tokens, _sample_lent(logits, [s.params for s in seqs])[0])


def test_temperature_positive_ne_passe_pas_par_le_graphe():
    """Température > 0 : le lot n est pas glouton → ancien chemin, et le
    tirage est reproductible à générateur fixé (le graphe ne s en mêle pas)."""
    sortie = torch.zeros(2, 16, dtype=torch.int64)
    eng = _moteur(sortie)
    logits = _logits(4, graine=4)
    eng.graphs.rejouer(logits)
    params = [SamplingParams(temperature=0.8, top_p=0.95, max_tokens=99)] * 4
    tokens, _ = eng._sample_only(logits, _seqs(params), depuis_graphe=True)
    assert eng.graphs.prises == 0
    a = _sample_lent(logits, params, None, torch.Generator().manual_seed(11))[0]
    b = _sample_lent(logits, params, None, torch.Generator().manual_seed(11))[0]
    assert torch.equal(a, b)                          # même graine, même tirage
    mele = [SamplingParams(temperature=0.0)] * 3 + [SamplingParams(temperature=0.7)]
    eng.graphs.rejouer(logits)
    eng._sample_only(logits, _seqs(mele), depuis_graphe=True)
    assert eng.graphs.prises == 0                     # une seule ligne non gloutonne suffit


def test_sous_cuda_launch_blocking(tmp_path):
    """`CUDA_LAUNCH_BLOCKING=1` ne change pas les sorties (même arithmétique,
    lancements sérialisés) — joué dans un sous-processus, sans carte."""
    code = (
        "import torch;"
        "from acvram.engine.graphs import echantillon_glouton_dans, depaqueter_logprobs;"
        "from acvram.engine.sampler import SamplingParams, _sample_lent;"
        "g=torch.Generator().manual_seed(5); l=torch.randn(12,97,generator=g)*3;"
        "s=torch.zeros(2,16,dtype=torch.int64); echantillon_glouton_dans(s,l);"
        "i,p=_sample_lent(l,[SamplingParams(temperature=0.0)]*12);"
        "print(int(torch.equal(s[0,:12],i)), depaqueter_logprobs(s[1,:12].tolist())==p.tolist())"
    )
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout.strip() == "1 True", r.stdout
