"""Le prefill découpé rend la main — sans changer un seul jeton.

Coupé par défaut (`ACVRAM_BUDGET_JETONS=0`). Les deux épreuves qui comptent :
la sortie est identique au jeton près, et pendant qu'une longue invite se
précalcule, les séquences déjà en cours continuent de produire — ce que le
découpage déjà présent dans le moteur ne faisait PAS, ses deux passes
s'enchaînant dans le même pas.
"""
import torch

from acvram.engine.loader import load_model
from acvram.engine.runner import Engine
from acvram.engine.sampler import SamplingParams


def _moteur(converted, n=4):
    charge = load_model(converted, dtype=torch.float32, device_override="cpu",
                        max_concurrent_seqs=n)
    charge.plan.kv_planned_seqs = n   # cf. test_server.py : pas de GPU ici, le kwarg est ignoré
    return Engine(charge, None, max_batch_size=n, max_model_len=512)


def _produire(converted, invite, budget, monkeypatch, max_tokens=6):
    monkeypatch.setenv("ACVRAM_BUDGET_JETONS", str(budget))
    e = _moteur(converted)
    p = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    return [t for o in e.generate(list(invite), p) for t in o.token_ids]


def test_le_budget_ne_change_aucun_jeton(converted, monkeypatch):
    """« Une optimisation qui change la sortie est un bogue. » L'invite fait
    40 jetons, le budget 7 : six tranches, dont une incomplète."""
    invite = [(i * 31 + 5) % 900 + 1 for i in range(40)]
    sans = _produire(converted, invite, 0, monkeypatch)
    avec = _produire(converted, invite, 7, monkeypatch)
    assert sans == avec, "le decoupage a change la sortie"
    # contrôle : le budget mord bien. À 7 jetons par tranche il faut
    # plusieurs pas là où l'invite entière en demandait un seul.
    monkeypatch.setenv("ACVRAM_BUDGET_JETONS", "7")
    e = _moteur(converted)
    seq = e.add_request(list(invite), SamplingParams(temperature=0.0, max_tokens=1))
    pas = 0
    while not seq.prefilled:
        e.step()
        pas += 1
        assert pas < 20, "le prefill ne termine pas"
    assert pas == 6, f"40 jetons par tranches de 7 font 6 passes, pas {pas}"


def test_une_longue_invite_ne_bloque_plus_les_sequences_en_cours(converted,
                                                                 monkeypatch):
    """Le fait qui justifie tout le chantier : la latence, pas le calcul."""
    monkeypatch.setenv("ACVRAM_BUDGET_JETONS", "7")
    e = _moteur(converted)
    court = e.add_request([3, 1, 4, 1, 5], SamplingParams(temperature=0.0,
                                                          max_tokens=32))
    e.step()                                   # le court est en decodage
    assert court.prefilled
    longue = e.add_request([(i * 17 + 3) % 900 + 1 for i in range(40)],
                           SamplingParams(temperature=0.0, max_tokens=2))
    produits = 0
    while not longue.prefilled:
        avant = len(court.output_ids)
        e.step()
        produits += len(court.output_ids) - avant
    assert produits >= 3, \
        f"le court n'a produit que {produits} jetons pendant le prefill long"


def test_a_zero_le_moteur_est_celui_d_avant(converted, monkeypatch):
    """Contrôle du contrôle : sans budget, une seule passe de prefill."""
    monkeypatch.setenv("ACVRAM_BUDGET_JETONS", "0")
    e = _moteur(converted)
    seq = e.add_request([(i * 31 + 5) % 900 + 1 for i in range(40)],
                        SamplingParams(temperature=0.0, max_tokens=1))
    e.step()
    assert seq.prefilled, "l'invite entiere doit passer en un pas"
