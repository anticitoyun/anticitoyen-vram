"""`sample` vectorisé sur le lot (opt-in ACVRAM_SAMPLER_LOT=1) == `_sample_lent` (l ancienne boucle par ligne, défaut) :
mêmes ids à générateur identique sur un lot factice mêlant glouton, top_k, top_p, min_p, température, pénalités ;
mêmes logprobs quand ils sont demandés ; glouton pur = argmax seul, zéros sans demande. À sec (processeur)."""
import copy, torch, pytest
from acvram.engine.sampler import SamplingParams, sample, _sample_lent

V = 4096


@pytest.fixture(autouse=True)
def _lot_actif(monkeypatch):
    monkeypatch.setenv("ACVRAM_SAMPLER_LOT", "1")     # les tests d équivalence jouent le vectorisé ; le défaut est testé à part


def _lot(seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(12, V, generator=g) * 4
    logits[3, 17] = logits[3].max() + 0.0     # égalité stricte : le premier indice gagne, des deux côtés
    params = [SamplingParams(temperature=0.0), SamplingParams(temperature=0.8), SamplingParams(temperature=0.7, top_k=40),
              SamplingParams(temperature=0.0), SamplingParams(temperature=1.0, top_p=0.9), SamplingParams(temperature=0.9, min_p=0.05),
              SamplingParams(temperature=0.6, top_k=10, top_p=0.5), SamplingParams(temperature=1.2, top_k=V + 5),
              SamplingParams(temperature=0.5, top_p=0.95, min_p=0.02, top_k=100), SamplingParams(temperature=1.0, repetition_penalty=1.3),
              SamplingParams(temperature=0.8, presence_penalty=0.5, frequency_penalty=0.2), SamplingParams(temperature=0.0, logprobs=1)]
    history = [[int(x) for x in torch.randint(0, V, (30,), generator=g)] for _ in params]
    return logits, params, history


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_memes_ids_que_le_temoin_a_generateur_identique(seed):
    logits, params, history = _lot(seed)
    ga, gb = torch.Generator().manual_seed(123), torch.Generator().manual_seed(123)
    ta, la = sample(logits.clone(), params, history, generator=ga)
    tb, lb = _sample_lent(logits.clone(), params, history, generator=gb)
    assert torch.equal(ta, tb), (ta.tolist(), tb.tolist())
    assert torch.equal(la, lb), "logprobs demandés par une ligne : calculés pour tout le lot, au bit"


def test_defaut_lent_et_opt_in_lot(monkeypatch):
    """Défaut = lent (verdict eea064fe) : sans variable, `sample` EST `_sample_lent` (mêmes logprobs, jamais zéros) ;
    ACVRAM_SAMPLER_LOT=1 bascule sur le vectorisé (glouton pur → zéros sans demande)."""
    from acvram.engine.sampler import sampler_texte
    logits = torch.randn(12, V); params = [SamplingParams(temperature=0.0)] * 12
    monkeypatch.delenv("ACVRAM_SAMPLER_LOT", raising=False)
    assert sampler_texte() == "lent"
    ta, la = sample(logits.clone(), params, [() for _ in params])
    tb, lb = _sample_lent(logits.clone(), params, [() for _ in params])
    assert torch.equal(ta, tb) and torch.equal(la, lb) and la.any()
    monkeypatch.setenv("ACVRAM_SAMPLER_LOT", "1")
    assert sampler_texte() == "lot"
    tc, lc = sample(logits.clone(), params, [() for _ in params])
    assert torch.equal(tc, tb) and not lc.any()


def test_glouton_pur_argmax_seul_et_zeros_sans_demande():
    logits = torch.randn(12, V).to(torch.bfloat16)
    params = [SamplingParams(temperature=0.0)] * 12
    t, lp = sample(logits, params, [() for _ in params])
    assert torch.equal(t, logits.argmax(-1)) and torch.equal(t, _sample_lent(logits, params, [() for _ in params])[0])
    assert lp.dtype == torch.float32 and not lp.any()
    params[4] = SamplingParams(temperature=0.0, logprobs=1)
    t2, lp2 = sample(logits, params, [() for _ in params])
    assert torch.equal(t2, t) and torch.equal(lp2, _sample_lent(logits, params, [() for _ in params])[1])


def test_doit_casser_si_l_ordre_des_filtres_change():
    """Le témoin applique top_k PUIS min_p PUIS top_p ; une ligne où l ordre compte rend d autres ids."""
    logits, _, _ = _lot(9)
    params = [SamplingParams(temperature=0.7, top_k=5, top_p=0.3)] * 12
    ta, _ = sample(logits.clone(), params, None, generator=torch.Generator().manual_seed(1))
    tb, _ = _sample_lent(logits.clone(), params, None, generator=torch.Generator().manual_seed(1))
    assert torch.equal(ta, tb)
