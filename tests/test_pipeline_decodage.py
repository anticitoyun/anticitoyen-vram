"""Bead runner (14/09) : recouvrement pas n+1 / rejeu n (`ACVRAM_PIPELINE`).

Les jetons émis doivent être identiques entre PIPELINE=0 et PIPELINE=1, SAUF
aux points où les deux meilleurs candidats sont quasi à égalité en bf16 —
là, `argmax` peut légitimement basculer d'un ordre de sommation à l'autre
(mémoire du 8/09 : « une divergence n'est pas une preuve »). Deux contrôles
de Jérôme, 14/09 soir, l'ont établi : (1) la même divergence existe déjà
entre `graphs.py` d'avant le pipeline et l'eager pur (bissection) ; (2) au
point de divergence, les logits du chemin qui bascule montrent un ÉCART
QUASI NUL entre le top-1 et le top-2 (TIE), l'autre chemin un écart de
l'ordre de 1-3 ulp bf16 — du bruit numérique légitime, pas un bogue.

Donc : une divergence de JETON n'est acceptée que si, À CE POINT, l'un des
deux chemins montre un top-1/top-2 à moins de `TOLERANCE_ULP` d'écart.
Sinon, le test échoue — c'est le « changement qui doit casser » (REGLES.md
règle 5) : une divergence FRANCHE (pas un tie) doit rester détectée.
"""
from __future__ import annotations

import os

import pytest

@pytest.fixture(autouse=True)
def _lot_du_test_pas_du_plan(monkeypatch):
    """Le converti du disque porte son plan (kv_planned_seqs) ; le test dimensionne son lot (T4 20/09)."""
    monkeypatch.setenv("ACVRAM_KV_PLAN_OVERRIDE", "1")
import torch
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../outils'))
from racine_modeles import racine_modeles as _racine_modeles, alias_absent as _alias_absent  # noqa: E402
_RACINE = _racine_modeles()   # ACVRAM_MODELES → ~/.config/acvram/modeles → littéral (20/09)


pytestmark = pytest.mark.gpu_requis

MODEL = _RACINE + "/Qwen3-Coder-30B-A3B-nvfp4"
N_SEQ_INIT = 11
# ~1-3 ulp bf16 à la magnitude des logits observés ce soir (12-17, exposant
# 3-4, ulp 0,0625-0,125) — marge large plutôt qu'ajustée au cas observé.
TOLERANCE_ULP = 0.4


def _invite(k: int, n: int = 128) -> list[int]:
    return [(1000 + k * 7919 + i * 13) % 150000 + 10 for i in range(n)]


def _rejouer(engine, capturer_logits: bool):
    """Rejoue le scénario (11 séquences, fins à des pas différents, une
    arrivée au pas 30) et rend (jetons par request_id, logits top1/top2 par
    (request_id, indice de jeton) si `capturer_logits`)."""
    from acvram.engine.sampler import SamplingParams

    engine._eos = set()
    n_pas_max = 200
    max_tokens = [40, 60, 80, 100, 120, 140, 160, 180, n_pas_max, n_pas_max, n_pas_max]
    id_vers_rid: dict[int, str] = {}
    for k in range(N_SEQ_INIT):
        rid = f"s{k}"
        seq = engine.add_request(_invite(k), SamplingParams(temperature=0.0,
                                                             max_tokens=max_tokens[k]),
                                 request_id=rid)
        id_vers_rid[seq.id] = rid

    tokens: dict[str, list[int]] = {f"s{k}": [] for k in range(N_SEQ_INIT)}
    logits_top2: dict[tuple, tuple] = {}

    if capturer_logits:
        # Espionne `_sample_only`, PAS `_emit` : le chemin recouvert
        # (`_pipeline_suite`) appelle `_sample_only` directement, sans
        # jamais passer par `_emit` (qui combine `_sample_only`+
        # `_consommer` — utilisé seulement par le pas synchrone et
        # l'amorçage). Un index par SÉQUENCE, incrémenté à chaque
        # ÉCHANTILLONNAGE : correspond au jeton de rang i qu'elle finira
        # par émettre, que la consommation soit immédiate ou différée
        # d'un pas (pipeline).
        indices_pas: dict[str, int] = {rid: 0 for rid in tokens}
        orig_sample = engine._sample_only

        def sample_espion(logits, seqs):
            top2 = torch.topk(logits.to(torch.float32), 2, dim=-1)
            for i, seq in enumerate(seqs):
                rid = id_vers_rid.get(seq.id)
                if rid is None:
                    continue
                j = indices_pas.get(rid, 0)
                logits_top2[(rid, j)] = (top2.values[i].tolist(), top2.indices[i].tolist())
                indices_pas[rid] = j + 1
            return orig_sample(logits, seqs)
        engine._sample_only = sample_espion

    arrivee_faite = False
    for pas in range(n_pas_max + 40):
        if not arrivee_faite and pas == 30:
            seq = engine.add_request(_invite(999), SamplingParams(temperature=0.0, max_tokens=50),
                                     request_id="arrivee")
            id_vers_rid[seq.id] = "arrivee"
            tokens["arrivee"] = []
            if capturer_logits:
                indices_pas["arrivee"] = 0
            arrivee_faite = True
        for out in engine.step():
            rid = id_vers_rid[out.sequence_id]
            tokens[rid].extend(out.token_ids)
        if not engine.running and not engine.waiting:
            break
    return tokens, logits_top2


@pytest.mark.skipif(bool(_alias_absent("Qwen3-Coder-30B-A3B-nvfp4")),
                    reason=_alias_absent("Qwen3-Coder-30B-A3B-nvfp4"))
def test_pipeline_bit_identique_a_egalite_pres():
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine

    os.environ["ACVRAM_REPIN"] = "0"
    loaded = load_model(MODEL, dtype=torch.bfloat16, max_model_len=1024)

    engine_a = Engine(loaded, None, max_batch_size=12, max_model_len=1024)
    engine_a.pipeline_actif = False
    tokens_a, _ = _rejouer(engine_a, capturer_logits=False)
    del engine_a
    torch.cuda.empty_cache()

    engine_b = Engine(loaded, None, max_batch_size=12, max_model_len=1024)
    engine_b.pipeline_actif = True
    tokens_b, logits_b = _rejouer(engine_b, capturer_logits=True)
    del engine_b
    torch.cuda.empty_cache()

    assert tokens_a.keys() == tokens_b.keys()
    franches = []
    for rid in tokens_a:
        # Ne vérifier que la PREMIÈRE divergence : une fois un jeton
        # basculé par un tie, les deux séquences continuent depuis des
        # CONTEXTES DIFFÉRENTS — comparer plus loin ne compare plus « même
        # calcul, ordre de sommes différent » mais deux générations sans
        # rapport, dont l'écart n'apprend plus rien sur le pipeline.
        n = min(len(tokens_a[rid]), len(tokens_b[rid]))
        for i in range(n):
            if tokens_a[rid][i] == tokens_b[rid][i]:
                continue
            vals, idx = logits_b.get((rid, i), (None, None))
            ecart = None if vals is None else vals[0] - vals[1]
            if ecart is None or ecart > TOLERANCE_ULP:
                franches.append((rid, i, tokens_a[rid][i], tokens_b[rid][i], ecart))
            break

    assert not franches, (
        "divergence(s) FRANCHE(S) (pas un tie bf16) entre PIPELINE=0 et 1 : "
        + ", ".join(f"{rid}[{i}] {ta} vs {tb} (écart top1/top2 = {e})"
                    for rid, i, ta, tb, e in franches))


@pytest.mark.parametrize("b", [1, 12])
@pytest.mark.skipif(bool(_alias_absent("Qwen3-Coder-30B-A3B-nvfp4")),
                    reason=_alias_absent("Qwen3-Coder-30B-A3B-nvfp4"))
def test_pipeline_par_defaut_ids_au_bit_b1_et_b12(b):
    """Maîtresse 21/09 (0.6.34) : ACVRAM_PIPELINE=1 par défaut ; contre le témoin ACVRAM_PIPELINE=0, MÊME forme de
    lot (b séquences, même invite, 48 jetons greedy), les ids sont identiques AU BIT — pas « à égalité près » :
    le pipeline rejoue le même graphe sur le même lot, seul l ordre des copies change."""
    from acvram.engine.loader import load_model
    from acvram.engine.runner import Engine
    from acvram.engine.sampler import SamplingParams

    os.environ["ACVRAM_REPIN"] = "0"
    loaded = load_model(MODEL, dtype=torch.bfloat16, max_model_len=1024)

    def rejouer(actif: bool):
        engine = Engine(loaded, None, max_batch_size=b, max_model_len=1024)
        engine.pipeline_actif = actif
        assert (" pipeline=1 " in engine.regime_ligne() + " ") == actif or engine.graphs is None
        engine._eos = set()
        ids = {}
        for k in range(b):
            seq = engine.add_request(_invite(k, 96), SamplingParams(temperature=0.0, max_tokens=48), request_id=f"s{k}")
            ids[seq.id] = []
        while engine.running or engine.waiting:
            for out in engine.step():
                ids[out.sequence_id].extend(out.token_ids)
        del engine
        torch.cuda.empty_cache()
        return ids
    temoin, pipeline = rejouer(False), rejouer(True)
    assert list(temoin.values()) == list(pipeline.values()), \
        [(k, next((i for i, (x, y) in enumerate(zip(temoin[k], pipeline[k])) if x != y), None)) for k in temoin]
