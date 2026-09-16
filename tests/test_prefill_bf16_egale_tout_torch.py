"""Équivalence du prefill (poste7-prefill-a8-verdict-17-09 § 3) : la PPL d'un
converti NVFP4 sous le régime par défaut (`ACVRAM_PREFILL=bf16`, noyaux
actifs) égale celle du chemin tout-torch à ± 0,002 — sur une tranche, ici le
mini-modèle dense du conftest (fenêtres de 64 > seuil GEMV 32, donc le chemin
de prefill de `nvfp4_matmul` est bien celui exercé).

Bras : (a) régime par défaut ; (b) le backend fusionné masqué par
`acvram.regime.masquer` — `DISABLE_KERNELS` sur carte, `cpu-avx2` sur
processeur — donc le backend `reference` ; (c) témoin qui DOIT différer, sur
carte seulement : `ACVRAM_PREFILL=w8a8`, l'ancien défaut, hors tolérance.
Sans (c), une tolérance qui passe ne prouverait pas que l'instrument voit
un changement de régime."""
import math
import os

import pytest
import torch

from acvram import regime
from acvram.kernels import backends


def _tokenizer_et_corpus(converted):
    """Tokeniseur mot-à-mot de 1024 entrées (comme test_calibration) et un
    corpus de 300 jetons tirés au sort : chaque position est un vrai jeton,
    pas un inconnu répété."""
    import json
    import random
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    vocab = {f"tok{i}": i for i in range(1024)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="tok0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.WordPiece(prefix="")
    tok.save(os.path.join(converted, "tokenizer.json"))
    json.dump({"eos_token": "tok0"}, open(os.path.join(converted, "tokenizer_config.json"), "w"))
    rng = random.Random(7)
    corpus = os.path.join(converted, "corpus.txt")
    open(corpus, "w").write(" ".join(f"tok{rng.randrange(1, 1024)}" for _ in range(300)))
    return corpus


def _ppl(converted, corpus, device, dtype):
    from acvram.evaluate import perplexity
    r = perplexity(converted, corpus_path=corpus, window=64, stride=64, max_tokens=256,
                   device=device, dtype=dtype)
    assert math.isfinite(r.perplexity)
    return r.perplexity


@pytest.fixture
def masques_propres():
    from acvram import kernels
    etat = (kernels._EXT, kernels._TRIED, kernels._ERROR, set(backends._MASQUES))
    try:
        yield
    finally:
        kernels._EXT, kernels._TRIED, kernels._ERROR = etat[:3]
        backends._MASQUES.clear(); backends._MASQUES.update(etat[3])
        backends._RESOLVED.clear()


def test_ppl_prefill_bf16_egale_tout_torch(converted, monkeypatch, masques_propres):
    corpus = _tokenizer_et_corpus(converted)
    from acvram import kernels
    cuda = torch.cuda.is_available() and kernels.get_extension() is not None
    device, dtype = ("cuda", torch.bfloat16) if cuda else ("cpu", torch.float32)
    if not cuda and "cpu-avx2" not in [b.name for b in backends.resolve("nvfp4", torch.device("cpu"))]:
        pytest.skip("ni extension CUDA ni noyaux processeur : un seul chemin, rien à comparer")
    monkeypatch.delenv("ACVRAM_PREFILL", raising=False)
    monkeypatch.delenv("ACVRAM_DISABLE_KERNELS", raising=False)
    defaut = _ppl(converted, corpus, device, dtype)
    if cuda:
        # le témoin d'abord, tant que l'extension est encore là
        monkeypatch.setenv("ACVRAM_PREFILL", "w8a8")
        w8a8 = _ppl(converted, corpus, device, dtype)
        monkeypatch.delenv("ACVRAM_PREFILL")
        assert abs(w8a8 / defaut - 1) > 0.002, f"témoin w8a8 = défaut ({w8a8:.4f} / {defaut:.4f}) : l'instrument ne voit pas le régime"
    regime.masquer(["DISABLE_KERNELS"] if cuda else ["cpu-avx2"])
    torch_pur = _ppl(converted, corpus, device, dtype)
    assert abs(defaut / torch_pur - 1) <= 0.002, f"défaut {defaut:.4f} ≠ tout-torch {torch_pur:.4f}"
