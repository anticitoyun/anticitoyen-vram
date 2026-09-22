"""`perplexity()` par tranches de tête (sage-p2-ppl-instrument-file-7h-19-09) :
la NLL par position calculée par `_pertes_par_tranches` (états cachés → tête
par tranches) égale à 10⁻⁶ celle de l'ancien chemin (logits de la fenêtre
entière en un appel), quelle que soit la taille de tranche — sur le modèle
jouet, à sec. Témoin cassant : une tranche décalée d'une position change la
NLL (le test ne peut pas rendre « vrai » par construction)."""
import os
import shutil
import sys
import pathlib

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from acvram.engine.loader import load_model                                     # noqa: E402
from acvram.engine.model import ForwardBatch                                    # noqa: E402
from acvram.evaluate import _pertes_par_tranches, perplexity                    # noqa: E402
from acvram.memory.kvcache import BLOCK_SIZE, BlockAllocator                  # noqa: E402


def _batch(chunk, blocks_per_window):
    alloc = BlockAllocator(blocks_per_window, enable_prefix_cache=False)
    blocks = alloc.allocate(blocks_per_window)
    n = len(chunk)
    slots = torch.tensor([blocks[i // BLOCK_SIZE] * BLOCK_SIZE + i % BLOCK_SIZE for i in range(n)], dtype=torch.long)
    return ForwardBatch(tokens=torch.tensor(chunk, dtype=torch.long), positions=torch.arange(n, dtype=torch.long),
                        seq_lens=[n], query_lens=[n], block_tables=[torch.tensor(blocks, dtype=torch.long)],
                        slot_mapping=slots, is_prefill=True)


def test_tranches_egalent_la_fenetre_entiere(converted):
    loaded = load_model(converted, dtype=torch.float32, device_override="cpu", max_model_len=128)
    model = loaded.model
    torch.manual_seed(7)
    chunk = torch.randint(5, 200, (61,)).tolist()
    batch = _batch(chunk, (len(chunk) + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)
    targets = torch.tensor(chunk[1:], dtype=torch.long)
    with torch.no_grad():
        ref = model(batch, logits_positions=batch.all_token_indices())[:-1].to(torch.float32)
        nll_ref = torch.nn.functional.cross_entropy(ref, targets, reduction="none")
        batch2 = _batch(chunk, (len(chunk) + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)
        h = model(batch2, return_hidden=True)
        for tranche in (1, 7, 16, 256):
            nll = _pertes_par_tranches(model, h, targets, 0, tranche)
            assert nll.shape == nll_ref.shape
            assert torch.allclose(nll, nll_ref, rtol=0, atol=1e-6), (tranche, (nll - nll_ref).abs().max())
        nll3 = _pertes_par_tranches(model, h, targets, 3, 16)               # first_new respecté
        assert torch.allclose(nll3, nll_ref[3:], rtol=0, atol=1e-6)
        # témoin cassant : la même tête sur des positions décalées d'un cran
        faux = torch.nn.functional.cross_entropy(model._logits_finaux(model._tete(h[1:])).float()[:-1], targets[:-1], reduction="none")
        assert not torch.allclose(faux, nll_ref[:-1], atol=1e-3)


def test_perplexity_de_bout_en_bout_inchangee(converted, tiny_checkpoint, monkeypatch):
    for fn in ("tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(tiny_checkpoint, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(converted, fn))
    if not os.path.isfile(os.path.join(converted, "tokenizer.json")):
        pytest.skip("no tokenizer available")
    import acvram.evaluate as EV
    monkeypatch.setattr(EV, "_PPL_TRANCHE", 5)
    r5 = perplexity(converted, window=64, stride=32, max_tokens=256, device="cpu", dtype=torch.float32)
    monkeypatch.setattr(EV, "_PPL_TRANCHE", 4096)
    r_tout = perplexity(converted, window=64, stride=32, max_tokens=256, device="cpu", dtype=torch.float32)
    assert abs(r5.perplexity - r_tout.perplexity) <= 1e-6 * max(1.0, r_tout.perplexity)
    assert r5.par_contexte == r_tout.par_contexte
