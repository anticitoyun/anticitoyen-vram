"""Perplexity, so format decisions rest on a measurement.

SNR on a weight matrix and cosine similarity between logit vectors are proxies.
They correlate with quality, they are cheap, and they are what the converter
reports per tensor -- but they cannot answer "is NVFP4 with a promotion floor
of 25 dB better than plain INT4 on this model". Perplexity can.

The evaluation is a straightforward sliding window: feed a window of tokens,
score the model's prediction of each token given everything before it, slide
by ``stride`` and only count the newly exposed positions so no token is scored
twice with different amounts of context.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

__all__ = ["EvalResult", "perplexity", "compare_models", "DEFAULT_CORPUS"]

DEFAULT_CORPUS = """The transformer architecture replaced recurrence with attention,
which made it possible to train on far longer sequences without the gradient
path growing with the sequence length. Quantization reduces the number of bits
each weight occupies; the difficulty is not the storage but the distribution,
because a handful of channels carry activations orders of magnitude larger
than the rest and a uniform grid spends most of its resolution on the wrong
values.

def sliding_window(tokens, size, stride):
    for start in range(0, len(tokens), stride):
        window = tokens[start:start + size]
        if len(window) < 2:
            return
        yield start, window

La memoire d'un ordinateur n'est pas un mur mais une hierarchie: registres,
caches, memoire vive, disque. Un modele de langage qui ne tient pas dans la
memoire video n'est pas pour autant hors de portee, a condition d'accepter que
certaines couches soient lues plus lentement que d'autres et de placer au bon
endroit celles qui comptent.

SELECT model, AVG(tokens_per_second) AS throughput
FROM benchmarks WHERE quantization IN ('nvfp4', 'int4') GROUP BY model;
"""


@dataclass
class EvalResult:
    model: str
    perplexity: float = 0.0
    nll: float = 0.0
    tokens: int = 0
    windows: int = 0
    seconds: float = 0.0
    weights_bytes: int = 0
    bits_per_weight: float = 0.0
    formats: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "perplexity": round(self.perplexity, 4),
            "nll": round(self.nll, 6),
            "tokens": self.tokens,
            "windows": self.windows,
            "seconds": round(self.seconds, 2),
            "weights_bytes": self.weights_bytes,
            "bits_per_weight": round(self.bits_per_weight, 3),
            "formats": self.formats,
        }


def _load_corpus(path: Optional[str]) -> str:
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    return DEFAULT_CORPUS


def perplexity(model_dir: str, corpus_path: Optional[str] = None,
               window: int = 512, stride: int = 256,
               max_tokens: int = 8192, device: Optional[str] = None,
               dtype: torch.dtype = torch.bfloat16,
               progress: Optional[Callable[[int, int], None]] = None
               ) -> EvalResult:
    """Sliding-window perplexity over a converted model."""
    from .engine.loader import load_model
    from .engine.model import ForwardBatch
    from .memory.kvcache import BLOCK_SIZE, BlockAllocator
    from .server.chat import load_tokenizer

    t0 = time.time()
    loaded = load_model(model_dir, dtype=dtype, device_override=device)
    tokenizer = load_tokenizer(model_dir)
    if tokenizer is None:
        raise ValueError(f"no tokenizer.json in {model_dir}; "
                         f"perplexity needs one to build the token stream")

    ids = tokenizer.encode(_load_corpus(corpus_path))[:max_tokens]
    if len(ids) < 16:
        raise ValueError("corpus is too short to evaluate")

    model = loaded.model
    result = EvalResult(model=os.path.basename(os.path.abspath(model_dir)))
    result.weights_bytes = model.nbytes
    n_params = loaded.spec.total_params
    result.bits_per_weight = (result.weights_bytes * 8 / n_params) if n_params else 0.0
    for entry in loaded.manifest.get("tensors", {}).values():
        f = entry.get("format", "?")
        result.formats[f] = result.formats.get(f, 0) + 1

    blocks_per_window = (window + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
    total_nll = 0.0
    counted = 0
    n_windows = max(1, (len(ids) - 1 + stride - 1) // stride)

    for w, start in enumerate(range(0, len(ids) - 1, stride)):
        chunk = ids[start:start + window]
        if len(chunk) < 2:
            break
        alloc = BlockAllocator(blocks_per_window, enable_prefix_cache=False)
        blocks = alloc.allocate(blocks_per_window)
        n = len(chunk)
        slots = torch.tensor([blocks[i // BLOCK_SIZE] * BLOCK_SIZE + i % BLOCK_SIZE
                              for i in range(n)], dtype=torch.long)
        batch = ForwardBatch(
            tokens=torch.tensor(chunk, dtype=torch.long),
            positions=torch.arange(n, dtype=torch.long),
            seq_lens=[n], query_lens=[n],
            block_tables=[torch.tensor(blocks, dtype=torch.long)],
            slot_mapping=slots, is_prefill=True)

        logits = model(batch, logits_positions=batch.all_token_indices())
        logits = logits[:-1].to(torch.float32)
        targets = torch.tensor(chunk[1:], dtype=torch.long, device=logits.device)

        # Only score positions this window exposes for the first time, so a
        # token is never counted twice with different amounts of context.
        first_new = 0 if start == 0 else max(0, (window - stride) - 1)
        if first_new >= logits.shape[0]:
            break
        nll = torch.nn.functional.cross_entropy(
            logits[first_new:], targets[first_new:], reduction="sum")
        total_nll += float(nll)
        counted += int(targets[first_new:].numel())
        result.windows += 1
        if progress:
            progress(w + 1, n_windows)
        if start + window >= len(ids):
            break

    result.tokens = counted
    result.nll = total_nll / max(1, counted)
    result.perplexity = math.exp(min(result.nll, 60.0))
    result.seconds = time.time() - t0
    return result


def compare_models(model_dirs: list[str], **kwargs: Any) -> list[EvalResult]:
    """Evaluate several converted models on the same corpus and rank them."""
    out = [perplexity(d, **kwargs) for d in model_dirs]
    return sorted(out, key=lambda r: r.perplexity)


def render(results: list[EvalResult]) -> str:
    if not results:
        return "no results"
    width = max(len(r.model) for r in results)
    lines = [f"  {'model':<{width}}  {'ppl':>9}  {'bpw':>6}  {'size':>10}  "
             f"{'tokens':>8}"]
    best = min(r.perplexity for r in results)
    for r in results:
        delta = "" if r.perplexity == best else f"  (+{100*(r.perplexity/best-1):.1f}%)"
        lines.append(f"  {r.model:<{width}}  {r.perplexity:9.3f}  "
                     f"{r.bits_per_weight:6.2f}  {r.weights_bytes/2**20:8.1f}MiB  "
                     f"{r.tokens:8d}{delta}")
    return "\n".join(lines)
