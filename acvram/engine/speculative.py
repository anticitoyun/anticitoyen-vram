"""Speculative decoding.

Decoding one token at batch 1 is memory bound: the machine reads every active
weight to produce a single token. Verifying K proposed tokens reads those same
weights *once*. So if something cheap can guess the next few tokens and be
right often enough, throughput multiplies by the number of guesses accepted --
for free, in the sense that the expensive part of the step did not get more
expensive.

Two proposers, deliberately different in cost:

``NGramProposer``
    Looks for the current suffix earlier in the context and proposes whatever
    followed it. Costs nothing, needs no model, and is useless in open-ended
    conversation -- but on code editing, RAG answers and summarisation, where
    the output quotes the input heavily, it is often right.

``DraftModelProposer``
    A small model running on a second device. On the target rig that device is
    the RTX 3080 Ti, which the placement planner deliberately leaves idle for
    any model that fits on the 5090. Turning idle silicon into a draft model
    is the best use available for it.

Acceptance is *exact*, not approximate. With a draft distribution q and the
target's p, a proposal x is accepted with probability min(1, p(x)/q(x)) and a
rejection resamples from the normalised positive part of (p - q). The n-gram
proposer has no distribution, so q is a point mass at its proposal: accept
with probability p(x), and on rejection resample from p with that token
removed. Both cases leave the output distribution identical to ordinary
sequential decoding -- speculation buys speed, never a different answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import torch

from .sampler import SamplingParams

__all__ = ["Proposal", "Proposer", "NGramProposer", "DraftModelProposer",
           "verify_proposal", "make_proposer"]


@dataclass
class Proposal:
    tokens: list[int]
    # [k, vocab] draft probabilities, or None for a proposer without a model
    probs: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.tokens)


class Proposer(Protocol):
    name: str

    def propose(self, seq: Any, k: int) -> Proposal: ...

    def commit(self, seq: Any, accepted: list[int]) -> None: ...

    def release(self, seq: Any) -> None: ...


# --------------------------------------------------------------------------
# n-gram / prompt lookup
# --------------------------------------------------------------------------


class NGramProposer:
    """Propose the continuation of the most recent repeated suffix.

    Searches from the longest n-gram down: a longer match is rarer but far
    more likely to be right, so trying 4 before 2 costs one extra scan and
    materially raises the acceptance rate.
    """

    name = "ngram"

    def __init__(self, max_ngram: int = 4, min_ngram: int = 2,
                 max_window: int = 4096) -> None:
        self.max_ngram = max_ngram
        self.min_ngram = min_ngram
        self.max_window = max_window

    def propose(self, seq: Any, k: int) -> Proposal:
        ids = seq.all_ids
        if len(ids) < self.min_ngram + 1:
            return Proposal([])
        window = ids[-self.max_window:]
        base = len(ids) - len(window)
        for n in range(min(self.max_ngram, len(window) - 1), self.min_ngram - 1, -1):
            suffix = window[-n:]
            # Search backwards for the most recent earlier occurrence.
            for start in range(len(window) - n - 1, -1, -1):
                if window[start:start + n] != suffix:
                    continue
                nxt = window[start + n:start + n + k]
                if nxt:
                    return Proposal(list(nxt))
                break
        del base
        return Proposal([])

    def commit(self, seq: Any, accepted: list[int]) -> None:
        return

    def release(self, seq: Any) -> None:
        return


# --------------------------------------------------------------------------
# draft model
# --------------------------------------------------------------------------


@dataclass
class _DraftState:
    blocks: list[int] = field(default_factory=list)
    length: int = 0                   # tokens the draft's cache already holds


class DraftModelProposer:
    """A small model with its own KV cache, kept in step with the target.

    The draft keeps its own paged cache so proposing K tokens costs K small
    decode steps rather than K re-prefills. Its state is synchronised lazily:
    whatever the target accepted since the last call is fed in one batch
    before the new proposals are generated, which also repairs the divergence
    left behind by a rejection.
    """

    name = "draft"

    def __init__(self, loaded: Any, temperature: float = 0.0,
                 max_model_len: int = 8192) -> None:
        from ..memory.kvcache import BLOCK_SIZE, BlockAllocator

        self.loaded = loaded
        self.model = loaded.model
        self.temperature = temperature
        self.max_model_len = max_model_len
        self.block_size = BLOCK_SIZE
        n_blocks = min((c.cfg.num_blocks for c in self.model.caches.values()),
                       default=512)
        self.allocator = BlockAllocator(n_blocks, enable_prefix_cache=False)
        self.state: dict[int, _DraftState] = {}

    # -- plumbing --------------------------------------------------------
    def _ensure_blocks(self, st: _DraftState, needed_tokens: int) -> bool:
        need = (needed_tokens + self.block_size - 1) // self.block_size
        if need <= len(st.blocks):
            return True
        extra = need - len(st.blocks)
        if self.allocator.num_free < extra:
            return False
        st.blocks.extend(self.allocator.allocate(extra))
        return True

    def _batch(self, st: _DraftState, tokens: list[int], start: int,
               prefill: bool):
        from .model import ForwardBatch
        slots = [st.blocks[(start + i) // self.block_size] * self.block_size
                 + (start + i) % self.block_size for i in range(len(tokens))]
        return ForwardBatch(
            tokens=torch.tensor(tokens, dtype=torch.long),
            positions=torch.arange(start, start + len(tokens), dtype=torch.long),
            seq_lens=[start + len(tokens)], query_lens=[len(tokens)],
            block_tables=[torch.tensor(st.blocks, dtype=torch.long)],
            slot_mapping=torch.tensor(slots, dtype=torch.long),
            is_prefill=prefill)

    def _sync(self, seq: Any) -> Optional[_DraftState]:
        st = self.state.setdefault(seq.id, _DraftState())
        ids = seq.all_ids
        if len(ids) > self.max_model_len:
            return None
        # Everything except the final token: that one is fed as the first step
        # of proposing, so its logits become the first proposal.
        target = len(ids) - 1
        if st.length >= target:
            return st
        if not self._ensure_blocks(st, len(ids) + 8):
            return None
        pending = ids[st.length:target]
        if pending:
            batch = self._batch(st, pending, st.length, prefill=len(pending) > 1)
            self.model(batch)
            st.length = target
        return st

    # -- interface -------------------------------------------------------
    def propose(self, seq: Any, k: int) -> Proposal:
        st = self._sync(seq)
        if st is None:
            return Proposal([])
        ids = seq.all_ids
        tokens: list[int] = []
        probs: list[torch.Tensor] = []
        cur = ids[-1]
        pos = len(ids) - 1
        for _ in range(k):
            if not self._ensure_blocks(st, pos + 2):
                break
            batch = self._batch(st, [cur], pos, prefill=False)
            logits = self.model(batch)[0].to(torch.float32)
            p = torch.softmax(logits / max(self.temperature, 1e-5), dim=-1)
            tok = int(p.argmax()) if self.temperature <= 0 else \
                int(torch.multinomial(p, 1))
            tokens.append(tok)
            probs.append(p)
            pos += 1
            st.length = pos
            cur = tok
        if not tokens:
            return Proposal([])
        return Proposal(tokens, torch.stack(probs))

    def commit(self, seq: Any, accepted: list[int]) -> None:
        # A rejection leaves the draft's cache holding tokens the target did
        # not take. Rewinding the length is enough: those slots are overwritten
        # on the next sync, and nothing reads past `length`.
        st = self.state.get(seq.id)
        if st is not None:
            st.length = min(st.length, len(seq.all_ids) - 1)

    def release(self, seq: Any) -> None:
        st = self.state.pop(seq.id, None)
        if st is not None and st.blocks:
            self.allocator.free(st.blocks)


# --------------------------------------------------------------------------
# acceptance
# --------------------------------------------------------------------------


def _target_probs(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    if params.temperature <= 0:
        out = torch.zeros_like(logits)
        out[int(logits.argmax())] = 1.0
        return out
    return torch.softmax(logits / params.temperature, dim=-1)


def verify_proposal(logits: torch.Tensor, proposal: Proposal,
                    params: SamplingParams,
                    generator: Optional[torch.Generator] = None
                    ) -> tuple[list[int], int]:
    """Accept a prefix of the proposal, then emit exactly one more token.

    ``logits`` is ``[k + 1, vocab]``: row *i* is the target's prediction for
    the position proposal token *i* would occupy, and the final row is the
    prediction that follows a fully accepted proposal.

    Returns the tokens to append and how many proposals were accepted. The
    result is always at least one token long, so a step can never stall.
    """
    k = len(proposal)
    accepted: list[int] = []
    logits = logits.to(torch.float32)

    for i in range(k):
        p = _target_probs(logits[i], params)
        x = proposal.tokens[i]
        q_x = 1.0 if proposal.probs is None else float(proposal.probs[i, x])
        p_x = float(p[x])
        if q_x <= 0:
            keep = False
        elif p_x >= q_x:
            keep = True
        else:
            keep = bool(torch.rand((), generator=generator).item() < p_x / q_x)

        if keep:
            accepted.append(x)
            continue

        # Rejected. Resample from the normalised positive part of (p - q) so
        # the overall distribution stays exactly the target's.
        residual = p.clone()
        if proposal.probs is None:
            residual[x] = 0.0
        else:
            residual = torch.clamp(p - proposal.probs[i].to(p.dtype), min=0.0)
        total = float(residual.sum())
        if total <= 0:
            residual = p.clone()
            residual[x] = 0.0
            total = float(residual.sum())
        if total <= 0:
            return accepted + [x], len(accepted)
        residual = residual / total
        tok = int(torch.multinomial(residual, 1, generator=generator))
        return accepted + [tok], len(accepted)

    # Every proposal accepted: the final row gives one free bonus token.
    p = _target_probs(logits[k], params)
    tok = int(p.argmax()) if params.temperature <= 0 else \
        int(torch.multinomial(p, 1, generator=generator))
    return accepted + [tok], len(accepted)


def make_proposer(kind: str, **kwargs: Any) -> Optional[Proposer]:
    if kind in ("", "none", "off"):
        return None
    if kind == "ngram":
        return NGramProposer(**{k: v for k, v in kwargs.items()
                                if k in ("max_ngram", "min_ngram", "max_window")})
    if kind == "draft":
        return DraftModelProposer(**{k: v for k, v in kwargs.items()
                                     if k in ("loaded", "temperature",
                                              "max_model_len")})
    raise ValueError(f"unknown speculator {kind!r}; use ngram, draft or none")
