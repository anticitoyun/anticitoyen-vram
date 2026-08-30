"""Token sampling: temperature, top-k, top-p, repetition and presence penalties.

Everything runs batched on the logits tensor. The order is the one the OpenAI
API implies: penalties adjust the raw logits, then temperature, then the
truncation filters, then a single multinomial draw.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

__all__ = ["SamplingParams", "sample"]


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    max_tokens: int = 512
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    seed: Optional[int] = None
    n: int = 1
    logprobs: Optional[int] = None

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


def _apply_penalties(logits: torch.Tensor, history: list[list[int]],
                     params: list[SamplingParams]) -> torch.Tensor:
    for i, (toks, p) in enumerate(zip(history, params)):
        if not toks:
            continue
        if p.repetition_penalty == 1.0 and p.presence_penalty == 0.0 \
                and p.frequency_penalty == 0.0:
            continue
        ids = torch.tensor(sorted(set(toks)), device=logits.device, dtype=torch.long)
        if p.repetition_penalty != 1.0:
            vals = logits[i, ids]
            # Dividing positives and multiplying negatives keeps the direction
            # of the penalty consistent regardless of the logit's sign.
            logits[i, ids] = torch.where(
                vals > 0, vals / p.repetition_penalty, vals * p.repetition_penalty)
        if p.presence_penalty:
            logits[i, ids] -= p.presence_penalty
        if p.frequency_penalty:
            counts = torch.bincount(
                torch.tensor(toks, device=logits.device, dtype=torch.long),
                minlength=logits.shape[-1]).to(logits.dtype)
            logits[i] -= p.frequency_penalty * counts
    return logits


def sample(logits: torch.Tensor, params: list[SamplingParams],
           history: Optional[list[list[int]]] = None,
           generator: Optional[torch.Generator] = None
           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(token_ids, logprobs_of_chosen)`` for a ``[batch, vocab]`` tensor."""
    logits = logits.to(torch.float32).clone()
    if history is not None:
        logits = _apply_penalties(logits, history, params)

    temps = torch.tensor([max(p.temperature, 1e-5) for p in params],
                         device=logits.device).unsqueeze(-1)
    greedy_mask = torch.tensor([p.greedy for p in params], device=logits.device)
    scaled = logits / temps

    for i, p in enumerate(params):
        if p.top_k and p.top_k < scaled.shape[-1]:
            kth = torch.topk(scaled[i], p.top_k).values[-1]
            scaled[i] = torch.where(scaled[i] < kth,
                                    torch.full_like(scaled[i], float("-inf")),
                                    scaled[i])
        if p.min_p > 0:
            probs = torch.softmax(scaled[i], dim=-1)
            thresh = p.min_p * probs.max()
            scaled[i] = torch.where(probs < thresh,
                                    torch.full_like(scaled[i], float("-inf")),
                                    scaled[i])
        if 0.0 < p.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(scaled[i], descending=True)
            cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            # Keep the first token that crosses the threshold, so top_p never
            # empties the distribution.
            drop = cumulative - torch.softmax(sorted_logits, dim=-1) > p.top_p
            sorted_logits[drop] = float("-inf")
            scaled[i] = torch.full_like(scaled[i], float("-inf")).scatter(
                0, sorted_idx, sorted_logits)

    probs = torch.softmax(scaled, dim=-1)
    drawn = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    argmax = scaled.argmax(dim=-1)
    tokens = torch.where(greedy_mask, argmax, drawn)
    logprobs = torch.log_softmax(scaled, dim=-1).gather(
        1, tokens.unsqueeze(-1)).squeeze(-1)
    return tokens, logprobs
