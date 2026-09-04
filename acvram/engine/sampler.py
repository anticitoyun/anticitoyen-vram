"""Échantillonnage des jetons : température, top-k, top-p, pénalités de
répétition et de présence.

Tout s'exécute par lot sur le tenseur de logits. L'ordre est celui qu'implique
l'API OpenAI : les pénalités ajustent les logits bruts, puis la température,
puis les filtres de troncature, puis un unique tirage multinomial.
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
            # Diviser les valeurs positives et multiplier les négatives garde
            # le sens de la pénalité cohérent quel que soit le signe du logit.
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
    """Rend ``(identifiants, logprobs des choisis)`` pour un tenseur ``[lot, vocabulaire]``."""
    penalites = history is not None and any(
        p.repetition_penalty != 1.0 or p.presence_penalty or p.frequency_penalty
        for p in params)
    # Le clone n'existe que pour les pénalités, qui écrivent en place ; sans
    # elles il recopiait 600 Ko de logits par jeton pour rien.
    logits = logits.to(torch.float32)
    if penalites:
        logits = _apply_penalties(logits.clone(), history, params)

    # Tout le lot en glouton, sans penalite : un argmax suffit. Le chemin
    # general fait un tri, deux softmax et un tirage multinomial sur tout le
    # vocabulaire — plusieurs millisecondes par pas que le decodage a
    # temperature nulle payait pour rien.
    if all(p.greedy for p in params):
        tokens = logits.argmax(dim=-1)
        # log p(choisi) = logit - logsumexp : une passe de réduction, là où
        # log_softmax matérialisait tout le vocabulaire avant d'en lire une
        # seule case.
        logprobs = (logits.gather(1, tokens.unsqueeze(-1)).squeeze(-1)
                    - torch.logsumexp(logits, dim=-1))
        return tokens, logprobs

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
            # On garde le premier jeton qui franchit le seuil, pour que top_p
            # ne vide jamais la distribution.
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
