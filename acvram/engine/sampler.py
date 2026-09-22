"""Échantillonnage des jetons : température, top-k, top-p, pénalités de
répétition et de présence.

Tout s'exécute par lot sur le tenseur de logits. L'ordre est celui qu'implique
l'API OpenAI : les pénalités ajustent les logits bruts, puis la température,
puis les filtres de troncature, puis un unique tirage multinomial.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import torch

__all__ = ["SamplingParams", "sample", "sampler_lot_actif", "sampler_texte"]


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
    # Même sémantique que vLLM/llama-server (sage-harnais-egal-ignore-eos-18-09) :
    # continue au-delà de l'EOS jusqu'à `max_tokens`, sans toucher aux
    # `stop`/`stop_token_ids` explicites de l'appelant. Sans lui, un harnais
    # comparatif qui force ignore_eos chez llama.cpp mais pas ici fait tomber
    # le lot acvram sous b à chaque séquence qui atteint l'EOS avant les
    # autres — asymétrie non neutre, cellule b=12 indécidable.
    ignore_eos: bool = False

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


def besoin_historique(params) -> bool:
    """L historique des jetons deja produits est-il LU par l echantillonnage ?

    Il ne l est que hors du chemin glouton : les penalites de repetition et de
    presence sont les seules a le consulter. L exposer ici plutot que de le
    tester chez l appelant evite que les deux conditions divergent — celle de
    `sample` et celle qui decide de CONSTRUIRE l historique. Un historique
    construit puis jamais lu coute une concatenation par sequence et par pas,
    proportionnelle au contexte total : a 4000 jetons c est un travail qui
    grandit a chaque jeton produit, pour rien.
    """
    return not all(p.greedy for p in params)



def sampler_lot_actif() -> bool:
    """Chemin vectorisé en OPT-IN (`ACVRAM_SAMPLER_LOT=1`) : verdict Manon 21/09 (eea064fe, 6 fenêtres) — b=12 lot 1 506,2
    contre lent 1 540,4 t/s, B/A 0,978, RÉFUTÉ ; GLM b=1 +0,9 % seulement. Le défaut reste l ancienne fonction
    (`_sample_lent`) tant que la cause n est pas nommée sur carte ; la ligne de régime porte `sampler=lent|lot`."""
    return os.environ.get("ACVRAM_SAMPLER_LOT", "0") == "1"


def sampler_texte() -> str:
    return "lot" if sampler_lot_actif() else "lent"


def _colonne(valeurs, device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(valeurs, device=device, dtype=dtype).unsqueeze(-1)


def sample(logits: torch.Tensor, params: list[SamplingParams],
           history: Optional[list[list[int]]] = None,
           generator: Optional[torch.Generator] = None
           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Rend ``(identifiants, logprobs des choisis)`` pour un tenseur ``[lot, vocabulaire]``.

    Vectorisé sur le lot (Laurine `laurine-h2-frontiere-pas-21-09` : 14 lancements par pas à la frontière, 0,43 ms
    à b=12, pas de glue MoE) : un topk, un softmax, un tri, un multinomial pour tout le lot, jamais une boucle par
    ligne. Chemin glouton (le décodage mesuré) : un argmax dans le dtype des logits, sans cast fp32 ni logsumexp ;
    les logprobs ne se calculent que si une séquence les demande (`params.logprobs`), sinon zéros — le serveur ne
    les sert pas. Équivalence : ids au bit avec `_sample_lent` à générateur identique, tests/test_sampler_lot.py.
    OPT-IN `ACVRAM_SAMPLER_LOT=1` (réfuté en service le 21/09, voir `sampler_lot_actif`)."""
    if not sampler_lot_actif():
        return _sample_lent(logits, params, history, generator)
    penalites = history is not None and any(
        p.repetition_penalty != 1.0 or p.presence_penalty or p.frequency_penalty
        for p in params)
    veut_logprobs = any(p.logprobs is not None for p in params)
    if penalites:
        logits = _apply_penalties(logits.to(torch.float32).clone(), history, params)

    if not besoin_historique(params):
        tokens = logits.argmax(dim=-1)                       # argmax exact dans le dtype d origine (cast fp32 inutile)
        if not veut_logprobs:
            return tokens, torch.zeros_like(tokens, dtype=torch.float32)
        l32 = logits.to(torch.float32)
        logprobs = (l32.gather(1, tokens.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(l32, dim=-1))
        return tokens, logprobs

    logits = logits.to(torch.float32)
    d = logits.device
    V = logits.shape[-1]
    temps = _colonne([max(p.temperature, 1e-5) for p in params], d)
    greedy_mask = torch.tensor([p.greedy for p in params], device=d)
    scaled = logits / temps
    neg_inf = torch.full_like(scaled, float("-inf"))

    ks = [p.top_k if 0 < p.top_k < V else 0 for p in params]
    if any(ks):                                              # k-ième plus grand par ligne, seuil −inf pour les lignes sans top_k
        kmax = max(ks)
        top = torch.topk(scaled, kmax, dim=-1).values
        kth = top.gather(1, _colonne([max(k, 1) - 1 for k in ks], d, torch.long))
        kth = torch.where(_colonne([bool(k) for k in ks], d, torch.bool), kth, torch.full_like(kth, float("-inf")))
        scaled = torch.where(scaled < kth, neg_inf, scaled)
    if any(p.min_p > 0 for p in params):
        probs = torch.softmax(scaled, dim=-1)
        thresh = _colonne([p.min_p for p in params], d) * probs.amax(dim=-1, keepdim=True)
        scaled = torch.where(probs < thresh, neg_inf, scaled)
    if any(0.0 < p.top_p < 1.0 for p in params):
        sorted_logits, sorted_idx = torch.sort(scaled, dim=-1, descending=True)
        sm = torch.softmax(sorted_logits, dim=-1)
        cumulative = sm.cumsum(dim=-1)
        # On garde le premier jeton qui franchit le seuil, pour que top_p ne vide jamais la distribution ;
        # 2.0 pour les lignes sans top_p : jamais franchi
        seuil = _colonne([p.top_p if 0.0 < p.top_p < 1.0 else 2.0 for p in params], d)
        sorted_logits = sorted_logits.masked_fill(cumulative - sm > seuil, float("-inf"))
        scaled = neg_inf.scatter(1, sorted_idx, sorted_logits)

    probs = torch.softmax(scaled, dim=-1)
    drawn = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    argmax = scaled.argmax(dim=-1)
    tokens = torch.where(greedy_mask, argmax, drawn)
    if not veut_logprobs:
        return tokens, torch.zeros_like(tokens, dtype=torch.float32)
    logprobs = torch.log_softmax(scaled, dim=-1).gather(1, tokens.unsqueeze(-1)).squeeze(-1)
    return tokens, logprobs

def _sample_lent(logits: torch.Tensor, params: list[SamplingParams],
           history: Optional[list[list[int]]] = None,
           generator: Optional[torch.Generator] = None
           ) -> tuple[torch.Tensor, torch.Tensor]:
    """L ancienne fonction, une boucle par ligne du lot — LE DÉFAUT EN SERVICE (verdict eea064fe) et le témoin du test
    d équivalence de `sample` (mêmes ids à générateur identique).""" 
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
    if not besoin_historique(params):
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
