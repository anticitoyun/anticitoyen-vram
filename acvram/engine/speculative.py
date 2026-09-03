"""Décodage spéculatif.

Décoder un jeton avec un lot de taille 1 est limité par la mémoire : la machine
lit tous les poids actifs pour produire un seul jeton. Vérifier K jetons
proposés lit ces mêmes poids *une seule fois*. Si donc quelque chose de bon
marché sait deviner les quelques jetons suivants et tombe juste assez souvent,
le débit est multiplié par le nombre de propositions acceptées — gratuitement,
au sens où la partie coûteuse de l'étape n'est pas devenue plus coûteuse.

Deux propositeurs, de coûts délibérément différents :

``NGramProposer``
    Cherche le suffixe courant plus tôt dans le contexte et propose ce qui le
    suivait. Ne coûte rien, ne demande aucun modèle, et ne sert à rien dans une
    conversation libre — mais sur l'édition de code, les réponses de RAG et le
    résumé, où la sortie recopie largement l'entrée, il tombe souvent juste.

``DraftModelProposer``
    Un petit modèle tournant sur un second appareil. Sur la machine cible, cet
    appareil est la RTX 3080 Ti, que le planificateur de placement laisse
    volontairement oisive pour tout modèle qui tient sur la 5090. Transformer du
    silicium inutilisé en modèle brouillon est le meilleur usage disponible.

L'acceptation est *exacte*, pas approchée. Avec une distribution de brouillon q
et celle de la cible p, une proposition x est acceptée avec la probabilité
min(1, p(x)/q(x)), et un rejet rééchantillonne dans la partie positive
normalisée de (p − q). Le propositeur par n-grammes n'a pas de distribution : q
est alors une masse de Dirac sur sa proposition, donc on accepte avec la
probabilité p(x) et, en cas de rejet, on rééchantillonne dans p privé de ce
jeton. Dans les deux cas la distribution de sortie reste identique à celle d'un
décodage séquentiel ordinaire — la spéculation achète de la vitesse, jamais une
réponse différente.
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
    # [k, vocabulaire] probabilités du brouillon, ou None sans modèle
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
    """Propose la suite du suffixe répété le plus récent.

    Cherche du n-gramme le plus long au plus court : une correspondance longue
    est plus rare mais bien plus souvent juste, si bien qu'essayer 4 avant 2
    coûte un balayage de plus et relève sensiblement le taux d'acceptation.

    L'index des n-grammes est tenu à jour au fil de l'eau, une entrée par
    jeton et par longueur : proposer coûte alors trois consultations de
    dictionnaire. Le balayage arrière d'une fenêtre de 4096 jetons, en Python
    pur et à chaque pas, coûtait à lui seul près d'un dixième du débit — plus
    que la spéculation ne rapportait sur du texte non répétitif.
    """

    name = "ngram"

    def __init__(self, max_ngram: int = 4, min_ngram: int = 2,
                 max_window: int = 4096, adaptatif: bool = True,
                 seuil: float = 0.15, fenetre: int = 24,
                 pause: int = 64) -> None:
        self.max_ngram = max_ngram
        self.min_ngram = min_ngram
        self.max_window = max_window
        # Une proposition rejetée n'est pas gratuite : le pas de vérification
        # coûte quelques pour cent de plus qu'un pas simple. Le proposeur
        # surveille son rendement et se met en veille quand il gagne moins de
        # « seuil » jeton par pas, en réessayant périodiquement.
        self.adaptatif = adaptatif
        self.seuil = seuil
        self.fenetre = fenetre
        self.pause = pause
        self._etat: dict[Any, dict] = {}

    def _cle(self, seq: Any):
        return seq.id if hasattr(seq, "id") else id(seq)

    def _st(self, seq: Any) -> dict:
        st = self._etat.get(self._cle(seq))
        if st is None:
            st = {"index": {n: {} for n in range(self.min_ngram, self.max_ngram + 1)},
                  "vus": {n: 0 for n in range(self.min_ngram, self.max_ngram + 1)},
                  "pas": 0, "gain": 0, "veille": 0}
            self._etat[self._cle(seq)] = st
        return st

    def _indexer(self, st: dict, ids) -> None:
        """Enregistre les n-grammes qui se terminent avant le suffixe courant."""
        L = len(ids)
        for n in range(self.min_ngram, self.max_ngram + 1):
            d = st["index"][n]
            j = st["vus"][n]
            fin = L - n                      # le suffixe courant n'est pas indexé
            while j < fin:
                d[tuple(ids[j:j + n])] = j + n
                j += 1
            st["vus"][n] = max(j, 0)

    def propose(self, seq: Any, k: int) -> Proposal:
        ids = seq.all_ids
        if len(ids) < self.min_ngram + 1 or k <= 0:
            return Proposal([])
        st = self._st(seq)
        self._indexer(st, ids)
        if self.adaptatif:
            if st["veille"] > 0:
                st["veille"] -= 1
                return Proposal([])
            # le pas est compté ici : un pas sans proposition pèse aussi dans
            # le rendement, et c'est le cas le plus fréquent en prose
            st["pas"] += 1
            if st["pas"] >= self.fenetre:
                if st["gain"] / st["pas"] < self.seuil:
                    st["veille"] = self.pause
                st["pas"] = st["gain"] = 0
        for n in range(min(self.max_ngram, len(ids) - 1), self.min_ngram - 1, -1):
            p = st["index"][n].get(tuple(ids[-n:]))
            if p is None:
                continue
            nxt = ids[p:p + k]
            if nxt:
                return Proposal(list(nxt))
        return Proposal([])

    def commit(self, seq: Any, accepted: list[int]) -> None:
        if not self.adaptatif:
            return
        st = self._st(seq)
        st["gain"] += max(0, len(accepted) - 1)

    def release(self, seq: Any) -> None:
        self._etat.pop(self._cle(seq), None)


# --------------------------------------------------------------------------
# draft model
# --------------------------------------------------------------------------


@dataclass
class _DraftState:
    blocks: list[int] = field(default_factory=list)
    length: int = 0                   # tokens the draft's cache already holds


class DraftModelProposer:
    """Un petit modèle avec son propre cache KV, tenu au pas avec la cible.

    Le brouillon garde son propre cache paginé, si bien que proposer K jetons
    coûte K petites étapes de décodage plutôt que K précalculs complets. Son
    état est synchronisé paresseusement : tout ce que la cible a accepté depuis
    le dernier appel lui est présenté en un lot avant de produire de nouvelles
    propositions, ce qui répare aussi la divergence laissée par un rejet.
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
        # Tout sauf le dernier jeton : celui-là est présenté à la première
        # étape de proposition, si bien que ses logits deviennent la première
        # proposition.
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
        # Un rejet laisse dans le cache du brouillon des jetons que la cible
        # n'a pas retenus. Rembobiner la longueur suffit : ces emplacements sont
        # écrasés à la prochaine synchronisation, et rien ne lit au-delà de
        # `length`.
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
    """Accepte un préfixe de la proposition, puis émet exactement un jeton de plus.

    ``logits`` vaut ``[k + 1, vocabulaire]`` : la ligne *i* est la prédiction de
    la cible pour la position qu'occuperait le jeton proposé *i*, et la dernière
    ligne est la prédiction qui suit une proposition entièrement acceptée.

    Rend les jetons à ajouter et le nombre de propositions acceptées. Le
    résultat fait toujours au moins un jeton, si bien qu'une étape ne peut
    jamais caler.
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

        # Rejeté. On rééchantillonne dans la partie positive normalisée de
        # (p − q), pour que la distribution globale reste exactement celle de la
        # cible.
        residual = p.clone()
        if proposal.probs is None:
            residual[x] = 0.0
        else:
            # Le brouillon peut vivre sur une autre carte que la cible : ses
            # probabilites arrivent sur son peripherique a lui.
            residual = torch.clamp(
                p - proposal.probs[i].to(device=p.device, dtype=p.dtype),
                min=0.0)
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

    # Toutes les propositions acceptées : la dernière ligne offre un jeton en prime.
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
