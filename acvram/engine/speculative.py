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

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import torch

from .sampler import SamplingParams

__all__ = ["Proposal", "Proposer", "NGramProposer", "DraftModelProposer",
           "verify_proposal", "make_proposer", "GardeSpeculation"]


class GardeSpeculation:
    """Decide si la speculation doit s'activer ce pas.

    Mesure du 14/09 (revue/verdict-cout-verification-ngram-b12-14-09.md) :
    a b_reel=12 la carte est deja pleine a largeur 1 par sequence, verifier
    jusqu'a spec_k+1 par sequence n'a plus de marge a absorber gratuitement
    -- le debit mesure a ete divise par deux (481,4 -> 241,4 t/s). A b_reel=1
    (revue/verdict-taux-ngram-code-13-09.md, taux 1,6137) la speculation
    gagne. Deux gardes, l'une a priori, l'autre mesuree :

    1. seuil de lot : au-dessus de `lot_max`, jamais eligible (le regime ou
       la carte n'a plus de marge commence a un lot connu, pas a deviner) ;
    2. garde glissante : meme sous le seuil, si le gain reel moyen sur les
       dernieres `fenetre` pas speculatifs tombe sous `gain_min` (jetons
       emis par pas / b_reel), la speculation se desactive -- le lot seul ne
       capture pas tous les regimes ou l'acceptation ne paie pas. Elle se
       reactive quand le lot redescend a nouveau sous le seuil apres etre
       monte au-dessus (transition haute -> basse) : une chance neuve, pas
       une desactivation permanente.
    """

    def __init__(self, lot_max: int, fenetre: int = 32, gain_min: float = 1.05):
        self.lot_max = lot_max
        self.gain_min = gain_min
        self._fenetre: deque = deque(maxlen=fenetre)
        self._desactive = False
        self._dernier_lot: int | None = None

    def eligible(self, b_reel: int) -> bool:
        if (self._dernier_lot is not None and self._dernier_lot > self.lot_max
                and b_reel <= self.lot_max):
            self._desactive = False
            self._fenetre.clear()
        self._dernier_lot = b_reel
        return b_reel <= self.lot_max and not self._desactive

    def enregistrer(self, jetons_emis: int, b_reel: int) -> None:
        """A appeler apres un pas ou la speculation a reellement tourne."""
        if b_reel <= 0:
            return
        self._fenetre.append(jetons_emis / b_reel)
        if len(self._fenetre) == self._fenetre.maxlen:
            moyenne = sum(self._fenetre) / len(self._fenetre)
            if moyenne < self.gain_min:
                self._desactive = True

    def etat_dict(self, mode: str) -> dict:
        """État de la garde : mode, activité, gain moyen glissant, seuil de lot.
        Destiné à la ligne de régime et à /metrics — le régime se porte par le
        nom, pas par la vigilance (REGLES §6)."""
        gain = round(sum(self._fenetre) / len(self._fenetre), 3) if self._fenetre else None
        return {
            "mode": mode,
            "garde_active": not self._desactive,
            "gain_moyen": gain,
            "lot_max": self.lot_max,
        }


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
        # En glouton, la vérification n'a pas besoin des probabilités du
        # brouillon (q_x vaut 1) : ni softmax sur 152 000 entrées, ni pile de
        # k x vocabulaire flottants à chaque pas.
        glouton = self.temperature <= 0
        for _ in range(k):
            if not self._ensure_blocks(st, pos + 2):
                break
            batch = self._batch(st, [cur], pos, prefill=False)
            logits = self.model(batch)[0]
            if glouton:
                tok = int(logits.argmax())
            else:
                p = torch.softmax(logits.to(torch.float32)
                                  / max(self.temperature, 1e-5), dim=-1)
                tok = int(torch.multinomial(p, 1))
                probs.append(p)
            tokens.append(tok)
            pos += 1
            st.length = pos
            cur = tok
        if not tokens:
            return Proposal([])
        return Proposal(tokens, None if glouton else torch.stack(probs))

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


# --------------------------------------------------------------------------
# tête de prédiction multi-jetons
# --------------------------------------------------------------------------


class MTPProposer:
    """Brouillon tiré de la couche ``nextn`` du modèle lui-même.

    Un modèle brouillon séparé coûte un modèle entier : mesuré ici, le débit
    tombait de 152 à 30 t/s malgré 78 % d'acceptation. La tête MTP, elle, est
    un unique bloc de transformeur — un soixante-quatrième d'un 27B — et elle a
    été entraînée avec le modèle, sur son propre état caché.

    Elle ne sert qu'au décodage d'une séquence à la fois : l'état caché de la
    cible est repris dans un tampon unique, sans mémoire de l'ordre des
    séquences d'un lot. C'est le cas interactif, celui où la latence compte.
    """

    name = "mtp"

    def __init__(self, model: Any, temperature: float = 0.0,
                 max_model_len: int = 8192) -> None:
        from ..memory.kvcache import BLOCK_SIZE, BlockAllocator

        self.model = model
        self.tete = model.mtp
        self.temperature = temperature
        self.max_model_len = max_model_len
        self.block_size = BLOCK_SIZE
        self.allocator = BlockAllocator(self.tete.cache.cfg.num_blocks,
                                        enable_prefix_cache=False)
        self.state: dict[int, _DraftState] = {}
        self._ligne = 0            # ligne du dernier lot vérifié à reprendre

    def _ensure_blocks(self, st: _DraftState, needed_tokens: int) -> bool:
        need = (needed_tokens + self.block_size - 1) // self.block_size
        if need <= len(st.blocks):
            return True
        extra = need - len(st.blocks)
        if self.allocator.num_free < extra:
            return False
        st.blocks.extend(self.allocator.allocate(extra))
        return True

    def _batch(self, st: _DraftState, tokens: list[int], start: int):
        from .model import ForwardBatch
        slots = [st.blocks[(start + i) // self.block_size] * self.block_size
                 + (start + i) % self.block_size for i in range(len(tokens))]
        return ForwardBatch(
            tokens=torch.tensor(tokens, dtype=torch.long),
            positions=torch.arange(start, start + len(tokens), dtype=torch.long),
            seq_lens=[start + len(tokens)], query_lens=[len(tokens)],
            block_tables=[torch.tensor(st.blocks, dtype=torch.long)],
            slot_mapping=torch.tensor(slots, dtype=torch.long),
            is_prefill=False)

    def _hidden_cible(self) -> Optional[torch.Tensor]:
        """État caché de la cible pour le dernier jeton *retenu*.

        Le pas de vérification traite le jeton réel puis les propositions ; les
        états cachés en ressortent tous. Prendre la dernière ligne reviendrait à
        se fier à l'état d'un jeton que la cible vient peut-être de rejeter — la
        tête partait alors d'un contexte imaginaire, et l'acceptation tombait de
        50 % (mesuré en teacher forcing) à 11 %.
        """
        h = getattr(self.model, "_mtp_hidden", None)
        if h is None or h.numel() == 0:
            return None
        # Le tampon est réservé plus large que le pas : seules les ``n``
        # premières lignes datent de ce pas (``n`` posé par le moteur).
        n = getattr(self.model, "_mtp_hidden_n", 0) or h.shape[0]
        h = h[:n].reshape(-1, h.shape[-1])
        i = min(self._ligne, h.shape[0] - 1)
        return h[i:i + 1]

    def _amorcer(self, seq: Any, st: _DraftState) -> bool:
        """Remplit le cache de la tête avec le contexte de l'invite.

        Sans cela la tête n'a qu'un jeton d'historique et son attention ne voit
        rien : mesuré, le taux d'acceptation tombait à 14 %. Le prefill d'une
        couche unique coûte un soixante-quatrième de celui du modèle.
        """
        hs = getattr(self.model, "_mtp_prefill", None)
        ids = seq.all_ids
        if hs is None or hs.shape[0] < len(ids) - 1:
            return False
        n = len(ids) - 1
        if not self._ensure_blocks(st, n + 8):
            return False
        emb = self.model.embed_tokens
        toks = torch.tensor(ids[1:n + 1], dtype=torch.long, device=emb.device)
        e = torch.nn.functional.embedding(toks, emb)
        self.tete(e.to(hs.dtype), hs[:n], self._batch(st, ids[1:n + 1], 0))
        st.length = n
        return True

    def propose(self, seq: Any, k: int) -> Proposal:
        ids = seq.all_ids
        if len(ids) > self.max_model_len:
            return Proposal([])
        h = self._hidden_cible()
        if h is None:
            return Proposal([])                        # avant le premier pas
        st = self.state.setdefault(seq.id, _DraftState())
        if st.length == 0 and len(ids) > 2 and not self._amorcer(seq, st):
            return Proposal([])
        st.length = min(st.length, len(ids) - 1)

        emb = self.model.embed_tokens
        tete, lm = self.tete, self.model.lm_head
        glouton = self.temperature <= 0
        tokens: list[int] = []
        probs: list[torch.Tensor] = []
        cur, pos = ids[-1], len(ids) - 1
        for _ in range(k):
            if not self._ensure_blocks(st, pos + 2):
                break
            e = torch.nn.functional.embedding(
                torch.tensor([cur], dtype=torch.long, device=emb.device), emb)
            sortie = tete(e.to(h.dtype), h, self._batch(st, [cur], pos))
            logits = lm(sortie.to(lm.qweight.qweight.device
                                  if hasattr(lm.qweight, "qweight")
                                  else sortie.device))[0]
            if glouton:
                tok = int(logits.argmax())
            else:
                p = torch.softmax(logits.to(torch.float32)
                                  / max(self.temperature, 1e-5), dim=-1)
                tok = int(torch.multinomial(p, 1))
                probs.append(p)
            tokens.append(tok)
            h = sortie[-1:]                            # la tête se relit
            cur, pos = tok, pos + 1
            st.length = pos
        if not tokens:
            return Proposal([])
        return Proposal(tokens, None if glouton else torch.stack(probs))

    def commit(self, seq: Any, accepted: list[int]) -> None:
        self._ligne = len(accepted)
        st = self.state.get(seq.id)
        if st is not None:
            st.length = min(st.length, len(seq.all_ids) - 1)

    def release(self, seq: Any) -> None:
        self._ligne = 0
        st = self.state.pop(seq.id, None)
        if st is not None and st.blocks:
            self.allocator.free(st.blocks)
