"""Ordonnancement et génération : lot continu sur un cache KV paginé.

Les requêtes arrivent n'importe quand et se terminent à des longueurs
différentes : le moteur fait donc tourner un lot *continu*. À chaque étape, il
admet toutes les nouvelles requêtes que les blocs KV libres peuvent payer,
décode un jeton pour tout ce qui tourne déjà, et évince les séquences dès
qu'elles s'arrêtent. Rien n'attend une frontière de lot.

Le moteur est monothread à dessein. Les couches du modèle sont réparties sur
deux GPU et la mémoire hôte, et une étape les touche toutes en série ; ajouter
des fils par-dessus ferait se disputer les mêmes appareils sans ajouter de
parallélisme. La concurrence vient du lot, pas des fils, et le serveur asyncio
fait entrer et sortir le travail par une file.
"""

from __future__ import annotations

import itertools
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

import torch

from ..memory.kvcache import BLOCK_SIZE, BlockAllocator
from .loader import LoadedModel
from .model import ForwardBatch
from .sampler import SamplingParams, sample
from .speculative import Proposal, verify_proposal

__all__ = ["Sequence", "GenerationOutput", "Engine", "EngineStats"]

_ids = itertools.count(1)


@dataclass
class Sequence:
    prompt_ids: list[int]
    params: SamplingParams
    request_id: str = ""
    id: int = field(default_factory=lambda: next(_ids))
    output_ids: list[int] = field(default_factory=list)
    blocks: list[int] = field(default_factory=list)
    finished: bool = False
    finish_reason: str = ""
    arrival: float = field(default_factory=time.time)
    first_token_at: float = 0.0
    cumulative_logprob: float = 0.0
    prefilled: bool = False
    cached_len: int = 0                 # jetons d'invite servis par le cache de préfixe
    hashes: list[int] = field(default_factory=list)
    n_accepted: int = 0                 # jetons spéculatifs acceptés
    n_proposed: int = 0

    @property
    def length(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    def blocks_needed(self, extra: int = 0) -> int:
        return (self.length + extra + BLOCK_SIZE - 1) // BLOCK_SIZE


@dataclass
class GenerationOutput:
    sequence_id: int
    request_id: str
    token_ids: list[int]
    text_delta: str = ""
    finished: bool = False
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class EngineStats:
    steps: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    running: int = 0
    waiting: int = 0
    kv_blocks_free: int = 0
    kv_refills: int = 0
    kv_blocks_total: int = 0
    cached_prompt_tokens: int = 0
    accepted_tokens: int = 0
    proposed_tokens: int = 0
    spec_steps: int = 0

    @property
    def decode_tok_s(self) -> float:
        return self.decode_tokens / self.decode_seconds if self.decode_seconds else 0.0

    @property
    def prefill_tok_s(self) -> float:
        return self.prefill_tokens / self.prefill_seconds if self.prefill_seconds else 0.0

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": self.decode_tokens,
            "decode_tok_s": round(self.decode_tok_s, 2),
            "prefill_tok_s": round(self.prefill_tok_s, 1),
            "running": self.running, "waiting": self.waiting,
            "kv_blocks_free": self.kv_blocks_free,
            "kv_blocks_total": self.kv_blocks_total,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "prefill_tokens_saved": self.cached_prompt_tokens,
            "accepted_tokens": self.accepted_tokens,
            "proposed_tokens": self.proposed_tokens,
            "acceptance_rate": round(self.acceptance_rate, 3),
            "tokens_per_step": round(self.tokens_per_step, 3),
        }

    @property
    def acceptance_rate(self) -> float:
        return (self.accepted_tokens / self.proposed_tokens
                if self.proposed_tokens else 0.0)

    @property
    def tokens_per_step(self) -> float:
        """Jetons décodés par étape de modèle. Au-dessus de 1, la spéculation a payé."""
        return self.decode_tokens / self.spec_steps if self.spec_steps else 1.0



def _trim_at_stop(joined: str, delta: str,
                  stops: list[str]) -> tuple[str, bool]:
    """Coupe le delta juste avant la première séquence d'arrêt rencontrée.

    L'API s'y engage : la chaîne d'arrêt est *exclue* de la sortie. La
    détection se fait sur le texte assemblé (une séquence peut chevaucher deux
    jetons) ; la coupe, elle, ne peut retrancher que le delta courant — un
    chevauchement sur un delta déjà livré en flux est perdu pour le client,
    comme chez les autres serveurs.
    """
    cut = -1
    for st in stops:
        if st:
            at = joined.find(st)
            if at >= 0 and (cut < 0 or at < cut):
                cut = at
    if cut < 0:
        return delta, False
    prev = len(joined) - len(delta)
    return (joined[prev:cut] if cut > prev else ""), True


class Engine:
    """Détient le modèle, l'allocateur de blocs et les files de requêtes."""

    def __init__(self, loaded: LoadedModel, tokenizer: Any = None,
                 max_batch_size: int = 16, max_model_len: int = 8192,
                 enable_prefix_cache: bool = True,
                 speculator: Any = None, spec_k: int = 4,
                 enable_cuda_graphs: bool = True,
                 host_kv_gib: float = 0.0) -> None:
        self.loaded = loaded
        self.model = loaded.model
        self.spec = loaded.spec
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_model_len = max_model_len

        n_blocks = min((c.cfg.num_blocks for c in self.model.caches.values()),
                       default=1024)
        self.allocator = BlockAllocator(n_blocks, enable_prefix_cache)
        self.speculator = speculator
        self.spec_k = spec_k
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []
        self.stats = EngineStats(kv_blocks_total=n_blocks)
        self._lock = threading.Lock()
        self._eos = self._eos_ids()
        # Étage hôte du cache KV : les blocs de préfixe évincés descendent en
        # RAM et remontent au réemploi, au lieu d'être recalculés.
        self.host_kv = None
        if host_kv_gib > 0 and self.model.caches:
            from ..memory.kvcache import HostKVPool
            self.host_kv = HostKVPool(int(host_kv_gib * 1024 ** 3))

            def _deverser(blk: int, h: int) -> None:
                self.host_kv.store(h, [c.export_block(blk)
                                       for c in self.model.caches.values()])
            self.allocator.spill_cb = _deverser

        self.graphs = None
        if enable_cuda_graphs:
            from .graphs import GraphRunner
            gr = GraphRunner(self.model, max_model_len)
            self.graphs = gr if gr.enabled else None

    # -- admission -------------------------------------------------------
    def _eos_ids(self) -> set[int]:
        ids: set[int] = set()
        cfg = self.loaded.manifest.get("model", {})
        raw = self.loaded.manifest.get("generation_config", {})
        # Le generation_config.json est recopie a cote du modele converti :
        # c'est lui qui porte <|im_end|> chez Qwen.
        disque: dict = {}
        d = getattr(self.loaded, "path", "") or ""
        gen_path = os.path.join(d, "generation_config.json") if d else ""
        if gen_path and os.path.isfile(gen_path):
            try:
                with open(gen_path, "r", encoding="utf-8") as fh:
                    disque = json.load(fh)
            except (OSError, json.JSONDecodeError):
                disque = {}
        for key in ("eos_token_id", "eos_token_ids"):
            for src in (cfg, raw, disque):
                v = src.get(key)
                if isinstance(v, int):
                    ids.add(v)
                elif isinstance(v, list):
                    ids.update(int(x) for x in v if isinstance(x, int))
        return ids

    def add_request(self, prompt_ids: list[int], params: SamplingParams,
                    request_id: str = "") -> Sequence:
        if len(prompt_ids) >= self.max_model_len:
            raise ValueError(
                f"invite de {len(prompt_ids)} jetons au-delà de max_model_len "
                f"{self.max_model_len}")
        seq = Sequence(list(prompt_ids), params, request_id)
        with self._lock:
            self.waiting.append(seq)
        return seq

    def abort(self, request_id: str) -> None:
        with self._lock:
            for seq in list(self.running) + list(self.waiting):
                if seq.request_id == request_id:
                    self._finish(seq, "abort")

    def _admit(self) -> list[Sequence]:
        """Fait entrer dans le lot autant de séquences en attente que les blocs le permettent."""
        admitted = []
        with self._lock:
            while self.waiting and len(self.running) < self.max_batch_size:
                seq = self.waiting[0]
                # On réserve l'invite plus un peu de marge, pour que les
                # premières étapes de décodage n'aient pas besoin aussitôt d'un
                # bloc supplémentaire.
                need = seq.blocks_needed(extra=BLOCK_SIZE)
                if need > self.allocator.num_free:
                    break
                self.waiting.pop(0)

                # On sert les blocs de tête que le cache détient déjà. Un bloc
                # est toujours retenu : une requête dont l'invite est
                # entièrement en cache a tout de même besoin d'un jeton à faire
                # traverser le modèle.
                hashes = BlockAllocator.block_hashes(seq.prompt_ids, BLOCK_SIZE)
                limit = max(0, (len(seq.prompt_ids) - 1) // BLOCK_SIZE)
                matched = self.allocator.match_prefix(hashes, limit=limit)
                # L'étage hôte prolonge la suite : chaque bloc suivant présent
                # en RAM remonte dans un bloc VRAM fraîchement alloué.
                if self.host_kv is not None:
                    while len(matched) < (limit or 0):
                        data = self.host_kv.fetch(hashes[len(matched)])
                        if data is None or self.allocator.num_free < 1:
                            break
                        blk = self.allocator.allocate(1)[0]
                        for c, d in zip(self.model.caches.values(), data):
                            c.import_block(blk, d)
                        self.allocator.register(blk, hashes[len(matched)])
                        matched.append(blk)
                        self.stats.kv_refills += 1
                seq.blocks = list(matched)
                seq.cached_len = len(matched) * BLOCK_SIZE
                seq.hashes = list(hashes[:len(matched)])
                seq.blocks.extend(self.allocator.allocate(need - len(matched)))
                self.stats.cached_prompt_tokens += seq.cached_len

                self.running.append(seq)
                admitted.append(seq)
        return admitted

    def _register_complete_blocks(self, seq: Sequence) -> None:
        """Publie les blocs désormais pleins, pour que des requêtes ultérieures les
        réutilisent.

        Uniquement des blocs complets : un bloc à moitié rempli, retrouvé par un
        hachage nommant un contenu qu'il ne porte pas encore, livrerait à une
        requête ultérieure des clés et des valeurs jamais écrites.
        """
        ids = seq.all_ids
        n_full = min(len(ids) // BLOCK_SIZE, len(seq.blocks))
        while len(seq.hashes) < n_full:
            i = len(seq.hashes)
            prev = seq.hashes[-1] if seq.hashes else 0
            span = tuple(ids[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE])
            h = hash((prev, span))
            seq.hashes.append(h)
            self.allocator.register(seq.blocks[i], h)

    def _grow(self, seq: Sequence, extra: int = 0) -> bool:
        need = seq.blocks_needed(extra=extra)
        if need <= len(seq.blocks):
            return True
        if self.allocator.num_free < need - len(seq.blocks):
            return False
        seq.blocks.extend(self.allocator.allocate(need - len(seq.blocks)))
        return True

    def _finish(self, seq: Sequence, reason: str) -> None:
        seq.finished = True
        if self.speculator is not None:
            self.speculator.release(seq)
        seq.finish_reason = reason
        if seq.blocks:
            self.allocator.free(seq.blocks)
            seq.blocks = []
        if seq in self.running:
            self.running.remove(seq)
        if seq in self.waiting:
            self.waiting.remove(seq)

    # -- batch construction ----------------------------------------------
    def _build_batch(self, seqs: list[Sequence], prefill: bool) -> ForwardBatch:
        tokens: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        query_lens: list[int] = []
        seq_lens: list[int] = []
        block_tables: list[torch.Tensor] = []

        for seq in seqs:
            if prefill:
                # On saute ce que le cache de préfixe détient déjà.
                ids = seq.prompt_ids[seq.cached_len:]
                start = seq.cached_len
            else:
                ids = [seq.output_ids[-1]] if seq.output_ids else [seq.prompt_ids[-1]]
                start = seq.length - 1
            for j, tok in enumerate(ids):
                pos = start + j
                tokens.append(tok)
                positions.append(pos)
                slots.append(seq.blocks[pos // BLOCK_SIZE] * BLOCK_SIZE
                             + pos % BLOCK_SIZE)
            query_lens.append(len(ids))
            seq_lens.append(start + len(ids))
            block_tables.append(torch.tensor(seq.blocks, dtype=torch.long))

        return ForwardBatch(
            tokens=torch.tensor(tokens, dtype=torch.long),
            positions=torch.tensor(positions, dtype=torch.long),
            seq_lens=seq_lens, query_lens=query_lens,
            block_tables=block_tables,
            slot_mapping=torch.tensor(slots, dtype=torch.long),
            is_prefill=prefill)

    # -- the step --------------------------------------------------------
    def step(self) -> list[GenerationOutput]:
        """Exécute une passe avant et rend ce qu'elle a produit."""
        new = self._admit()
        outputs: list[GenerationOutput] = []

        # On précalcule les séquences nouvellement admises une par une. Mêler
        # une longue invite à un lot de décodage bloquerait derrière elle toutes
        # les séquences en cours.
        for seq in new:
            t0 = time.perf_counter()
            batch = self._build_batch([seq], prefill=True)
            logits = self.model(batch)
            self.stats.prefill_seconds += time.perf_counter() - t0
            self.stats.prefill_tokens += len(seq.prompt_ids) - seq.cached_len
            seq.prefilled = True
            outputs += self._emit(logits, [seq])
            self._register_complete_blocks(seq)

        decodable = [s for s in self.running if s.prefilled and not s.finished]
        if decodable:
            t0 = time.perf_counter()
            if self.speculator is not None:
                outputs += self._speculative_decode(decodable)
            else:
                outputs += self._plain_decode(decodable)
            self.stats.decode_seconds += time.perf_counter() - t0
            self.stats.spec_steps += 1
            for seq in decodable:
                if not seq.finished:
                    self._register_complete_blocks(seq)

        self.stats.steps += 1
        self.stats.running = len(self.running)
        self.stats.waiting = len(self.waiting)
        self.stats.kv_blocks_free = self.allocator.num_free
        return outputs

    def _plain_decode(self, decodable: list[Sequence]) -> list[GenerationOutput]:
        for seq in decodable:
            if not self._grow(seq):
                self._finish(seq, "length")
        decodable = [s for s in decodable if not s.finished]
        if not decodable:
            return []
        batch = self._build_batch(decodable, prefill=False)
        logits = self.graphs.run(batch) if self.graphs is not None else None
        if logits is None:
            logits = self.model(batch)
        self.stats.decode_tokens += len(decodable)
        return self._emit(logits, decodable)

    def _speculative_decode(self, decodable: list[Sequence]
                            ) -> list[GenerationOutput]:
        """Proposer, vérifier en une passe avant, garder le préfixe accepté.

        Tout le lot est vérifié ensemble bien que les propositions diffèrent en
        longueur : le chemin d'attention gère déjà un bloc de requêtes par
        séquence avec son propre décalage absolu, la machinerie même dont le
        cache de préfixe avait besoin.
        """
        proposals: dict[int, Proposal] = {}
        for seq in decodable:
            budget = max(0, seq.params.max_tokens - len(seq.output_ids) - 1)
            k = min(self.spec_k, budget)
            prop = self.speculator.propose(seq, k) if k > 0 else Proposal([])
            # Une proposition qui dépasserait la limite de contexte est rognée
            # plutôt qu'abandonnée : une spéculation plus courte paie encore.
            room = self.max_model_len - seq.length - 1
            if len(prop) > room:
                prop = Proposal(prop.tokens[:max(0, room)],
                                None if prop.probs is None
                                else prop.probs[:max(0, room)])
            proposals[seq.id] = prop
            if not self._grow(seq, extra=len(prop)):
                self._finish(seq, "length")

        decodable = [s for s in decodable if not s.finished]
        if not decodable:
            return []

        batch = self._build_spec_batch(decodable, proposals)
        flat = self.graphs.run(batch) if self.graphs is not None else None
        if flat is None:
            flat = self.model(batch, logits_positions=batch.all_token_indices())

        outputs: list[GenerationOutput] = []
        cursor = 0
        for seq in decodable:
            prop = proposals[seq.id]
            width = len(prop) + 1
            rows = flat[cursor:cursor + width]
            cursor += width
            tokens, n_acc = verify_proposal(rows, prop, seq.params)
            self.stats.proposed_tokens += len(prop)
            self.stats.accepted_tokens += n_acc
            self.stats.decode_tokens += len(tokens)
            seq.n_proposed += len(prop)
            seq.n_accepted += n_acc
            self.speculator.commit(seq, tokens)
            outputs.append(self._append(seq, tokens))
        return outputs

    def _build_spec_batch(self, seqs: list[Sequence],
                          proposals: dict[int, Proposal]) -> ForwardBatch:
        """Le lot de vérification : le dernier vrai jeton, puis les propositions.

        Présenter le dernier jeton produit à côté des propositions ne coûte
        rien de plus : ses clés et valeurs n'ont jamais été écrites, puisqu'un
        jeton n'entre dans le cache qu'au moment où on le présente. Les K+1
        positions sont donc exactement les K+1 prédictions nécessaires.
        """
        tokens: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        query_lens: list[int] = []
        seq_lens: list[int] = []
        block_tables: list[torch.Tensor] = []

        for seq in seqs:
            prop = proposals[seq.id]
            start = seq.length - 1
            block = [seq.output_ids[-1] if seq.output_ids else seq.prompt_ids[-1]]
            block += list(prop.tokens)
            for j, tok in enumerate(block):
                pos = start + j
                tokens.append(tok)
                positions.append(pos)
                slots.append(seq.blocks[pos // BLOCK_SIZE] * BLOCK_SIZE
                             + pos % BLOCK_SIZE)
            query_lens.append(len(block))
            seq_lens.append(start + len(block))
            block_tables.append(torch.tensor(seq.blocks, dtype=torch.long))

        return ForwardBatch(
            tokens=torch.tensor(tokens, dtype=torch.long),
            positions=torch.tensor(positions, dtype=torch.long),
            seq_lens=seq_lens, query_lens=query_lens,
            block_tables=block_tables,
            slot_mapping=torch.tensor(slots, dtype=torch.long),
            is_prefill=False)

    def _append(self, seq: Sequence, tokens: list[int]) -> GenerationOutput:
        """Ajoute plusieurs jetons acceptés, en s'arrêtant au premier qui termine."""
        reason = ""
        kept: list[int] = []
        for tok in tokens:
            seq.output_ids.append(int(tok))
            kept.append(int(tok))
            if not seq.first_token_at:
                seq.first_token_at = time.time()
            if tok in self._eos or tok in seq.params.stop_token_ids:
                reason = "stop"
                break
            if len(seq.output_ids) >= seq.params.max_tokens:
                reason = "length"
                break
            if seq.length >= self.max_model_len:
                reason = "length"
                break

        text = self._decode_delta(seq, len(kept)) if self.tokenizer else ""
        if not reason and seq.params.stop and text:
            text, coupe = _trim_at_stop(self._decode_all(seq), text,
                                        seq.params.stop)
            if coupe:
                reason = "stop"
        if reason:
            self._finish(seq, reason)
        return GenerationOutput(
            sequence_id=seq.id, request_id=seq.request_id, token_ids=kept,
            text_delta=text, finished=bool(reason), finish_reason=reason,
            prompt_tokens=len(seq.prompt_ids),
            completion_tokens=len(seq.output_ids))

    def _emit(self, logits: torch.Tensor,
              seqs: list[Sequence]) -> list[GenerationOutput]:
        params = [s.params for s in seqs]
        history = [s.all_ids for s in seqs]
        tokens, logprobs = sample(logits, params, history)
        out = []
        for seq, tok, lp in zip(seqs, tokens.tolist(), logprobs.tolist()):
            seq.output_ids.append(int(tok))
            seq.cumulative_logprob += float(lp)
            if not seq.first_token_at:
                seq.first_token_at = time.time()

            reason = ""
            if tok in self._eos or tok in seq.params.stop_token_ids:
                reason = "stop"
            elif len(seq.output_ids) >= seq.params.max_tokens:
                reason = "length"
            elif seq.length >= self.max_model_len:
                reason = "length"

            text = ""
            if self.tokenizer is not None:
                text = self._decode_delta(seq)
            if not reason and seq.params.stop and text:
                text, coupe = _trim_at_stop(self._decode_all(seq), text,
                                            seq.params.stop)
                if coupe:
                    reason = "stop"

            if reason:
                self._finish(seq, reason)
            out.append(GenerationOutput(
                sequence_id=seq.id, request_id=seq.request_id,
                token_ids=[int(tok)], text_delta=text,
                finished=bool(reason), finish_reason=reason,
                prompt_tokens=len(seq.prompt_ids),
                completion_tokens=len(seq.output_ids)))
        return out

    def _decode_delta(self, seq: Sequence, n_new: int = 1) -> str:
        """Décode au fil de l'eau, en respectant les séquences UTF-8 multi-jetons.

        Décoder le seul dernier jeton couperait les caractères multi-octets et
        émettrait des caractères de remplacement en plein mot : on décode donc
        une petite fenêtre et on ne rend que ce qui est nouveau.
        """
        tok = self.tokenizer
        span = max(8, n_new + 4)
        window = seq.output_ids[-span:]
        prev = seq.output_ids[-span:-n_new] if n_new else seq.output_ids[-span:]
        try:
            full = tok.decode(window)
            head = tok.decode(prev) if prev else ""
        except Exception:                            # noqa: BLE001
            return ""
        return full[len(head):] if full.startswith(head) else full

    def _decode_all(self, seq: Sequence) -> str:
        try:
            return self.tokenizer.decode(seq.output_ids)
        except Exception:                            # noqa: BLE001
            return ""

    # -- convenience -----------------------------------------------------
    def generate(self, prompt_ids: list[int], params: SamplingParams
                 ) -> Iterator[GenerationOutput]:
        """Générateur bloquant pour une requête unique. Utilisé par le CLI et les tests."""
        seq = self.add_request(prompt_ids, params)
        while not seq.finished:
            for out in self.step():
                if out.sequence_id == seq.id:
                    yield out
            if not self.running and not self.waiting:
                break

    @property
    def idle(self) -> bool:
        return not self.running and not self.waiting
