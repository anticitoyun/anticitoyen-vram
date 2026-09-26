"""Décodage à un pas de retard (`ACVRAM_PIPELINE=1`, bead runner 14/09) : le lot suivant est préparé et son rejeu
enfilé pendant que le rejeu courant tourne, la lecture des jetons est différée d un pas. Déplacement PUR depuis
`runner.py` (4 bis 3/3, 21/09) : `PipelineDecodage` est un mixin d `Engine`, corps octet pour octet."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..memory.kvcache import BLOCK_SIZE
from .model import ForwardBatch

if TYPE_CHECKING:                                   # annotations seulement (`from __future__ import annotations`)
    from .runner import GenerationOutput, Sequence

__all__ = ["PipelineDecodage"]


class PipelineDecodage:
    """Mixin d `Engine` : lot sur device, amorce, suite et pas recouvert du pipeline (voir l en-tête du module)."""

    def _build_batch_device(self, seqs: list[Sequence],
                            tokens_dev: torch.Tensor) -> ForwardBatch:
        """Comme `_build_batch(seqs, prefill=False)`, mais le jeton à
        plonger vient d'un tenseur DEVICE (l'argmax du pas précédent, jamais
        rapatrié) plutôt que de `seq.output_ids[-1]` — c'est justement ce qui
        permet de lancer ce pas SANS attendre que le pas précédent soit
        rapatrié sur l'hôte (bead runner, 14/09, accord d'interface avec
        poste4 côté `graphs.py`).

        Appelée AVANT `_consommer` (recouvrement : c'est justement pour ça
        qu'on n'attend pas) — `output_ids` ne porte PAS encore ce jeton,
        contrairement à `_build_batch`. Convention DÉCALÉE d'un cran :
        `pos = seq.length` (pas `- 1`), et l'appelant doit réserver le bloc
        avec `_grow(seq, extra=1)`."""
        positions: list[int] = []
        slots: list[int] = []
        query_lens: list[int] = []
        seq_lens: list[int] = []
        block_tables: list[torch.Tensor] = []
        for seq in seqs:
            pos = seq.length
            positions.append(self._pos_decodage(seq, pos))
            slots.append(seq.blocks[pos // BLOCK_SIZE] * BLOCK_SIZE
                         + pos % BLOCK_SIZE)
            query_lens.append(1)
            seq_lens.append(pos + 1)
            block_tables.append(torch.tensor(seq.blocks, dtype=torch.long))
        return ForwardBatch(
            tokens=tokens_dev,
            positions=torch.tensor(positions, dtype=torch.long),
            seq_lens=seq_lens, query_lens=query_lens,
            block_tables=block_tables,
            slot_mapping=torch.tensor(slots, dtype=torch.long),
            is_prefill=False,
            seq_ids=[s.id for s in seqs], gdn_store=self.gdn_states)

    def _apres_echantillon(self, tokens_dev: torch.Tensor, logprobs_dev: torch.Tensor):
        """Levier 2 (poste1-levier-2-conception-22-09) : après le clone du
        levier 1, enfiler sur le MÊME flux une copie non bloquante vers un
        tampon HÔTE ÉPINGLÉ, puis enregistrer l événement — dernier enfilé,
        il couvre le clone ET la copie. `_consommer` lira l épinglé sans rien
        enfiler : aujourd hui son `.tolist()` est une copie D2H rangée
        DERRIÈRE le rejeu n+1 lancé juste avant, et l hôte y attend 6,7 ms
        au lieu de préparer n+2 (trou_gpu 173 µs = suite_prep + lancement).

        Double tampon par PARITÉ du pas : la copie de n+1 est enfilée avant
        la lecture de n (même `step()`), un seul tampon serait une course
        gagnée par 6,7 ms de marge. L invariant qui porte la parité — au plus
        un pas en vol — est rendu impossible à sauter par `lu` : cibler un
        tampon non encore lu lève, jamais un avertissement."""
        epingle = None
        if (getattr(self, "rapatriement_epingle", False) and logprobs_dev.dtype == torch.int64
                and getattr(self.graphs, "sampler_graphe", False)):
            paquet = logprobs_dev._base if logprobs_dev._base is not None else torch.stack((tokens_dev, logprobs_dev))
            n = paquet.shape[1]
            p = self._parite_epingle
            self._parite_epingle ^= 1
            slot = self._epingles[p]
            if slot is not None and not slot["lu"]:
                raise RuntimeError(f"tampon épinglé de parité {p} réutilisé avant lecture : plus d un pas en vol")
            # Le tampon fait EXACTEMENT [2, n] : une vue `[:, :n]` d un tampon
            # plus large n est pas contiguë, et une copie carte → hôte non
            # contiguë passe par un tampon paginable, donc SYNCHRONE — l hôte
            # attendait derrière le rejeu n+1 dans `_pipeline_suite` (poste2
            # 22/09, `verdict-levier2-2-frontiere` : suite_prep 151 → 6 860 µs,
            # trou_gpu inchangé). Un tampon par (parité, n) ; n change à chaque
            # recomposition, rarement.
            if slot is None or tuple(slot["tenseur"].shape) != (2, n):
                t = torch.empty(2, n, dtype=torch.int64)
                if paquet.is_cuda:
                    t = t.pin_memory()
                slot = self._epingles[p] = {"tenseur": t, "lu": True, "n": 0}
            cible = slot["tenseur"]
            if not (cible.is_contiguous() and paquet.is_contiguous() and tuple(cible.shape) == tuple(paquet.shape)):
                raise RuntimeError("rapatriement épinglé : source et cible doivent être contiguës et de même forme "
                                   f"(sinon la copie est synchrone) — {tuple(paquet.shape)} → {tuple(cible.shape)}")
            if paquet.is_cuda and not cible.is_pinned():
                raise RuntimeError("rapatriement épinglé : cible non épinglée, la copie serait synchrone")
            cible.copy_(paquet, non_blocking=True)
            slot["lu"], slot["n"] = False, n
            epingle = slot
        evenement = torch.cuda.Event()
        evenement.record()
        return epingle, evenement

    def _pipeline_amorcer(self, decodable: list[Sequence]) -> list[GenerationOutput]:
        """Un pas NORMAL (synchrone, comme `_plain_decode_sync`), qui pose ou
        REPOSE l'état du pipeline plutôt que de le poursuivre en
        recouvrement — au tout premier pas, et chaque fois que la
        composition du lot change (une séquence finit, une autre est
        admise) : `_plain_decode_pipeline` retombe ici plutôt que de
        deviner. Un seul pas de recouvrement perdu à chaque recomposition,
        jamais de famine pour une arrivée."""
        for seq in decodable:
            if not self._grow(seq):
                self._finish_budget_epuise(seq)
        decodable = [s for s in decodable if not s.finished]
        if not decodable:
            return []
        batch = self._build_batch(decodable, prefill=False)
        ok = self.graphs.preparer(batch)
        self.stats.decode_tokens += len(decodable)
        if not ok:
            self._sonde_eager(batch)
            logits = self.model(batch)
            return self._emit(logits, decodable)
        logits = self.graphs.rejouer_suivant()
        tokens_dev, logprobs_dev = self._sample_only(logits, decodable, depuis_graphe=True)
        epingle, evenement = self._apres_echantillon(tokens_dev, logprobs_dev)
        self._pipeline_pendiente = {
            "seqs": decodable, "tokens_dev": tokens_dev,
            "logprobs_dev": logprobs_dev, "event": evenement, "epingle": epingle,
            "tops": self._tops_si_demande(logits, decodable)}
        return []

    def _pipeline_suite(self, roster: list[Sequence],
                        tokens_dev: torch.Tensor) -> list[GenerationOutput]:
        """Prépare et lance le pas SUIVANT en recouvrement — enfilée AVANT
        que `_consommer` n'ait rapatrié le pas courant (c'est le
        recouvrement : `_plain_decode_pipeline` appelle celle-ci D'ABORD).
        `roster` est le sous-ensemble encore actif tel que connu au pas
        PRÉCÉDENT (`output_ids` ne porte pas encore son jeton de ce pas-ci),
        `tokens_dev` ses jetons DEVICE alignés dans le même ordre. `_grow`
        avec `extra=1` : réserve le bloc du jeton pas encore ajouté à
        `output_ids` (cf. `_build_batch_device`)."""
        for seq in roster:
            if not self._grow(seq, extra=1):
                self._finish_budget_epuise(seq)
        vivants = [s for s in roster if not s.finished]
        if not vivants:
            return []
        if len(vivants) != len(roster):
            mask = torch.tensor([not s.finished for s in roster],
                                device=tokens_dev.device)
            tokens_dev = tokens_dev[mask]
        batch = self._build_batch_device(vivants, tokens_dev)
        ok = self.graphs.preparer(batch)
        self.stats.decode_tokens += len(vivants)
        if not ok:
            self._sonde_eager(batch)
            logits = self.model(batch)
            return self._emit(logits, vivants)
        logits = self.graphs.rejouer_suivant()
        tokens_dev2, logprobs_dev2 = self._sample_only(logits, vivants, depuis_graphe=True)
        epingle, evenement = self._apres_echantillon(tokens_dev2, logprobs_dev2)
        self._pipeline_pendiente = {
            "seqs": vivants, "tokens_dev": tokens_dev2,
            "logprobs_dev": logprobs_dev2, "event": evenement, "epingle": epingle,
            "tops": self._tops_si_demande(logits, vivants)}
        return []

    def _plain_decode_pipeline(self, decodable: list[Sequence]) -> list[GenerationOutput]:
        """Décodage à un pas de retard (`ACVRAM_PIPELINE=1`, bead runner
        14/09) : le pas n+1 est préparé et lancé AVANT de rapatrier les
        jetons du pas n — c'est le recouvrement lui-même, `.tolist()`
        continue d'attendre le GPU (chef : « le gain vient du
        recouvrement, pas de la suppression de l'attente »).

        `decodable` est calculé par `step()` AVANT l'appel — donc avant que
        le rapatriement ci-dessous ait pu faire finir une séquence. Il ne
        sert qu'à détecter un changement de composition (une arrivée) ; la
        composition RÉELLE du pas en vol vient de `_pipeline_pendiente`."""
        pend = self._pipeline_pendiente
        if pend is None:
            return self._pipeline_amorcer(decodable)

        self._pipeline_pendiente = None
        roster_avant = [s for s in pend["seqs"] if not s.finished]
        # `.id`, pas `in`/`==` : `Sequence` est un dataclass à égalité par
        # champs (output_ids inclus) — comparer les objets eux-mêmes serait
        # à la fois faux (deux séquences ne sont jamais "égales" en ce sens)
        # et lent (compare tous les champs, listes comprises).
        ids_pend = {s.id for s in pend["seqs"]}
        nouveaux = [s for s in decodable if s.id not in ids_pend]

        if nouveaux or not roster_avant:
            # Recomposition du lot : pas de rejeu à enfiler par avance (sa
            # forme dépend de la nouvelle composition), donc rien à
            # recouvrir ici — on synchronise puis on retombe sur le pas
            # normal, comme documenté.
            pend["event"].synchronize()
            outputs = self._consommer(pend["tokens_dev"], pend["logprobs_dev"], pend["seqs"], epingle=pend.get("epingle"), tops=pend.get("tops"))
            outputs += self._pipeline_amorcer(roster_avant + nouveaux)
            return outputs

        # Lot stable : enfiler le rejeu n+1 D'ABORD (il ne lit que
        # `tokens_dev`, un tenseur DEVICE écrit par le rejeu n — l'ordre du
        # flux CUDA garantit la dépendance, aucun événement requis ici) puis
        # SEULEMENT ENSUITE synchroniser pour rapatrier les jetons du pas n
        # — sinon le rejeu n+1 ne part qu'après tout le travail hôte de
        # `_consommer`, et il n'y a plus rien à recouvrir (ce qui était le
        # bogue : le +0,6 % mesuré le 14/09 venait de cet ordre inversé).
        mask = [not s.finished for s in pend["seqs"]]
        tokens_dev = (pend["tokens_dev"] if all(mask) else
                     pend["tokens_dev"][torch.tensor(mask, device=pend["tokens_dev"].device)])
        outputs = self._pipeline_suite(roster_avant, tokens_dev)

        pend["event"].synchronize()
        outputs = self._consommer(pend["tokens_dev"], pend["logprobs_dev"], pend["seqs"], epingle=pend.get("epingle"), tops=pend.get("tops")) + outputs
        return outputs
