"""Graphes CUDA pour le pas de décodage.

Le profil de Qwen3-14B montrait ~68 ms de Python par jeton pour ~30 ms de
calcul GPU : plus de la moitié du temps partait en lancements de noyaux et en
navette d'objets Python. Un graphe CUDA capture une fois la séquence complète
des noyaux d'un pas de décodage, puis la rejoue pour le prix d'un seul appel.

Ce que la capture exige — et comment on l'obtient :

* **Des formes fixes.** Un graphe est capturé par *godet* ``(lot, blocs KV)`` :
  le lot est pris tel quel (un serveur local décode presque toujours à 1), le
  nombre de blocs est arrondi à la puissance de deux supérieure. La table de
  blocs est complétée avec le bloc 0 — lu pour rien, masqué par ``seq_lens``.
* **Des adresses stables.** Les entrées vivent dans des tampons statiques dont
  seul le contenu change avant chaque rejeu ; le cache RoPE est étendu à la
  longueur maximale *avant* la capture pour ne jamais être réalloué.
* **Aucun scalaire tiré des données.** C'est le rôle des chemins
  ``decode_fixed`` : longueurs et positions restent des tenseurs.

L'écriture du cache KV pendant l'échauffement et la capture est volontairement
idempotente — mêmes emplacements, mêmes valeurs — si bien qu'exécuter le pas
deux fois puis le rejouer produit exactement l'état qu'un pas ordinaire aurait
produit. Un test l'affirme jeton par jeton.

Ce qui n'est **pas** capturé, à dessein : le prefill (formes libres), la
spéculation (plusieurs positions par séquence), les modèles à experts (le
routage dépend des données), les couches réparties sur plusieurs appareils et
les poids streamés (le préchargement change les adresses).
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from ..memory.kvcache import BLOCK_SIZE, bucket_blocks
from .layers import QuantLinear
from .model import ForwardBatch, MoEBlock

__all__ = ["GraphRunner"]

# Au-delà, les godets les moins récents ne sont plus capturés : chaque graphe
# retient sa mémoire d'activations, et un serveur qui voit trente formes de
# lot différentes est un serveur de lots — le prefill y domine de toute façon.
MAX_GRAPHS = 16


class GraphRunner:
    """Capture paresseuse et rejeu des pas de décodage purs."""

    def __init__(self, model, max_model_len: int) -> None:
        self.model = model
        self.max_model_len = max_model_len
        self.device: Optional[torch.device] = None
        self.graphs: dict[tuple[int, int], dict] = {}
        self._pool = None
        self.enabled = self._eligible()
        self.replays = 0
        self.captures = 0
        self._last_key: Optional[tuple[int, int]] = None

    # -- éligibilité -----------------------------------------------------
    def _eligible(self) -> bool:
        if os.environ.get("ACVRAM_DISABLE_CUDA_GRAPHS"):
            return False
        if not torch.cuda.is_available():
            return False
        m = self.model
        devs = {l.device for l in m.layers} | {l.mlp_device for l in m.layers}
        devs.add(m.norm.weight.device)
        head = getattr(m.lm_head.qweight, "qweight", None)
        devs.add(head.device if head is not None else m.norm.weight.device)
        if len(devs) != 1 or next(iter(devs)).type != "cuda":
            return False                     # pipeline multi-appareils : eager
        for mod in m.modules():
            if isinstance(mod, MoEBlock):
                return False                 # routage dependant des donnees
            if isinstance(mod, QuantLinear) and mod.streamed is not None:
                return False                 # les adresses changent en vol
        if len(m.caches) != len(m.layers):
            return False
        if any(m.caches[i].k.device != next(iter(devs))
               for i in range(len(m.layers))):
            return False
        self.device = next(iter(devs))
        return True

    # -- exécution -------------------------------------------------------
    def run(self, batch: ForwardBatch) -> Optional[torch.Tensor]:
        """Logits du lot, ou None si ce lot n'est pas rejouable en graphe."""
        if not self.enabled or batch.is_prefill:
            return None
        if any(ql != 1 for ql in batch.query_lens):
            return None                      # verification speculative : eager
        b = batch.batch_size
        nblk = bucket_blocks(max(t.shape[0] for t in batch.block_tables))
        if nblk * BLOCK_SIZE > self.max_model_len + BLOCK_SIZE:
            nblk = bucket_blocks((self.max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
        key = (b, nblk)
        self._last_key = key

        entry = self.graphs.get(key)
        if entry is None:
            if len(self.graphs) >= MAX_GRAPHS:
                return None
            entry = self._capture(b, nblk, batch)
            self.graphs[key] = entry
            self.replays += 1                # la capture rejoue deja une fois
            return entry["out"].clone()

        self._fill(entry, batch)
        entry["graph"].replay()
        self.replays += 1
        return entry["out"].clone()

    # -- tampons ---------------------------------------------------------
    def _embed(self, batch: ForwardBatch) -> torch.Tensor:
        """Le plongement, hors graphe, sur l'appareil où réside la table."""
        m = self.model
        idx = batch.tokens.to(m.embed_tokens.device)
        return torch.nn.functional.embedding(idx, m.embed_tokens).to(m.dtype)

    def _fill(self, entry: dict, batch: ForwardBatch) -> None:
        b, nblk = entry["key"]
        entry["x"].copy_(self._embed(batch).to(self.device), non_blocking=True)
        entry["positions"].copy_(batch.positions, non_blocking=True)
        entry["slots"].copy_(batch.slot_mapping, non_blocking=True)
        entry["seq_lens"].copy_(
            torch.tensor(batch.seq_lens, dtype=torch.long), non_blocking=True)
        # Table completee au godet avec le bloc 0 : lu, dequantifie, masque.
        tables = entry["tables"]
        tables.zero_()
        for i, t in enumerate(batch.block_tables):
            tables[i, : t.shape[0]].copy_(t, non_blocking=True)

    def _capture(self, b: int, nblk: int, batch: ForwardBatch) -> dict:
        m = self.model
        d = self.device
        h = m.spec.hidden_size
        entry = {
            "key": (b, nblk),
            "x": torch.zeros(b, h, dtype=m.dtype, device=d),
            "positions": torch.zeros(b, dtype=torch.long, device=d),
            "slots": torch.zeros(b, dtype=torch.long, device=d),
            "tables": torch.zeros(b, nblk, dtype=torch.long, device=d),
            "seq_lens": torch.zeros(b, dtype=torch.long, device=d),
        }
        self._fill(entry, batch)
        max_pos = min(self.max_model_len + 1, nblk * BLOCK_SIZE + 1)

        # Le cache RoPE doit exister a sa taille finale avant la capture :
        # une extension pendant un rejeu pointerait un tenseur abandonne.
        for layer in m.layers:
            layer.self_attn.rope(entry["positions"], d, m.dtype,
                                 max_pos=self.max_model_len + 1)

        def step() -> torch.Tensor:
            return m.decode_fixed(entry["x"], entry["positions"],
                                  entry["slots"], entry["tables"],
                                  entry["seq_lens"], max_pos)

        # Echauffement sur un flux annexe (exige par la capture), puis capture.
        # Les ecritures KV de ces passes sont identiques a celle du pas reel :
        # les rejouer n'ajoute rien, n'efface rien.
        torch.cuda.synchronize(d)
        side = torch.cuda.Stream(d)
        side.wait_stream(torch.cuda.current_stream(d))
        with torch.cuda.stream(side):
            with torch.inference_mode():
                for _ in range(2):
                    step()
        torch.cuda.current_stream(d).wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            if self._pool is None:
                with torch.cuda.graph(graph):
                    entry["out"] = step()
                self._pool = graph.pool()
            else:
                with torch.cuda.graph(graph, pool=self._pool):
                    entry["out"] = step()
        entry["graph"] = graph
        self.captures += 1
        graph.replay()                       # la capture n'execute pas : rejouer
        return entry
