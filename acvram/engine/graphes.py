"""Graphes CUDA côté moteur : capture d avance des godets (`warm_graphs`) et sonde eager des entrées
(`_sonde_eager`, pendant de `graphs._sonde_entrees`). Déplacement PUR depuis `runner.py` (4 bis 2/3, 21/09) :
`GraphesMoteur` est un mixin d `Engine`, corps octet pour octet ; le GraphRunner lui-même reste dans `graphs.py`."""

from __future__ import annotations

import os

import torch

from .model import ForwardBatch
from .sampler import SamplingParams

__all__ = ["GraphesMoteur"]


class GraphesMoteur:
    """Mixin d `Engine` : capture d avance des graphes et sonde eager (voir l en-tête du module)."""

    def _sonde_eager(self, batch: ForwardBatch) -> None:
        """ACVRAM_TRACE_ENTREES=1 : le pendant eager de graphs._sonde_entrees —
        jetons, positions, seq_lens et le plongement que le modèle va calculer."""
        if not os.environ.get("ACVRAM_TRACE_ENTREES"):
            return
        torch.cuda.synchronize()
        n = min(4, batch.batch_size)
        m = self.model
        idx = batch.tokens[:n]
        if torch.is_tensor(idx):
            idx = idx.to(m.embed_tokens.device)
        else:
            idx = torch.tensor(list(idx), device=m.embed_tokens.device)
        x = torch.nn.functional.embedding(idx, m.embed_tokens).to(m.dtype)
        if m.spec.embedding_multiplier != 1.0:
            x = x * m.spec.embedding_multiplier
        print(f"[ENTREES-EAGER] jetons={idx.tolist()} positions={batch.positions[:n].tolist()} "
              f"seq_lens={list(batch.seq_lens[:n])} | x[:4]={[[round(v, 5) for v in r] for r in x[:, :4].float().tolist()]}",
              flush=True)

    def warm_graphs(self, max_len: int = 2048) -> int:
        """Capture d'avance les graphes de décodage des godets jusqu'à
        ``max_len`` jetons : une capture coûte 40 à 130 ms, mieux vaut la
        payer au démarrage qu'au milieu d'une réponse. Rend le nombre de
        captures faites."""
        if self.graphs is None:
            return 0
        avant = self.graphs.captures
        L = 128
        while L <= min(max_len, self.max_model_len - 4):
            # longueur choisie pour que prefill + 2 jetons restent dans le
            # godet de L/16 blocs (puissance de deux)
            for _ in self.generate([1] * (L - 2), SamplingParams(max_tokens=2,
                                                                 temperature=0.0)):
                pass
            if (self.est_hybride and self.speculator is not None
                    and os.environ.get("ACVRAM_WARM_SPEC", "1") != "0"):
                # motif répété : le proposeur n-gramme spécule dès le
                # premier pas, d'où la capture du graphe de forme k+1
                ids = ([5, 6, 7, 8] * (L // 4))[:L - 2]
                for _ in self.generate(ids, SamplingParams(max_tokens=self.spec_k + 2,
                                                           temperature=0.0)):
                    pass
            L *= 2
        return self.graphs.captures - avant
