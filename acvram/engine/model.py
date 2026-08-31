"""Le transformeur lui-même : famille llama, dense et à mélange d'experts,
réparti sur les étages de mémoire.

Le périmètre est délibéré. Il couvre l'architecture qu'emploie à peu près tout
modèle à poids ouverts de la gamme de tailles qui vaut la peine d'être exécutée
sur cette machine — RMSNorm, RoPE, attention à requêtes groupées, SwiGLU, et
optionnellement un bloc à mélange d'experts creux — plutôt que d'essayer d'être
une ménagerie universelle. Llama, Mistral, Qwen2/3, Mixtral et DeepSeek y
entrent tous.

Ce qui est propre à ce projet, c'est qu'une couche sait sur quel appareil elle
s'exécute et si ses poids sont résidents ou transférés, et que le bloc à mélange
d'experts ne touche que les experts choisis par le routeur — ce qui fait de la
mémoire vive un endroit raisonnable pour garder les 120 autres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..memory.kvcache import PagedKVCache, bucket_blocks
from .config import ModelSpec
from .layers import (QuantLinear, RMSNorm, RotaryEmbedding, apply_rope,
                     attention, batched_decode_attention,
                     decode_attention_fixed, repeat_kv)

__all__ = ["Attention", "MLP", "MoEBlock", "DecoderLayer", "ACVRamModel",
           "ForwardBatch"]


@dataclass
class ForwardBatch:
    """Une étape de travail, prefill ou décodage.

    ``seq_lens`` est la longueur de contexte totale de chaque séquence *après*
    ajout des jetons de ce lot : c'est ce dont l'attention a besoin pour savoir
    quelle quantité d'histoire lire.
    """

    tokens: torch.Tensor              # [total_tokens] flattened across sequences
    positions: torch.Tensor           # [total_tokens]
    seq_lens: list[int]
    query_lens: list[int]
    block_tables: list[torch.Tensor]
    slot_mapping: torch.Tensor        # [total_tokens]
    is_prefill: bool

    @property
    def batch_size(self) -> int:
        return len(self.seq_lens)

    # Les index d'une etape (positions, slots) naissent sur l'hote et sont lus
    # par chaque couche. Les transferer a chaque couche coutait ~0,7 ms par
    # appel — plus que le calcul de la couche elle-meme. On les copie donc une
    # fois par peripherique et par etape ; le lot est reconstruit a chaque
    # etape, le cache ne peut pas devenir obsolete.
    def positions_on(self, device: torch.device) -> torch.Tensor:
        cache = self.__dict__.setdefault("_pos_cache", {})
        t = cache.get(device)
        if t is None:
            t = cache[device] = self.positions.to(device, non_blocking=True)
        return t

    def fixed_decode_views(self, device: torch.device
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """Table de blocs complétée au godet et longueurs, en tenseurs.

        C'est la forme sous laquelle le décodage pur consomme le lot — la même
        pour le chemin eager et pour le graphe CUDA, précisément pour que le
        second reproduise le premier au bit près. Mémorisé par périphérique,
        comme les index d'étape.
        """
        cache = self.__dict__.setdefault("_fixed_cache", {})
        got = cache.get(device)
        if got is None:
            n = bucket_blocks(max(t.shape[0] for t in self.block_tables))
            tables = torch.zeros(len(self.block_tables), n, dtype=torch.long)
            for i, t in enumerate(self.block_tables):
                tables[i, : t.shape[0]] = t
            got = cache[device] = (
                tables.to(device, non_blocking=True),
                torch.tensor(self.seq_lens, dtype=torch.long).to(
                    device, non_blocking=True))
        return got

    def slots_on(self, device: torch.device) -> torch.Tensor:
        cache = self.__dict__.setdefault("_slot_cache", {})
        t = cache.get(device)
        if t is None:
            t = cache[device] = self.slot_mapping.to(device, non_blocking=True)
        return t

    @property
    def is_decode(self) -> bool:
        return not self.is_prefill

    @property
    def q_offsets(self) -> list[int]:
        """Position absolue à laquelle commence le bloc de requêtes de chaque séquence."""
        return [s - q for s, q in zip(self.seq_lens, self.query_lens)]

    def last_token_indices(self) -> torch.Tensor:
        out, pos = [], 0
        for qlen in self.query_lens:
            pos += qlen
            out.append(pos - 1)
        return torch.tensor(out, dtype=torch.long)

    def all_token_indices(self) -> torch.Tensor:
        return torch.arange(self.tokens.shape[0], dtype=torch.long)


class Attention(nn.Module):
    def __init__(self, spec: ModelSpec, q: QuantLinear, k: QuantLinear,
                 v: QuantLinear, o: QuantLinear, rope: RotaryEmbedding,
                 q_norm: Optional[nn.Module] = None,
                 k_norm: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = q, k, v, o
        # Qwen3, Gemma 3 et Olmo 2 normalisent Q et K par tete, avant la RoPE.
        # L'ordre compte : normaliser apres ferait tourner un vecteur puis
        # ecraserait sa norme, ce que le modele n'a pas appris.
        self.q_norm, self.k_norm = q_norm, k_norm
        self.n_heads = spec.num_attention_heads
        self.n_kv_heads = spec.num_key_value_heads
        self.head_dim = spec.head_dim
        self.n_rep = self.n_heads // max(1, self.n_kv_heads)
        self.scale = self.head_dim ** -0.5
        self.rope = rope

    def forward(self, x: torch.Tensor, batch: ForwardBatch,
                cache: Optional[PagedKVCache]) -> torch.Tensor:
        t = x.shape[0]
        q = self.q_proj(x).view(t, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(t, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(t, self.n_kv_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        cos, sin = self.rope(batch.positions_on(x.device), x.device, x.dtype,
                             max_pos=max(batch.seq_lens))
        q, k = apply_rope(q, k, cos, sin)

        if cache is not None:
            cache.write(batch.slots_on(x.device), k, v)

        if batch.is_decode:
            return self._decode(q, k, v, batch, cache, t)
        return self._prefill(q, k, v, batch, cache, t)

    def decode_fixed(self, x: torch.Tensor, positions: torch.Tensor,
                     slots: torch.Tensor, block_tables: torch.Tensor,
                     seq_lens: torch.Tensor, max_pos: int,
                     cache: PagedKVCache) -> torch.Tensor:
        """Le pas de décodage à formes fixes — le chemin que capture le graphe.

        Même mathématique que ``forward`` en décodage, mais aucun scalaire
        Python tiré des données : positions, emplacements, tables et longueurs
        sont des tenseurs dont seul le *contenu* change entre deux rejeux.
        ``max_pos`` majore les positions (la longueur maximale du godet) : il ne
        sert qu'à garantir que le cache RoPE est déjà assez grand.
        """
        b = x.shape[0]
        q = self.q_proj(x).view(b, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(b, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, self.n_kv_heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        cos, sin = self.rope(positions, x.device, x.dtype, max_pos=max_pos)
        q, k = apply_rope(q, k, cos, sin)
        cache.write(slots, k, v)
        kk, vv = cache.gather_fixed(block_tables, q.dtype)
        out = decode_attention_fixed(q, kk, vv, seq_lens, self.n_rep, self.scale)
        return self.o_proj(out.reshape(b, self.n_heads * self.head_dim))

    def _prefill(self, q, k, v, batch: ForwardBatch,
                 cache: Optional[PagedKVCache], t: int) -> torch.Tensor:
        out = torch.empty_like(q)
        start = 0
        for i, qlen in enumerate(batch.query_lens):
            end = start + qlen
            offset = batch.seq_lens[i] - qlen
            if cache is not None and offset > 0:
                # Une partie de cette séquence est déjà en cache : un préfixe
                # servi, ou un morceau antérieur. On la relit et on masque selon
                # le décalage absolu de la requête.
                kk, vv = cache.gather(batch.block_tables[i].to(q.device),
                                      batch.seq_lens[i], q.dtype)
            else:
                # Rien avant : on utilise les clés qu'on vient de calculer
                # plutôt que de les relire par le cache. Cela évite un
                # aller-retour de quantification sur chaque jeton de prefill, ce
                # qui est à la fois plus rapide et un peu plus précis.
                kk, vv = k[start:end], v[start:end]
            kk = repeat_kv(kk, self.n_rep)
            vv = repeat_kv(vv, self.n_rep)
            out[start:end] = attention(q[start:end], kk, vv, True, self.scale,
                                       q_offset=offset)
            start = end
        return self.o_proj(out.reshape(t, self.n_heads * self.head_dim))

    def _decode(self, q, k, v, batch: ForwardBatch,
                cache: Optional[PagedKVCache], t: int) -> torch.Tensor:
        """Décodage, et vérification spéculative, pour tout le lot d'un coup."""
        if cache is None:
            return self._prefill(q, k, v, batch, cache, t)

        if all(ql == 1 for ql in batch.query_lens):
            # Décodage pur : le chemin à formes fixes, celui-là même que le
            # graphe CUDA capture — un seul gather vectorisé, pas de boucle
            # Python, et une sortie identique au bit près entre eager et rejeu.
            tables, lens = batch.fixed_decode_views(q.device)
            kk, vv = cache.gather_fixed(tables, q.dtype)
            out = decode_attention_fixed(q, kk, vv, lens, self.n_rep,
                                         self.scale)
            return self.o_proj(out.reshape(t, self.n_heads * self.head_dim))

        keys, values = [], []
        for i in range(batch.batch_size):
            kk, vv = cache.gather(batch.block_tables[i].to(q.device),
                                  batch.seq_lens[i], q.dtype)
            keys.append(kk)
            values.append(vv)

        # Vérification spéculative : plusieurs positions de requête par
        # séquence, chacune attendant sur son propre préfixe. Toujours causal,
        # toujours décalé.
        out = torch.empty_like(q)
        start = 0
        for i, qlen in enumerate(batch.query_lens):
            end = start + qlen
            offset = batch.seq_lens[i] - qlen
            out[start:end] = attention(
                q[start:end], repeat_kv(keys[i], self.n_rep),
                repeat_kv(values[i], self.n_rep), True, self.scale,
                q_offset=offset)
            start = end
        return self.o_proj(out.reshape(t, self.n_heads * self.head_dim))


class MLP(nn.Module):
    def __init__(self, gate: QuantLinear, up: QuantLinear, down: QuantLinear) -> None:
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = gate, up, down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEBlock(nn.Module):
    """Mélange d'experts creux.

    Seuls les ``top_k`` experts vers lesquels un jeton a été routé sont évalués :
    le coût d'une couche est donc indépendant du nombre d'experts qu'elle
    possède. C'est ce qui permet à 128 experts de résider en mémoire vive pendant
    que le modèle décode à un rythme exploitable — par jeton, la machine en lit
    8, pas 128.
    """

    def __init__(self, router: QuantLinear, experts: list[MLP], top_k: int,
                 shared: Optional[MLP] = None,
                 norm_topk_prob: bool = True) -> None:
        super().__init__()
        self.router = router
        self.experts = nn.ModuleList(experts)
        self.top_k = top_k
        self.shared = shared
        self.norm_topk_prob = norm_topk_prob
        self._stack_state = "?"                # ? | oui | non
        self._stacks = None

    # ------------------------------------------------------------------
    # Pile d'experts pour le chemin groupé. Les qweight/échelles de tous les
    # experts d'une projection sont recopiés dans un tenseur [E, ...] contigu,
    # puis chaque expert reçoit une *vue* de la pile — la mémoire n'est pas
    # doublée, et la boucle par expert du prefill continue de marcher.
    # ------------------------------------------------------------------
    def _try_build_stacks(self) -> bool:
        from ..quant.int4 import INT4Tensor
        from ..quant.nvfp4 import NVFP4Tensor

        def one(projs):
            ws = [p.qweight for p in projs]
            if any(p.streamed is not None for p in projs):
                return None
            if any(p.scaler is not None and not p.scaler.is_identity
                   for p in projs):
                return None                    # échelle AWQ par expert : repli
            if all(isinstance(w, NVFP4Tensor) for w in ws):
                if len({(w.shape, w.padded_in) for w in ws}) != 1:
                    return None
                qw = torch.stack([w.qweight for w in ws]).contiguous()
                bs = torch.stack([w.block_scale.view(torch.uint8)
                                  for w in ws]).contiguous()
                gs = torch.tensor([w.global_scale_float() for w in ws],
                                  dtype=torch.float32, device=qw.device)
                for e, w in enumerate(ws):     # vues : une seule mémoire
                    w.qweight = qw[e]
                    w.block_scale = bs[e].view(torch.float8_e4m3fn)
                return ("nvfp4", qw, bs, gs, ws[0].padded_in, ws[0].shape[0])
            if all(isinstance(w, INT4Tensor) for w in ws):
                if len({(w.shape, w.padded_in, w.group_size) for w in ws}) != 1:
                    return None
                qw = torch.stack([w.qweight for w in ws]).contiguous()
                sc = torch.stack([w.scales for w in ws]).contiguous()
                zr = torch.stack([w.zeros for w in ws]).contiguous()
                for e, w in enumerate(ws):
                    w.qweight, w.scales, w.zeros = qw[e], sc[e], zr[e]
                return ("int4", qw, sc, zr, ws[0].padded_in,
                        ws[0].group_size, ws[0].shape[0])
            return None

        piles = {}
        for nom in ("gate_proj", "up_proj", "down_proj"):
            pile = one([getattr(e, nom) for e in self.experts])
            if pile is None:
                return False
            piles[nom] = pile
        self._stacks = piles
        return True

    def _grouped(self, x32: torch.Tensor, pile, expert_ids, token_ids):
        from .. import kernels
        if pile[0] == "nvfp4":
            _, qw, bs, gs, k, m = pile
            return kernels.nvfp4_gemv_grouped(x32, qw, bs, gs, expert_ids,
                                              token_ids, k)[:, :m]
        _, qw, sc, zr, k, gsz, m = pile
        return kernels.int4_gemv_grouped(x32, qw, sc, zr, expert_ids,
                                         token_ids, k, gsz)[:, :m]

    def _forward_grouped(self, x, topw, topi):
        t = x.shape[0]
        eid = topi.reshape(-1).to(torch.int32)
        tok = torch.arange(t, device=x.device,
                           dtype=torch.int32).repeat_interleave(self.top_k)
        x32 = x.to(torch.float32)
        g = self._grouped(x32, self._stacks["gate_proj"], eid, tok)
        u = self._grouped(x32, self._stacks["up_proj"], eid, tok)
        act = F.silu(g) * u                     # [G, I] fp32
        seq = torch.arange(eid.shape[0], device=x.device, dtype=torch.int32)
        d = self._grouped(act, self._stacks["down_proj"], eid, seq)
        d = d * topw.reshape(-1, 1).to(d.dtype)
        # Chaque jeton possède exactement top_k lignes contiguës : une somme
        # sur cet axe remplace l'index_add_ atomique — déterministe, plus
        # rapide, et rejouable dans un graphe CUDA sans écart d'un rejeu à
        # l'autre.
        return d.view(t, self.top_k, -1).sum(dim=1).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, h = x.shape
        logits = self.router(x.to(torch.float32))
        weights = F.softmax(logits, dim=-1)
        topw, topi = torch.topk(weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topw = topw / topw.sum(dim=-1, keepdim=True)
        topw = topw.to(x.dtype)

        # Chemin groupé : trois lancements pour toute la couche, quel que soit
        # le nombre d'experts touchés. La boucle par expert reste le chemin des
        # grands lots de prefill (le regroupement par expert y redevient
        # rentable) et le repli des piles hétérogènes.
        if x.is_cuda and t <= 8 and self._stack_state != "non":
            if self._stack_state == "?":
                self._stack_state = "oui" if self._try_build_stacks() else "non"
            if self._stack_state == "oui":
                y = self._forward_grouped(x, topw, topi)
                if self.shared is not None:
                    y = y + self.shared(x)
                return y

        out = torch.zeros_like(x)
        # On regroupe les jetons par expert, pour que chaque expert fasse un
        # seul produit matriciel par lot au lieu d'un par jeton.
        flat_expert = topi.reshape(-1)
        flat_weight = topw.reshape(-1)
        flat_token = torch.arange(t, device=x.device).repeat_interleave(self.top_k)
        for e in flat_expert.unique().tolist():
            sel = flat_expert == e
            tok = flat_token[sel]
            y = self.experts[e](x[tok])
            out.index_add_(0, tok, y * flat_weight[sel].unsqueeze(-1))
        if self.shared is not None:
            out = out + self.shared(x)
        return out

    def prefetch(self) -> None:
        for expert in self.experts:
            for lin in (expert.gate_proj, expert.up_proj, expert.down_proj):
                lin.prefetch()


class DecoderLayer(nn.Module):
    def __init__(self, index: int, attn: Attention, mlp: nn.Module,
                 input_norm: RMSNorm, post_norm: RMSNorm, device: torch.device,
                 mlp_device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.index = index
        self.self_attn = attn
        self.mlp = mlp
        self.input_layernorm = input_norm
        self.post_attention_layernorm = post_norm
        self.device = device
        # L'attention et le MLP n'ont pas à s'exécuter sur le même appareil.
        # Quand les poids du MLP résident en mémoire vive, il est en général
        # moins coûteux de les y calculer que de les copier par le PCIe — voir
        # PlannerOptions.host_exec.
        self.mlp_device = mlp_device or device

    def forward(self, x: torch.Tensor, batch: ForwardBatch,
                cache: Optional[PagedKVCache]) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), batch, cache)
        h = self.post_attention_layernorm(x)
        if self.mlp_device != self.device:
            # Seul l'état caché traverse le bus : [jetons, dimension], quelques
            # kilooctets par jeton décodé face à des gigaoctets de poids.
            y = self.mlp(h.to(self.mlp_device)).to(x.device, non_blocking=True)
        else:
            y = self.mlp(h)
        return x + y

    def decode_fixed(self, x: torch.Tensor, positions: torch.Tensor,
                     slots: torch.Tensor, block_tables: torch.Tensor,
                     seq_lens: torch.Tensor, max_pos: int,
                     cache: PagedKVCache) -> torch.Tensor:
        x = x + self.self_attn.decode_fixed(self.input_layernorm(x), positions,
                                            slots, block_tables, seq_lens,
                                            max_pos, cache)
        return x + self.mlp(self.post_attention_layernorm(x))

    def prefetch(self) -> None:
        for m in self.modules():
            if isinstance(m, QuantLinear) and m.streamed is not None:
                m.prefetch()


class ACVRamModel(nn.Module):
    """Le modèle assemblé, ses couches réparties sur plusieurs appareils."""

    def __init__(self, spec: ModelSpec, embed: torch.Tensor,
                 layers: list[DecoderLayer], norm: RMSNorm,
                 lm_head: QuantLinear, caches: dict[int, PagedKVCache],
                 dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.spec = spec
        self.embed_tokens = embed          # kept as a plain tensor: it is a gather
        self.layers = nn.ModuleList(layers)
        self.norm = norm
        self.lm_head = lm_head
        self.caches = caches
        self.dtype = dtype

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, return_hidden: bool = False,
                logits_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Logits du dernier jeton de chaque séquence.

        Avec ``return_hidden``, ce sont les états cachés normalisés qui sont
        rendus, pour tous les jetons et non seulement le dernier — c'est sur eux
        que /v1/embeddings fait sa moyenne. Sauter lm_head évite aussi le produit
        matriciel le plus coûteux du modèle : plonger un document coûte donc
        nettement moins que d'engendrer à partir de lui.
        """
        idx = batch.tokens.to(self.embed_tokens.device)
        x = F.embedding(idx, self.embed_tokens).to(self.dtype)

        current = None
        for i, layer in enumerate(self.layers):
            if layer.device != current:
                x = x.to(layer.device, non_blocking=True)
                current = layer.device
            # On lance le transfert de la couche suivante avant d'exécuter
            # celle-ci, pour que la copie PCIe d'une couche résidant en mémoire
            # vive se cache derrière du vrai travail.
            if i + 1 < len(self.layers):
                self.layers[i + 1].prefetch()
            x = layer(x, batch, self.caches.get(i))

        x = self.norm(x.to(self.norm.weight.device))
        if return_hidden:
            return x
        # La vérification spéculative et la perplexité ont toutes deux besoin
        # de logits ailleurs qu'à la position finale : les lignes qui atteignent
        # lm_head sont donc un paramètre. Cela compte, car lm_head est le plus
        # gros produit matriciel du modèle, et l'exécuter sur chaque jeton de
        # prefill au lieu d'un seul coûte du temps réel.
        idx = (batch.last_token_indices() if logits_positions is None
               else logits_positions)
        x = x[idx.to(x.device)]
        head_dev = getattr(self.lm_head.qweight, "qweight", None)
        target = head_dev.device if head_dev is not None else x.device
        return self.lm_head(x.to(target))

    def decode_fixed(self, x: torch.Tensor, positions: torch.Tensor,
                     slots: torch.Tensor, block_tables: torch.Tensor,
                     seq_lens: torch.Tensor, max_pos: int) -> torch.Tensor:
        """Logits d'un pas de décodage pur, à formes fixes.

        Le plongement est déjà fait — ``x`` est l'état caché d'entrée sur le
        périphérique des couches : l'indexation de la table de plongement vit
        hors du graphe, sur l'appareil où elle réside.
        """
        for i, layer in enumerate(self.layers):
            x = layer.decode_fixed(x, positions, slots, block_tables,
                                   seq_lens, max_pos, self.caches[i])
        x = self.norm(x)
        return self.lm_head(x)

    @property
    def nbytes(self) -> int:
        total = self.embed_tokens.numel() * self.embed_tokens.element_size()
        for m in self.modules():
            if isinstance(m, QuantLinear):
                total += m.nbytes
        return total
