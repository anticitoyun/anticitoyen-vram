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

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import kernels
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
    seq_ids: list[int] = None         # identités des séquences (états GDN)
    gdn_store: dict = None            # {layer_idx: {seq_id: état récurrent}}

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
                 k_norm: Optional[nn.Module] = None,
                 output_gate: bool = False,
                 n_kv_heads: Optional[int] = None,
                 head_dim: Optional[int] = None,
                 scale: Optional[float] = None,
                 v_norm_eps: Optional[float] = None,
                 k_eq_v: bool = False,
                 window: int = 0) -> None:
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = q, k, v, o
        # qwen3-next : q_proj sort, par tête, [q | porte] ; la sortie de
        # l'attention est multipliée par sigmoïde(porte) avant o_proj.
        self.output_gate = output_gate
        # Qwen3, Gemma 3 et Olmo 2 normalisent Q et K par tete, avant la RoPE.
        # L'ordre compte : normaliser apres ferait tourner un vecteur puis
        # ecraserait sa norme, ce que le modele n'a pas appris.
        self.q_norm, self.k_norm = q_norm, k_norm
        self.n_heads = spec.num_attention_heads
        self.n_kv_heads = n_kv_heads or spec.num_key_value_heads
        self.head_dim = head_dim or spec.head_dim
        self.n_rep = self.n_heads // max(1, self.n_kv_heads)
        self.scale = scale if scale is not None \
            else (spec.attention_multiplier or self.head_dim ** -0.5)
        # Gemma 4 : v normalisé (RMS sans poids), v = k brut sur les couches
        # globales (k_eq_v), fenêtre glissante sur les couches locales
        self.v_norm_eps = v_norm_eps
        self.k_eq_v = k_eq_v
        self.window = window
        self.rope = rope

    def _kv(self, x: torch.Tensor, t: int):
        k = self.k_proj(x).view(t, self.n_kv_heads, self.head_dim)
        v = k if self.k_eq_v else self.v_proj(x).view(t, self.n_kv_heads, self.head_dim)
        if self.v_norm_eps is not None:
            v32 = v.to(torch.float32)
            v = (v32 * torch.rsqrt(v32.pow(2).mean(-1, keepdim=True)
                                   + self.v_norm_eps)).to(x.dtype)
        return k, v

    def forward(self, x: torch.Tensor, batch: ForwardBatch,
                cache: Optional[PagedKVCache]) -> torch.Tensor:
        t = x.shape[0]
        gate = None
        if self.output_gate:
            # par tête : [q_h | porte_h] — l'ordre du point de contrôle HF,
            # conservé par le convertisseur GGUF (vérifié : l'ordre plat
            # dégénère immédiatement, celui-ci non)
            qg = self.q_proj(x).view(t, self.n_heads, 2 * self.head_dim)
            q, gate = qg[..., :self.head_dim].contiguous(), \
                qg[..., self.head_dim:]
        else:
            q = self.q_proj(x).view(t, self.n_heads, self.head_dim)
        k, v = self._kv(x, t)

        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        if self.rope is not None:
            cos, sin = self.rope(batch.positions_on(x.device), x.device, x.dtype,
                                 max_pos=max(batch.seq_lens))
            q, k = apply_rope(q, k, cos, sin)

        if cache is not None:
            cache.write(batch.slots_on(x.device), k, v)

        if batch.is_decode:
            return self._decode(q, k, v, batch, cache, t, gate)
        return self._prefill(q, k, v, batch, cache, t, gate)

    def decode_fixed(self, x: torch.Tensor, positions: torch.Tensor,
                     slots: torch.Tensor, block_tables: torch.Tensor,
                     seq_lens: torch.Tensor, max_pos: int,
                     cache: PagedKVCache, q_len: int = 1) -> torch.Tensor:
        """Le pas de décodage à formes fixes — le chemin que capture le graphe.

        Même mathématique que ``forward`` en décodage, mais aucun scalaire
        Python tiré des données : positions, emplacements, tables et longueurs
        sont des tenseurs dont seul le *contenu* change entre deux rejeux.
        ``max_pos`` majore les positions (la longueur maximale du godet).
        ``q_len`` > 1 est le pas de vérification spéculative : chaque séquence
        pose q_len positions, chacune voyant son propre préfixe causal — le
        noyau paginé le gère nativement, et c'est lui qui est exigé ici (le
        repli déquantifier-puis-SDPA ne connaît que q_len = 1).
        """
        b = x.shape[0]
        gate = None
        if self.output_gate:                 # porte fusionnée dans q (qwen35)
            qg = self.q_proj(x).view(b, self.n_heads, 2 * self.head_dim)
            q, gate = qg[..., :self.head_dim].contiguous(), qg[..., self.head_dim:]
        else:
            q = self.q_proj(x).view(b, self.n_heads, self.head_dim)
        k, v = self._kv(x, b)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        if self.rope is not None:
            cos, sin = self.rope(positions, x.device, x.dtype, max_pos=max_pos)
            q, k = apply_rope(q, k, cos, sin)
        cache.write(slots, k, v)
        out = kernels.paged_attention(q, cache, block_tables, seq_lens,
                                      self.n_rep, self.scale, q_len=q_len,
                                      window=self.window)
        if out is None:                    # cache non int8, ou pas de noyau
            if q_len != 1:
                raise RuntimeError("verification speculative a formes fixes "
                                   "sans noyau pagine : chemin inéligible")
            kk, vv = cache.gather_fixed(block_tables, q.dtype)
            out = decode_attention_fixed(q, kk, vv, seq_lens, self.n_rep,
                                         self.scale, window=self.window)
        out = self._gated(out.to(x.dtype), gate, b)
        return self.o_proj(out.reshape(b, self.n_heads * self.head_dim))

    def _gated(self, out: torch.Tensor, gate, t: int) -> torch.Tensor:
        if gate is not None:
            out = out * torch.sigmoid(gate.reshape(out.shape))
        return out

    def _prefill(self, q, k, v, batch: ForwardBatch,
                 cache: Optional[PagedKVCache], t: int,
                 gate=None) -> torch.Tensor:
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
                                       q_offset=offset, window=self.window)
            start = end
        out = self._gated(out, gate, t)
        return self.o_proj(out.reshape(t, self.n_heads * self.head_dim))

    def _decode(self, q, k, v, batch: ForwardBatch,
                cache: Optional[PagedKVCache], t: int,
                gate=None) -> torch.Tensor:
        """Décodage, et vérification spéculative, pour tout le lot d'un coup."""
        if cache is None:
            return self._prefill(q, k, v, batch, cache, t, gate)

        if all(ql == 1 for ql in batch.query_lens):
            # Décodage pur : le chemin à formes fixes, celui-là même que le
            # graphe CUDA capture — un seul gather vectorisé, pas de boucle
            # Python, et une sortie identique au bit près entre eager et rejeu.
            tables, lens = batch.fixed_decode_views(q.device)
            out = kernels.paged_attention(q, cache, tables, lens,
                                          self.n_rep, self.scale,
                                          window=self.window)
            if out is None:                # cache non int8, ou pas de noyau
                kk, vv = cache.gather_fixed(tables, q.dtype)
                out = decode_attention_fixed(q, kk, vv, lens, self.n_rep,
                                             self.scale, window=self.window)
            out = self._gated(out.to(q.dtype), gate, t)
            return self.o_proj(out.reshape(t, self.n_heads * self.head_dim))

        # Vérification spéculative : plusieurs positions de requête par
        # séquence, chacune ne voyant que son propre préfixe. Quand toutes les
        # séquences vérifient le même nombre de positions — le cas normal —
        # le noyau paginé les traite en un lancement, chaque ligne de requête
        # avec sa longueur causale propre.
        ql = batch.query_lens[0]
        if all(q_ == ql for q_ in batch.query_lens):
            tables, lens = batch.fixed_decode_views(q.device)
            out = kernels.paged_attention(q, cache, tables, lens,
                                          self.n_rep, self.scale, q_len=ql,
                                          window=self.window)
            if out is not None:
                out = self._gated(out.to(q.dtype), gate, t)
                return self.o_proj(out.reshape(t, self.n_heads * self.head_dim))

        keys, values = [], []
        for i in range(batch.batch_size):
            kk, vv = cache.gather(batch.block_tables[i].to(q.device),
                                  batch.seq_lens[i], q.dtype)
            keys.append(kk)
            values.append(vv)

        out = torch.empty_like(q)
        start = 0
        for i, qlen in enumerate(batch.query_lens):
            end = start + qlen
            offset = batch.seq_lens[i] - qlen
            out[start:end] = attention(
                q[start:end], repeat_kv(keys[i], self.n_rep),
                repeat_kv(values[i], self.n_rep), True, self.scale,
                q_offset=offset, window=self.window)
            start = end
        out = self._gated(out, gate, t)
        return self.o_proj(out.reshape(t, self.n_heads * self.head_dim))


class MLP(nn.Module):
    def __init__(self, gate: QuantLinear, up: QuantLinear, down: QuantLinear,
                 act: str = "silu") -> None:
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = gate, up, down
        self.act = act

    def _act(self, g: torch.Tensor) -> torch.Tensor:
        if self.act in ("gelu_pytorch_tanh", "gelu_tanh"):
            return F.gelu(g, approximate="tanh")
        if self.act == "gelu":
            return F.gelu(g)
        return F.silu(g)

    gate_up: Optional[nn.Module] = None

    def fuse(self) -> bool:
        """gate et up lisent la même entrée : une GEMV INT8 empilée au lieu
        de deux (les NVFP4 ont une échelle globale par tenseur : non empilés)."""
        from .layers import stack_int8_linears
        self.gate_up = stack_int8_linears([self.gate_proj, self.up_proj])
        return self.gate_up is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_up is not None and x.shape[0] <= 8:
            gu = self.gate_up(x)
            g, u = gu.split(gu.shape[-1] // 2, dim=-1)
            return self.down_proj(self._act(g) * u)
        return self.down_proj(self._act(self.gate_proj(x)) * self.up_proj(x))


class MLP2(nn.Module):
    """MLP sans porte (Nemotron-H) : down(act(up(x))), act = ReLU² ou GELU."""

    def __init__(self, up: QuantLinear, down: QuantLinear, act: str = "relu2") -> None:
        super().__init__()
        self.up_proj, self.down_proj = up, down
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up_proj(x)
        if self.act == "relu2":
            h = F.relu(h); h = h * h
        elif self.act.startswith("gelu"):
            h = F.gelu(h, approximate="tanh")
        else:
            h = F.silu(h)
        return self.down_proj(h)


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
                 norm_topk_prob: bool = True,
                 shared_gate: Optional[torch.Tensor] = None,
                 scoring: str = "softmax",
                 score_bias: Optional[torch.Tensor] = None,
                 routed_scale: float = 1.0) -> None:
        super().__init__()
        self.router = router
        self.experts = nn.ModuleList(experts)
        self.top_k = top_k
        self.shared = shared
        # porte sigmoïde de l'expert partagé (Qwen3-Next) : vecteur [1, d]
        self.shared_gate = shared_gate
        self.norm_topk_prob = norm_topk_prob
        # routage DeepSeek (kimi-linear) : scores sigmoïde, biais de sélection
        # (e_score_correction_bias) hors des poids, renormalisation, échelle
        self.scoring = scoring
        self.score_bias = score_bias
        self.routed_scale = routed_scale
        self._stack_state = "?"                # ? | oui | non
        self._stacks = None

    # ------------------------------------------------------------------
    # Pile d'experts pour le chemin groupé. Les qweight/échelles de tous les
    # experts d'une projection sont recopiés dans un tenseur [E, ...] contigu,
    # puis chaque expert reçoit une *vue* de la pile — la mémoire n'est pas
    # doublée, et la boucle par expert du prefill continue de marcher.
    # ------------------------------------------------------------------
    def _noms_experts(self) -> tuple:
        return ("gate_proj", "up_proj", "down_proj") if hasattr(self.experts[0], "gate_proj") \
            else ("up_proj", "down_proj")

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
        for nom in self._noms_experts():
            pile = one([getattr(e, nom) for e in self.experts])
            if pile is None:
                return False
            piles[nom] = pile
        self._stacks = piles
        return True

    def _grouped(self, x32: torch.Tensor, pile, expert_ids, token_ids):
        if pile[0] == "nvfp4":
            _, qw, bs, gs, k, m = pile
            return kernels.nvfp4_gemv_grouped(x32, qw, bs, gs, expert_ids,
                                              token_ids, k)[:, :m]
        _, qw, sc, zr, k, gsz, m = pile
        return kernels.int4_gemv_grouped(x32, qw, sc, zr, expert_ids,
                                         token_ids, k, gsz)[:, :m]

    # -- prefill : GEMM groupées sur les jetons triés par expert -------------
    def _pile_bf16(self, pile) -> Optional[torch.Tensor]:
        """La pile d'experts d'une projection déquantifiée en bf16 [E, M, K]
        (transitoire : ~850 Mo par projection pour 180 experts de 1024x2304)."""
        if pile[0] != "nvfp4":
            return None
        _, qw, bs, gs, k, m = pile
        E, M = qw.shape[0], qw.shape[1]
        from ..quant.formats import NVFP4Tensor
        plat = NVFP4Tensor.__new__(NVFP4Tensor)
        plat.qweight = qw.view(E * M, -1)
        plat.block_scale = bs.view(E * M, -1)
        plat.global_scale = torch.ones((), dtype=torch.float32, device=qw.device)
        plat.padded_in = k
        plat.shape = (E * M, k)
        w = kernels.nvfp4_dequant(plat, torch.bfloat16).view(E, M, k)
        w.mul_(gs.to(torch.bfloat16).view(E, 1, 1))
        return w[:, :m, :]

    def _forward_prefill_grouped(self, x, topw, topi) -> Optional[torch.Tensor]:
        if not hasattr(torch, "_grouped_mm") or self._stacks is None:
            return None
        if "gate_proj" not in self._stacks:
            return None                                # experts sans porte : boucle
        pg, pu, pd = (self._stacks[n] for n in ("gate_proj", "up_proj", "down_proj"))
        if any(p[0] != "nvfp4" for p in (pg, pu, pd)):
            return None
        t, k = topi.shape
        E = pg[1].shape[0]
        flat_e = topi.reshape(-1).to(torch.int64)
        flat_t = torch.arange(t, device=x.device).repeat_interleave(k)
        ordre = torch.argsort(flat_e, stable=True)
        offs = torch.cumsum(torch.bincount(flat_e, minlength=E), 0).to(torch.int32)
        xs = x[flat_t[ordre]].to(torch.bfloat16).contiguous()        # [G, H]
        wg = self._pile_bf16(pg); g = torch._grouped_mm(xs, wg.transpose(1, 2), offs=offs); del wg
        wu = self._pile_bf16(pu); u = torch._grouped_mm(xs, wu.transpose(1, 2), offs=offs); del wu
        act = (F.silu(g.to(torch.float32)) * u.to(torch.float32)).to(torch.bfloat16)
        wd = self._pile_bf16(pd); d = torch._grouped_mm(act, wd.transpose(1, 2), offs=offs); del wd
        d = d.to(torch.float32) * topw.reshape(-1)[ordre].to(torch.float32).unsqueeze(-1)
        inv = torch.empty_like(ordre); inv[ordre] = torch.arange(ordre.numel(), device=x.device)
        return d[inv].view(t, k, -1).sum(dim=1).to(x.dtype)

    def _forward_grouped(self, x, topw, topi):
        t = x.shape[0]
        eid = topi.reshape(-1).to(torch.int32)
        tok = torch.arange(t, device=x.device,
                           dtype=torch.int32).repeat_interleave(self.top_k)
        if "gate_proj" not in self._stacks:            # experts sans porte (ReLU²)
            u = self._grouped(x.to(torch.float32), self._stacks["up_proj"], eid, tok)
            act = F.relu(u); act = act * act
            seq = torch.arange(eid.shape[0], device=x.device, dtype=torch.int32)
            d = self._grouped(act, self._stacks["down_proj"], eid, seq)
            d = d * topw.reshape(-1, 1).to(d.dtype)
            return d.view(t, self.top_k, -1).sum(dim=1).to(x.dtype)
        pg, pu = self._stacks["gate_proj"], self._stacks["up_proj"]
        ext = kernels.get_extension()
        if (pg[0] == "nvfp4" and pu[0] == "nvfp4" and ext is not None
                and hasattr(ext, "nvfp4_gemv_grouped_gateup")
                and pg[4] * 4 <= 48 * 1024):
            # gate, up et SiLU·up en un lancement, activation bf16 lue telle quelle
            act = ext.nvfp4_gemv_grouped_gateup(
                pg[1], pg[2], pg[3], pu[1], pu[2], pu[3], eid, tok,
                x.contiguous(), pg[4])[:, :pg[5]]
        else:
            x32 = x.to(torch.float32)
            g = self._grouped(x32, pg, eid, tok)
            u = self._grouped(x32, pu, eid, tok)
            act = F.silu(g) * u                 # [G, I] fp32
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
        # le routeur est un petit poids en clair : on garde sa copie fp32
        # plutôt que de la reconvertir à chaque pas
        w32 = getattr(self, "_router_w32", None)
        if w32 is None and hasattr(self.router.qweight, "weight"):
            w32 = self.router.qweight.weight.to(torch.float32)
            self._router_w32 = w32
        logits = (F.linear(x.to(torch.float32), w32) if w32 is not None
                  else self.router(x.to(torch.float32)))
        ext = kernels.get_extension() if x.is_cuda else None
        if (ext is not None and hasattr(ext, "moe_route")
                and logits.shape[-1] <= 1024 and self.top_k <= 32):
            # un lancement : scores, biais, top-k, renormalisation, échelle
            bias = self.score_bias if self.score_bias is not None \
                else torch.empty(0, device=x.device)
            topw, topi = ext.moe_route(logits, bias, self.top_k,
                                       self.scoring == "sigmoid",
                                       bool(self.norm_topk_prob),
                                       float(self.routed_scale))
        else:
            if self.scoring == "sigmoid":
                scores = torch.sigmoid(logits)
                sel = scores if self.score_bias is None else scores + self.score_bias
                _, topi = torch.topk(sel, self.top_k, dim=-1)
                topw = scores.gather(-1, topi)
            else:
                weights = F.softmax(logits, dim=-1)
                topw, topi = torch.topk(weights, self.top_k, dim=-1)
            if self.norm_topk_prob:
                topw = topw / topw.sum(dim=-1, keepdim=True)
            if self.routed_scale != 1.0:
                topw = topw * self.routed_scale
        topw = topw.to(x.dtype)

        # Chemin groupé : trois lancements pour toute la couche, quel que soit
        # le nombre d'experts touchés. La boucle par expert reste le chemin des
        # grands lots de prefill (le regroupement par expert y redevient
        # rentable) et le repli des piles hétérogènes.
        if x.is_cuda and self._stack_state != "non":
            if self._stack_state == "?":
                self._stack_state = "oui" if self._try_build_stacks() else "non"
            if self._stack_state == "oui":
                if t <= _MOE_GROUPED_MAX:
                    y = self._forward_grouped(x, topw, topi)
                else:
                    y = self._forward_prefill_grouped(x, topw, topi)
                if y is not None:
                    if self.shared is not None:
                        y = y + self._shared_out(x)
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
            out = out + self._shared_out(x)
        return out

    def _shared_out(self, x: torch.Tensor) -> torch.Tensor:
        y = self.shared(x)
        if self.shared_gate is not None:
            y = y * torch.sigmoid(x @ self.shared_gate.t())
        return y

    def prefetch(self) -> None:
        for expert in self.experts:
            for lin in (expert.gate_proj, expert.up_proj, expert.down_proj):
                lin.prefetch()


# Jetons au-delà desquels le MoE repasse de la GEMV groupée (une paire
# (jeton, expert) par tranche de grille) à la boucle par expert : la boucle
# coûte ~0,6 ms par expert visité, la GEMV groupée relit les poids de
# l'expert pour chaque jeton — croisement mesuré vers quelques milliers.
_MOE_GROUPED_MAX = int(os.environ.get("ACVRAM_MOE_GROUPED_MAX", "32"))

# Marque, dans le magasin d'états, une séquence dont l'état réside dans les
# tampons fixes d'une couche (chemin graphes) plutôt qu'en tuple fonctionnel.
_STATIC = object()


class DecoderLayerGDN(nn.Module):
    """Bloc à récurrence linéaire : Gated DeltaNet à la place de l'attention.

    L'état (convolution + matrice delta) vit par séquence dans
    ``batch.gdn_store[index]`` — porté par le moteur, hors du cache paginé.
    """

    def __init__(self, index: int, gdn: nn.Module, mlp: nn.Module,
                 input_norm: "RMSNorm", post_norm: "RMSNorm",
                 device: torch.device) -> None:
        super().__init__()
        self.index = index
        self.linear_attn = gdn
        self.mlp = mlp
        self.input_layernorm = input_norm
        self.post_attention_layernorm = post_norm
        self.device = device
        self.mlp_device = device

    def forward(self, x: torch.Tensor, batch: ForwardBatch,
                cache=None) -> torch.Tensor:
        h = self.input_layernorm(x)
        store = batch.gdn_store.setdefault(self.index, {}) \
            if batch.gdn_store is not None else {}
        sorties = []
        start = 0
        for i, ql in enumerate(batch.query_lens):
            sid = batch.seq_ids[i] if batch.seq_ids else i
            etat = store.get(sid)
            if etat is _STATIC:               # l'état vit dans les tampons fixes
                etat = self.linear_attn.static_export(self.static)
                self.static_owner = None
            y, etat = self.linear_attn(h[start:start + ql], etat)
            store[sid] = etat
            sorties.append(y)
            start += ql
        x = x + torch.cat(sorties).to(x.dtype)
        if self.mlp is None:
            return x
        return x + self.mlp(self.post_attention_layernorm(x))

    # -- chemin à formes fixes (graphes CUDA), une séquence ----------------
    static: Optional[dict] = None
    static_owner: Optional[int] = None
    static_bucket: int = 0

    def static_bind(self, sid: int, store: dict, max_len: int,
                    dtype: torch.dtype) -> None:
        """Amène l'état de ``sid`` dans les tampons fixes de la couche.

        L'état du propriétaire précédent est exporté vers le magasin s'il y
        vit encore ; celui de ``sid`` est chargé (ou remis à zéro). Le magasin
        note alors que l'état de ``sid`` réside dans les tampons.
        """
        la = self.linear_attn
        if self.static is None:
            if hasattr(la, "rank"):           # MLA : cache latent borné
                self.static = la.new_static(self.device, max_len, dtype)
            else:
                self.static = la.new_static(self.device)
        if self.static_owner == sid and store.get(sid) is _STATIC:
            return
        prev = self.static_owner
        if prev is not None and prev != sid and store.get(prev) is _STATIC:
            store[prev] = la.static_export(self.static)
        etat = store.get(sid)
        la.static_load(self.static, None if etat is _STATIC else etat)
        store[sid] = _STATIC
        self.static_owner = sid

    def decode_fixed(self, x: torch.Tensor, positions: torch.Tensor,
                     slots: torch.Tensor, block_tables: torch.Tensor,
                     seq_lens: torch.Tensor, max_pos: int,
                     cache, q_len: int = 1) -> torch.Tensor:
        h = self.input_layernorm(x)
        la = self.linear_attn
        if hasattr(la, "rank"):
            y = la.decode_static(h, self.static, self.static_bucket)
        else:
            y = la.decode_static(h, self.static)
        x = x + y.to(x.dtype)
        if self.mlp is None:
            return x
        return x + self.mlp(self.post_attention_layernorm(x))

    def prefetch(self) -> None:
        pass


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
        self.residual_multiplier = 1.0        # granite : résidu atténué

    def forward(self, x: torch.Tensor, batch: ForwardBatch,
                cache: Optional[PagedKVCache]) -> torch.Tensor:
        r = self.residual_multiplier
        if self.self_attn is None:                     # couche MLP seule (Nemotron-H)
            y = self.mlp(self.input_layernorm(x))
            return x + (y if r == 1.0 else y * r)
        a = self.self_attn(self.input_layernorm(x), batch, cache)
        x = x + (a if r == 1.0 else a * r)
        if self.mlp is None:                           # couche d'attention seule
            return x
        h = self.post_attention_layernorm(x)
        if self.mlp_device != self.device:
            # Seul l'état caché traverse le bus : [jetons, dimension], quelques
            # kilooctets par jeton décodé face à des gigaoctets de poids.
            y = self.mlp(h.to(self.mlp_device)).to(x.device, non_blocking=True)
        else:
            y = self.mlp(h)
        return x + (y if r == 1.0 else y * r)

    def decode_fixed(self, x: torch.Tensor, positions: torch.Tensor,
                     slots: torch.Tensor, block_tables: torch.Tensor,
                     seq_lens: torch.Tensor, max_pos: int,
                     cache: PagedKVCache, q_len: int = 1) -> torch.Tensor:
        r = self.residual_multiplier
        if self.self_attn is None:
            y = self.mlp(self.input_layernorm(x))
            return x + (y if r == 1.0 else y * r)
        a = self.self_attn.decode_fixed(self.input_layernorm(x), positions,
                                        slots, block_tables, seq_lens,
                                        max_pos, cache, q_len)
        x = x + (a if r == 1.0 else a * r)
        if self.mlp is None:
            return x
        y = self.mlp(self.post_attention_layernorm(x))
        return x + (y if r == 1.0 else y * r)

    def prefetch(self) -> None:
        for m in self.modules():
            if isinstance(m, QuantLinear) and m.streamed is not None:
                m.prefetch()


class DecoderLayerGemma(nn.Module):
    """Bloc Gemma 4 : normes avant ET après l'attention et le MLP, puis un
    scalaire de sortie par couche (layer_scalar)."""

    def __init__(self, index: int, attn: Attention, mlp: nn.Module,
                 input_norm: RMSNorm, post_attn_norm: RMSNorm,
                 pre_ffn_norm: RMSNorm, post_ffn_norm: RMSNorm,
                 out_scale: Optional[torch.Tensor], device: torch.device) -> None:
        super().__init__()
        self.index = index
        self.self_attn = attn
        self.mlp = mlp
        self.input_layernorm = input_norm
        self.post_attention_layernorm = post_attn_norm
        self.pre_feedforward_layernorm = pre_ffn_norm
        self.post_feedforward_layernorm = post_ffn_norm
        self.out_scale = out_scale
        self.device = device
        self.mlp_device = device
        self.residual_multiplier = 1.0

    def _reste(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = x + self.post_attention_layernorm(a)
        y = self.mlp(self.pre_feedforward_layernorm(x))
        x = x + self.post_feedforward_layernorm(y)
        if self.out_scale is not None:
            x = x * self.out_scale.to(x.dtype)
        return x

    def forward(self, x: torch.Tensor, batch: ForwardBatch,
                cache: Optional[PagedKVCache]) -> torch.Tensor:
        return self._reste(x, self.self_attn(self.input_layernorm(x), batch, cache))

    def decode_fixed(self, x: torch.Tensor, positions: torch.Tensor,
                     slots: torch.Tensor, block_tables: torch.Tensor,
                     seq_lens: torch.Tensor, max_pos: int,
                     cache: PagedKVCache, q_len: int = 1) -> torch.Tensor:
        a = self.self_attn.decode_fixed(self.input_layernorm(x), positions,
                                        slots, block_tables, seq_lens,
                                        max_pos, cache, q_len)
        return self._reste(x, a)

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
        if self.spec.embedding_multiplier != 1.0:
            x = x * self.spec.embedding_multiplier

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
        logits = self.lm_head(x.to(target))
        return self._logits_finaux(logits)

    def _logits_finaux(self, logits: torch.Tensor) -> torch.Tensor:
        if self.spec.logits_scaling != 1.0:
            logits = logits / self.spec.logits_scaling
        c = self.spec.final_logit_softcapping
        if c:
            logits = torch.tanh(logits.to(torch.float32) / c) * c
        return logits

    def decode_fixed(self, x: torch.Tensor, positions: torch.Tensor,
                     slots: torch.Tensor, block_tables: torch.Tensor,
                     seq_lens: torch.Tensor, max_pos: int,
                     q_len: int = 1) -> torch.Tensor:
        """Logits d'un pas de décodage pur, à formes fixes.

        Le plongement est déjà fait — ``x`` est l'état caché d'entrée sur le
        périphérique des couches : l'indexation de la table de plongement vit
        hors du graphe, sur l'appareil où elle réside.
        """
        for i, layer in enumerate(self.layers):
            x = layer.decode_fixed(x, positions, slots, block_tables,
                                   seq_lens, max_pos, self.caches.get(i), q_len)
        x = self.norm(x)
        return self._logits_finaux(self.lm_head(x))

    @property
    def nbytes(self) -> int:
        total = self.embed_tokens.numel() * self.embed_tokens.element_size()
        for m in self.modules():
            if isinstance(m, QuantLinear):
                total += m.nbytes
        return total
