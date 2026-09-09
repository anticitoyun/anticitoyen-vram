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
from ..memory import trace_routage as _trace_routage
from ..memory.kvcache import PagedKVCache, bucket_blocks
from .config import ModelSpec
from .layers import (QuantLinear, RMSNorm, RotaryEmbedding, add_norm, apply_rope,
                     rope_fusee,
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
        # Déclaré ici et pas au niveau de la classe : un attribut de classe
        # masque le module enregistré par nn.Module.__setattr__, et la
        # projection empilée resterait invisible (self.qkv_proj toujours None).
        self.qkv_proj = None
        self.qkv_tailles = ()

    # q, k et v lisent la même entrée : au décodage, trois GEMV dont deux
    # minuscules (têtes KV groupées) coûtent plus que la seule grande qui
    # les contient toutes.
    def fuse(self) -> bool:
        from .layers import (stack_int8_linears, stack_nvfp4_linears,
                             stack_plain_linears)
        lins = [self.q_proj, self.k_proj] + ([] if self.k_eq_v else [self.v_proj])
        if any(l is None for l in lins):
            return False
        self.qkv_proj = (stack_int8_linears(lins) or stack_nvfp4_linears(lins)
                         or stack_plain_linears(lins))
        if self.qkv_proj is None:
            return False
        self.qkv_tailles = tuple(l.qweight.shape[0] for l in lins)
        return True

    def _proj(self, x: torch.Tensor, t: int):
        """q, k, v (et la porte de sortie) : une GEMV empilée si possible."""
        if self.qkv_proj is not None and t <= 8:
            p = torch.split(self.qkv_proj(x), self.qkv_tailles, dim=-1)
            qr, kr = p[0], p[1]
            vr = kr if self.k_eq_v else p[2]
        else:
            qr, kr = self.q_proj(x), self.k_proj(x)
            vr = kr if self.k_eq_v else self.v_proj(x)
        gate = None
        if self.output_gate:
            # par tête : [q_h | porte_h] — l'ordre du point de contrôle HF,
            # conservé par le convertisseur GGUF (vérifié : l'ordre plat
            # dégénère immédiatement, celui-ci non)
            qg = qr.view(t, self.n_heads, 2 * self.head_dim)
            q, gate = qg[..., :self.head_dim].contiguous(), qg[..., self.head_dim:]
        else:
            q = qr.view(t, self.n_heads, self.head_dim)
        k = kr.view(t, self.n_kv_heads, self.head_dim)
        v = k if self.k_eq_v else vr.view(t, self.n_kv_heads, self.head_dim)
        if self.v_norm_eps is not None:
            v32 = v.to(torch.float32)
            v = (v32 * torch.rsqrt(v32.pow(2).mean(-1, keepdim=True)
                                   + self.v_norm_eps)).to(x.dtype)
        return q, k, v, gate

    def forward(self, x: torch.Tensor, batch: ForwardBatch,
                cache: Optional[PagedKVCache]) -> torch.Tensor:
        t = x.shape[0]
        q, k, v, gate = self._proj(x, t)

        r = None
        if self.rope is not None:
            pos = batch.positions_on(x.device)
            mx = max(batch.seq_lens)
            r = rope_fusee(q, k, self.rope, pos, mx, self.q_norm, self.k_norm)
        if r is not None:
            q, k = r                       # normes par tête comprises
        else:
            if self.q_norm is not None:
                q = self.q_norm(q)
            if self.k_norm is not None:
                k = self.k_norm(k)
            if self.rope is not None:
                cos, sin = self.rope(pos, x.device, x.dtype, max_pos=mx)
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
        q, k, v, gate = self._proj(x, b)
        r = None
        if self.rope is not None:
            r = rope_fusee(q, k, self.rope, positions, max_pos,
                           self.q_norm, self.k_norm)
        if r is not None:
            q, k = r                       # normes par tête comprises
        else:
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
        self.gate_up = None       # attribut d'instance : voir Attention.fuse

    def _act(self, g: torch.Tensor) -> torch.Tensor:
        if self.act in ("gelu_pytorch_tanh", "gelu_tanh"):
            return F.gelu(g, approximate="tanh")
        if self.act == "gelu":
            return F.gelu(g)
        return F.silu(g)

    def fuse(self) -> bool:
        """gate et up lisent la même entrée : une GEMV empilée au lieu de deux.

        Les NVFP4 ont chacun leur échelle globale ; le noyau en accepte une par
        ligne de sortie, ce qui les empile sans réarrondi (v0.4.62)."""
        from .layers import (stack_int8_linears, stack_nvfp4_linears,
                             stack_plain_linears)
        paire = [self.gate_proj, self.up_proj]
        self.gate_up = (stack_int8_linears(paire) or stack_nvfp4_linears(paire)
                        or stack_plain_linears(paire))
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


_SYNC_COUCHES = bool(os.environ.get("ACVRAM_SYNC_COUCHES"))


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
        # Index de la couche, pose par le chargeur. -1 quand personne ne l'a
        # pose : la trace de routage l'ecrit tel quel plutot que d'inventer un
        # numero, et une trace pleine de -1 se voit tout de suite.
        self.index_couche = -1
        # activation des experts : le chemin groupé la reproduit (SiLU par
        # défaut, GELU-tanh pour Gemma 4)
        a = getattr(experts[0], "act", "silu") if experts else "silu"
        self.act = "gelu_tanh" if str(a).startswith("gelu") else "silu"

    def _act(self, g: torch.Tensor) -> torch.Tensor:
        if self.act == "gelu_tanh":
            return F.gelu(g, approximate="tanh")
        return F.silu(g)

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
        try:
            for nom in self._noms_experts():
                pile = one([getattr(e, nom) for e in self.experts])
                if pile is None:
                    return False
                piles[nom] = pile
        except torch.OutOfMemoryError:
            # la pile d'une projection double transitoirement sa mémoire ;
            # un modèle qui remplit la carte (80B) reste sur la boucle par
            # expert plutôt que de mourir ici
            torch.cuda.empty_cache()
            print("[acvram] piles d'experts : mémoire GPU insuffisante, boucle par expert",
                  flush=True)
            return False
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
        # échelle globale par expert appliquée dans le noyau (une passe de
        # moins sur ~30 Go de bf16 au prefill d'un 30B)
        w = kernels.nvfp4_dequant(plat, torch.bfloat16,
                                  gscale_rows=gs.reshape(-1).to(torch.float32),
                                  rows_per_group=M).view(E, M, k)
        return w[:, :m, :]

    @staticmethod
    def _tuiles(cnt: torch.Tensor, bt: int = 16):
        """Découpe chaque expert en tuiles de ``bt`` jetons consécutifs.

        Renvoie (expert, premier jeton, compte) par tuile — la grille du noyau
        de GEMM groupée, qui ne connaît qu'un expert par bloc."""
        dev = cnt.device
        starts = torch.cumsum(cnt, 0) - cnt
        ntiles = (cnt + bt - 1) // bt
        tot = int(ntiles.sum())
        if tot == 0:
            vide = torch.zeros(0, dtype=torch.int32, device=dev)
            return vide, vide, vide
        tile_e = torch.repeat_interleave(
            torch.arange(cnt.numel(), device=dev), ntiles)
        base = torch.cumsum(ntiles, 0) - ntiles
        idx = torch.arange(tot, device=dev) - torch.repeat_interleave(base, ntiles)
        t0 = starts[tile_e] + idx * bt
        n = torch.clamp(cnt[tile_e] - idx * bt, max=bt)
        return (tile_e.to(torch.int32), t0.to(torch.int32), n.to(torch.int32))

    def _gemm(self, pile, xs, tiles):
        _, qw, bs, gs, k, m = pile
        return kernels.get_extension().nvfp4_gemm_grouped(
            qw, bs, gs, xs, tiles[0], tiles[1], tiles[2], k)[:, :m]

    def _forward_prefill_grouped(self, x, topw, topi) -> Optional[torch.Tensor]:
        if self._stacks is None or "gate_proj" not in self._stacks:
            return None                                # experts sans porte : boucle
        pg, pu, pd = (self._stacks[n] for n in ("gate_proj", "up_proj", "down_proj"))
        if any(p[0] != "nvfp4" for p in (pg, pu, pd)):
            return None
        ext = kernels.get_extension()
        # La GEMM groupée relit les poids d'un expert une fois par tuile de
        # 16 jetons ; la déquantification, elle, les écrit puis les relit en
        # bf16 une seule fois quel que soit le lot. Le premier gagne tant que
        # les experts reçoivent peu de jetons — croisement mesuré entre 64
        # et 128 sur Qwen3-Coder-30B (512 j : +33 %, 1024 j : +5 %, 2048 j : -25 %).
        par_expert = topi.numel() / max(1, pg[1].shape[0])
        direct = (ext is not None and hasattr(ext, "nvfp4_gemm_grouped")
                  and not os.environ.get("ACVRAM_PREFILL_DEQUANT")
                  and par_expert <= _MOE_GEMM_MAX
                  and pg[4] % 64 == 0 and pd[4] % 64 == 0)
        if not direct and not hasattr(torch, "_grouped_mm"):
            return None
        t, k = topi.shape
        E = pg[1].shape[0]
        flat_e = topi.reshape(-1).to(torch.int64)
        flat_t = torch.arange(t, device=x.device).repeat_interleave(k)
        ordre = torch.argsort(flat_e, stable=True)
        cnt = torch.bincount(flat_e, minlength=E)
        xs = x[flat_t[ordre]].to(torch.bfloat16).contiguous()        # [G, H]
        if direct:
            # les poids restent en 4 bits : plus de pile bf16 intermédiaire
            # (trois passes de plusieurs Gio par couche en moins)
            tiles = self._tuiles(cnt)
            if xs.shape[1] != pg[4]:                   # entrée rembourrée
                xs = F.pad(xs, (0, pg[4] - xs.shape[1])).contiguous()
            g = self._gemm(pg, xs, tiles)
            u = self._gemm(pu, xs, tiles)
            act = (self._act(g.to(torch.float32))
                   * u.to(torch.float32)).to(torch.bfloat16)
            if act.shape[1] != pd[4]:
                act = F.pad(act, (0, pd[4] - act.shape[1]))
            d = self._gemm(pd, act.contiguous(), tiles)
        else:
            offs = torch.cumsum(cnt, 0).to(torch.int32)
            wg = self._pile_bf16(pg); g = torch._grouped_mm(xs, wg.transpose(1, 2), offs=offs); del wg
            wu = self._pile_bf16(pu); u = torch._grouped_mm(xs, wu.transpose(1, 2), offs=offs); del wu
            act = (self._act(g.to(torch.float32)) * u.to(torch.float32)).to(torch.bfloat16)
            wd = self._pile_bf16(pd); d = torch._grouped_mm(act, wd.transpose(1, 2), offs=offs); del wd
        d = d.to(torch.float32) * topw.reshape(-1)[ordre].to(torch.float32).unsqueeze(-1)
        inv = torch.empty_like(ordre); inv[ordre] = torch.arange(ordre.numel(), device=x.device)
        return d[inv].view(t, k, -1).sum(dim=1).to(x.dtype)

    def _forward_grouped(self, x, topw, topi):
        t = x.shape[0]
        ext = kernels.get_extension()
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
        if (pg[0] == "nvfp4" and pu[0] == "nvfp4" and ext is not None
                and hasattr(ext, "nvfp4_gemv_grouped_gateup")
                and pg[4] * 4 <= 48 * 1024):
            # gate, up et SiLU·up en un lancement, activation bf16 lue telle quelle
            act = ext.nvfp4_gemv_grouped_gateup(
                pg[1], pg[2], pg[3], pu[1], pu[2], pu[3], eid, tok,
                x.contiguous(), pg[4],
                1 if self.act == "gelu_tanh" else 0)[:, :pg[5]]
        else:
            x32 = x.to(torch.float32)
            g = self._grouped(x32, pg, eid, tok)
            u = self._grouped(x32, pu, eid, tok)
            act = self._act(g) * u              # [G, I] fp32
        seq = torch.arange(eid.shape[0], device=x.device, dtype=torch.int32)
        d = self._grouped(act, self._stacks["down_proj"], eid, seq)
        # Chaque jeton possède exactement top_k lignes contiguës : une somme
        # sur cet axe remplace l'index_add_ atomique — déterministe, plus
        # rapide, et rejouable dans un graphe CUDA sans écart d'un rejeu à
        # l'autre. Pondération, somme et conversion tiennent en un lancement.
        if (ext is not None and hasattr(ext, "moe_reduce")
                and d.dtype == torch.float32 and x.dtype == torch.bfloat16):
            tw = topw.reshape(-1)
            if tw.dtype != torch.float32:
                tw = tw.to(torch.float32)
            return ext.moe_reduce(d.contiguous(), tw.contiguous(), self.top_k)
        d = d * topw.reshape(-1, 1).to(d.dtype)
        return d.view(t, self.top_k, -1).sum(dim=1).to(x.dtype)

    def _router_logits(self, x: torch.Tensor) -> torch.Tensor:
        # Le routeur est un petit poids en clair : sa copie est gardée dans le
        # type de l'entrée. En bf16 le produit accumule quand même en fp32
        # (cuBLAS) mais évite de convertir l'état caché à chaque couche.
        cache = getattr(self, "_router_w", None)
        if cache is None:
            cache = {}
            self._router_w = cache
        w = cache.get(x.dtype)
        if w is None and hasattr(self.router.qweight, "weight"):
            w = self.router.qweight.weight.to(x.dtype)
            cache[x.dtype] = w
        return F.linear(x, w) if w is not None else self.router(x)

    def _route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Poids et indices des top-k experts par jeton."""
        logits = self._router_logits(x)
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
        # Trace de routage : un test de booleen quand elle est eteinte, et le
        # module ne touche a rien de plus. Sous trace, elle synchronise — c'est
        # le prix d'une mesure d'ordre, et elle n'est jamais active en service.
        if _trace_routage.actif():
            _trace_routage.noter(self.index_couche, topi)
        return topw, topi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, h = x.shape
        # topw reste en fp32 : il sort du routage ainsi et y retourne pour la
        # réduction pondérée ; l'aller-retour en bf16 coûtait deux copies par
        # couche pour rien
        topw, topi = self._route(x)

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

        if t == 1 and not os.environ.get("ACVRAM_MOE_DECODE_MASQUES"):
            # Décodage, un jeton : les masques par expert (nonzero, index)
            # coûtaient une trentaine de synchronisations hôte par couche —
            # le profil du 8/09 y voyait 2,7 ms de processeur pour 1,2 ms de
            # carte. Une seule synchronisation (la liste des experts routés),
            # toutes les copies lancées d'abord, puis les GEMV.
            ids = topi.reshape(-1).tolist()          # unique synchronisation
            poids = topw.reshape(-1).to(x.dtype)
            for e in ids:
                exp = self.experts[e]
                for lin in (getattr(exp, "gate_proj", None), exp.up_proj, exp.down_proj):
                    if lin is not None:
                        lin.precharger()
            # Accumulateur en float32. Dix termes sommes en bf16 laissent
            # 5,4e-3 d'ecart relatif rien qu'en changeant leur ordre — mesure
            # du 8/09/2026, a poids et ponderations identiques ; en float32 le
            # meme changement d'ordre donne zero exact. C'est ce bruit-la qui
            # faisait diverger ce chemin de celui par masques, qui somme dans
            # l'ordre trie de `unique()`. Le cout est un tenseur [t, cache] par
            # couche, la ou chaque expert en produit deja un.
            out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
            for j, e in enumerate(ids):
                out += (self.experts[e](x) * poids[j]).to(torch.float32)
            out = out.to(x.dtype)
            if self.shared is not None:
                out = out + self._shared_out(x)
            return out

        # Meme accumulateur float32 que le chemin direct, et pour la meme
        # raison : sans lui les deux chemins ne rendent pas le meme vecteur.
        out = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
        # On regroupe les jetons par expert, pour que chaque expert fasse un
        # seul produit matriciel par lot au lieu d'un par jeton.
        flat_expert = topi.reshape(-1)
        flat_weight = topw.reshape(-1).to(x.dtype)   # topw est en fp32
        flat_token = torch.arange(t, device=x.device).repeat_interleave(self.top_k)
        for e in flat_expert.unique().tolist():
            sel = flat_expert == e
            tok = flat_token[sel]
            y = self.experts[e](x[tok])
            out.index_add_(0, tok, (y * flat_weight[sel].unsqueeze(-1)).to(torch.float32))
        out = out.to(x.dtype)
        if self.shared is not None:
            out = out + self._shared_out(x)
        return out

    def _shared_out(self, x: torch.Tensor) -> torch.Tensor:
        y = self.shared(x)
        if self.shared_gate is not None:
            y = y * torch.sigmoid(x @ self.shared_gate.t())
        return y

    def prefetch(self) -> None:
        # Seul l'expert partagé sert à chaque jeton et peut être préchargé.
        # Les experts routés ne se connaissent qu'après le routage : leur
        # QuantLinear, sur un pool, ignore de toute façon le préchargement.
        if self.shared is not None and not os.environ.get("ACVRAM_SANS_PRECHARGE"):
            for lin in self.shared.modules():
                if isinstance(lin, QuantLinear):
                    lin.prefetch()


# Jetons au-delà desquels le MoE repasse de la GEMV groupée (une paire
# (jeton, expert) par tranche de grille) à la boucle par expert : la boucle
# coûte ~0,6 ms par expert visité, la GEMV groupée relit les poids de
# l'expert pour chaque jeton — croisement mesuré vers quelques milliers.
_MOE_GROUPED_MAX = int(os.environ.get("ACVRAM_MOE_GROUPED_MAX", "32"))

# Jetons par expert au-delà desquels le prefill repasse de la GEMM groupée
# NVFP4 à la déquantification en bf16 suivie de torch._grouped_mm.
_MOE_GEMM_MAX = float(os.environ.get("ACVRAM_MOE_GEMM_MAX", "64"))

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
                 device: torch.device,
                 mlp_device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.index = index
        self.linear_attn = gdn
        self.mlp = mlp
        # L'index descend jusqu'au bloc d'experts : lui seul sait quels
        # experts il route, et la trace a besoin de savoir DE QUELLE COUCHE.
        # Pose ici, au seul endroit qui connaisse l'index, plutot qu'aux trois
        # sites de construction du bloc.
        if hasattr(mlp, "index_couche"):
            mlp.index_couche = index
        self.input_layernorm = input_norm
        self.post_attention_layernorm = post_norm
        self.device = device
        self.mlp_device = mlp_device or device   # experts en RAM hôte : autre appareil
        # spéculation : états photographiés après chaque jeton du lot
        # vérifié, pour revenir à celui du dernier jeton accepté
        self.static_hist: Optional[dict] = None
        self.static_hist_len = 0
        # tampons fixes : un créneau par séquence du lot (graphes b > 1)
        self.statics: list = []
        self.static_owners: list = []

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
            if etat is _STATIC:               # l'état vit dans un créneau fixe
                etat = self._reprendre(sid)
            y, etat = self.linear_attn(h[start:start + ql], etat)
            store[sid] = etat
            sorties.append(y)
            start += ql
        x = x + torch.cat(sorties).to(x.dtype)
        if self.mlp is None:
            return x
        return self._mlp(x)

    # -- chemin à formes fixes (graphes CUDA), une séquence ----------------
    static_bucket: int = 0

    @property
    def static(self) -> Optional[dict]:
        """Créneau 0 (une séquence : décodage simple et spéculation)."""
        return self.statics[0] if self.statics else None

    def _nouveau_static(self, max_len: int, dtype: torch.dtype) -> dict:
        la = self.linear_attn
        if hasattr(la, "rank"):               # MLA : cache latent borné
            return la.new_static(self.device, max_len, dtype)
        return la.new_static(self.device)

    def _reprendre(self, sid: int):
        """Sort l'état de ``sid`` de son créneau (retour au chemin eager)."""
        if sid not in self.static_owners:
            return None
        slot = self.static_owners.index(sid)
        self.static_owners[slot] = None
        return self.linear_attn.static_export(self.statics[slot])

    def static_bind(self, slot: int, sid: int, store: dict, max_len: int,
                    dtype: torch.dtype) -> None:
        """Amène l'état de ``sid`` dans le créneau ``slot`` de la couche.

        L'état du propriétaire précédent du créneau est exporté vers le
        magasin s'il y vit encore ; celui de ``sid`` est chargé depuis le
        magasin, depuis un autre créneau (le lot a changé d'ordre) ou remis
        à zéro. Le magasin note alors que l'état de ``sid`` réside dans les
        tampons.
        """
        la = self.linear_attn
        while len(self.statics) <= slot:
            self.statics.append(self._nouveau_static(max_len, dtype))
            self.static_owners.append(None)
        if self.static_owners[slot] == sid and store.get(sid) is _STATIC:
            return
        prev = self.static_owners[slot]
        if prev is not None and prev != sid and store.get(prev) is _STATIC:
            store[prev] = la.static_export(self.statics[slot])
        self.static_owners[slot] = None
        etat = store.get(sid)
        if etat is _STATIC:
            etat = self._reprendre(sid)
        la.static_load(self.statics[slot], etat)
        store[sid] = _STATIC
        self.static_owners[slot] = sid

    def decode_fixed(self, x: torch.Tensor, positions: torch.Tensor,
                     slots: torch.Tensor, block_tables: torch.Tensor,
                     seq_lens: torch.Tensor, max_pos: int,
                     cache, q_len: int = 1) -> torch.Tensor:
        h = self.input_layernorm(x)
        y = self._la_decode(h, q_len)
        x = x + y.to(x.dtype)
        if self.mlp is None:
            return x
        return self._mlp(x)

    def _la_decode(self, h: torch.Tensor, q_len: int) -> torch.Tensor:
        """Attention linéaire sur les tampons fixes ; ``q_len`` > 1 (lot de
        vérification spéculative) déroule les jetons un à un et photographie
        l'état après chacun dans ``static_hist``."""
        la = self.linear_attn
        if hasattr(la, "rank"):
            un = lambda t, st: la.decode_static(t, st, self.static_bucket)
        else:
            un = lambda t, st: la.decode_static(t, st)
        b = h.shape[0] // q_len
        if b > 1:                              # un créneau par séquence
            return torch.cat([un(h[i:i + 1], self.statics[i]) for i in range(b)], dim=0)
        if q_len == 1:
            return un(h, self.static)
        hist = self.ensure_hist(q_len)
        ys = []
        for j in range(q_len):
            ys.append(un(h[j:j + 1], self.static))
            for k, v in hist.items():
                v[j].copy_(self.static[k])
        return torch.cat(ys, dim=0)

    def ensure_hist(self, q_len: int) -> dict:
        """Historique alloué une fois pour toutes (les graphes capturés y
        écrivent) : le cache latent MLA en est exclu, sa longueur suffit."""
        if self.static_hist is None or self.static_hist_len < q_len:
            if os.environ.get("ACVRAM_TRACE_PTRS"):
                print(f"[hist] couche {self.index} : allocation de static_hist({q_len}), "
                      f"ancien={self.static_hist_len}, pendant une capture : "
                      f"{torch.cuda.is_current_stream_capturing()}", flush=True)
            self.static_hist = {
                k: torch.zeros((q_len,) + tuple(v.shape), dtype=v.dtype, device=v.device)
                for k, v in self.static.items() if k not in ("cache", "scores")}
            self.static_hist_len = q_len
        return self.static_hist

    def rollback(self, n_consumed: int) -> None:
        """Ramène l'état au ``n_consumed``-ième jeton du dernier lot vérifié."""
        for k, v in self.static_hist.items():
            self.static[k].copy_(v[n_consumed - 1])

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        h = self.post_attention_layernorm(x)
        if self.mlp_device != self.device:
            return x + self.mlp(h.to(self.mlp_device)).to(x.device, non_blocking=True)
        return x + self.mlp(h)

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
        # L'index descend jusqu'au bloc d'experts : lui seul sait quels
        # experts il route, et la trace a besoin de savoir DE QUELLE COUCHE.
        # Pose ici, au seul endroit qui connaisse l'index, plutot qu'aux trois
        # sites de construction du bloc.
        if hasattr(mlp, "index_couche"):
            mlp.index_couche = index
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
            h = self.input_layernorm(x)
            if self.mlp_device != self.device:
                y = self.mlp(h.to(self.mlp_device)).to(x.device, non_blocking=True)
            else:
                y = self.mlp(h)
            return x + (y if r == 1.0 else y * r)
        a = self.self_attn(self.input_layernorm(x), batch, cache)
        if self.mlp is None:                           # couche d'attention seule
            return x + (a if r == 1.0 else a * r)
        # somme résiduelle et normalisation en un lancement
        x, h = add_norm(x, a, self.post_attention_layernorm, r)
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
            h = self.input_layernorm(x)
            y = self.mlp(h.to(self.mlp_device)).to(x.device) if self.mlp_device != self.device \
                else self.mlp(h)
            return x + (y if r == 1.0 else y * r)
        a = self.self_attn.decode_fixed(self.input_layernorm(x), positions,
                                        slots, block_tables, seq_lens,
                                        max_pos, cache, q_len)
        if self.mlp is None:
            return x + (a if r == 1.0 else a * r)
        x, h = add_norm(x, a, self.post_attention_layernorm, r)
        y = self.mlp(h)
        return x + (y if r == 1.0 else y * r)

    def decode_fixed_res(self, x: torch.Tensor, delta: Optional[torch.Tensor],
                         positions: torch.Tensor, slots: torch.Tensor,
                         block_tables: torch.Tensor, seq_lens: torch.Tensor,
                         max_pos: int, cache: PagedKVCache, q_len: int = 1):
        """Pas de décodage à résidu différé : reçoit (x, delta) et rend
        (x, delta). La somme résiduelle de la couche précédente est absorbée
        par la première normalisation — un lancement de moins par couche."""
        r = self.residual_multiplier
        if delta is None:
            h = self.input_layernorm(x)
        else:
            x, h = add_norm(x, delta, self.input_layernorm, r)
        a = self.self_attn.decode_fixed(h, positions, slots, block_tables,
                                        seq_lens, max_pos, cache, q_len)
        x, h2 = add_norm(x, a, self.post_attention_layernorm, r)
        return x, self.mlp(h2)

    def prefetch(self) -> None:
        for m in self.modules():
            if isinstance(m, QuantLinear) and m.streamed is not None:
                m.prefetch()


class DecoderLayerParallel(DecoderLayerGDN):
    """Falcon-H1 : attention ET Mamba2 sur la même entrée normée, sommées."""

    def __init__(self, index: int, attn: "Attention", mamba: nn.Module,
                 mlp: nn.Module, input_norm: "RMSNorm", post_norm: "RMSNorm",
                 device: torch.device) -> None:
        super().__init__(index, mamba, mlp, input_norm, post_norm, device)
        self.self_attn = attn

    def forward(self, x: torch.Tensor, batch: ForwardBatch,
                cache=None) -> torch.Tensor:
        h = self.input_layernorm(x)
        a = self.self_attn(h, batch, cache)
        store = batch.gdn_store.setdefault(self.index, {}) \
            if batch.gdn_store is not None else {}
        sorties = []
        start = 0
        for i, ql in enumerate(batch.query_lens):
            sid = batch.seq_ids[i] if batch.seq_ids else i
            etat = store.get(sid)
            if etat is _STATIC:
                etat = self._reprendre(sid)
            y, etat = self.linear_attn(h[start:start + ql], etat)
            store[sid] = etat
            sorties.append(y)
            start += ql
        x = x + a + torch.cat(sorties).to(x.dtype)
        return self._mlp(x)

    def decode_fixed(self, x: torch.Tensor, positions: torch.Tensor,
                     slots: torch.Tensor, block_tables: torch.Tensor,
                     seq_lens: torch.Tensor, max_pos: int,
                     cache, q_len: int = 1) -> torch.Tensor:
        h = self.input_layernorm(x)
        a = self.self_attn.decode_fixed(h, positions, slots, block_tables,
                                        seq_lens, max_pos, cache, q_len)
        m = self._la_decode(h, q_len)
        x = x + a + m.to(x.dtype)
        return self._mlp(x)


class MoEBlockGemma(MoEBlock):
    """MoE de Gemma 4 (26B-A4B) : routeur sur x normalisé (RMS sans poids)
    × échelle × h^-½, softmax, top-k sans renormalisation, poids × échelle
    par expert ; experts GELU-tanh, reproduite par le chemin groupé."""

    def __init__(self, router: QuantLinear, experts: list, top_k: int,
                 router_scale: torch.Tensor, per_expert_scale: torch.Tensor,
                 eps: float) -> None:
        super().__init__(router, experts, top_k, None, norm_topk_prob=False)
        self.router_scale = router_scale.to(torch.float32)
        self.per_expert_scale = per_expert_scale.to(torch.float32).reshape(-1)
        self.eps = eps
        self.root = float(router_scale.numel()) ** -0.5

    def _route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x32 = x.to(torch.float32)
        xr = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        xr = xr * self.router_scale * self.root
        probs = F.softmax(self._router_logits(xr), dim=-1)
        topw, topi = torch.topk(probs, self.top_k, dim=-1)
        return topw * self.per_expert_scale[topi], topi


class DecoderLayerGemma(nn.Module):
    """Bloc Gemma 4 : normes avant ET après l'attention et le MLP, puis un
    scalaire de sortie par couche (layer_scalar)."""

    def __init__(self, index: int, attn: Attention, mlp: nn.Module,
                 input_norm: RMSNorm, post_attn_norm: RMSNorm,
                 pre_ffn_norm: RMSNorm, post_ffn_norm: RMSNorm,
                 out_scale: Optional[torch.Tensor], device: torch.device,
                 moe: Optional[nn.Module] = None,
                 post_ffn_norm_1: Optional[RMSNorm] = None,
                 post_ffn_norm_2: Optional[RMSNorm] = None,
                 pre_ffn_norm_2: Optional[RMSNorm] = None) -> None:
        super().__init__()
        self.index = index
        self.self_attn = attn
        self.mlp = mlp
        self.input_layernorm = input_norm
        self.post_attention_layernorm = post_attn_norm
        self.pre_feedforward_layernorm = pre_ffn_norm
        self.post_feedforward_layernorm = post_ffn_norm
        self.out_scale = out_scale
        # 26B-A4B : MoE en parallèle du MLP dense, chacun avec sa norme de
        # sortie, sommés avant post_feedforward_layernorm
        self.moe = moe
        self.post_feedforward_layernorm_1 = post_ffn_norm_1
        self.post_feedforward_layernorm_2 = post_ffn_norm_2
        self.pre_feedforward_layernorm_2 = pre_ffn_norm_2
        self.device = device
        self.mlp_device = device
        self.residual_multiplier = 1.0

    def _reste(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = x + self.post_attention_layernorm(a)
        y = self.mlp(self.pre_feedforward_layernorm(x))
        if self.moe is not None:
            y = self.post_feedforward_layernorm_1(y) + self.post_feedforward_layernorm_2(
                self.moe(self.pre_feedforward_layernorm_2(x)))
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
        # Tête de prédiction multi-jetons, quand le modèle en porte une : le
        # chargeur la pose ici. Le brouillon spéculatif a besoin de l'état
        # caché normalisé du dernier jeton ; on le recopie dans un tampon
        # statique pour que la capture du graphe de décodage l'emporte avec
        # elle — une affectation Python, elle, ne serait pas rejouée.
        self.mtp = None
        self._mtp_hidden: Optional[torch.Tensor] = None
        self._mtp_hidden_n: int = 0
        self._mtp_prefill: Optional[torch.Tensor] = None

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
            if _SYNC_COUCHES:
                # Diagnostic : une faute CUDA asynchrone remonte au premier
                # point de synchronisation, loin de son origine. Synchroniser
                # après chaque couche la fait remonter avec le bon index.
                try:
                    torch.cuda.synchronize(layer.device)
                except Exception as exc:
                    raise RuntimeError(f"faute CUDA après la couche {i} "
                                       f"({type(layer).__name__} sur {layer.device}) : {exc}") from exc

        brut = x
        x = self.norm(x.to(self.norm.weight.device))
        if return_hidden:
            return x
        if self.mtp is not None:
            brut = brut.to(x.device)     # la tête MTP lit l'état AVANT la norme finale
            if batch.is_prefill:
                # Le brouillon MTP a besoin du contexte entier pour amorcer son
                # propre cache : au prefill on garde tous les etats, pas
                # seulement celui du dernier jeton.
                self._mtp_prefill = brut.detach()
            self._garder_hidden(brut[(batch.last_token_indices() if logits_positions
                                      is None else logits_positions).to(brut.device)])
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
        # La tête de sortie est rembourrée à un multiple de 64 lignes pour ses
        # noyaux ; ces colonnes n'ont pas de jeton et ne doivent jamais gagner
        # l'argmax — sur muse-glimmer-30b (202 048 jetons, tête de 202 112)
        # l'une d'elles sortait vers le 250e jeton et faisait tomber le
        # plongement suivant sur un indice hors table.
        v = self.spec.vocab_size
        if v and logits.shape[-1] > v:
            logits = logits[..., :v]
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
        if self._res_differe():
            delta = None
            for i, layer in enumerate(self.layers):
                x, delta = layer.decode_fixed_res(
                    x, delta, positions, slots, block_tables, seq_lens,
                    max_pos, self.caches.get(i), q_len)
            x, h = add_norm(x, delta, self.norm,
                            self.layers[-1].residual_multiplier)
            if self.mtp is not None:
                self._garder_hidden(x)
            return self._logits_finaux(self.lm_head(h))
        for i, layer in enumerate(self.layers):
            x = layer.decode_fixed(x, positions, slots, block_tables,
                                   seq_lens, max_pos, self.caches.get(i), q_len)
        if self.mtp is not None:
            self._garder_hidden(x)
        x = self.norm(x)
        return self._logits_finaux(self.lm_head(x))

    # Lignes réservées d'avance pour l'état caché que lit la tête MTP : un lot
    # de vérification spéculative en pose k+1 par séquence.
    MTP_HIDDEN_LIGNES = 16

    def reserver_hidden(self, lignes: int, h: Optional[torch.Tensor] = None) -> None:
        """Réserve, hors de toute capture, le tampon où ``_garder_hidden``
        recopie l'état caché. Le tampon ne bouge plus ensuite.

        Il a été alloué à la volée, par ``clone()``, à la forme du pas courant :
        capturé dans un graphe de vérification (5 lignes), puis réalloué par le
        pas suivant (1 ligne), il laissait au graphe l'adresse d'un tenseur
        rendu au pool — le rejeu écrivait dans une page morte (``memcpy32_post``,
        Warp MMU Fault, reproduit sur Qwen3.8-27B le 5/09/2026), et le brouillon
        MTP lisait entre-temps un état périmé."""
        ref = h if h is not None else self._mtp_hidden
        if ref is None:
            return
        cap = max(lignes, self.MTP_HIDDEN_LIGNES)
        b = self._mtp_hidden
        if (b is not None and b.shape[0] >= cap and b.shape[1:] == ref.shape[1:]
                and b.dtype == ref.dtype and b.device == ref.device):
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("tampon MTP réservé pendant une capture de graphe : "
                               "appeler reserver_hidden() avant la capture")
        self._mtp_hidden = torch.zeros((cap,) + tuple(ref.shape[1:]),
                                       dtype=ref.dtype, device=ref.device)

    def _garder_hidden(self, h: torch.Tensor) -> None:
        """Recopie l'état caché normalisé dans les ``n`` premières lignes du
        tampon réservé. Une copie, et non une référence : sous graphe CUDA le
        tenseur source est réécrit à chaque rejeu, et une affectation Python ne
        serait jouée qu'à la capture. Le nombre de lignes valides est posé par
        le moteur (``_mtp_hidden_n``), lui aussi hors graphe."""
        h = h.detach()
        n = h.shape[0]
        b = self._mtp_hidden
        if b is None or b.shape[0] < n or b.shape[1:] != h.shape[1:] \
                or b.dtype != h.dtype or b.device != h.device:
            self.reserver_hidden(n, h)
            b = self._mtp_hidden
        b[:n].copy_(h)
        self._mtp_hidden_n = n

    def _res_differe(self) -> bool:
        """Le chemin à résidu différé n'est pris que si toutes les couches
        sont des blocs attention+MLP ordinaires — les hybrides et Gemma 4 ont
        leurs propres enchaînements de normes."""
        v = getattr(self, "_res_ok", None)
        if v is None:
            v = all(type(l) is DecoderLayer and l.self_attn is not None
                    and l.mlp is not None and l.mlp_device == l.device
                    and hasattr(l, "decode_fixed_res") for l in self.layers) \
                and type(self.norm).__name__ == "RMSNorm"
            self._res_ok = v
        return v

    @property
    def nbytes(self) -> int:
        total = self.embed_tokens.numel() * self.embed_tokens.element_size()
        for m in self.modules():
            if isinstance(m, QuantLinear):
                total += m.nbytes
        return total
