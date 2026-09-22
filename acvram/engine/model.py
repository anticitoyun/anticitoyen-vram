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

import sys

import time

import torch

import torch.nn as nn

import torch.nn.functional as F

from .. import kernels

from ..memory import trace_routage as _trace_routage

from ..memory.kvcache import PagedKVCache, bucket_blocks

from ..memory import kv_lm4

from ..quant.calibrate import fwht_activations

from .config import ModelSpec

from .lot import ForwardBatch   # noqa: F401  (scission 21/09 : réexport, le lot vit dans engine/lot.py)

from .deepstack import ajouter_deepstack, disperser_images, niveaux_deepstack   # noqa: F401  (scission 21/09, module 2 : réexport)

from .attention import (Attention, MLP, MLP2, SEUIL_FUSION, _ROPE_KV,   # noqa: F401  (scission 21/09, module 3 : réexport)
                        _multi_projection, _multi_utilisable)

from .layers import (QuantLinear, RMSNorm, RotaryEmbedding, add_norm, apply_rope,
                     rope_fusee,
                     attention, batched_decode_attention,
                     decode_attention_fixed, repeat_kv)

__all__ = ["Attention", "MLP", "MoEBlock", "DecoderLayer", "ACVRamModel",
           "ForwardBatch"]

_SYNC_COUCHES = bool(os.environ.get("ACVRAM_SYNC_COUCHES"))

# ACVRAM_TRACE_COUCHES=1 : une ligne par couche, après synchronisation, avec
# l'horodatage — pour situer un pas qui ne rend jamais la main (70B en exil,
# 17/09 : 33 min sans une ligne, GPU 0 %, pile Python illisible sans ptrace).
# Lent (une synchronisation par couche) : diagnostic seulement.
_TRACE_COUCHES = bool(os.environ.get("ACVRAM_TRACE_COUCHES"))

def _trace_couche(quoi: str, i: int, layer) -> None:
    if not _TRACE_COUCHES:
        return
    if layer.device.type == "cuda":
        torch.cuda.synchronize(layer.device)
    exile = any(getattr(m, "streamed", None) is not None for m in layer.modules())
    print(f"[acvram] {quoi} couche {i:3d} {'flux' if exile else 'résidente'} "
          f"t={time.monotonic():.3f}", file=sys.stderr, flush=True)

# Scission 22/09, module 4 : le bloc MoE vit dans engine/moe.py — réexport
# intégral, mêmes objets. Un masquage de réglage vise acvram.engine.moe.
from .moe import (   # noqa: F401
    MoEBlock, _MOE_GROUPED_MAX, _MOE_GEMM_MAX,
    _MOE_MMA, _PREFILL_GROUPED, _PREFILL_A4,
    _PREFILL_W8R, _PREFILL_A8, _PREFILL_A8_FMT,
    _E2M1, _E2M1_MILIEUX, fausse_quant_a8,
    fausse_quant_nvfp4, _MOE_MMA_BT, _MOE_MMA_ETAGES,
    _MOE_MMA_KS, _MOE_DECODE_MMA, _MOE_DECODE_MMA_BT,
    _MOE_AWQ_TEMOIN, _QA_COMPTE, _QA_COMPTEURS,
    _qa_compteurs, _qa_imprime, _MOE_DECODE_MMA_MIN_T,
    _ROUTE_PREP, _MOE_DECODE_FUSED, _MOE_FUSED_TN,
    _MOE_FUSED_ATOMIQUE, _MOE_FUSED_ETAGES, _MOE_ROUTE_PACK,
    _MOE_GEMV, _GEMV_LAYOUT, _MOE_DECODE_MMA_MARLIN,
    _MARLIN_DISTINCT, _TRACE_ROUTAGE, _ROUTAGES,
    _ROUTAGE_TEMOIN, _TEMOINS_ROUTAGE, temoins_routage,
    _DOUBLE_DIAG, _mla_glue, _COLLE_MOE,
    _BORNES_EXPERTS, _comptes_tries, _colle_moe_triton,
    MoEBlockGemma, _DUMP_MOE,
)

# Poste F, fusion (3b) : norme d'entrée absorbée par le GEMV int8 q/k/v — RÉFUTÉ
# (poste3 6dbb1bb : exact, −47 lancements, mais +0,31 ms/pas : chaque bloc du
# GEMV recalcule la norme) ; témoin nommé, jamais défaut (acvram_kernels.cu)
_NORME_FUSEE = os.environ.get("ACVRAM_NORME_FUSEE", "0") == "1"

# Décodage MLA sous graphes : 0 = boucle par créneau ; 1 = le noyau
# d'attention batché seul (bead 6wa, bit-identique, −28,5 % sur le pas
# GLM-42B b=12) ; 2 = tout decode_static batché, projections comprises
# (numérique de forward_batch, à mesurer avant d'en faire le défaut).
# 2 = tout le chemin MLA du décodage batché sur le lot (projections en un
# GEMM M=b, normes, RoPE, cat, écriture du latent en un lancement) — défaut
# depuis poste7-duel-verdict-16-09 § 6.2 (15 534 lancements/pas à b=12 en
# séquence par séquence) ; 1 = noyau d'attention seul batché ; 0 = boucle.
_MLA_BATCH = int(os.environ.get("ACVRAM_MLA_BATCH", "2"))

# Marque, dans le magasin d'états, une séquence dont l'état réside dans les
# tampons fixes d'une couche (chemin graphes) plutôt qu'en tuple fonctionnel.
_STATIC = object()

def _sid_fantome(sid) -> bool:
    """Vrai pour une sentinelle de rembourrage (`graphs._bind_hybrid` : -1,
    -2, ...). Les identités réelles viennent de `runner.Sequence.id`, un
    compteur à partir de 0 : aucune n'est négative."""
    return isinstance(sid, int) and sid < 0

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
        la = self.linear_attn
        # Lot hors créneau, décodage pur : les projections/RoPE/normes ne
        # dépendent que de x, un seul appel pour les b séquences au lieu de
        # b — seul ext.mla_decode a encore besoin du cache, par séquence.
        # Conservateur : au moindre doute (créneau actif, prefill mélangé,
        # noyau absent), la boucle inchangée ci-dessous.
        # GDN (17/09) : même contrat, `forward_batch` = un lancement fla pour
        # les b séquences (poste7-priorite-apres-campagne-17-09 § 2)
        if (hasattr(la, "forward_batch") and batch.gdn_store is not None
                and all(ql == 1 for ql in batch.query_lens)
                and la.peut_batcher_decode(h)):
            sids = [batch.seq_ids[i] if batch.seq_ids else i
                   for i in range(len(batch.query_lens))]
            etats = [store.get(sid) for sid in sids]
            if not any(e is _STATIC for e in etats):
                y, etats_new = la.forward_batch(h, etats)
                for sid, e in zip(sids, etats_new):
                    store[sid] = e
                x = x + y.to(x.dtype)
                if self.mlp is None:
                    return x
                return self._mlp(x)
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
        # Créneau de rembourrage (godet > lot réel, `graphs._bind_hybrid`) :
        # sid négatif, hors de tout lot. Son état n'appartient à personne : il
        # part à ZÉRO à chaque liaison, avance avec les entrées factices du
        # godet tant que le créneau reste du rembourrage (jamais lu, jamais
        # exporté), et n'écrit RIEN dans le magasin. Avant le 19/09, la
        # sentinelle passait par le chemin ordinaire : à l'éviction son état
        # (avancé par les rejeux) était exporté sous `store[-1]`, une clé
        # absente de tout lot, puis RECHARGÉ dans le créneau de rembourrage
        # suivant — neutre au premier cycle seulement (chantier-c4-19-09).
        fantome = _sid_fantome(sid)
        if self.static_owners[slot] == sid and (fantome or store.get(sid) is _STATIC):
            return
        prev = self.static_owners[slot]
        if (prev is not None and prev != sid and not _sid_fantome(prev)
                and store.get(prev) is _STATIC):
            store[prev] = la.static_export(self.statics[slot])
        self.static_owners[slot] = None
        etat = None if fantome else store.get(sid)
        if etat is _STATIC:
            etat = self._reprendre(sid)
        la.static_load(self.statics[slot], etat)
        if not fantome:
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

    def decode_fixed_res(self, x: torch.Tensor, delta: Optional[torch.Tensor],
                         positions: torch.Tensor, slots: torch.Tensor,
                         block_tables: torch.Tensor, seq_lens: torch.Tensor,
                         max_pos: int, cache, q_len: int = 1):
        """C15 : ``decode_fixed`` à résidu différé pour les couches MLA (le
        même contrat que ``DecoderLayer.decode_fixed_res``) : la somme
        résiduelle de la couche précédente est absorbée par la première norme
        (``add_norm`` = rmsnorm_bf16 avec résidu, acvram_kernels.cu
        ``rmsnorm_bf16_kernel`` : ``bf16(fp32(res) + fp32(y))`` — l'addition
        bf16 de torch, au bit), celle de l'attention par la seconde : deux
        lancements de moins par couche. Pris par ``ACVRamModel._res_differe``
        sous ACVRAM_MLA_GLUE ≥ 1 seulement."""
        if delta is None:
            h = self.input_layernorm(x)
        else:
            x, h = add_norm(x, delta, self.input_layernorm)
        y = self._la_decode(h, q_len).to(x.dtype)
        if self.mlp is None:
            return x, y
        x, h2 = add_norm(x, y, self.post_attention_layernorm)
        return x, self.mlp(h2)

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
            if q_len == 1 and hasattr(la, "decode_static_batch") and not hasattr(la, "rank") \
                    and la.peut_batcher_decode(h):
                return la.decode_static_batch(h, self.statics[:b])      # GDN : un lancement fla
            if q_len == 1 and hasattr(la, "rank") and _MLA_BATCH:
                ext = kernels.get_extension()
                if ext is not None and hasattr(ext, "mla_decode_batch"):
                    ptrs, scores, len_ptrs = self._mla_lot(b)
                    if _MLA_BATCH >= 2:
                        return la.decode_static_batch_complet(h, self.statics[:b], self.static_bucket,
                                                              ptrs, scores, len_ptrs)
                    return la.decode_static_batch(h, self.statics[:b], self.static_bucket, ptrs, scores)
            return torch.cat([un(h[i:i + 1], self.statics[i]) for i in range(b)], dim=0)
        if q_len == 1:
            if hasattr(la, "rank") and _mla_glue() >= 2:
                # C15, niveau 2 : à b=1 aussi, la préparation en un noyau
                # (mla_prep_batch) et l'écriture du latent en un lancement —
                # la numérique du lot (servie à b=12), pas celle de la boucle
                ext = kernels.get_extension()
                if ext is not None and hasattr(ext, "mla_decode_batch"):
                    ptrs, scores, len_ptrs = self._mla_lot(1)
                    if os.environ.get("ACVRAM_MLA_ECRIT_TORCH") == "1":
                        len_ptrs = None                # sonde niveau 2 : écriture du latent en torch (_ecrit_ligne)
                    return la.decode_static_batch_complet(h, self.statics[:1], self.static_bucket,
                                                          ptrs, scores, len_ptrs)
            return un(h, self.static)
        hist = self.ensure_hist(q_len)
        ys = []
        for j in range(q_len):
            ys.append(un(h[j:j + 1], self.static))
            for k, v in hist.items():
                v[j].copy_(self.static[k])
        return torch.cat(ys, dim=0)

    def _mla_lot(self, b: int):
        """Table d'adresses des caches des ``b`` premiers créneaux et tampon de
        scores partagé, créés une fois par jeu d'adresses (bead 6wa, spec §3-A).
        Les caches par créneau ne sont jamais réalloués : la clé est stable, et
        la création tombe dans l'échauffement eager qui précède toute capture
        de graphe (graphs.py, _capture) — jamais dans la capture elle-même."""
        cle = tuple((self.statics[i]["cache"].data_ptr(), self.statics[i]["len"].data_ptr()) for i in range(b))
        cache = self.__dict__.setdefault("_mla_lots", {})
        entree = cache.get(cle)
        if entree is None:
            from .mla import _refuser_en_capture
            _refuser_en_capture("tables d'adresses du lot MLA (_mla_lot)")
            dev = self.statics[0]["cache"].device
            ptrs = torch.tensor([c for c, _ in cle], dtype=torch.int64, device=dev)
            # les longueurs aussi : un tenseur 0-d par créneau, adresse stable
            # (mla_ecrit_latent les avance sur la carte, un lancement pour b)
            len_ptrs = torch.tensor([self.statics[i]["len"].data_ptr() for i in range(b)],
                                    dtype=torch.int64, device=dev)
            sc0 = self.statics[0]["scores"]
            scores = torch.zeros(b, sc0.shape[0], sc0.shape[1], dtype=torch.float32, device=dev)
            entree = cache[cle] = (ptrs, scores, len_ptrs)
        return entree

    def ensure_hist(self, q_len: int) -> dict:
        """Historique alloué une fois pour toutes (les graphes capturés y
        écrivent) : le cache latent MLA en est exclu, sa longueur suffit."""
        if self.static_hist is None or self.static_hist_len < q_len:
            if os.environ.get("ACVRAM_TRACE_PTRS"):
                print(f"[hist] couche {self.index} : allocation de static_hist({q_len}), "
                      f"ancien={self.static_hist_len}, pendant une capture : "
                      f"{torch.cuda.is_current_stream_capturing()}", flush=True)
            # `static["lot"]` (lot_etats.nouveau_static : (numéro de lot, rang)) est un
            # tuple, pas un tenseur : le décodage spéculatif sur un hybride GDN mourait ici
            # (`AttributeError: 'tuple' object has no attribute 'shape'`, verdict-mtp-exact-
            # 19-09) — seuls les tenseurs d'état ont un historique
            self.static_hist = {
                k: torch.zeros((q_len,) + tuple(v.shape), dtype=v.dtype, device=v.device)
                for k, v in self.static.items() if k not in ("cache", "scores") and isinstance(v, torch.Tensor)}
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

    def forward_res(self, x: torch.Tensor, delta: Optional[torch.Tensor],
                    batch: ForwardBatch, cache: Optional[PagedKVCache]):
        """C15-prefill : `forward` à résidu différé — reçoit (x, delta) et rend
        (x, y) ; la somme ``x + delta`` de la couche précédente est absorbée par
        la première normalisation (`add_norm`, un lancement de moins et 8,4 Mo
        de moins relus par couche à L = 2 047), comme `decode_fixed_res` au
        décodage. Même arithmétique que `forward` : bf16(fp32(x) + fp32(delta))
        puis la norme (rmsnorm_bf16_kernel avec résidu, mult = 1 — réservé au
        multiplicateur 1,0 : y·r puis + arrondit deux fois, le noyau une)."""
        assert self.residual_multiplier == 1.0
        if delta is None:
            h = self.input_layernorm(x)
        else:
            x, h = add_norm(x, delta, self.input_layernorm, 1.0)
        a = self.self_attn(h, batch, cache)
        x, h2 = add_norm(x, a, self.post_attention_layernorm, 1.0)
        return x, self.mlp(h2)

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
        # Créneaux fantômes du remplissage godet (bucket_batch, graphs.py) :
        # `slots` porte déjà la sentinelle -1 posée par `_fill` (bead pds,
        # 14/09) — la même qui protège le cache KV. Dense (MLP simple) n'a
        # pas de routage dépendant des données, `valid` n'y sert à rien.
        y = (self.mlp(h, valid=slots >= 0) if isinstance(self.mlp, MoEBlock)
             else self.mlp(h))
        return x + (y if r == 1.0 else y * r)

    def decode_fixed_res(self, x: torch.Tensor, delta: Optional[torch.Tensor],
                         positions: torch.Tensor, slots: torch.Tensor,
                         block_tables: torch.Tensor, seq_lens: torch.Tensor,
                         max_pos: int, cache: PagedKVCache, q_len: int = 1,
                         valid: Optional[torch.Tensor] = None):
        """Pas de décodage à résidu différé : reçoit (x, delta) et rend
        (x, delta). La somme résiduelle de la couche précédente est absorbée
        par la première normalisation — un lancement de moins par couche.
        ``valid`` (= ``slots >= 0``, C15 niveau 3) est calculé une fois par pas
        par le modèle au lieu d'une fois par couche."""
        r = self.residual_multiplier
        a = None
        if (_NORME_FUSEE and delta is not None and x.is_cuda and x.dtype == torch.bfloat16
                and type(self.input_layernorm).__name__ == "RMSNorm"
                and self.self_attn.norme_fusee_possible(x.shape[0])):
            # Poste F (3b) : la norme d'entrée dans le GEMV q/k/v — un
            # lancement de moins par couche, x_out écrit par le noyau
            res = self.self_attn.decode_fixed_norme(x, delta, self.input_layernorm, r, positions,
                                                    slots, block_tables, seq_lens, max_pos, cache, q_len)
            if res is not None:
                x, a = res
        if a is None:
            if delta is None:
                h = self.input_layernorm(x)
            else:
                x, h = add_norm(x, delta, self.input_layernorm, r)
            a = self.self_attn.decode_fixed(h, positions, slots, block_tables,
                                            seq_lens, max_pos, cache, q_len)
        x, h2 = add_norm(x, a, self.post_attention_layernorm, r)
        # Créneaux fantômes : même garde que decode_fixed ci-dessus.
        if isinstance(self.mlp, MoEBlock):
            y = self.mlp(h2, valid=(slots >= 0) if valid is None else valid)
        else:
            y = self.mlp(h2)
        return x, y

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
        if _trace_routage.actif():
            # Pièce 27(a) : masque de modalité de CETTE passe (image/texte/spécial),
            # aligné sur la séquence réellement fournie ; lu par `noter`.
            _trace_routage.poser_modalites(_trace_routage.modalites_du_lot(
                batch.tokens, getattr(self.spec, "image_token_id", None),
                [p for i in range(len(batch.seq_lens)) for p in (batch.images_de(i) or [])]))
        idx = batch.tokens.to(self.embed_tokens.device)
        x = F.embedding(idx, self.embed_tokens).to(self.dtype)
        if self.spec.embedding_multiplier != 1.0:
            x = x * self.spec.embedding_multiplier
        if batch.images is not None:
            # Après le multiplicateur : HF Gemma disperse les traits d'image
            # dans des embeddings déjà mis à l'échelle (masked_scatter), les
            # traits eux-mêmes ne sont pas multipliés.
            x = disperser_images(x, batch)

        current = None
        # C15-prefill : résidu différé au préfill eager — (x, delta) d'une couche
        # à la suivante, la somme faite par add_norm de la couche suivante (et
        # par celle de la norme finale) ; réservé aux blocs ordinaires à
        # multiplicateur 1,0, sinon `forward` (le chemin d'avant, au bit).
        differe = kernels.prefill_compact("residu") and all(
            type(l) is DecoderLayer and l.self_attn is not None and l.mlp is not None
            and l.mlp_device == l.device and l.residual_multiplier == 1.0 for l in self.layers)
        delta = None
        # Deepstack (Qwen3-VL) : n niveaux à ajouter après les couches 0..n−1
        # aux lignes image ; 0 sans image, un entier comparé par couche.
        n_deep = niveaux_deepstack(batch) if batch.deepstack is not None else 0
        for i, layer in enumerate(self.layers):
            if layer.device != current:
                if delta is not None:
                    x, delta = x + delta, None
                x = x.to(layer.device, non_blocking=True)
                current = layer.device
            # On lance le transfert de la couche suivante avant d'exécuter
            # celle-ci, pour que la copie PCIe d'une couche résidant en mémoire
            # vive se cache derrière du vrai travail.
            if i + 1 < len(self.layers):
                self.layers[i + 1].prefetch()
            if differe:
                x, delta = layer.forward_res(x, delta, batch, self.caches.get(i))
            else:
                x = layer(x, batch, self.caches.get(i))
            if i < n_deep:
                x, delta = ajouter_deepstack(x, delta, batch, i)
            _trace_couche("forward", i, layer)
            if _SYNC_COUCHES:
                # Diagnostic : une faute CUDA asynchrone remonte au premier
                # point de synchronisation, loin de son origine. Synchroniser
                # après chaque couche la fait remonter avec le bon index.
                try:
                    torch.cuda.synchronize(layer.device)
                except Exception as exc:
                    raise RuntimeError(f"faute CUDA après la couche {i} "
                                       f"({type(layer).__name__} sur {layer.device}) : {exc}") from exc

        if delta is None:
            brut = x
            x = self.norm(x.to(self.norm.weight.device))
        elif x.device == self.norm.weight.device:
            # résidu différé : la dernière somme dans la norme finale
            brut, x = add_norm(x, delta, self.norm, 1.0)
        else:
            brut = x + delta
            x = self.norm(brut.to(self.norm.weight.device))
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
        # LES LOGITS SE PRODUISENT EN FP32, ET C'EST L'ENTREE QU'ON CONVERTIT.
        # Le dtype de sortie des noyaux suit celui de x — nvfp4_gemv fait
        # `out = torch::empty({N, M}, xc.options())` — donc passer x en float
        # bascule le produit sur le chemin float de bout en bout. Convertir la
        # SORTIE ne restaurerait rien : la perte est dans l'accumulation, pas
        # dans un arrondi final.
        #
        # Ce que l'arrondi bf16 coutait, mesure sur quatre pas de GLM-4.7 :
        # logits centres sur 95,6 avec une amplitude de 23, donc dans
        # l'intervalle [64, 128) ou le pas bf16 vaut 0,5 — QUARANTE-CINQ
        # niveaux distincts pour 151 936 jetons, et une marge top1-top2 de un
        # a deux ULP.
        #
        # Et l'erreur n'est pas seulement du bruit. logsumexp est convexe,
        # donc par Jensen l'arrondi introduit un BIAIS de 1/2 sigma^2
        # (1 - somme p^2) qui NE DECROIT PAS avec la longueur du corpus :
        # verifie par simulation, 0,009875 nat mesure contre 0,010410 predit,
        # soit +1,05 % sur la perplexite. Le bruit, lui, decroit en 1/racine(N)
        # et vaut 0,09 % sur nos etalons de 148 920 jetons.
        #
        # Le biais depend du PAS, donc de la magnitude des logits, donc du
        # modele : 1,05 % sur GLM-4.7 contre 0,001 % sur un modele centre sur
        # zero, un rapport de 1024. C'est donc un biais SYSTEMATIQUE ENTRE
        # MODELES, qui fausse exactement les comparaisons de perplexite que
        # nous faisons. En fp32 pres de 116 le pas tombe a 7,6e-06 et le biais
        # a 2,4e-12 nat : la question ne se pose plus.
        #
        # Le cout est nul : l'etat cache fait quelques milliers d'elements, et
        # le vecteur de sortie 151 936 flottants, soit 594 Kio par jeton.
        # ACVRAM_LOGITS_BF16=1 retablit l'ancien comportement : il rend le
        # correctif MESURABLE par A/B sans recompiler, et sert de repli si le
        # cout en temps s'averait sensible. Un correctif qu'on ne peut pas
        # comparer a son absence n'est pas evaluable.
        logits = self._tete(x)
        if logits.dtype != torch.float32:
            # Un noyau qui rend autre chose que ce qu'on lui a donne annule le
            # correctif en silence. On le dit une fois plutot que de le taire.
            if not getattr(self, "_dit_logits_dtype", False):
                self._dit_logits_dtype = True
                print(f"[acvram] les logits sortent en {logits.dtype} malgre "
                      "une entree fp32 : le noyau de la tete impose son type, "
                      "et le gain de resolution n'est pas acquis.", flush=True)
        return self._logits_finaux(logits)

    def _tete(self, x: torch.Tensor) -> torch.Tensor:
        """L'entrée de ``lm_head`` en fp32 sur l'appareil de la tête — LE MÊME
        texte pour le chemin eager et le chemin à formes fixes (graphes,
        ACVRAM_GRAPHS_EAGER). Le fixe gardait la tête en bf16 (a5fac1c n'avait
        porté le fp32 qu'en eager) : l'argmax basculait dès le 2e pas de
        décodage, 64-68 % de jetons justes à l'arbitre prefill = décodage
        (poste7-duel-verdict § 13, REGLES §7). ACVRAM_LOGITS_BF16=1 : témoin."""
        head_dev = getattr(self.lm_head.qweight, "qweight", None)
        target = head_dev.device if head_dev is not None else x.device
        if os.environ.get("ACVRAM_LOGITS_BF16") == "1":
            return self.lm_head(x.to(target))
        # Tete INT8 au decodage (poste7-duel-verdict par. 14 (ii)) : x reste en
        # bf16, le GEMV accumule et SORT en fp32 — memes produits, meme ordre
        # de sommes que x.to(float32) : logits egaux au bit, sans la conversion
        # de h ni le double trafic de x en fp32 (0,85 -> ~0,2 ms attendu).
        # ACVRAM_TETE_FP32_ENTREE=1 : temoin (l ancienne conversion).
        w = self.lm_head.qweight
        lin = self.lm_head
        if (x.dtype == torch.bfloat16 and getattr(w, "format", "") == "int8"
                and head_dev is not None and head_dev.is_cuda
                and getattr(lin, "scaler", None) is None and getattr(lin, "streamed", None) is None
                and getattr(lin, "bias", None) is None
                and kernels.tete_int8_entree_bf16(x.shape[0])
                and os.environ.get("ACVRAM_TETE_FP32_ENTREE") != "1"):
            return kernels.int8_matmul(x.to(target), w, sortie_fp32=True)
        # Tête NVFP4 à b ≥ DENSE_NVFP4_MIN_M (poste7-gemm-dense-palier2-non-
        # ouvert-17-09 § 2) : la même GEMM dense étroite que les projections
        # (poids lus une fois par pas), logits accumulés en fp32 ; la GEMV fp32
        # relisait 0,6 Go par séquence — 18 % du pas Qwen3.8 b=12 (poste3 0690bd4).
        if (x.dtype == torch.bfloat16 and getattr(w, "format", "") == "nvfp4"
                and head_dev is not None and head_dev.is_cuda
                and getattr(lin, "streamed", None) is None and getattr(lin, "bias", None) is None
                and kernels._DENSE_NVFP4 == "triton" and kernels._DENSE_NVFP4_MIN_M <= x.shape[0] <= 32
                and os.environ.get("ACVRAM_TETE_FP32_ENTREE") != "1"):
            from ..kernels import gemm_dense_etroit as _gde
            if _gde.disponible():
                xt = x.to(target)
                sc = getattr(lin, "scaler", None)
                if sc is not None and not sc.is_identity:
                    xt = sc.apply(xt)
                return _gde.gemm_dense_etroit(xt, w, sortie_fp32=True)[:, : w.shape[0]]
        return self.lm_head(x.to(target, dtype=torch.float32))

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
            # C15 niveau 3 : `slots >= 0` (fantômes du godet) une fois par pas
            # au lieu d'une fois par couche (48 nœuds → 1) ; témoin : None,
            # chaque couche le recalcule.
            # (sans couche MoE — jouet dense ou MLA seul — ce `ge` serait un nœud de plus,
            # pas de moins : test_mla_glue_c15 le comptait, 20/09)
            if "_a_des_moe" not in self.__dict__:
                self.__dict__["_a_des_moe"] = any(isinstance(l.mlp, MoEBlock) for l in self.layers)
            valid = (slots >= 0) if kernels.glue_compact("kv") and self.__dict__["_a_des_moe"] else None
            for i, layer in enumerate(self.layers):
                if valid is not None and type(layer) is DecoderLayer:
                    x, delta = layer.decode_fixed_res(
                        x, delta, positions, slots, block_tables, seq_lens,
                        max_pos, self.caches.get(i), q_len, valid=valid)
                    continue
                x, delta = layer.decode_fixed_res(
                    x, delta, positions, slots, block_tables, seq_lens,
                    max_pos, self.caches.get(i), q_len)
            x, h = add_norm(x, delta, self.norm,
                            getattr(self.layers[-1], "residual_multiplier", 1.0))
            if self.mtp is not None:
                self._garder_hidden(x)
            return self._logits_finaux(self._tete(h))
        for i, layer in enumerate(self.layers):
            x = layer.decode_fixed(x, positions, slots, block_tables,
                                   seq_lens, max_pos, self.caches.get(i), q_len)
            _trace_couche("décodage", i, layer)
        if self.mtp is not None:
            self._garder_hidden(x)
        x = self.norm(x)
        return self._logits_finaux(self._tete(x))

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
            def ordinaire(l) -> bool:
                return (type(l) is DecoderLayer and l.self_attn is not None
                        and l.mlp is not None and l.mlp_device == l.device
                        and hasattr(l, "decode_fixed_res"))

            def mla_glue(l) -> bool:
                # C15 : couche MLA (GLM, DeepSeek) sous ACVRAM_MLA_GLUE ≥ 1 —
                # jamais GDN/Mamba ni les blocs parallèles, qui gardent leurs
                # enchaînements de normes
                return (type(l) is DecoderLayerGDN and hasattr(l.linear_attn, "rank")
                        and l.mlp is not None and l.mlp_device == l.device)
            glue = _mla_glue() >= 1
            v = all(ordinaire(l) or (glue and mla_glue(l)) for l in self.layers) \
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

    def usage_routage(self) -> dict:
        """Histogramme de routage par couche — À LA DEMANDE seulement.

        `MoEBlock._compter_routage` tourne à chaque pas ; ceci ne l'est PAS :
        c'est le point de lecture (dump fin de requête, endpoint /routage),
        jamais appelé depuis le chemin de décodage. Rend `{index_couche:
        tenseur}` pour chaque couche MoE qui a déjà tourné au moins une fois ;
        les couches denses ou pas encore sollicitées n'y figurent pas. Les
        tenseurs restent sur leur device — `.cpu()` (et donc la synchronisation
        qu'il implique) est décidé par l'appelant, pas ici."""
        return {m.index_couche: m._usage_routage for m in self.modules()
                if isinstance(m, MoEBlock) and m._usage_routage is not None}

    def nbytes_detail(self) -> dict:
        """Decompose `nbytes` pour que l'ecart a la prevision se NOMME.

        Le 10/09/2026, la prevision statique tiree des manifestes (embedding au
        dtype de chargement + somme des `QuantLinear.nbytes`) a donne 15,9994
        bits/poids pour Llama-2-7b-fp16pur quand le releve d'evaluation en
        portait 26,987 — un facteur 1,687. Meme forme pour les deux dossiers
        quantifies, mais a 10,3 % seulement (8,3391 prevu contre 9,198 releve ;
        4,7235 contre 5,215). Un ecart qui n'est pas proportionnel aux octets
        stockes : ce n'est donc pas une erreur de densite.

        Cette decomposition teste l'hypothese de tete : un meme objet de poids
        compte plusieurs fois, parce que `modules()` le rencontre sous
        plusieurs `QuantLinear` (projections fusionnees exposees aussi comme
        tranches, tete liee a l'embedding). `vus_plusieurs_fois` rend le compte
        exact des octets comptes en double ; s'il est nul, l'hypothese tombe et
        `par_classe` dit ou les octets sont reellement.
        """
        vus: dict[int, dict] = {}
        par_classe: dict[str, int] = {}
        n_ql = 0
        for nom, m in self.named_modules():
            if not isinstance(m, QuantLinear):
                continue
            n_ql += 1
            q = m.qweight
            o = getattr(q, "nbytes", 0)
            fmt = getattr(q, "format", type(q).__name__)
            par_classe[fmt] = par_classe.get(fmt, 0) + o
            e = vus.setdefault(id(q), {"octets": o, "fmt": fmt, "noms": []})
            e["noms"].append(nom)
        double = sum(e["octets"] * (len(e["noms"]) - 1) for e in vus.values())
        # Le compte par IDENTITE D'OBJET a rendu 0 a la premiere mesure
        # (230 QuantLinear, 230 objets distincts) : l'hypothese « un meme objet
        # rencontre plusieurs fois » est morte. Il reste 461 455 360 octets
        # au-dessus de la somme des tenseurs du DISQUE (7 223 386 112 mesures
        # contre 6 761 930 752 attendus pour l'int8 de Llama-2-7B), et le
        # detecteur ne pouvait pas les voir : deux objets DISTINCTS portant des
        # octets equivalents — une projection fusionnee materialisee a cote de
        # ses tranches — s'accordent avec « 0 objet vu deux fois ».
        #
        # D'ou ce second compte, par NOM et par forme : il attribue les octets
        # a des tenseurs nommes, donc il peut nommer les 461 Mo au lieu de les
        # constater. Note au passage : 230 QuantLinear pour 225 tenseurs
        # stockes au manifeste — cinq de plus, et 461 455 360 / 5 = 92 291 072,
        # soit exactement la taille d'un gate_up fusionne en int8. C'est une
        # PISTE, pas une explication : elle attend ce releve pour etre nommee.
        # OCTETS REELLEMENT ALLOUES, par stockage unique. `nbytes` somme des
        # `numel()` : une VUE y compte ses elements comme si elle possedait ses
        # octets. Or les quatre empileurs remplacent les originaux par des
        # tranches de la pile — c'est le but — donc `nbytes` compte deux fois
        # tout ce qui est fusionne.
        #
        # C'est ce qui explique ENTIEREMENT l'ecart que je cherchais depuis ce
        # matin. Llama-2-7b-fp16pur, tout PlainTensor en bf16 :
        #
        #   base (embed + 257 denses)          13 476 302 848
        #   vues q, k, v                        3 221 225 472
        #   vues gate, up                       5 771 362 304
        #   embed compte une seconde fois         262 144 000
        #   PREVU                              22 731 034 624
        #   MESURE                             22 731 030 528   ecart 4 096 o
        #
        # Les 4 096 octets restants sont les 2 048 parametres que le manifeste
        # compte en trop (6 738 417 664 contre 6 738 415 616 reels). Il n'y a
        # donc plus rien d'inexplique dans le 26,987 bits/poids : ce n'etait ni
        # un cache, ni un tampon, ni un doublon accidentel — c'etait l'unite de
        # mesure.
        vus_stockage: dict[int, int] = {}
        for nom, m in self.named_modules():
            if not isinstance(m, QuantLinear):
                continue
            q = m.qweight
            for champ in ("qweight", "weight", "scales", "zeros",
                          "block_scale", "global_scale_rows"):
                t = getattr(q, champ, None)
                if t is None or not hasattr(t, "untyped_storage"):
                    continue
                st = t.untyped_storage()
                vus_stockage[st.data_ptr()] = st.nbytes()
        st_emb = self.embed_tokens.untyped_storage()
        vus_stockage[st_emb.data_ptr()] = st_emb.nbytes()
        octets_stockage = sum(vus_stockage.values())

        par_forme: dict[str, int] = {}
        for e in vus.values():
            for nom in e["noms"]:
                par_forme[nom] = e["octets"]
        gros = sorted(par_forme.items(), key=lambda kv: -kv[1])[:12]
        emb = self.embed_tokens.numel() * self.embed_tokens.element_size()
        return {
            "total": self.nbytes,
            "embed_tokens": emb,
            "embed_dtype": str(self.embed_tokens.dtype),
            "quantlinear": n_ql,
            "objets_distincts": len(vus),
            "octets_comptes_en_double": double,
            "total_sans_doubles": emb + sum(e["octets"] for e in vus.values()),
            "par_format": par_classe,
            "vus_plusieurs_fois": [
                {"octets": e["octets"], "fmt": e["fmt"], "noms": e["noms"]}
                for e in vus.values() if len(e["noms"]) > 1],
            "octets_stockage_uniques": octets_stockage,
            "stockages_distincts": len(vus_stockage),
            "octets_dupliques_par_les_vues": self.nbytes - octets_stockage,
            "douze_plus_gros": [{"nom": n, "octets": o} for n, o in gros],
            "octets_par_nom_total": sum(par_forme.values()),
        }
