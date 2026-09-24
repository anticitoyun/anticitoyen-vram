"""Les couches du transformeur : `DecoderLayer` (dense et MoE),
`DecoderLayerGDN` (attention linéaire Gated DeltaNet et MLA), sa variante
parallèle et celle de Gemma 4, avec les témoins qu elles lisent à l import
(`_MLA_BATCH`, `_NORME_FUSEE`, `_STATIC`).

Déplacement PUR depuis `engine/model.py` (scission 22/09, module 5, après
`lot.py`, `deepstack.py`, `attention.py` et `moe.py`) : pas une ligne de
logique changée, les noms restent importables depuis `engine.model`, qui les
réexporte — `model.DecoderLayer is couches.DecoderLayer`.

Comme pour `moe.py`, un réexport ne conserve pas un `setattr` sur un scalaire :
`ACVRAM_MLA_BATCH` et `ACVRAM_NORME_FUSEE` se masquent désormais sur
`acvram.engine.couches` (cibles `lu_a` de `regime.VARIABLES` migrées dans le
même commit ; `tests/test_regime_cibles_lu_a.py` refuse une cible qui ne
définit pas la variable).
"""

from __future__ import annotations

from typing import Optional

import contextlib
import os
import torch
import torch.nn as nn

from .. import kernels
from ..memory.kvcache import PagedKVCache
from .attention import Attention
from .layers import QuantLinear, RMSNorm, add_norm
from .lot import ForwardBatch
from .moe import MoEBlock, MoEBlockGemma, _mla_glue

__all__ = [
    "_NORME_FUSEE", "_MLA_BATCH", "_STATIC", "_sid_fantome",
    "DecoderLayerGDN", "DecoderLayer", "DecoderLayerParallel", "DecoderLayerGemma",
]

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
_GDN_PREFILL_LOT = os.environ.get("ACVRAM_GDN_PREFILL_LOT", "0") == "1"
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
        # Préfill de plusieurs séquences (pièce 150 bis, opt-in) : projections
        # du lot en un appel au lieu d'une par séquence ; voir forward_lot.
        if (_GDN_PREFILL_LOT and hasattr(la, "forward_lot")
                and len(batch.query_lens) > 1
                and not all(ql == 1 for ql in batch.query_lens)):
            sids = [batch.seq_ids[i] if batch.seq_ids else i
                    for i in range(len(batch.query_lens))]
            etats = []
            for sid in sids:
                etat = store.get(sid)
                if etat is _STATIC:
                    etat = self._reprendre(sid)
                etats.append(etat)
            y, etats_new = la.forward_lot(h, etats, list(batch.query_lens))
            for sid, e in zip(sids, etats_new):
                store[sid] = e
            x = x + y.to(x.dtype)
            if self.mlp is None:
                return x
            return self._mlp(x)
        sorties = []
        start = 0
        # Pièce 172 (B') : les séquences de la boucle partagent le poids déquantifié de chaque linéaire (au bit :
        # kernels.depaquetage_partage) ; les GEMM restent une par séquence. Une seule séquence : rien à partager, et
        # garder les poids de la couche vivants ne ferait que monter le pic (chef, 25/09) — portée fermée.
        with (kernels.depaquetage_partage() if len(batch.query_lens) > 1 else contextlib.nullcontext()):
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
