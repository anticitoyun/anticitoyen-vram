"""Cache clés-valeurs paginé et quantifié.

La pagination, reprise de vLLM : le cache est une réserve de blocs de taille
fixe et chaque séquence détient une liste d'indices de blocs. Rien n'est
contigu, si bien qu'une séquence peut croître sans réserver d'avance son
contexte du pire cas, et qu'une séquence terminée rend ses blocs
immédiatement. Sur une carte de 32 Go servant plusieurs conversations à la
fois, c'est la différence entre quatre séquences simultanées et quarante.

La quantification par-dessus : les clés et les valeurs sont stockées sur 8 bits
avec une échelle fp16 par (bloc, position, tête), soit deux fois moins que le
fp16 pour environ 0,4 % de surcoût. Les deux GPU reçoivent un stockage sur
8 bits, mais pas les mêmes 8 bits :

    RTX 5090    FP8 E4M3 — plage dynamique plus large, pas de point zéro
    RTX 3080 Ti INT8     — Ampere n'a pas de FP8, donc entier symétrique

Le cache est propre à chaque couple (couche, appareil) : une couche s'exécutant
sur cuda:1 garde ses blocs sur cuda:1, si bien que l'attention ne traverse
jamais le bus PCIe.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from . import kv_lm4

__all__ = ["KVCacheConfig", "PagedKVCache", "BlockAllocator"]

BLOCK_SIZE = 16


def bucket_blocks(n: int) -> int:
    """Arrondit un nombre de blocs au godet supérieur (puissances de deux).

    Le chemin de décodage à formes fixes — eager comme graphe CUDA — travaille
    sur ces godets : les deux doivent employer la même formule, c'est elle qui
    rend leurs sorties identiques au bit près.
    """
    b = 8
    while b < n:
        b <<= 1
    return b


class BlockAllocator:
    """Liste de blocs libres, avec réutilisation adressée par le contenu.

    Deux rôles dans un seul objet, parce qu'ils se disputent la même ressource :

    * **Allocation.** Les blocs sont distribués et rendus ; une séquence qui se
      termine libère les siens immédiatement.
    * **Cache de préfixe.** Le contenu d'un bloc complet est entièrement
      déterminé par les jetons qui l'ont produit *et* par tous ceux qui
      précèdent : un bloc peut donc être adressé par le hachage chaîné de sa
      tranche de jetons. Deux requêtes partageant une consigne système
      partagent alors ses blocs, et la seconde évite complètement de
      précalculer cette tranche.

    Un bloc libéré dont le contenu reste identifiable ne rejoint pas la liste
    des libres : il part en fin de file LRU et n'est recyclé que lorsque la
    réserve s'épuise. C'est ce qui fait survivre le cache entre les requêtes
    sans jamais refuser une allocation qu'il aurait pu servir.
    """

    def __init__(self, num_blocks: int, enable_prefix_cache: bool = True) -> None:
        self.num_blocks = num_blocks
        self.enable_prefix_cache = enable_prefix_cache
        self._free: list[int] = list(range(num_blocks))
        self._refs: dict[int, int] = {}
        self._hash_of: dict[int, int] = {}      # block id  -> chained hash
        self._by_hash: dict[int, int] = {}      # chained hash -> block id
        self._lru: OrderedDict[int, None] = OrderedDict()   # evictable blocks
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.spill_cb = None              # branche par le moteur (etage hote)

    # -- capacity --------------------------------------------------------
    @property
    def num_free(self) -> int:
        """Blocs obtenables sans attendre : libres plus récupérables."""
        return len(self._free) + len(self._lru)

    @property
    def num_truly_free(self) -> int:
        return len(self._free)

    @property
    def num_cached(self) -> int:
        return len(self._by_hash)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    # -- allocation ------------------------------------------------------
    def allocate(self, n: int = 1) -> list[int]:
        if n > self.num_free:
            raise MemoryError(
                f"cache KV épuisé : {n} blocs demandés, {self.num_free} "
                f"disponibles sur {self.num_blocks}")
        out: list[int] = []
        for _ in range(n):
            if self._free:
                blk = self._free.pop()
            else:
                blk = self._evict_one()
            self._refs[blk] = 1
            out.append(blk)
        return out

    def _evict_one(self) -> int:
        """Recycle le bloc en cache libéré le moins récemment.

        Si un déversoir est branché (``spill_cb``), le contenu du bloc part
        vers l'étage hôte avant le recyclage — il reste retrouvable par son
        hachage, au prix d'une remontée PCIe au lieu d'un prefill.
        """
        blk, _ = self._lru.popitem(last=False)
        h = self._hash_of.pop(blk, None)
        if h is not None and self._by_hash.get(h) == blk:
            del self._by_hash[h]
            if self.spill_cb is not None:
                self.spill_cb(blk, h)
        self.evictions += 1
        return blk

    def free(self, blocks: list[int]) -> None:
        for blk in blocks:
            n = self._refs.get(blk, 1) - 1
            if n > 0:
                self._refs[blk] = n
                continue
            self._refs.pop(blk, None)
            if blk in self._hash_of:
                # Contenu identifiable : on le garde pour qu'il soit retrouvé.
                self._lru[blk] = None
                self._lru.move_to_end(blk)
            else:
                self._free.append(blk)

    def incref(self, blocks: list[int]) -> None:
        for blk in blocks:
            self._refs[blk] = self._refs.get(blk, 0) + 1
            self._lru.pop(blk, None)

    def reset(self) -> None:
        self._free = list(range(self.num_blocks))
        self._refs.clear()
        self._hash_of.clear()
        self._by_hash.clear()
        self._lru.clear()

    # -- prefix cache ----------------------------------------------------
    @staticmethod
    def block_hashes(token_ids: Sequence[int], block_size: int = BLOCK_SIZE,
                     ) -> list[int]:
        """Hachages chaînés, un par bloc *complet*.

        Le chaînage compte : un bloc contenant les mêmes 16 jetons dans deux
        contextes différents ne contient pas les mêmes clés et valeurs, puisque
        l'attention a vu une histoire différente. Hacher la seule tranche
        servirait volontiers le cache d'une séquence à une autre.
        """
        out: list[int] = []
        prev = 0
        n_full = len(token_ids) // block_size
        for i in range(n_full):
            span = tuple(token_ids[i * block_size:(i + 1) * block_size])
            prev = hash((prev, span))
            out.append(prev)
        return out

    def match_prefix(self, hashes: Sequence[int],
                     limit: Optional[int] = None) -> list[int]:
        """Plus longue suite de blocs de tête déjà présents dans le cache.

        ``limit`` plafonne le nombre de blocs servis : une requête dont
        l'invite est entièrement en cache a tout de même besoin d'au moins un
        jeton à faire traverser le modèle, sans quoi il n'y a rien pour produire
        des logits.
        """
        if not self.enable_prefix_cache:
            return []
        if limit is not None:
            hashes = hashes[:limit]
        matched: list[int] = []
        for h in hashes:
            blk = self._by_hash.get(h)
            if blk is None:
                break
            matched.append(blk)
        if matched:
            self.incref(matched)
            self.hits += len(matched)
        self.misses += len(hashes) - len(matched)
        return matched

    def register(self, block: int, chained_hash: int) -> None:
        """Publie un bloc rempli, pour que des requêtes ultérieures le retrouvent.

        Appelé uniquement sur un bloc *complet*. Un bloc partiellement rempli
        serait retrouvé par un hachage décrivant un contenu qu'il ne porte pas
        encore.
        """
        if not self.enable_prefix_cache:
            return
        existing = self._by_hash.get(chained_hash)
        if existing is not None and existing != block:
            return                     # quelqu'un d'autre l'a publié avant
        self._hash_of[block] = chained_hash
        self._by_hash[chained_hash] = block

    def stats(self) -> dict:
        return {
            "blocks_total": self.num_blocks,
            "blocks_free": self.num_truly_free,
            "blocks_reclaimable": len(self._lru),
            "blocks_cached": self.num_cached,
            "prefix_hits": self.hits,
            "prefix_misses": self.misses,
            "prefix_hit_rate": round(self.hit_rate, 4),
            "evictions": self.evictions,
        }


@dataclass
class KVCacheConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    num_blocks: int
    block_size: int = BLOCK_SIZE
    dtype: str = "int8"                 # int8 | fp8_e4m3 | fp16 | bf16 | lm4 (lm3, lm2 : témoins)
    device: str = "cuda:0"

    @property
    def quantized(self) -> bool:
        return self.dtype in ("int8", "fp8_e4m3") or self.rotated

    @property
    def rotated(self) -> bool:
        """4 bits par rotation (`kv_lm4`) : deux codes par octet, échelle
        fp16 par (jeton, tête) comme int8, mais D/2 octets de stockage."""
        return self.dtype in kv_lm4.FORMATS

    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            "int8": torch.int8,
            "fp8_e4m3": torch.float8_e4m3fn,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "lm4": torch.uint8, "lm3": torch.uint8, "lm2": torch.uint8,
        }[self.dtype]

    @property
    def storage_dim(self) -> int:
        """Dernière dimension des tenseurs k/v : D, ou D/2 emballé."""
        return self.head_dim // 2 if self.rotated else self.head_dim

    def bytes_per_block(self) -> int:
        elems = 2 * self.block_size * self.num_kv_heads * self.head_dim
        width = 0.5 if self.rotated else 1 if self.quantized else 2
        scale_bytes = (2 * self.block_size * self.num_kv_heads * 2
                       if self.quantized else 0)
        return int(elems * width) + scale_bytes

    def total_bytes(self) -> int:
        return self.bytes_per_block() * self.num_blocks * self.num_layers

    @property
    def capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size



class HostKVPool:
    """Étage hôte du cache KV : les blocs de préfixe évincés de la VRAM
    descendent ici (mémoire épinglée, toujours en INT8 + échelles) au lieu
    d'être perdus, et remontent quand une requête les redemande.

    C'est le troisième étage de la hiérarchie — VRAM chaude, RAM froide — et
    la brique qui permet aux longues conversations de garder leur préfixe :
    recharger un bloc par le PCIe (~30 µs) coûte trois ordres de grandeur de
    moins que recalculer son prefill.

    Le pool est indexé par le hachage chaîné des blocs, le même que le cache
    de préfixe VRAM : un bloc n'y est stocké qu'évincé *publié*, donc
    identifiable. Éviction LRU sous budget d'octets.
    """

    def __init__(self, budget_bytes: int) -> None:
        from collections import OrderedDict
        self.budget = budget_bytes
        self._data: "OrderedDict[int, list[tuple]]" = OrderedDict()
        self._bytes = 0
        self.spills = 0
        self.refills = 0
        self.evictions = 0

    @staticmethod
    def _tuple_bytes(t: tuple) -> int:
        return sum(x.numel() * x.element_size() for x in t if x is not None)

    def store(self, chained_hash: int, per_layer: list[tuple]) -> None:
        if self.budget <= 0 or chained_hash in self._data:
            return
        taille = sum(self._tuple_bytes(t) for t in per_layer)
        while self._bytes + taille > self.budget and self._data:
            _, vieux = self._data.popitem(last=False)
            self._bytes -= sum(self._tuple_bytes(t) for t in vieux)
            self.evictions += 1
        if self._bytes + taille > self.budget:
            return
        self._data[chained_hash] = per_layer
        self._bytes += taille
        self.spills += 1

    def fetch(self, chained_hash: int):
        got = self._data.get(chained_hash)
        if got is not None:
            self._data.move_to_end(chained_hash)
            self.refills += 1
        return got

    def __contains__(self, chained_hash: int) -> bool:
        return chained_hash in self._data

    def stats(self) -> dict:
        return {"blocks": len(self._data),
                "bytes": self._bytes, "budget": self.budget,
                "spills": self.spills, "refills": self.refills,
                "evictions": self.evictions}


class PagedKVCache:
    """Le cache d'une seule couche, résidant sur un seul appareil."""

    def __init__(self, cfg: KVCacheConfig, device: Optional[str] = None) -> None:
        self.cfg = cfg
        self.device = torch.device(device or cfg.device)
        shape = (cfg.num_blocks, cfg.block_size, cfg.num_kv_heads, cfg.storage_dim)
        dt = cfg.torch_dtype
        self.k = torch.zeros(shape, dtype=dt, device=self.device)
        self.v = torch.zeros(shape, dtype=dt, device=self.device)
        if cfg.quantized:
            sshape = (cfg.num_blocks, cfg.block_size, cfg.num_kv_heads)
            self.k_scale = torch.zeros(sshape, dtype=torch.float16, device=self.device)
            self.v_scale = torch.zeros(sshape, dtype=torch.float16, device=self.device)
        else:
            self.k_scale = self.v_scale = None

    # -- quantization ----------------------------------------------------
    def _quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """``x`` vaut [jetons, têtes_kv, dim_tête] -> stockage + échelle par tête."""
        if not self.cfg.quantized:
            return x.to(self.cfg.torch_dtype), None
        if self.cfg.rotated:
            return kv_lm4.quantifier(x, self.cfg.dtype)
        amax = x.abs().amax(dim=-1, keepdim=True).to(torch.float32)
        if self.cfg.dtype == "int8":
            scale = (amax / 127.0).clamp(min=1e-8)
            q = (x.to(torch.float32) / scale).round().clamp(-127, 127).to(torch.int8)
        else:                                        # fp8_e4m3, max 448
            scale = (amax / 448.0).clamp(min=1e-8)
            q = (x.to(torch.float32) / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        return q, scale.squeeze(-1).to(torch.float16)

    def _dequantize(self, q: torch.Tensor,
                    scale: Optional[torch.Tensor],
                    dtype: torch.dtype) -> torch.Tensor:
        if scale is None:
            return q.to(dtype)
        if self.cfg.rotated:
            return kv_lm4.dequantifier(q, scale, self.cfg.dtype, dtype)
        return (q.to(torch.float32) * scale.to(torch.float32).unsqueeze(-1)).to(dtype)

    # -- I/O -------------------------------------------------------------
    def write(self, slot_mapping: torch.Tensor, k: torch.Tensor,
              v: torch.Tensor, positions: Optional[torch.Tensor] = None) -> None:
        """Disperse les nouvelles clés et valeurs dans leurs emplacements.

        ``slot_mapping[i]`` est la position à plat ``bloc × taille_bloc +
        décalage`` du jeton ``i`` ; la calculer du côté de l'ordonnanceur garde
        ici une unique dispersion vectorisée au lieu d'une boucle par séquence.
        ``positions`` (position absolue du jeton ``i``, PAS son décalage dans
        le bloc) ne sert qu'au diagnostic lm4 ci-dessous.
        """
        bs = self.cfg.block_size
        # Diagnostic lm4 (poste7-kv-lm4-clos-17-09 § 1, poste2 6a660fb) : quand
        # ACVRAM_KV_LM4_SEUL ou ACVRAM_KV_LM4_PUITS est posée, k et v sont
        # remplacés par leur aller-retour (lm4 sur le côté / les positions
        # actifs, int8 par amax ailleurs) AVANT d'entrer dans le cache, quel
        # que soit son format : le porteur attendu est `int8` (le bras de
        # référence — int8∘int8 est idempotent, int8∘lm4 ≈ lm4 à 0,7 %), et
        # le noyau paginé continue de servir la lecture. Eager seulement
        # (`.any()` hôte dans quantifier_diagnostic) ; le régime le dit.
        if kv_lm4.diagnostic_actif():
            k = kv_lm4.quantifier_diagnostic(k, "k", positions)
            v = kv_lm4.quantifier_diagnostic(v, "v", positions)
        # Chemin fusionné : amax, quantification et dispersion en un noyau.
        # Le chemin PyTorch demandait une vingtaine de lancements par couche
        # sur des tenseurs de quelques centaines de valeurs.
        if (self.cfg.quantized and self.cfg.dtype == "int8" and k.is_cuda
                and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
                and self.k_scale is not None):
            from ..kernels import get_extension
            ext = get_extension()
            if ext is not None and hasattr(ext, "kv_write_int8"):
                sm = slot_mapping if slot_mapping.dtype == torch.int64 \
                    else slot_mapping.to(torch.int64)
                # C15 niveau 3 : k et v sont des tranches de la projection
                # q/k/v empilée ([T, H, D] à pas de jeton libre) — le noyau
                # les lit en place ; le témoin (ACVRAM_GLUE_COMPACT=0) garde
                # les deux copies contiguës d'avant (2 nœuds par couche).
                from ..kernels import glue_compact
                if not glue_compact("kv"):
                    k, v = k.contiguous(), v.contiguous()
                ext.kv_write_int8(k, v, sm, self.k.view(-1, *self.k.shape[2:]),
                                  self.v.view(-1, *self.v.shape[2:]),
                                  self.k_scale.view(-1, self.k_scale.shape[-1]),
                                  self.v_scale.view(-1, self.v_scale.shape[-1]), bs)
                return
        # SENTINELLE : un emplacement NEGATIF designe une ligne de
        # rembourrage — calculee pour donner au lot une forme reguliere, et
        # qui ne doit RIEN ecrire. Le noyau CUDA fusionne la porte deja
        # (`acvram_kernels.cu:1831`, « if (slot < 0) return; ») ; ce chemin de
        # repli ne la portait pas, et la symetrie n existait donc qu a moitie.
        #
        # Sans cette garde l indexation negative de PyTorch REBOUCLE au lieu
        # d ignorer : avec bs = 16, `-1` donne `blk = -1` et `off = 15`, soit
        # le DERNIER bloc, decalage 15. L ecriture est reelle, dans un bloc
        # qui appartient a une autre sequence, et le cache de prefixe la
        # republie ensuite. Corruption silencieuse, a distance, sans lien
        # visible avec ce qui l a produite.
        #
        # Le repli sert les caches non-int8, le processeur et l absence
        # d extension : c est la majorite des configurations, ET celle des
        # essais. Une epreuve d equivalence ecrite sur la foi du seul `.cu`
        # aurait tourne ici, sur le chemin non garde, et serait passee.
        if bool((slot_mapping < 0).any()):
            gardes = slot_mapping >= 0
            slot_mapping, k, v = slot_mapping[gardes], k[gardes], v[gardes]
            if slot_mapping.numel() == 0:
                return
        kq, ks = self._quantize(k)
        vq, vs = self._quantize(v)
        blk = torch.div(slot_mapping, bs, rounding_mode="floor")
        off = slot_mapping % bs
        self.k[blk, off] = kq
        self.v[blk, off] = vq
        if self.k_scale is not None:
            self.k_scale[blk, off] = ks
            self.v_scale[blk, off] = vs

    def gather(self, block_table: torch.Tensor, length: int,
               dtype: torch.dtype = torch.float16) -> tuple[torch.Tensor, torch.Tensor]:
        """Relit le cache d'une séquence sous forme dense ``[longueur, têtes, dim]``."""
        bs = self.cfg.block_size
        n_blocks = (length + bs - 1) // bs
        blocks = block_table[:n_blocks]
        k = self.k[blocks].reshape(-1, self.cfg.num_kv_heads, self.cfg.storage_dim)
        v = self.v[blocks].reshape(-1, self.cfg.num_kv_heads, self.cfg.storage_dim)
        ks = vs = None
        if self.k_scale is not None:
            ks = self.k_scale[blocks].reshape(-1, self.cfg.num_kv_heads)
            vs = self.v_scale[blocks].reshape(-1, self.cfg.num_kv_heads)
            ks, vs = ks[:length], vs[:length]
        return (self._dequantize(k[:length], ks, dtype),
                self._dequantize(v[:length], vs, dtype))

    def export_block(self, blk: int) -> tuple:
        """Copie un bloc vers l'hôte (mémoire épinglée), échelles comprises."""
        def pin(t):
            return t.detach().to("cpu", non_blocking=False).pin_memory() \
                if t.is_cuda else t.detach().clone()
        return (pin(self.k[blk]), pin(self.v[blk]),
                None if self.k_scale is None else pin(self.k_scale[blk]),
                None if self.v_scale is None else pin(self.v_scale[blk]))

    def import_block(self, blk: int, data: tuple) -> None:
        k, v, ks, vs = data
        self.k[blk].copy_(k, non_blocking=True)
        self.v[blk].copy_(v, non_blocking=True)
        if self.k_scale is not None and ks is not None:
            self.k_scale[blk].copy_(ks, non_blocking=True)
            self.v_scale[blk].copy_(vs, non_blocking=True)

    def gather_fixed(self, block_tables: torch.Tensor,
                     dtype: torch.dtype = torch.float16
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        """Relit tout un lot d'un coup, à forme fixe : ``[lot, N×bloc, têtes, dim]``.

        Contrairement à ``gather``, aucune longueur Python n'intervient : la
        table est déjà complétée à ``N`` blocs par l'appelant, et c'est à lui de
        masquer les positions au-delà de la vraie longueur. C'est ce qui rend ce
        chemin capturable dans un graphe CUDA — chaque forme, chaque adresse est
        connue à la capture.
        """
        b, n = block_tables.shape
        k = self.k[block_tables]          # [b, n, bs, hkv, d]
        v = self.v[block_tables]
        k = k.reshape(b, n * self.cfg.block_size, self.cfg.num_kv_heads,
                      self.cfg.storage_dim)
        v = v.reshape(b, n * self.cfg.block_size, self.cfg.num_kv_heads,
                      self.cfg.storage_dim)
        ks = vs = None
        if self.k_scale is not None:
            ks = self.k_scale[block_tables].reshape(b, -1, self.cfg.num_kv_heads)
            vs = self.v_scale[block_tables].reshape(b, -1, self.cfg.num_kv_heads)
        return (self._dequantize(k, ks, dtype), self._dequantize(v, vs, dtype))

    @property
    def nbytes(self) -> int:
        n = self.k.numel() * self.k.element_size() * 2
        if self.k_scale is not None:
            n += self.k_scale.numel() * 2 * 2
        return n
