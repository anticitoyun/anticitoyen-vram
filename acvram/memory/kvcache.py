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
from . import kv_canal
from . import kv_k8v4

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
    def sel_images(images, a: int, b: int) -> tuple:
        """Les sha256 des images dont la plage [debut, fin) touche [a, b),
        dans l'ordre des positions ; () sans image (clé texte d'avant).

        ``images`` : itérable de (debut, fin, sha256, …) ou d'objets à
        attributs ``debut``, ``fin``, ``sha256`` (frontière tolérante avec la
        pièce (b), REGLES § 4 : deux images ≠, même texte → deux clés)."""
        if not images:
            return ()
        sel = []
        for im in images:
            if isinstance(im, (tuple, list)):
                d, f, sha = im[0], im[1], im[2]
            else:
                d, f, sha = im.debut, im.fin, im.sha256
            if int(d) < b and int(f) > a:
                sel.append((int(d), int(f), str(sha)))
        sel.sort()
        return tuple(sel)

    @staticmethod
    def hash_bloc(prev: int, span: tuple, sel: tuple = ()) -> int:
        """Hachage chaîné d'un bloc : sans image, exactement `hash((prev, span))`
        — le texte seul garde sa clé d'avant."""
        return hash((prev, span, sel)) if sel else hash((prev, span))

    @staticmethod
    def block_hashes(token_ids: Sequence[int], block_size: int = BLOCK_SIZE,
                     images=None) -> list[int]:
        """Hachages chaînés, un par bloc *complet*.

        Le chaînage compte : un bloc contenant les mêmes 16 jetons dans deux
        contextes différents ne contient pas les mêmes clés et valeurs, puisque
        l'attention a vu une histoire différente. Hacher la seule tranche
        servirait volontiers le cache d'une séquence à une autre.

        ``images`` (multimodal P1) : les jetons d'une image sont des
        marqueurs identiques d'une image à l'autre ; sans le sha256 des
        pixels dans la clé, le préfixe « <image> » d'une autre image rendrait
        le KV d'une autre image sans erreur.
        """
        out: list[int] = []
        prev = 0
        n_full = len(token_ids) // block_size
        for i in range(n_full):
            a, b = i * block_size, (i + 1) * block_size
            span = tuple(token_ids[a:b])
            prev = BlockAllocator.hash_bloc(prev, span,
                                            BlockAllocator.sel_images(images, a, b))
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
    dtype: str = "int8"                 # int8 | fp8_e4m3 | fp16 | bf16 | lm4 (lm3, lm2 : témoins) | k8v4 (pièce 104)
    device: str = "cuda:0"
    # C5-b : clés int8 à échelle par canal sur chaque bloc (kv_canal) ; None =
    # ACVRAM_KV_INT8_CANAL, et seulement en int8. `rangs` = lignes bf16 de la
    # réserve des blocs courants, par couche (None = ACVRAM_KV_CANAL_RANGS).
    canal: Optional[bool] = None
    rangs: Optional[int] = None
    # Repli 104 (1) : positions puits gardées en V int8 (kv_k8v4.PUITS), k8v4 seulement ; None = ACVRAM_KV_PUITS.
    puits: Optional[int] = None

    def __post_init__(self) -> None:
        if self.puits is None:
            self.puits = kv_k8v4.PUITS
        self.puits = int(self.puits) if self.dtype == kv_k8v4.FORMAT else 0
        if self.puits not in kv_k8v4.PUITS_VALEURS or (self.puits and self.puits != self.block_size):
            raise ValueError(f"puits={self.puits} : 0 ou un bloc ({self.block_size}) seulement")
        if self.canal is None:
            self.canal = kv_canal.ACTIF
        self.canal = bool(self.canal) and self.dtype == "int8"
        if self.rangs is None:
            self.rangs = kv_canal.RANGS

    @property
    def quantized(self) -> bool:
        return self.dtype in ("int8", "fp8_e4m3", kv_k8v4.FORMAT) or self.rotated

    @property
    def k8v4(self) -> bool:
        """Pièce 104 : K int8 par jeton (chemin int8 intact), V int4 par groupe de 32 (`kv_k8v4`)."""
        return self.dtype == kv_k8v4.FORMAT

    @property
    def nom_format(self) -> str:
        """Le format tel que la ligne de régime le nomme."""
        return "int8-canal16" if self.canal else self.dtype

    def octets_tampon(self) -> int:
        """Réserve bf16 des blocs courants (K seul) + pile des lignes libres."""
        if not self.canal:
            return 0
        return self.rangs * self.block_size * self.num_kv_heads * self.head_dim * 2 + self.rangs * 4

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
            kv_k8v4.FORMAT: torch.int8,          # K ; V : `torch_dtype_v`
        }[self.dtype]

    @property
    def torch_dtype_v(self) -> torch.dtype:
        """Type de stockage de V : celui de K, sauf k8v4 (quartets emballés)."""
        return torch.uint8 if self.k8v4 else self.torch_dtype

    @property
    def storage_dim(self) -> int:
        """Dernière dimension des tenseurs k/v : D, ou D/2 emballé."""
        return self.head_dim // 2 if self.rotated else self.head_dim

    @property
    def storage_dim_v(self) -> int:
        """Dernière dimension de V : `storage_dim`, sauf k8v4 (D/2, deux codes par octet)."""
        return self.head_dim // 2 if self.k8v4 else self.storage_dim

    @property
    def echelles_v(self) -> tuple[int, ...]:
        """Forme des échelles de V au-delà de (bloc, jeton, tête) : () par jeton, (D/32,) en k8v4."""
        return (kv_k8v4.groupes(self.head_dim),) if self.k8v4 else ()

    def bytes_per_block(self) -> int:
        if self.k8v4:
            # réserve des puits (v1 de mesure) : V int8 + échelle half pour chaque ligne de chaque bloc
            reserve = self.block_size * self.num_kv_heads * (self.head_dim + 2) if self.puits else 0
            return self.block_size * kv_k8v4.octets_par_jeton_couche(self.num_kv_heads, self.head_dim) + reserve
        elems = 2 * self.block_size * self.num_kv_heads * self.head_dim
        width = 0.5 if self.rotated else 1 if self.quantized else 2
        scale_bytes = (2 * self.block_size * self.num_kv_heads * 2
                       if self.quantized else 0)
        if self.canal:
            # sc E4M3 [HKV, D] par bloc (128 o par tête pour D = 128) + tampon_de int32
            scale_bytes += self.num_kv_heads * self.head_dim + 4
        return int(elems * width) + scale_bytes

    def total_bytes(self) -> int:
        return (self.bytes_per_block() * self.num_blocks + self.octets_tampon()) * self.num_layers

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
        # k8v4 : V emballé [.., D/2] uint8 et une échelle par groupe de 32 canaux ;
        # partout ailleurs V a la forme de K.
        self.v = torch.zeros(shape[:-1] + (cfg.storage_dim_v,), dtype=cfg.torch_dtype_v,
                             device=self.device)
        if cfg.quantized:
            sshape = (cfg.num_blocks, cfg.block_size, cfg.num_kv_heads)
            self.k_scale = torch.zeros(sshape, dtype=torch.float16, device=self.device)
            self.v_scale = torch.zeros(sshape + cfg.echelles_v, dtype=torch.float16,
                                       device=self.device)
        else:
            self.k_scale = self.v_scale = None
        # Repli 104 (1) : réserve des puits, V int8 [NB, 16, HKV, D] + échelle half [NB, 16, HKV], indexée comme K
        self.puits_v = self.puits_vs = None
        if cfg.puits:
            self.puits_v = torch.zeros(shape, dtype=torch.int8, device=self.device)
            self.puits_vs = torch.zeros(shape[:-1], dtype=torch.float16, device=self.device)
        # C5-b (kv_canal) : échelles de K par canal, réserve bf16 des blocs
        # courants et sa pile de lignes libres — tout sur l'appareil, adresses
        # fixes : les noyaux d'écriture allouent et rendent les lignes eux-mêmes
        # (capturable). Un bloc jamais écrit lit sc = 1 (par jeton) et ks = 0.
        self.k_scale_canal = self.tampon = self.tampon_de = None
        self.tampon_libres = self.tampon_sommet = None
        if cfg.canal:
            self.k_scale_canal = torch.full((cfg.num_blocks, cfg.num_kv_heads, cfg.head_dim),
                                            kv_canal.SC_PAR_JETON, dtype=torch.uint8,
                                            device=self.device).view(torch.float8_e4m3fn)
            self.tampon = torch.zeros((cfg.rangs, cfg.block_size, cfg.num_kv_heads, cfg.head_dim),
                                      dtype=torch.bfloat16, device=self.device)
            self.tampon_de = torch.full((cfg.num_blocks,), -1, dtype=torch.int32, device=self.device)
            self.tampon_libres = torch.arange(cfg.rangs, dtype=torch.int32, device=self.device)
            self.tampon_sommet = torch.full((1,), cfg.rangs, dtype=torch.int32, device=self.device)

    @property
    def canal(self) -> bool:
        return self.k_scale_canal is not None

    # -- quantization ----------------------------------------------------
    def _quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """``x`` vaut [jetons, têtes_kv, dim_tête] -> stockage + échelle par tête."""
        if not self.cfg.quantized:
            return x.to(self.cfg.torch_dtype), None
        if self.cfg.rotated:
            return kv_lm4.quantifier(x, self.cfg.dtype)
        if self.cfg.k8v4:
            return kv_k8v4.quantifier_k(x)       # int8 par jeton, division IEEE (au bit du noyau)
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

    # k8v4 : K suit le chemin int8 (`_quantize`/`_dequantize`), V le jumeau par groupe.
    def _quantize_v(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.cfg.k8v4:
            return kv_k8v4.quantifier_v(x)
        return self._quantize(x)

    def _dequantize_v(self, q: torch.Tensor, scale: Optional[torch.Tensor],
                      dtype: torch.dtype) -> torch.Tensor:
        if self.cfg.k8v4:
            return kv_k8v4.dequantifier_v(q, scale, dtype)
        return self._dequantize(q, scale, dtype)

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
        # C5-b : clés par canal (kv_canal) — trois noyaux (rôles, écriture,
        # fermeture) sur carte, le jumeau torch ailleurs ; V par jeton dans
        # les deux cas. La sentinelle (slot < 0) est portée par les deux.
        if self.canal:
            if k.is_cuda and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16:
                from ..kernels import get_extension
                ext = get_extension()
                if ext is not None and hasattr(ext, "kv_write_int8_canal"):
                    sm = slot_mapping if slot_mapping.dtype == torch.int64 \
                        else slot_mapping.to(torch.int64)
                    ext.kv_write_int8_canal(
                        k, v, sm, self.k.view(-1, *self.k.shape[2:]),
                        self.v.view(-1, *self.v.shape[2:]),
                        self.k_scale.view(-1, self.k_scale.shape[-1]),
                        self.v_scale.view(-1, self.v_scale.shape[-1]),
                        self.k_scale_canal.view(torch.uint8), self.tampon, self.tampon_de,
                        self.tampon_libres, self.tampon_sommet, bs)
                    return
            kv_canal.ecrire(self, slot_mapping, k, v)
            return
        # k8v4 (pièce 104) : un noyau, K int8 par jeton + V int4 par groupe de 32 ;
        # sans le symbole, le jumeau torch (chemin de repli ci-dessous) écrit la
        # même chose au bit.
        if (self.cfg.k8v4 and k.is_cuda and k.dtype == torch.bfloat16
                and v.dtype == torch.bfloat16):
            from ..kernels import get_extension, glue_compact
            ext = get_extension()
            if self.puits_v is not None and ext is not None and hasattr(ext, "kv_write_k8v4_puits"):
                if positions is None:
                    raise RuntimeError("k8v4 + puits : l'écriture exige les positions absolues")
                sm = slot_mapping if slot_mapping.dtype == torch.int64 \
                    else slot_mapping.to(torch.int64)
                pos = positions if positions.dtype == torch.int64 else positions.to(torch.int64)
                if not glue_compact("kv"):
                    k, v = k.contiguous(), v.contiguous()
                ext.kv_write_k8v4_puits(k, v, sm, pos.reshape(-1).contiguous(),
                                        self.k.view(-1, *self.k.shape[2:]),
                                        self.v.view(-1, *self.v.shape[2:]),
                                        self.k_scale.view(-1, self.k_scale.shape[-1]),
                                        self.v_scale.view(-1, *self.v_scale.shape[2:]),
                                        self.puits_v.view(-1, *self.puits_v.shape[2:]),
                                        self.puits_vs.view(-1, self.puits_vs.shape[-1]), bs, self.cfg.puits)
                return
            if self.puits_v is None and ext is not None and hasattr(ext, "kv_write_k8v4"):
                sm = slot_mapping if slot_mapping.dtype == torch.int64 \
                    else slot_mapping.to(torch.int64)
                if not glue_compact("kv"):
                    k, v = k.contiguous(), v.contiguous()
                ext.kv_write_k8v4(k, v, sm, self.k.view(-1, *self.k.shape[2:]),
                                  self.v.view(-1, *self.v.shape[2:]),
                                  self.k_scale.view(-1, self.k_scale.shape[-1]),
                                  self.v_scale.view(-1, *self.v_scale.shape[2:]), bs)
                return
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
        if self.puits_v is not None and positions is None:
            raise RuntimeError("k8v4 + puits : l'écriture exige les positions absolues")
        if bool((slot_mapping < 0).any()):
            gardes = slot_mapping >= 0
            slot_mapping, k, v = slot_mapping[gardes], k[gardes], v[gardes]
            if positions is not None:
                positions = positions.reshape(-1)[gardes]
            if slot_mapping.numel() == 0:
                return
        kq, ks = self._quantize(k)
        vq, vs = self._quantize_v(v)
        blk = torch.div(slot_mapping, bs, rounding_mode="floor")
        off = slot_mapping % bs
        self.k[blk, off] = kq
        self.v[blk, off] = vq
        if self.k_scale is not None:
            self.k_scale[blk, off] = ks
            self.v_scale[blk, off] = vs
        if self.puits_v is not None:
            # jumeau du noyau : V int8 par jeton (division IEEE, comme K) aux positions < puits, EN PLUS du k8v4
            p = positions.reshape(-1) < self.cfg.puits
            if bool(p.any()):
                pq, ps = kv_k8v4.quantifier_k(v[p])
                self.puits_v[blk[p], off[p]] = pq
                self.puits_vs[blk[p], off[p]] = ps

    def gather(self, block_table: torch.Tensor, length: int,
               dtype: torch.dtype = torch.float16) -> tuple[torch.Tensor, torch.Tensor]:
        """Relit le cache d'une séquence sous forme dense ``[longueur, têtes, dim]``."""
        bs = self.cfg.block_size
        n_blocks = (length + bs - 1) // bs
        blocks = block_table[:n_blocks]
        k = self.k[blocks].reshape(-1, self.cfg.num_kv_heads, self.cfg.storage_dim)
        v = self.v[blocks].reshape(-1, self.cfg.num_kv_heads, self.cfg.storage_dim_v)
        ks = vs = None
        if self.k_scale is not None:
            ks = self.k_scale[blocks].reshape(-1, self.cfg.num_kv_heads)
            vs = self.v_scale[blocks].reshape(-1, self.cfg.num_kv_heads, *self.cfg.echelles_v)
            ks, vs = ks[:length], vs[:length]
        if self.canal:
            kd = kv_canal.dequantifier_cache(self, blocks, dtype)
            return (kd.reshape(-1, self.cfg.num_kv_heads, self.cfg.head_dim)[:length],
                    self._dequantize_v(v[:length], vs, dtype))
        vd = self._dequantize_v(v[:length], vs, dtype)
        if self.puits_v is not None and length > 0:
            n = min(length, self.cfg.puits)
            pv = self.puits_v[blocks[:1]].reshape(-1, self.cfg.num_kv_heads, self.cfg.head_dim)[:n]
            ps = self.puits_vs[blocks[:1]].reshape(-1, self.cfg.num_kv_heads)[:n]
            vd = vd.clone()
            vd[:n] = self._dequantize(pv, ps, dtype)
        return (self._dequantize(k[:length], ks, dtype), vd)

    def export_block(self, blk: int) -> tuple:
        """Copie un bloc vers l'hôte (mémoire épinglée), échelles comprises."""
        if self.puits_v is not None:
            raise RuntimeError("k8v4 + puits : l'étage KV hôte n'est pas porté (la réserve des puits resterait sur la carte)")

        def pin(t):
            return t.detach().to("cpu", non_blocking=False).pin_memory() \
                if t.is_cuda else t.detach().clone()
        # Seuls des blocs COMPLETS (publiés) descendent : jamais de bloc courant,
        # donc jamais de ligne de la réserve ; les échelles par canal suivent.
        return (pin(self.k[blk]), pin(self.v[blk]),
                None if self.k_scale is None else pin(self.k_scale[blk]),
                None if self.v_scale is None else pin(self.v_scale[blk]),
                None if self.k_scale_canal is None else pin(self.k_scale_canal[blk].view(torch.uint8)))

    def import_block(self, blk: int, data: tuple) -> None:
        k, v, ks, vs = data[:4]
        self.k[blk].copy_(k, non_blocking=True)
        self.v[blk].copy_(v, non_blocking=True)
        if self.k_scale is not None and ks is not None:
            self.k_scale[blk].copy_(ks, non_blocking=True)
            self.v_scale[blk].copy_(vs, non_blocking=True)
        if self.k_scale_canal is not None and len(data) > 4 and data[4] is not None:
            self.k_scale_canal.view(torch.uint8)[blk].copy_(data[4], non_blocking=True)
            self.tampon_de[blk] = -1

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
                      self.cfg.storage_dim_v)
        ks = vs = None
        if self.k_scale is not None:
            ks = self.k_scale[block_tables].reshape(b, -1, self.cfg.num_kv_heads)
            vs = self.v_scale[block_tables].reshape(b, -1, self.cfg.num_kv_heads,
                                                    *self.cfg.echelles_v)
        if self.canal:
            kd = kv_canal.dequantifier_cache(self, block_tables, dtype)
            return (kd.reshape(b, n * self.cfg.block_size, self.cfg.num_kv_heads, self.cfg.head_dim),
                    self._dequantize_v(v, vs, dtype))
        vd = self._dequantize_v(v, vs, dtype)
        if self.puits_v is not None:
            p = self.cfg.puits
            pv = self.puits_v[block_tables[:, 0]].reshape(b, p, self.cfg.num_kv_heads, self.cfg.head_dim)
            ps = self.puits_vs[block_tables[:, 0]].reshape(b, p, self.cfg.num_kv_heads)
            vd = torch.cat((self._dequantize(pv, ps, dtype), vd[:, p:]), dim=1)
        return (self._dequantize(k, ks, dtype), vd)

    @property
    def nbytes(self) -> int:
        n = self.k.numel() * self.k.element_size() + self.v.numel() * self.v.element_size()
        if self.k_scale is not None:
            n += (self.k_scale.numel() + self.v_scale.numel()) * 2
        if self.puits_v is not None:
            n += self.puits_v.numel() + self.puits_vs.numel() * 2
        if self.canal:
            n += (self.k_scale_canal.numel() + self.tampon_de.numel() * 4
                  + self.tampon.numel() * 2 + self.tampon_libres.numel() * 4 + 4)
        return n
