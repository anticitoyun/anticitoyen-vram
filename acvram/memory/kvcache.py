"""Paged, quantized KV cache.

Paging (the vLLM idea): the cache is a pool of fixed-size blocks and each
sequence owns a list of block indices. Nothing is contiguous, so a sequence
can grow without reserving its worst-case context up front, and finished
sequences return their blocks immediately. On a 32 GB card serving several
conversations at once this is the difference between four concurrent
sequences and forty.

Quantization on top of that: keys and values are stored at 8 bits with one
fp16 scale per (block, position, head), which is 2x smaller than fp16 for
about 0.4% of overhead. Both GPUs get 8-bit storage, but not the same 8 bits:

    RTX 5090   FP8 E4M3 -- wider dynamic range, no zero point needed
    RTX 3080Ti INT8     -- Ampere has no FP8, so symmetric integer instead

The cache is per (layer, device): a layer executing on cuda:1 keeps its
blocks on cuda:1, so attention never reaches across the PCIe bus.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

__all__ = ["KVCacheConfig", "PagedKVCache", "BlockAllocator"]

BLOCK_SIZE = 16


class BlockAllocator:
    """Free list over a fixed pool of blocks, with content-addressed reuse.

    Two jobs in one object, because they are the same resource:

    * **Allocation.** Blocks are handed out and returned; a sequence that
      finishes releases its blocks immediately.
    * **Prefix caching.** A full block's contents are entirely determined by
      the tokens that produced it *and* everything before them, so a block can
      be addressed by the chained hash of its token span. Two requests sharing
      a system prompt then share its blocks outright, and the second request
      skips prefilling that span altogether.

    A freed block whose contents are still identifiable is not returned to the
    free list: it goes to the back of an LRU queue and is only recycled when
    the pool runs dry. That is what makes the cache survive between requests
    without ever refusing an allocation it could have served.
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

    # -- capacity --------------------------------------------------------
    @property
    def num_free(self) -> int:
        """Blocks obtainable without waiting -- free plus reclaimable."""
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
                f"KV cache exhausted: {n} blocks requested, {self.num_free} "
                f"available of {self.num_blocks}")
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
        """Recycle the least recently released cached block."""
        blk, _ = self._lru.popitem(last=False)
        h = self._hash_of.pop(blk, None)
        if h is not None and self._by_hash.get(h) == blk:
            del self._by_hash[h]
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
                # Identifiable contents: keep it around to be matched again.
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
        """Chained hashes, one per *complete* block.

        Chaining matters: a block holding the same 16 tokens in two different
        contexts does not hold the same keys and values, because attention saw
        different history. Hashing the span alone would happily serve one
        sequence's cache to another.
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
        """Longest run of leading blocks already in the cache.

        ``limit`` caps how many blocks may be served, because a request whose
        prompt is entirely cached still needs at least one token to run
        through the model -- there has to be something to produce logits from.
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
        """Publish a filled block so later requests can match it.

        Only ever called on a *complete* block. A partially filled block would
        be matched by a hash describing content it does not yet hold.
        """
        if not self.enable_prefix_cache:
            return
        existing = self._by_hash.get(chained_hash)
        if existing is not None and existing != block:
            return                     # someone else published it first
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
    dtype: str = "int8"                 # int8 | fp8_e4m3 | fp16 | bf16
    device: str = "cuda:0"

    @property
    def quantized(self) -> bool:
        return self.dtype in ("int8", "fp8_e4m3")

    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            "int8": torch.int8,
            "fp8_e4m3": torch.float8_e4m3fn,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.dtype]

    def bytes_per_block(self) -> int:
        elems = 2 * self.block_size * self.num_kv_heads * self.head_dim
        width = 1 if self.quantized else 2
        scale_bytes = (2 * self.block_size * self.num_kv_heads * 2
                       if self.quantized else 0)
        return elems * width + scale_bytes

    def total_bytes(self) -> int:
        return self.bytes_per_block() * self.num_blocks * self.num_layers

    @property
    def capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size


class PagedKVCache:
    """One layer's cache, living on one device."""

    def __init__(self, cfg: KVCacheConfig, device: Optional[str] = None) -> None:
        self.cfg = cfg
        self.device = torch.device(device or cfg.device)
        shape = (cfg.num_blocks, cfg.block_size, cfg.num_kv_heads, cfg.head_dim)
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
        """``x`` is [n_tokens, n_kv_heads, head_dim] -> storage + per-head scale."""
        if not self.cfg.quantized:
            return x.to(self.cfg.torch_dtype), None
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
        return (q.to(torch.float32) * scale.to(torch.float32).unsqueeze(-1)).to(dtype)

    # -- I/O -------------------------------------------------------------
    def write(self, slot_mapping: torch.Tensor, k: torch.Tensor,
              v: torch.Tensor) -> None:
        """Scatter new keys/values into their slots.

        ``slot_mapping[i]`` is the flat position ``block * block_size + offset``
        for token ``i``; computing it on the scheduler side keeps this a single
        vectorised scatter instead of a per-sequence loop.
        """
        kq, ks = self._quantize(k)
        vq, vs = self._quantize(v)
        bs = self.cfg.block_size
        blk = torch.div(slot_mapping, bs, rounding_mode="floor")
        off = slot_mapping % bs
        self.k[blk, off] = kq
        self.v[blk, off] = vq
        if self.k_scale is not None:
            self.k_scale[blk, off] = ks
            self.v_scale[blk, off] = vs

    def gather(self, block_table: torch.Tensor, length: int,
               dtype: torch.dtype = torch.float16) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one sequence's cache back as dense ``[length, heads, dim]``."""
        bs = self.cfg.block_size
        n_blocks = (length + bs - 1) // bs
        blocks = block_table[:n_blocks]
        k = self.k[blocks].reshape(-1, self.cfg.num_kv_heads, self.cfg.head_dim)
        v = self.v[blocks].reshape(-1, self.cfg.num_kv_heads, self.cfg.head_dim)
        ks = vs = None
        if self.k_scale is not None:
            ks = self.k_scale[blocks].reshape(-1, self.cfg.num_kv_heads)
            vs = self.v_scale[blocks].reshape(-1, self.cfg.num_kv_heads)
            ks, vs = ks[:length], vs[:length]
        return (self._dequantize(k[:length], ks, dtype),
                self._dequantize(v[:length], vs, dtype))

    @property
    def nbytes(self) -> int:
        n = self.k.numel() * self.k.element_size() * 2
        if self.k_scale is not None:
            n += self.k_scale.numel() * 2 * 2
        return n
