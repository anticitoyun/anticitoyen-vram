"""Placement planner: which layer lives where, and in which format.

The rig has three memory tiers with wildly different characteristics:

    tier        size    read bandwidth   format
    ---------   -----   --------------   ---------
    RTX 5090     32 GB      ~1790 GB/s   NVFP4   (4.50 bpw)
    RTX 3080 Ti  12 GB       ~912 GB/s   INT4    (4.16 bpw)
    DDR5 host    96 GB    ~70 GB/s local
                          but only as fast as the PCIe link when streamed

Decode of a single token is memory-bound: the time to produce one token is
essentially the time to read every active weight once. So the planner's job
is to minimise total read time subject to the capacity of each tier, and the
crucial asymmetry is that a *streamed* layer is limited by its PCIe link, not
by VRAM bandwidth -- roughly 55 GB/s on a Gen5 x16 slot, and only ~7 GB/s on
a chipset-attached Gen4 x4 slot.

Two structural facts drive every decision here:

* A sparse MoE layer only reads ``top_k`` experts per token. A 235B MoE with
  8 of 128 experts active reads ~5% of its weights, so host RAM becomes a
  perfectly reasonable place for experts -- this is what makes 96 GB of DDR5
  worth more than it looks.
* GeForce boards have no NVLink and NVIDIA disables PCIe P2P on them, so a
  GPU-to-GPU handoff goes through pinned host memory. That handoff is one
  hidden-state vector per token (a few KB), so it is cheap -- but it means
  the pipeline should cross between GPUs exactly once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional

from ..engine.config import ModelSpec
from ..hardware.detect import Rig
from ..quant.formats import bits_per_weight

__all__ = ["Tier", "LayerPlacement", "Plan", "PlannerOptions", "plan_placement"]

MB = 1024 ** 2
GB = 1024 ** 3

# Per-GPU overhead we cannot allocate: CUDA context, cuBLAS/cuDNN workspaces,
# the allocator's own bookkeeping, and whatever the display server holds.
CUDA_CONTEXT_RESERVE = 800 * MB
FRAGMENTATION_MARGIN = 0.03

# Nameplate dense throughput, used only to rank prefill options. These are
# estimates, not measurements: `acvram bench` replaces them with real numbers.
TFLOPS = {
    120: {"nvfp4": 838.0, "fp8": 419.0, "bf16": 209.0, "fp16": 209.0},
    89:  {"fp8": 660.0, "bf16": 330.0, "fp16": 330.0},
    86:  {"int8": 136.0, "fp16": 68.0, "bf16": 68.0},
    80:  {"int8": 125.0, "fp16": 62.0, "bf16": 62.0},
}


@dataclass
class Tier:
    name: str                    # "cuda:0", "cuda:1", "cpu"
    kind: str                    # "gpu" | "host"
    device_index: int
    capacity: int                # usable bytes for weights, after reserves
    weight_format: str
    kv_format: str
    read_bandwidth: float        # GB/s when the weights are resident here
    link_bandwidth: float        # GB/s host -> this device
    sm: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LayerPlacement:
    """Where one transformer block lives.

    Attention and MLP are placed independently. That distinction is the whole
    point for a sparse MoE: the attention block of a Qwen3-235B layer is ~71 M
    parameters while its 128 experts are ~2.4 B, so pinning attention in VRAM
    costs almost nothing and keeps the latency-critical path off the PCIe bus,
    while the experts -- of which only 8 are read per token -- live in host RAM.
    """

    index: int
    exec_device: str             # where the maths runs
    attn_storage: str            # exec_device or "cpu"
    mlp_storage: str             # exec_device or "cpu"
    fmt: str
    attn_bytes: int
    mlp_bytes: int
    mlp_active_bytes: int        # what one token actually reads from the MLP
    is_moe: bool = False
    cached_expert_fraction: float = 0.0
    mlp_exec: str = "gpu"        # "gpu" (stream the weights in) or "cpu"

    @property
    def total_bytes(self) -> int:
        return self.attn_bytes + self.mlp_bytes

    @property
    def streamed(self) -> bool:
        return self.attn_storage == "cpu" or self.mlp_storage == "cpu"

    @property
    def resident_bytes(self) -> int:
        n = 0
        if self.attn_storage != "cpu":
            n += self.attn_bytes
        if self.mlp_storage != "cpu":
            n += self.mlp_bytes
        return n

    def to_dict(self) -> dict:
        d = asdict(self)
        d["streamed"] = self.streamed
        return d


@dataclass
class PlannerOptions:
    max_model_len: int = 8192
    max_concurrent_seqs: int = 8
    kv_bits: int = 8
    reserve_per_gpu: int = CUDA_CONTEXT_RESERVE
    allow_host_tier: bool = True
    host_fraction: float = 0.85           # of MemAvailable we are willing to use
    kv_vram_fraction: float = 0.35        # ceiling on VRAM spent on KV cache
    expert_cache_fraction: float = 0.5    # of leftover VRAM, for hot MoE experts
    force_format: Optional[str] = None
    group_size: int = 128
    pin_attention: bool = True            # keep attention off the host tier
    gpus: Optional[str] = None            # "auto" | "all" | "0" | "0,1"
    host_exec: str = "auto"               # auto | stream | cpu
    host_compute_gb_s: float = 70.0       # DDR5 streaming reads, measured by bench


@dataclass
class Plan:
    model: str
    tiers: list[Tier] = field(default_factory=list)
    layers: list[LayerPlacement] = field(default_factory=list)
    embed_device: str = "cpu"
    lm_head_device: str = "cuda:0"
    kv_bytes_per_token: int = 0
    kv_budget: dict[str, int] = field(default_factory=dict)
    kv_max_tokens: int = 0
    expert_cache_bytes: dict[str, int] = field(default_factory=dict)
    stage_ranges: dict[str, tuple[int, int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    est_decode_tok_s: float = 0.0
    est_prefill_tok_s: float = 0.0
    est_bytes_per_token: int = 0
    total_weight_bytes: int = 0
    bytes_per_tier: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "tiers": [t.to_dict() for t in self.tiers],
            "layers": [l.to_dict() for l in self.layers],
            "embed_device": self.embed_device,
            "lm_head_device": self.lm_head_device,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "kv_budget": self.kv_budget,
            "kv_max_tokens": self.kv_max_tokens,
            "expert_cache_bytes": self.expert_cache_bytes,
            "stage_ranges": {k: list(v) for k, v in self.stage_ranges.items()},
            "total_weight_bytes": self.total_weight_bytes,
            "bytes_per_tier": self.bytes_per_tier,
            "est_decode_tok_s": round(self.est_decode_tok_s, 2),
            "est_prefill_tok_s": round(self.est_prefill_tok_s, 1),
            "est_bytes_per_token": self.est_bytes_per_token,
            "warnings": self.warnings,
        }

    def device_of_layer(self, i: int) -> str:
        return self.layers[i].exec_device

    def render(self) -> str:
        lines = [f"placement plan for {self.model}", ""]
        w = max([len(t.name) for t in self.tiers] + [6])
        lines.append(f"  {'tier':<{w}}  {'format':<9} {'capacity':>10} "
                     f"{'weights':>10} {'KV':>10}  {'stage':<12}")
        for t in self.tiers:
            used = self.bytes_per_tier.get(t.name, 0)
            kv = self.kv_budget.get(t.name, 0)
            rng = self.stage_ranges.get(t.name)
            stage = f"layers {rng[0]}-{rng[1]}" if rng else "-"
            lines.append(f"  {t.name:<{w}}  {t.weight_format:<9} {_h(t.capacity):>10} "
                         f"{_h(used):>10} {_h(kv):>10}  {stage:<12}")
        lines.append("")
        host_attn = [l.index for l in self.layers if l.attn_storage == "cpu"]
        host_mlp = [l.index for l in self.layers if l.mlp_storage == "cpu"]
        lines.append(f"  weights total      {_h(self.total_weight_bytes)}")
        lines.append(f"  read per token     {_h(self.est_bytes_per_token)}")
        lines.append(f"  KV per token       {_h(self.kv_bytes_per_token)}"
                     f"  -> {self.kv_max_tokens:,} tokens cached")
        lines.append(f"  embeddings         {self.embed_device}")
        lines.append(f"  lm_head            {self.lm_head_device}")
        if host_mlp:
            lines.append(f"  MLP in host RAM    {_compact_ranges(host_mlp)}")
            on_cpu = [l.index for l in self.layers if l.mlp_exec == "cpu"]
            if on_cpu:
                lines.append(f"    computed on CPU  {_compact_ranges(on_cpu)}"
                             f"  (DDR5 is wider than the PCIe link)")
            streamed = [l.index for l in self.layers
                        if l.mlp_storage == "cpu" and l.mlp_exec == "gpu"]
            if streamed:
                lines.append(f"    streamed to GPU  {_compact_ranges(streamed)}")
        if host_attn:
            lines.append(f"  attn in host RAM   {_compact_ranges(host_attn)}")
        for dev, b in self.expert_cache_bytes.items():
            if b:
                lines.append(f"  expert cache       {_h(b)} on {dev}")
        lines.append("")
        lines.append(f"  estimated decode   {self.est_decode_tok_s:.1f} tok/s  (batch 1)")
        lines.append(f"  estimated prefill  {self.est_prefill_tok_s:,.0f} tok/s")
        for warn in self.warnings:
            lines.append(f"  ! {warn}")
        return "\n".join(lines)


def _h(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def _compact_ranges(nums: list[int]) -> str:
    if not nums:
        return "-"
    nums = sorted(nums)
    out, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        out.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = n
    out.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(out)


# --------------------------------------------------------------------------
# tier construction
# --------------------------------------------------------------------------


def build_tiers(rig: Rig, opts: PlannerOptions) -> list[Tier]:
    tiers: list[Tier] = []
    for g in sorted(rig.gpus, key=lambda x: (-x.vram_bandwidth_gbps, x.index)):
        caps = g.caps
        fmt = opts.force_format or (caps.weight_format if caps else "int4_awq")
        kvf = caps.kv_format if caps else "int8"
        usable = int((g.total_mem - opts.reserve_per_gpu) * (1 - FRAGMENTATION_MARGIN))
        tiers.append(Tier(
            name=f"cuda:{g.index}", kind="gpu", device_index=g.index,
            capacity=max(0, usable), weight_format=fmt, kv_format=kvf,
            read_bandwidth=g.vram_bandwidth_gbps,
            link_bandwidth=g.host_link_gbps,
            sm=caps.sm if caps else 0,
        ))
    if opts.allow_host_tier:
        avail = rig.host.available or rig.host.total
        tiers.append(Tier(
            name="cpu", kind="host", device_index=-1,
            capacity=int(avail * opts.host_fraction),
            weight_format=tiers[0].weight_format if tiers else "int4_awq",
            kv_format="fp16", read_bandwidth=70.0,
            link_bandwidth=max((t.link_bandwidth for t in tiers), default=25.0),
        ))
    return tiers


def _bytes(n_params: float, fmt: str, group_size: int) -> int:
    return int(n_params * bits_per_weight(fmt, group_size=group_size) / 8)


# --------------------------------------------------------------------------
# planner
# --------------------------------------------------------------------------


def plan_placement(spec: ModelSpec, rig: Rig,
                   opts: Optional[PlannerOptions] = None) -> Plan:
    """Assign every tensor to a tier and estimate what it will cost.

    The order of decisions is deliberate. KV cache is sized first because it
    scales with traffic and a model that cannot hold its context is useless
    however well its weights fit. Attention comes next because it is small and
    sits on the latency-critical path. MLP and expert weights compete for what
    is left, and whatever loses goes to host RAM, where the cost of a miss is
    a PCIe transfer rather than an out-of-memory error.
    """
    opts = opts or PlannerOptions()
    tiers = build_tiers(rig, opts)
    gpu_tiers = [t for t in tiers if t.kind == "gpu"]
    host_tier = next((t for t in tiers if t.kind == "host"), None)
    plan = Plan(model=spec.name, tiers=tiers)

    if not gpu_tiers:
        plan.warnings.append("no CUDA device detected; planning a CPU-only run")
        gpu_tiers = []

    remaining = {t.name: float(t.capacity) for t in tiers}

    # ---- 1. KV cache -----------------------------------------------------
    kv_per_tok = spec.kv_bytes_per_token(opts.kv_bits)
    plan.kv_bytes_per_token = kv_per_tok
    if gpu_tiers and kv_per_tok:
        wanted = kv_per_tok * opts.max_model_len * opts.max_concurrent_seqs
        pool = sum(remaining[t.name] for t in gpu_tiers)
        target = min(wanted, pool * opts.kv_vram_fraction)
        for t in gpu_tiers:
            share = target * remaining[t.name] / max(1.0, pool)
            plan.kv_budget[t.name] = int(share)
            remaining[t.name] -= share
        plan.kv_max_tokens = int(sum(plan.kv_budget.values()) // max(1, kv_per_tok))
        if plan.kv_max_tokens < opts.max_model_len:
            plan.warnings.append(
                f"KV budget holds {plan.kv_max_tokens:,} tokens, less than one "
                f"full context of {opts.max_model_len:,}; lower --max-model-len "
                f"or raise --kv-vram-fraction")

    # ---- 2. pipeline stages ---------------------------------------------
    # Contiguous ranges, sized in proportion to each GPU's post-KV capacity so
    # the pipeline crosses a GPU boundary exactly once. Crossing is cheap (one
    # hidden state through pinned host memory) but must not happen per layer.
    n = spec.num_layers
    stage_of: list[str] = []
    if gpu_tiers:
        weights_pool = sum(max(0.0, remaining[t.name]) for t in gpu_tiers)
        cursor = 0
        for i, t in enumerate(gpu_tiers):
            if i == len(gpu_tiers) - 1:
                count = n - cursor
            else:
                frac = max(0.0, remaining[t.name]) / max(1.0, weights_pool)
                count = max(1, int(round(frac * n)))
                count = min(count, n - cursor - (len(gpu_tiers) - i - 1))
            if count > 0:
                plan.stage_ranges[t.name] = (cursor, cursor + count - 1)
            stage_of.extend([t.name] * count)
            cursor += count
        stage_of = stage_of[:n] + [gpu_tiers[-1].name] * max(0, n - len(stage_of))
    else:
        stage_of = ["cpu"] * n

    # ---- 3. attention: pinned to its stage GPU when it fits ---------------
    placements: list[LayerPlacement] = []
    used = {t.name: 0.0 for t in tiers}
    for layer in spec.layers:
        dev = stage_of[layer.index]
        fmt = next((t.weight_format for t in tiers if t.name == dev), "int4_awq")
        a_bytes = _bytes(layer.attn_params + layer.norm_params, fmt, opts.group_size)
        m_bytes = _bytes(layer.mlp_params, fmt, opts.group_size)
        m_active = _bytes(layer.active_params - layer.attn_params - layer.norm_params,
                          fmt, opts.group_size)
        attn_storage = dev
        if opts.pin_attention and dev != "cpu" and remaining[dev] >= a_bytes:
            remaining[dev] -= a_bytes
            used[dev] += a_bytes
        elif host_tier is not None:
            attn_storage = "cpu"
            remaining["cpu"] -= a_bytes
            used["cpu"] += a_bytes
        else:
            remaining[dev] -= a_bytes
            used[dev] += a_bytes
        placements.append(LayerPlacement(
            index=layer.index, exec_device=dev, attn_storage=attn_storage,
            mlp_storage="pending", fmt=fmt, attn_bytes=a_bytes,
            mlp_bytes=m_bytes, mlp_active_bytes=m_active, is_moe=layer.is_moe))

    # ---- 4. embeddings and lm_head ---------------------------------------
    # The embedding table is a gather of one row per token: leaving it in host
    # RAM costs a few KB of PCIe traffic. lm_head is a full vocabulary GEMM on
    # every step, so it earns its place on the fastest GPU.
    embed_bytes = spec.embed_params * 2
    fastest = gpu_tiers[0].name if gpu_tiers else "cpu"
    head_fmt = gpu_tiers[0].weight_format if gpu_tiers else "int4_awq"
    head_bytes = _bytes(spec.lm_head_params, head_fmt, opts.group_size)
    if gpu_tiers and remaining[fastest] > head_bytes:
        plan.lm_head_device = fastest
        remaining[fastest] -= head_bytes
        used[fastest] += head_bytes
    else:
        plan.lm_head_device = "cpu"
        used["cpu"] = used.get("cpu", 0.0) + head_bytes
    plan.embed_device = "cpu" if host_tier else fastest
    if plan.embed_device != "cpu":
        remaining[plan.embed_device] -= embed_bytes
        used[plan.embed_device] += embed_bytes
    else:
        used["cpu"] += embed_bytes

    # ---- 5. MLP / experts: fill VRAM front to back, spill the rest --------
    for lp in placements:
        dev = lp.exec_device
        if dev != "cpu" and remaining.get(dev, 0) >= lp.mlp_bytes:
            remaining[dev] -= lp.mlp_bytes
            used[dev] += lp.mlp_bytes
            lp.mlp_storage = dev
        elif host_tier is not None and remaining["cpu"] >= lp.mlp_bytes:
            remaining["cpu"] -= lp.mlp_bytes
            used["cpu"] += lp.mlp_bytes
            lp.mlp_storage = "cpu"
        else:
            lp.mlp_storage = "cpu"
            used["cpu"] += lp.mlp_bytes
            remaining["cpu"] -= lp.mlp_bytes

    if host_tier is not None and remaining["cpu"] < 0:
        plan.warnings.append(
            f"model overflows every tier by {_h(-remaining['cpu'])}; "
            f"use a smaller model, or add --kv-vram-fraction 0.15 to free VRAM")

    # ---- 5b. how host-resident weights get computed ----------------------
    # A layer left in host RAM can be copied to the GPU or computed in place.
    # Both are memory bound and read the same bytes, so the faster one is
    # simply whichever bus is wider: PCIe 5.0 x16 gives ~54 GB/s, dual-channel
    # DDR5-6000 gives ~70 GB/s of streaming reads. Computing in place also
    # leaves the GPU free rather than making it wait on a copy.
    for lp in placements:
        if lp.mlp_storage != "cpu":
            lp.mlp_exec = "gpu"
            continue
        if opts.host_exec == "stream":
            lp.mlp_exec = "gpu"
        elif opts.host_exec == "cpu":
            lp.mlp_exec = "cpu"
        else:
            link = next((t.link_bandwidth for t in tiers
                         if t.name == lp.exec_device), 25.0)
            lp.mlp_exec = "cpu" if opts.host_compute_gb_s > link else "gpu"

    # ---- 6. hot-expert cache --------------------------------------------
    # Whatever VRAM survived becomes an LRU cache for the experts that stayed
    # in host RAM. Routing is not uniform in practice, so a cache holding a
    # tenth of the experts serves noticeably more than a tenth of the reads.
    host_moe = [lp for lp in placements if lp.is_moe and lp.mlp_storage == "cpu"]
    if host_moe and gpu_tiers:
        for t in gpu_tiers:
            spare = max(0.0, remaining[t.name]) * opts.expert_cache_fraction
            if spare > 64 * MB:
                plan.expert_cache_bytes[t.name] = int(spare)
                remaining[t.name] -= spare
        cache_total = sum(plan.expert_cache_bytes.values())
        host_expert_bytes = sum(lp.mlp_bytes for lp in host_moe)
        if host_expert_bytes:
            raw = cache_total / host_expert_bytes
            # Mild routing skew: hot experts are hit more often than their
            # share. Capped at 1 -- never claim more than a full hit rate.
            hit = min(1.0, raw * 1.3)
            for lp in host_moe:
                lp.cached_expert_fraction = hit

    plan.layers = placements
    plan.bytes_per_tier = {k: int(v) for k, v in used.items()}
    plan.total_weight_bytes = int(sum(used.values()))
    _estimate(spec, plan, tiers, opts)
    return plan


def _estimate(spec: ModelSpec, plan: Plan, tiers: list[Tier],
              opts: PlannerOptions) -> None:
    by_name = {t.name: t for t in tiers}
    fastest_link = max((t.link_bandwidth for t in tiers if t.kind == "gpu"),
                       default=25.0)
    LAUNCH = 15e-6

    seconds = 0.0
    bytes_read = 0
    for lp in plan.layers:
        t = by_name.get(lp.exec_device)
        if t is None or t.kind != "gpu":
            # CPU execution: bound by DDR bandwidth.
            seconds += (lp.attn_bytes + lp.mlp_active_bytes) / (70.0 * 1e9)
            bytes_read += lp.attn_bytes + lp.mlp_active_bytes
            continue

        # attention
        if lp.attn_storage == "cpu":
            seconds += lp.attn_bytes / (min(t.link_bandwidth, fastest_link) * 1e9)
        else:
            seconds += lp.attn_bytes / (t.read_bandwidth * 1e9)
        bytes_read += lp.attn_bytes

        # MLP / experts
        active = lp.mlp_active_bytes
        if lp.mlp_storage == "cpu" and lp.mlp_exec == "cpu":
            # Computed where it lies: bounded by DDR bandwidth, plus a hidden
            # state crossing the bus in each direction (a few KB -- noise).
            seconds += active / (opts.host_compute_gb_s * 1e9)
            bytes_read += active
        elif lp.mlp_storage == "cpu":
            from_cache = active * lp.cached_expert_fraction
            from_host = active - from_cache
            t_copy = from_host / (t.link_bandwidth * 1e9)
            t_math = active / (t.read_bandwidth * 1e9)
            seconds += max(t_copy, t_math)      # prefetch overlaps compute
            bytes_read += int(from_host)
        else:
            seconds += active / (t.read_bandwidth * 1e9)
            bytes_read += active
        seconds += LAUNCH

    crossings = sum(1 for a, b in zip(plan.layers, plan.layers[1:])
                    if a.exec_device != b.exec_device)
    seconds += crossings * 60e-6
    plan.est_bytes_per_token = bytes_read
    plan.est_decode_tok_s = 1.0 / seconds if seconds > 0 else 0.0

    # prefill: compute-bound, and a streamed layer's weights are read once for
    # the whole batch rather than once per token
    pf = 0.0
    for lp in plan.layers:
        t = by_name.get(lp.exec_device)
        if t is None or t.kind != "gpu":
            continue
        layer = spec.layers[lp.index]
        table = TFLOPS.get(t.sm, TFLOPS[86])
        peak = table.get(lp.fmt) or table.get("bf16") or 60.0
        pf += (2.0 * layer.active_params) / (peak * 1e12 * 0.55)
    plan.est_prefill_tok_s = 1.0 / pf if pf > 0 else 0.0


# --------------------------------------------------------------------------
# outer search
# --------------------------------------------------------------------------


def auto_plan(spec: ModelSpec, rig: Rig,
              opts: Optional[PlannerOptions] = None,
              verbose: bool = False) -> tuple[Plan, list[dict]]:
    """Search the small space of sane configurations and keep the best.

    ``plan_placement`` answers "given these knobs, where does everything go".
    It cannot answer the two questions that actually decide throughput:

    * Should the second GPU be used at all? Pipelining a model that already
      fits on the 5090 onto the 3080 Ti makes single-stream decode *slower*,
      because the stages run in sequence and half of them now read at
      912 GB/s instead of 1790 GB/s. A second GPU earns its place only when it
      keeps weights out of host RAM.
    * How much VRAM should the KV cache get? Every gigabyte given to the cache
      is a gigabyte of weights pushed onto the PCIe bus, and a weight read
      over PCIe costs ~30x what it costs from VRAM. Past the point where one
      full context fits, more cache is close to worthless for a single stream.

    Both are answered by trying the handful of combinations and ranking them
    on estimated decode throughput, rejecting anything that cannot hold one
    full context or that overflows the machine.
    """
    base = opts or PlannerOptions()
    n_gpus = len([g for g in rig.gpus])
    candidates: list[tuple[Plan, dict]] = []
    trials: list[dict] = []

    kv_fractions = [0.06, 0.10, 0.15, 0.22, 0.30, 0.40, 0.55]
    gpu_counts = _gpu_counts(base.gpus, n_gpus)

    for used_gpus in gpu_counts:
        sub = _subset_rig(rig, used_gpus)
        for kvf in kv_fractions:
            o = PlannerOptions(**{**base.__dict__, "kv_vram_fraction": kvf})
            p = plan_placement(spec, sub, o)
            overflow = any("overflows every tier" in w for w in p.warnings)
            ctx_ok = p.kv_max_tokens >= base.max_model_len
            host_bytes = p.bytes_per_tier.get("cpu", 0)
            rec = {
                "gpus": used_gpus, "kv_fraction": kvf,
                "decode_tok_s": round(p.est_decode_tok_s, 2),
                "kv_tokens": p.kv_max_tokens,
                "host_bytes": host_bytes,
                "feasible": (not overflow) and ctx_ok,
                "overflow": overflow, "context_ok": ctx_ok,
            }
            trials.append(rec)
            if rec["feasible"]:
                candidates.append((p, rec))

    if not candidates:
        # Nothing satisfies every constraint. Report which constraint bit, and
        # by how much, rather than silently returning a plan that cannot run.
        fits = [t for t in trials if not t["overflow"]]
        if fits:
            best = max(fits, key=lambda t: t["decode_tok_s"])
            o = PlannerOptions(**{**base.__dict__,
                                  "kv_vram_fraction": best["kv_fraction"]})
            sub = _subset_rig(rig, best["gpus"])
            p = plan_placement(spec, sub, o)
            p.warnings.append(
                f"no configuration holds a full {base.max_model_len:,}-token "
                f"context; the cache was shortened to {p.kv_max_tokens:,} tokens "
                f"so the weights fit")
            return p, trials

        # It does not fit anywhere. Say what it would take.
        p = plan_placement(spec, rig, base)
        capacity = sum(t.capacity for t in p.tiers)
        short = p.total_weight_bytes - capacity
        need_bpw = capacity * 8 / max(1, spec.total_params)
        p.warnings.append(
            f"{spec.name} does not fit on this machine: {_h(p.total_weight_bytes)} "
            f"of weights against {_h(capacity)} of usable capacity, "
            f"{_h(max(0, short))} short.")
        p.warnings.append(
            f"it would need {need_bpw:.2f} bits per weight or less; the "
            f"formats here are {bits_per_weight('nvfp4'):.2f} (NVFP4) and "
            f"{bits_per_weight('int4_awq', group_size=base.group_size):.2f} (INT4). "
            f"Options: --host-fraction 0.95, a smaller model, or more RAM.")
        return p, trials

    best_plan, best_rec = max(candidates, key=lambda pr: pr[1]["decode_tok_s"])
    if best_rec["gpus"] < n_gpus:
        idle = [f"cuda:{g.index}" for g in
                sorted(rig.gpus, key=lambda g: -g.vram_bandwidth_gbps)[best_rec["gpus"]:]]
        best_plan.warnings.append(
            f"{', '.join(idle)} left idle on purpose: the model fits without it "
            f"and adding a slower stage to the pipeline would cost throughput. "
            f"Use it for a second model, or force it with --gpus all.")
    if verbose:
        best_plan.warnings.append(
            f"chose {best_rec['gpus']} GPU(s), kv_fraction={best_rec['kv_fraction']} "
            f"from {len(trials)} candidates")
    return best_plan, trials


def _subset_rig(rig: Rig, n_gpus: int) -> Rig:
    """A copy of the rig exposing only the ``n_gpus`` fastest boards."""
    return Rig(
        gpus=sorted(rig.gpus, key=lambda g: -g.vram_bandwidth_gbps)[:n_gpus],
        host=rig.host, cpu=rig.cpu, driver_version=rig.driver_version,
        cuda_version=rig.cuda_version, kernel=rig.kernel, distro=rig.distro,
        p2p_matrix=rig.p2p_matrix, source=rig.source,
    )


def _gpu_counts(spec: Optional[str], n_gpus: int) -> list[int]:
    """Which GPU counts the search is allowed to consider.

    ``auto`` (the default) tries every count and lets throughput decide, which
    is what leaves a slower second card idle when the model does not need it.
    ``all`` forces every card in -- useful when you would rather have the VRAM
    headroom for a longer context than the last few tokens per second.
    """
    if not n_gpus:
        return [0]
    if spec in (None, "", "auto"):
        return list(range(1, n_gpus + 1))
    if spec == "all":
        return [n_gpus]
    try:
        wanted = {int(x) for x in spec.replace(" ", "").split(",") if x != ""}
    except ValueError:
        raise ValueError(f"--gpus expects auto, all, or indices like 0,1; "
                         f"got {spec!r}") from None
    if not wanted:
        return [n_gpus]
    return [min(n_gpus, max(wanted) + 1)]
