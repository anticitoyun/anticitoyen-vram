# Architecture

## The path a model takes

```
  HF checkpoint (safetensors, bf16)
        |
        |  acvram plan     ->  where does every tensor go?
        v
  placement plan  ---------------------------------+
        |                                          |
        |  acvram convert  ->  quantize per        |
        v                      destination device  |
  acvram shards + manifest.json                    |
        |                                          |
        |  acvram serve                            |
        v                                          v
  loader ------> placed model ------> engine ------> OpenAI HTTP
```

The plan is computed **before** conversion and stored **inside** the converted
model's manifest. That is what removes an entire class of bug: at load time
there is never a question of whether this GPU can read this tensor, because
the tensor was written for that GPU.

## The planner

`memory/tiering.py`. Given a `ModelSpec` and a `Rig`, it produces a `Plan`.

It works from a memory-bound model of decode: producing one token means
reading every *active* weight once, so the time is

```
  sum over layers of:
      resident   ->  active_bytes / vram_bandwidth
      streamed   ->  max(active_bytes / pcie_bandwidth,
                         active_bytes / vram_bandwidth)      # prefetch overlaps
  + one host-staged hop per GPU boundary crossing
```

`active_bytes` is the whole layer for a dense block, and only the routed
experts for an MoE block — which is the entire reason a 235B MoE and a 70B
dense model behave so differently on the same machine.

Four decisions, in order:

1. **KV cache budget.** Taken first, because it scales with traffic while
   weights do not.
2. **Pipeline stages.** Contiguous layer ranges per GPU, sized so the pipeline
   crosses a GPU boundary exactly once. GeForce boards have no NVLink and
   NVIDIA disables PCIe P2P on them, so a crossing stages through pinned host
   memory — cheap for one hidden state per token, ruinous if it happened per
   layer.
3. **Attention placement.** Pinned to its stage's GPU whenever it fits. It is
   small and it is on the latency-critical path.
4. **MLP and experts.** Fill VRAM front to back; the remainder goes to host
   RAM. Leftover VRAM becomes an LRU cache for hot experts.

`auto_plan` then searches over (number of GPUs used) × (KV cache fraction) and
ranks by estimated decode throughput, rejecting anything that overflows or
cannot hold one full context.

## Formats

| | NVFP4 | INT4 |
|---|---|---|
| element | E2M1 (4 bit) | uint4 |
| scale | FP8 E4M3, one per 16 | FP16, one per 128 |
| zero point | none (symmetric) | uint4, one per 128 |
| global | FP32 per tensor | none |
| bits/weight | 4.50 | 4.156 |
| dequantized as | `level * e4m3(block) * global` | `(q - zero) * scale` |

The FP32 global scale in NVFP4 exists because E4M3 saturates at 448: it is
chosen as `amax / (6 * 448)` so the largest block scale lands exactly on that
ceiling, whatever the tensor's dynamic range.

## Kernels

Two entry points per format:

* `*_dequant` — materialise the matrix in a compute dtype, then hand it to
  cuBLAS. This is the **prefill** path: the dequantization cost is amortised
  over the batch, and cuBLAS beats anything hand-rolled.
* `*_gemv` — fused dequantize-and-multiply for one to eight rows. This is the
  **decode** path, purely memory bound; the point is to read 4-bit weights
  from global memory and never write a 16-bit copy.

The threshold between them is `gemv_threshold=8` in `kernels/__init__.py`.

`kernels/__init__.py` compiles the extension on first use, emitting code for
exactly the architectures present. If the build fails for any reason it warns
once and falls back to the PyTorch reference — slow, but numerically identical
and enough for the test suite to run anywhere.

## Engine

`engine/runner.py` runs a continuous batch: at each step it admits whatever
new requests the free KV blocks can pay for, prefills them one at a time (a
long prompt mixed into a decode batch would stall everything behind it),
decodes one token for everything running, and frees a sequence's blocks the
moment it stops.

It is single-threaded on purpose. The model's layers are spread across two
GPUs and host memory and a step touches them in sequence; threads would
contend without adding parallelism. Concurrency comes from batching. The
asyncio server bridges to it with one background thread and per-request
queues.

## Streaming weights

`engine/layers.py:StreamedWeight`. Host-resident weights live in **pinned**
memory — pageable memory would force the driver to stage the copy
synchronously and the overlap would vanish — and are copied on a side CUDA
stream into a double buffer. `ACVRamModel.forward` starts layer *i+1*'s
transfer before running layer *i*, so a streamed layer costs
`max(copy, compute)` rather than their sum. That is exactly what the planner's
cost model assumes; if you change one, change the other.

## Prefix caching

`memory/kvcache.py:BlockAllocator` is both the free list and the prefix cache,
because they compete for the same blocks.

A block's contents are determined by the tokens that produced it *and* every
token before them, so blocks are addressed by a chained hash:

```
h_0 = hash((0,     tokens[0:16]))
h_1 = hash((h_0,   tokens[16:32]))
h_i = hash((h_{i-1}, tokens[16i:16i+16]))
```

Chaining is not decoration. The same 16 tokens appearing in two different
contexts do not produce the same keys and values, because attention saw
different history; hashing the span alone would happily serve one sequence's
cache to another.

Only **complete** blocks are published. A half-filled block matched by a hash
naming content it does not hold yet would hand a later request keys and values
that were never written.

On release, a block whose contents are still identifiable goes to the back of
an LRU queue instead of the free list, and is recycled only when the pool runs
dry — so the cache survives between requests without ever refusing an
allocation it could have served.

One block is always held back from a match: a request whose prompt is entirely
cached still needs a token to run through the model, or there is nothing to
produce logits from.

## Speculative decoding

`engine/speculative.py`. The verification batch is `[last produced token] +
[K proposals]`, fed at absolute positions `n-1 .. n+K-1`. Feeding the last
produced token is not overhead: its keys and values were never written,
because a token only enters the cache when it is fed. So K+1 positions give
exactly the K+1 predictions needed.

Rejected positions leave stale entries in the cache beyond the sequence's
length. Nothing reads past `seq_len`, and the next step overwrites them.

Acceptance uses the standard rejection rule. For a proposer with no
distribution (n-gram), q is a point mass at the proposal, so the accept
probability is `p(x)` and a rejection resamples from `p` with that token
removed — which is exact, not an approximation.

## The host tier as a compute device

`DecoderLayer` carries two devices: attention runs on `self.device`, the MLP on
`self.mlp_device`. When the planner decides a host-resident MLP is better
computed in place, only the hidden state crosses the bus — `[tokens, hidden]`,
a few kilobytes per decoded token against gigabytes of weights.

The decision is made in `plan_placement`, comparing the executing GPU's
measured host link against `PlannerOptions.host_compute_gb_s`. `acvram bench
--what bandwidth` prints both numbers and the resulting recommendation.

## Kernel layout

Three backends, all numerically identical:

| backend | file | when |
|---|---|---|
| CUDA | `acvram_kernels.cu` | weights on a GPU |
| CPU | `acvram_cpu.cpp` (ctypes) | weights in host RAM |
| reference | `quant/*.py` | anything that failed to build |

The CUDA GEMV gets its speed from three things: `uint2` loads carrying 16
packed weights (exactly one NVFP4 scale block, and a whole number of INT4
groups, so a thread never straddles a scale boundary); four output rows per
block, so the activation slice is read once and reused; and split-K when the
matrix is too short to fill the device, which the small projections of a GQA
attention block always are.
