# What is and is not done

Written honestly, because the project this replaces claimed a working GPU
memory extension and shipped a kernel module that printed the numbers you
passed it on the command line.

## Done since 0.1.0

| | what it does | verified by |
|---|---|---|
| Speculative decoding | n-gram and draft-model proposers, exact acceptance | output identity + a 40k-draw distribution test |
| Prefix caching | chained-hash block reuse with LRU eviction | output identity, prefill-token accounting |
| Host-tier CPU compute | AVX2/scalar 4-bit GEMV, ctypes, no build deps | numerics against the reference, compiled and run here |
| Rewritten CUDA GEMV | 8-byte vector loads, 4 rows/block, split-K | nibble decode validated on CPU; **kernel itself unbuilt** |
| FP4 tensor-core GEMM | probed `torch._scaled_mm` path with fallback | probe reports its own reason in `acvram doctor` |
| Mixed precision | SNR-driven promotion, capped at 15% of tensors | promotion raises measured SNR, cap enforced |
| Perplexity harness | `acvram eval`, sliding window | runs, finite, ranks |
| Offset causal mask | correct attention for chunked/cached prefill | shown to differ from the built-in flag |
| Batched decode attention | one SDPA call per step instead of one per sequence | equals the per-sequence result |

The offset causal mask deserves a note: `F.scaled_dot_product_attention(
is_causal=True)` aligns its triangle to the top left, which is only correct
when the query covers the whole sequence. Prefix caching and chunked prefill
both produce an offset query block, and the built-in flag would have masked
the wrong cells — silently, with plausible-looking output. There is a test
asserting the two disagree.

## Still not run on the target hardware

Everything here was written and tested on a laptop with an i5-3230M and a
GeForce GT 740M on driver 470 — no CUDA toolkit, no Blackwell, no Ampere.

* **The CUDA kernels have never been compiled.** `acvram_kernels.cu` matches
  the PyTorch reference numerics by construction and its trickiest part (the
  nibble decode) was validated by compiling that function alone on the CPU,
  but nvcc has never seen the file. Expect to fix compilation errors on first
  build. The reference path is numerically identical, so nothing is broken
  meanwhile — only slow.
* **No measured throughput exists.** Every tok/s figure is the planner's
  estimate from nameplate bandwidths. `acvram bench` replaces them.
* **The AVX2 CPU path has never executed.** It compiles, and the scalar path
  that shares its structure is tested, but this laptop's CPU predates AVX2.
* **The FP4 tensor-core path has never bound.** It probes `torch._scaled_mm`
  at runtime and reports why it failed; on a real 5090 with a recent torch it
  may bind, or the layout may need adjusting in `_swizzle_scales`.

## Remaining work, roughly in order of value

1. **A paged-attention CUDA kernel.** Decode still gathers each sequence's
   cache into a dense tensor before attending. That dequantizes the whole
   context per layer per step. A kernel reading the block table directly would
   remove it. This is now the largest structural inefficiency left.
2. **Hot-expert LRU cache.** `tiering.py` reserves VRAM for it and models its
   hit rate; nothing implements it. An MoE model still re-fetches every routed
   expert per token.
3. **Chunked prefill.** The machinery exists — the offset mask and partial
   prefill both work — but the scheduler does not split a long prompt, so a
   very long prefill still blocks decoding.
4. **EAGLE-style speculation.** The n-gram proposer is free but only helps on
   repetitive output; a trained draft head would raise acceptance on open-ended
   text without needing a separate model.
5. **Grouped GEMM for MoE.** The expert loop is Python, one call per unique
   routed expert.
6. **GPTQ-style error compensation.** Calibration is AWQ's cheap variant; there
   is no second-order weight update and no propagation of a layer's error into
   the next layer's calibration.

## Known limitations

* **No authentication, no rate limiting.** Bind to `127.0.0.1` (the default).
* **No tool calling, no structured output.** `tools` and `response_format` are
  accepted and ignored.
* **`n > 1` produces one completion.**
* **Model coverage** is llama-family dense and MoE: RMSNorm, RoPE, GQA, SwiGLU.
  That covers Llama, Mistral, Qwen2/3, Mixtral, DeepSeek. Not covered: sliding-
  window attention, Mamba/hybrid blocks, MLA, vision towers.
* **Perplexity on an untrained model is meaningless.** The harness reports
  near-uniform perplexity for random weights, which is correct and also a
  reminder that it only discriminates on a real checkpoint.

## First hour on the real machine

```bash
acvram doctor                    # what built, what did not, and why
acvram bench --what bandwidth    # the two PCIe links and DDR; feeds --host-gb-s
acvram bench --what kernels      # CUDA and CPU kernels against the reference
pytest -q                        # the reference path must still pass
acvram plan ~/models/Qwen3-32B   # does the plan match docs/HARDWARE.md
```
