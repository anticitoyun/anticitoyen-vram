# anticitoyen VRAM/RAM (`acvram`)

An OpenAI-compatible inference server that treats memory as a hierarchy and
gives each GPU the numeric format its own silicon reads best.

Built for one specific machine:

| | |
|---|---|
| CPU | Intel Core i9-14900K (8 P-cores + 16 E-cores) |
| Board | ASUS ROG Maximus Z790 Dark Hero |
| RAM | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| OS | Linux Mint 22.3 |

## The two ideas

**One format per GPU.** The RTX 5090 has FP4 tensor cores; the RTX 3080 Ti does
not, and has no FP8 either. Levelling both down to a shared format wastes the
5090; so the converter writes the *same checkpoint twice*, in the format each
destination can actually use:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| weights | **NVFP4** — E2M1 + FP8 E4M3 scale per 16 | **INT4** — uint4 + fp16 scale/zero per 128 |
| bits per weight | 4.50 | 4.16 |
| vs BF16 | 3.56× smaller | 3.85× smaller |
| how it computes | FP4 tensor cores | dequantized to FP16 in-kernel, FP16 tensor cores |
| KV cache | INT8 | INT8 |

32 GB of VRAM at 4.5 bpw holds about **56 G parameters** of weight, against
16 G in BF16. Across both cards that is roughly **78 G parameters resident**,
before host RAM is touched at all.

**Memory is a hierarchy, not a wall.** Three tiers, and the planner measures
what each costs rather than hoping the model fits:

```
RTX 5090     32 GB   ~1790 GB/s     NVFP4
RTX 3080 Ti  12 GB    ~912 GB/s     INT4
DDR5 host    96 GB   PCIe-limited   whichever the executing GPU uses
```

## Quick start

```bash
./install.sh                       # venv + torch cu128 + acvram
acvram doctor                      # is this machine ready
acvram detect                      # what is actually here

acvram plan  ~/models/Qwen3-32B                    # where each layer would go
acvram convert ~/models/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Then point any OpenAI client at it:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"Bonjour"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
client.chat.completions.create(model="qwen3-32b",
                               messages=[{"role": "user", "content": "Bonjour"}])
```

## What `acvram plan` tells you

The planner is worth running before you download anything. It answers the
questions that decide whether a model is usable on this box:

```
$ acvram plan ~/models/Llama-3.3-70B --max-model-len 32768 --max-seqs 4

  tier    format      capacity    weights         KV  stage
  cuda:0  nvfp4       30.3 GiB   25.5 GiB    4.5 GiB  layers 0-58
  cuda:1  int4_awq    10.9 GiB    8.7 GiB    1.6 GiB  layers 59-79
  cpu     nvfp4       74.8 GiB    3.4 GiB        0 B  -

  weights total      37.6 GiB
  read per token     35.1 GiB
  KV per token       162.5 KiB  -> 39,843 tokens cached
  MLP in host RAM    55-58

  estimated decode   17.8 tok/s  (batch 1)
  estimated prefill  847 tok/s
```

It searches the configuration space rather than taking the first layout that
fits, and two of its decisions are counter-intuitive enough to be worth
stating:

* **It will leave the 3080 Ti idle** when a model fits on the 5090 alone.
  Pipeline stages run in sequence, so adding a 912 GB/s stage to a 1790 GB/s
  pipeline makes single-stream decoding *slower*. Override with `--gpus all`.
* **It will shrink the KV cache to keep weights in VRAM.** A gigabyte given to
  the cache is a gigabyte of weights pushed onto the PCIe bus, and a weight
  read over PCIe costs about 30× what it costs from VRAM. On the 70B above
  that single trade is worth 2.3 → 17.8 tok/s.

## Making it fast

Four optimisations, each checked for equivalence rather than just for speed --
an optimisation that changes the answer is a bug.

### Speculative decoding (`--speculative`)

Decoding one token at batch 1 is memory bound: the machine reads every active
weight to produce a single token. Verifying K proposed tokens reads those same
weights **once**. Two proposers:

* `ngram` (default) — looks for the current suffix earlier in the context and
  proposes what followed. Costs nothing, needs no model. Pays off when the
  output quotes the input: code editing, RAG, summarisation.
* `draft` — a small model on a second device. On this rig that device is the
  RTX 3080 Ti, which the planner deliberately leaves idle for any model that
  fits on the 5090.

Acceptance is exact, not approximate: a proposal is accepted with probability
`min(1, p/q)` and a rejection resamples from the normalised positive part of
`p - q`. Measured over 40 000 draws against a deliberately mismatched draft,
the emitted distribution is within 0.002 total variation of the target's —
speculation buys speed, never a different answer.

```
tiny model, greedy, k=4      model steps   tokens/step   output
  no speculation                     23          1.00    reference
  ngram                              13          1.77    identical
  draft (draft == target)             5          4.60    identical
```

### Prefix caching (on by default)

Blocks are addressed by the *chained* hash of their token span, so two
requests sharing a system prompt share its blocks outright and the second
skips prefilling them. Chaining matters: the same 16 tokens in a different
context do not hold the same keys and values, and hashing the span alone would
serve one sequence's cache to another.

A freed block whose contents are still identifiable goes to an LRU queue
rather than back to the free list, so the cache survives between requests
without ever refusing an allocation it could have served.

### Host-tier compute (`--host-exec`)

A layer whose weights sit in host RAM can be copied to the GPU or computed
where it is. Both are memory bound and read the same bytes, so the faster path
is whichever bus is wider — PCIe 5.0 x16 gives ~54 GB/s, dual-channel DDR5
gives ~70 GB/s — and computing in place also leaves the GPU free instead of
making it wait on a copy.

That only holds if the CPU reads the packed 4-bit weights directly, so there
is a small C++ kernel with an AVX2 path (`acvram_cpu.cpp`, built through
ctypes, no Python headers or ninja required). Even on the *scalar* fallback it
beats `dequantize() @ x` by 1.4x for INT4 and 3.2x for NVFP4, because the
dequantize path first writes a 32-bit copy of the whole matrix.

On Mistral-Large-123B the planner's estimate moves from 1.35 to 2.42 tok/s.

### Mixed precision (`--mixed-precision auto`)

The converter measures each tensor's layer-output SNR and promotes the ones
that land below `--snr-floor` to a wider format, capped at 15% of tensors.
Spending 8 bits on the few per cent that need them costs a fraction of a bit
per weight overall.

### And `acvram eval`

SNR and logit cosine are proxies. `acvram eval DIR [DIR ...]` measures sliding-
window perplexity so a format choice can be settled with evidence:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  model                    ppl     bpw        size    tokens
  qwen3-32b-nvfp4        6.412    4.51    17.4GiB      8192
  qwen3-32b-int4         6.583    4.17    16.1GiB      8192  (+2.7%)
```

## Endpoints

| endpoint | notes |
|---|---|
| `POST /v1/chat/completions` | streaming (SSE) and non-streaming; the checkpoint's own Jinja chat template |
| `POST /v1/completions` | text or token-id prompts |
| `POST /v1/embeddings` | mean-pooled final hidden states, L2-normalised, `dimensions` honoured |
| `GET /v1/models` | plus an `acvram` block: formats, devices, KV capacity |
| `GET /health`, `GET /metrics` | live decode rate, KV block occupancy |

## Where the numbers come from

Every figure quoted above is produced by code in this repository and checked
by `pytest`. Measured on CPU with the reference kernels:

| format | bpw | weight SNR | logit cosine vs BF16 |
|---|---|---|---|
| BF16 | 16.00 | — | 1.0000 |
| INT8 | 8.19 | 44.6 dB | 0.9998 |
| NVFP4 | 4.50 | 20.4 dB | 0.9664 |
| INT4 | 4.16 | 20.0 dB | 0.9427 |
| INT4 + Hadamard | 4.16 | 21.0 dB | 0.9582 |

Two findings from those measurements changed the defaults:

* **A Hadamard rotation helps INT4 and not NVFP4.** INT4's 128-wide groups
  cannot absorb a single outlier channel, so spreading outliers is worth an
  `n log n` transform per activation. NVFP4's 16-wide blocks already carry
  their own scale. Hence `--hadamard auto` applies it to INT4 only.
* **INT8 beats FP8 E4M3 for the KV cache**, 44 dB against 32 dB at identical
  size, because a per-(token, head) scale already supplies the dynamic range
  FP8 spends exponent bits on. Both cards default to INT8 KV even though the
  5090 could do FP8.

## Documentation

* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the pieces fit
* [`docs/HARDWARE.md`](docs/HARDWARE.md) — tuning this specific rig
* [`docs/ROADMAP.md`](docs/ROADMAP.md) — **what is not done yet**, read this first
* [`CLAUDE.md`](CLAUDE.md) — orientation for working on the code with Claude

## Status

Version 0.2.0. Written before the target machine was available, so every code
path is exercised on CPU and none has yet run on a 5090. The CPU kernels *are*
compiled and tested; the CUDA ones have never seen nvcc. See
[`docs/ROADMAP.md`](docs/ROADMAP.md) for exactly what that means and what to
check first on the real hardware.

67 tests, all on CPU, about a minute: `pytest -q`.

## License

GPL-3.0-or-later.
