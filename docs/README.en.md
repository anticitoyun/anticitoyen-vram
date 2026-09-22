<p align="center"><img src="logo-acvram.png" alt="acvram" width="420"></p>

# anticitoyen VRAM/RAM (`acvram`)

🌐 [Français](../README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Català](README.ca.md) · [Čeština](README.cs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Ελληνικά](README.el.md) · [English](README.en.md) · [Esperanto](README.eo.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Suomi](README.fi.md) · [עברית](README.he.md) · [हिन्दी](README.hi.md) · [Magyar](README.hu.md) · [Bahasa Indonesia](README.id.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk bokmål](README.nb.md) · [Nederlands](README.nl.md) · [Polski](README.pl.md) · [Português](README.pt.md) · [Română](README.ro.md) · [Русский](README.ru.md) · [Svenska](README.sv.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [中文](README.zh.md)

> Support: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)

An OpenAI-API-compatible inference gateway that treats memory as a hierarchy
and gives each GPU the numeric format its silicon reads best.

Designed for one specific machine:

| | |
|---|---|
| Processor | Intel Core i9-14900K (8 P-cores + 16 E-cores) |
| Motherboard | ASUS ROG Maximus Z790 Dark Hero |
| Memory | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| System | Ubuntu 26.04 LTS (CUDA 13); both cards on PCIe x8/x8, capped at 400 W / 275 W |

## The two ideas

**One format per GPU.** The RTX 5090 has FP4 tensor cores; the RTX 3080 Ti
has none, and no FP8 either. Aligning both on a common format would waste the
5090. The converter therefore writes *the same model twice*, in the format each
destination can actually exploit:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| weights | **NVFP4** — E2M1 + FP8 E4M3 scale every 16 | **INT4** — uint4 + fp16 scale and zero every 128 |
| bits per weight | 4.50 | 4.16 |
| versus BF16 | ×3.56 smaller | ×3.85 smaller |
| compute mode | FP4 tensor cores | dequantized to FP16 in the kernel, FP16 tensor cores |
| KV cache | INT8 | INT8 |

32 GB of VRAM at 4.5 bits per weight hold about **56 billion parameters**,
against 16 billion in BF16. Across both cards, that is roughly **78 billion
resident parameters** before touching system RAM at all.

**Memory is a hierarchy, not a wall.** Three tiers, and the planner measures
what each one costs instead of hoping the model fits:

```
RTX 5090     32 Go   ~1790 Go/s     NVFP4
RTX 3080 Ti  12 Go    ~912 Go/s     INT4
DDR5 hôte    96 Go   limité par le PCIe ou la DDR
```

## Quick start

```bash
./install.sh                       # environnement virtuel + torch cu128 + acvram
acvram doctor                      # cette machine est-elle prête, et pour quoi
acvram detect                      # qu'y a-t-il réellement ici

acvram plan  ~/modeles/Qwen3-32B                    # où irait chaque couche
acvram convert ~/modeles/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Any OpenAI client then plugs in:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"Bonjour"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="inutilise")
client.chat.completions.create(model="qwen3-32b",
                               messages=[{"role": "user", "content": "Bonjour"}])
```

## What `acvram plan` says

The planner is worth running before any download. It answers the questions
that decide whether a model is usable on this machine:

```
$ acvram plan ~/modeles/Llama-3.3-70B --max-model-len 32768 --max-seqs 4

  etage   format      capacite      poids         KV  tranche
  cuda:0  nvfp4        30,3 Gio   25,5 Gio   4,5 Gio  couches 0-58
  cuda:1  int4_awq     10,9 Gio    8,7 Gio   1,6 Gio  couches 59-79
  cpu     nvfp4        74,8 Gio    3,4 Gio       0 o  -

  poids au total     37,6 Gio
  lu par jeton       35,1 Gio
  KV par jeton       162,5 Kio  -> 39 843 jetons en cache
  MLP en RAM hote    55-58

  decodage estime    17,8 jetons/s  (lot de 1)
  prefill estime     847 jetons/s
```

It explores the configuration space instead of keeping the first one that
fits, and two of its decisions are counter-intuitive enough to be spelled out:

* **It leaves the 3080 Ti unused** when a model fits on the 5090 alone. The
  slices of a pipeline run in series: adding a 912 GB/s stage to a 1790 GB/s
  pipeline slows single-stream decoding. Force it with `--gpus all`.
* **It shrinks the KV cache to keep the weights in VRAM.** Every gigabyte
  given to the cache is a gigabyte of weights pushed onto the PCIe bus, and
  reading a weight over PCIe costs about thirty times what it costs from VRAM.
  On the 70B above, this single trade-off moves from 2.3 to 17.8 tokens/s.

## Going fast

Four optimizations, each verified by an equivalence proof and not just a
stopwatch: an optimization that changes the answer is a bug.

### Speculative decoding (`--speculative`)

Decoding one token with a batch of size 1 is memory-bound: the machine reads
every active weight to produce a single token. Verifying K proposed tokens
reads those same weights **once**. Two proposers:

* `ngram` (default) — looks for the current suffix earlier in the context and
  proposes what followed. Costs nothing, needs no model. Pays off when the
  output copies the input: code editing, RAG, summarization.
* `draft` — a small model on a second device. On this rig that device is the
  RTX 3080 Ti, which the planner deliberately leaves idle for any model that
  fits on the 5090.

Acceptance is exact, not approximate: a proposal is accepted with probability
`min(1, p/q)` and a rejection resamples from the normalized positive part of
`p - q`. Measured over 40,000 draws against a deliberately miscalibrated
draft, the emitted distribution stays within 0.002 total variation of the
target — speculation buys speed, never a different answer.

```
modele jouet, glouton, k=4    etapes   jetons/etape   sortie
  sans speculation                23           1,00   reference
  n-grammes                       13           1,77   identique
  brouillon (= cible)              5           4,60   identique
```

### Prefix cache (on by default)

Blocks are addressed by the *chained* hash of their token slice: two requests
that share a system prompt share its blocks, and the second no longer has to
precompute them. Chaining is essential: the same sixteen tokens in a different
context do not hold the same keys and values, and hashing the slice alone
would serve one sequence's cache to another.

A freed block whose content remains identifiable joins an LRU queue rather
than the free list: the cache thus survives across requests without ever
refusing an allocation it could have served.

### Host-tier compute (`--host-exec`)

A layer whose weights live in RAM can be copied to the GPU or computed in
place. Both paths are memory-bound and read the same bytes: the faster one is
the one with the wider bus — PCIe 5.0 x16 gives about 54 GB/s, dual-channel
DDR5 about 70 GB/s — and computing in place also leaves the GPU free instead
of making it wait for a copy.

This only pays if the processor reads the 4-bit packed weights directly. Hence
a small C++ kernel with an AVX2 path (`acvram_cpu.cpp`, loaded via ctypes, no
Python headers or ninja). Even on its **scalar** fallback branch, it beats
`dequantize() @ x` by a factor of 1.44 in INT4 and 3.21 in NVFP4, because the
latter first writes a 32-bit copy of the whole matrix.

On Mistral-Large-123B, the planner's estimate goes from 1.35 to
2.42 tokens/s.

### Mixed precision (`--snr-floor`, off by default)

The converter measures the signal-to-noise ratio at each layer's output for
every tensor and can promote to a wider format those that fall below
`--snr-floor`, within a limit of 15% of tensors and a price ceiling
(`--promotion-cout-max`, in added mebibytes).

The floor is **zero by default**: nothing is promoted. Decoding is bound by
memory bandwidth, and the measurement on `Huihui-Qwen3.8-27B` settles it — a
25 dB floor costs 13.4% of memory and 10.6% of throughput (18.50 GiB and
41.8 t/s against 16.02 and 46.2) for 2.0% of perplexity (42.591 against
43.447, 16,383-token corpus). `--snr-floor 25` restores the old behaviour when
quality matters more than speed.

### And `acvram eval`

Signal-to-noise ratio and logit cosine are approximations.
`acvram eval DIR [DIR ...]` measures sliding-window perplexity, so that a
format choice is settled on evidence:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  modele                   ppl     bpp      taille    jetons
  qwen3-32b-nvfp4        6,412    4,51    17,4 Gio      8192
  qwen3-32b-int4         6,583    4,17    16,1 Gio      8192  (+2,7 %)
```

## HTTP endpoints

| endpoint | notes |
|---|---|
| `POST /v1/chat/completions` | SSE stream or single response; uses the model's chat template |
| `POST /v1/completions` | prompt as text or token ids |
| `POST /v1/embeddings` | mean-pooled final hidden states, L2-normalized, `dimensions` honoured |
| `GET /v1/models` | plus an `acvram` block: formats, devices, KV cache capacity |
| `GET /health`, `GET /metrics` | decode throughput, KV block occupancy |

The field names in these responses stay in English: it is the OpenAI protocol,
and translating them would break every existing client.

## Where the numbers come from

Every value quoted above is produced by code in this repository and checked
by `pytest`. Measurements made on the CPU with the reference kernels:

| format | bits/weight | weight SNR | logit cosine vs BF16 |
|---|---|---|---|
| BF16 | 16.00 | — | 1.0000 |
| INT8 | 8.19 | 44.6 dB | 0.9998 |
| NVFP4 | 4.50 | 20.4 dB | 0.9664 |
| INT4 | 4.16 | 20.0 dB | 0.9427 |
| INT4 + Hadamard | 4.16 | 21.0 dB | 0.9582 |

Two findings from these measurements changed the defaults:

* **A Hadamard rotation helps INT4 and not NVFP4.** INT4's groups of 128
  cannot absorb an isolated outlier channel, so spreading the extreme values is
  worth an n log n transform per activation. NVFP4's blocks of 16 already
  carry their own scale. Hence `--hadamard auto`, which applies it to INT4
  only.
* **INT8 beats FP8 E4M3 for the KV cache**, 44 dB against 32 dB at the same
  size, because a per-(token, head) scale already provides the dynamic range
  FP8 spends exponent bits on. Both cards therefore use an INT8 KV cache, even
  though the 5090 could do FP8.

## Documentation

* [`REPRISE.md`](../REPRISE.md) — **resuming the project on another machine**
* [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — how the pieces fit together
* [`docs/MATERIEL.md`](MATERIEL.md) — tuning this specific machine
* [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) — **what is not done**, read first
* [`CONVENTIONS.md`](../CONVENTIONS.md) — working conventions for the code (language, style, checks before pushing)

## Measured results (22/09/2026, RTX 5090 at 400 W, ≥ 20 s regime on the energy meter)

Qwen3-Coder-30B-A3B in NVFP4 (experts) + INT8 (attention, head), same
protocol for every engine (`outils/`, one card, `energie.py`):

| | acvram 0.6.35 | vLLM 0.29 (CUTLASS FP4) | llama.cpp (sm_120) |
|---|---|---|---|
| decode, 12 sequences | **1,625.5 t/s** | 1,596.1 t/s | — |
| decode, 1 sequence | **380.8 t/s** | 290.6 t/s | 323.6 t/s |
| prefill pp2048 | **22,707 tokens/s** | 21,054 | 8,671 (TabbyAPI, withdrawn) |

Throughput of the day (station 1030, eco regime `-lgc 2700`, pipeline in
service; greedy sampling captured in the CUDA graph, on by default in 0.6.35).
The b=12 is the official sealed cell (median of 6 interleaved windows). The
vLLM 1,596.1 figure is a frozen reference from 21/09 (vLLM not replayed that
day): the +1.84 % gap holds at equal reference, not as a re-measurement of both
the same morning. The J/token is still being re-measured
(`outils/gpu/mesure/banc-4moteurs.py`) — a figure without a regime is not
published.

On the morning of 14/09 acvram was at 630 t/s and 0.619 J/token on the same
cell: the gains come from Blackwell's native FP4 MMA
(`mma.sync … kind::mxf4nvf4`, ×7.9 over bf16), from MoE as grouped GEMM per
batch bucket, from single-kernel routing (3,677 → 1,517 launches per step)
and from a narrow tensor-core GEMM for the projections. Every figure has its
note in `acvram-memoire/revue/` with the prediction sealed before the
measurement, the instrument and its regime — a figure without a regime is not
published.

Where acvram is ahead: MLA models (GLM-4.7-Flash) in native sm_120 NVFP4,
which vLLM only serves in FP8 (b=1: 165.35 t/s in service); models that do
not fit in VRAM; and, since 0.6.35, large-batch decoding of a MoE that fits in
VRAM — b=12 rises from 1,540 (0.6.34) to 1,625.5 t/s, i.e. +1.84 % ahead of the
frozen vLLM reference (1,596.1). The gap stays narrow and at a frozen reference;
the energy gap is to be re-measured.

## Status

Version 0.6.35. Everything runs on the 5090: CUDA kernels compiled for
`sm_120a` (native FP4) and `sm_86`, CUDA graphs, NVFP4/INT8/INT4
quantization, HTTP server. Guard rails in place: the card is invisible to
working sessions (`CUDA_VISIBLE_DEVICES` empty) and only `outils/carte.sh`
lends it, under lock, to one measurement at a time; a watcher logs every
access outside the lock; an energy measurement covering more than one card or
less than 10 s is invalidated; a model loaded in degraded mode says so and does
not enter a duel.

640 tests (`pytest -q`, one minute on the CPU; GPU tests only run under
`carte.sh`). Work tracking: `acvram-memoire/` (rules, directory, notebooks,
a review of 180 notes).

## Support

acvram is developed on personal hardware. If the project is useful to you:
**Support: [buymeacoffee.com/anticitoyen](https://buymeacoffee.com/anticitoyen)**.

## License

GPL-3.0 or later.
