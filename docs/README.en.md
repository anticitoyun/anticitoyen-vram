<p align="center">
  <img src="../docs/logo-acvram.png" alt="acvram" width="200">
</p>

# anticitoyen VRAM/RAM (`acvram`)

<p align="center">
  <a href="https://github.com/anticitoyun/anticitoyen-vram/releases/latest"><img src="https://img.shields.io/github/v/release/anticitoyun/anticitoyen-vram" alt="Release"></a>
  <a href="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml"><img src="https://github.com/anticitoyun/anticitoyen-vram/actions/workflows/tests.yml/badge.svg" alt="CI"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/licence-GPL--3.0--or--later-blue.svg" alt="Licence GPL-3.0-or-later"></a>
  <a href="https://buymeacoffee.com/anticitoyen"><img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-support-FFDD00?logo=buymeacoffee&logoColor=black" alt="Buy Me a Coffee"></a>
</p>

An OpenAI-API-compatible inference gateway that treats memory as a hierarchy, gives each GPU the numeric format its silicon reads best, and optimizes every token in joules as much as in seconds.

<div align="center">

[🇫🇷 Français](../README.md) · **🇬🇧 English** · [🇸🇦 العربية](README.ar.md) · [🇧🇩 বাংলা](README.bn.md) · [🇪🇸 Català](README.ca.md) · [🇨🇿 Čeština](README.cs.md) · [🇩🇰 Dansk](README.da.md) · [🇩🇪 Deutsch](README.de.md) · [🇬🇷 Ελληνικά](README.el.md) · [🌐 Esperanto](README.eo.md) · [🇪🇸 Español](README.es.md) · [🇮🇷 فارسی](README.fa.md) · [🇫🇮 Suomi](README.fi.md) · [🇮🇱 עברית](README.he.md) · [🇮🇳 हिन्दी](README.hi.md) · [🇭🇺 Magyar](README.hu.md) · [🇮🇩 Bahasa Indonesia](README.id.md) · [🇮🇹 Italiano](README.it.md) · [🇯🇵 日本語](README.ja.md) · [🇰🇷 한국어](README.ko.md) · [🇳🇴 Norsk bokmål](README.nb.md) · [🇳🇱 Nederlands](README.nl.md) · [🇵🇱 Polski](README.pl.md) · [🇵🇹 Português](README.pt.md) · [🇷🇴 Română](README.ro.md) · [🇷🇺 Русский](README.ru.md) · [🇸🇪 Svenska](README.sv.md) · [🇹🇭 ไทย](README.th.md) · [🇹🇷 Türkçe](README.tr.md) · [🇺🇦 Українська](README.uk.md) · [🇻🇳 Tiếng Việt](README.vi.md) · [🇨🇳 中文](README.zh.md)

</div>

<p align="center"><img src="captures/resultats-22-09.png" alt="Throughput and energy comparison against vLLM and llama.cpp" width="720"></p>

---

## Contents

- [Two ideas](#idees)
- [Quick start](#demarrage)
- [Install](#installer)
- [What `acvram plan` says](#plan)
- [Going fast](#optimisations)
- [HTTP endpoints](#http)
- [Where the numbers come from](#chiffres)
- [Documentation](#documentation)
- [Measured results](#resultats)
- [Status](#etat)
- [Credits](#credits)
- [Licence](#licence)
- [Support the project](#soutien)

---

<a id="idees"></a>

## Two ideas

Built for one specific machine:

| | |
|---|---|
| CPU | Intel Core i9-14900K (8 P-cores + 16 E-cores) |
| Motherboard | ASUS ROG Maximus Z790 Dark Hero |
| Memory | 96 GB DDR5 |
| GPU 0 | ASUS RTX 5090 Astral LC OC, 32 GB — Blackwell, `sm_120` |
| GPU 1 | ASUS RTX 3080 Ti, 12 GB — Ampere, `sm_86` |
| System | Ubuntu 26.04 LTS (CUDA 13); both cards at PCIe x8/x8, power-capped 400 W / 275 W |

**One format per GPU.** The RTX 5090 has FP4 tensor cores; the RTX 3080 Ti has neither those nor FP8. Aligning both on a common format would waste the 5090. So the converter writes *the same model twice*, in the format each destination can actually exploit:

| | RTX 5090 | RTX 3080 Ti |
|---|---|---|
| weights | **NVFP4** — E2M1 + FP8 E4M3 scale every 16 | **INT4** — uint4 + fp16 scale and zero-point every 128 |
| bits per weight | 4.50 | 4.16 |
| vs. BF16 | ×3.56 smaller | ×3.85 smaller |
| compute mode | FP4 tensor cores | dequantised to FP16 inside the kernel, FP16 tensor cores |
| KV cache | INT8 | INT8 |

32 GB of VRAM at 4.5 bits per weight holds roughly **56 billion parameters**, versus 16 billion in BF16. Across both cards, that is approximately **78 billion resident parameters** before touching host memory at all.

**Memory is a hierarchy, not a wall.** Three tiers, and the planner measures what each one actually costs instead of hoping the model fits:

```
RTX 5090     32 GB   ~1790 GB/s     NVFP4
RTX 3080 Ti  12 GB    ~912 GB/s     INT4
Host DDR5    96 GB   limited by PCIe or DDR
```

---

<a id="demarrage"></a>

## Quick start

```bash
./install.sh                       # virtual environment + torch cu128 + acvram
acvram doctor                      # is this machine ready, and for what
acvram detect                      # what is actually here

acvram plan  ~/models/Qwen3-32B                     # where each layer would go
acvram convert ~/models/Qwen3-32B -o ~/acv/qwen3-32b
acvram serve ~/acv/qwen3-32b --port 8000
```

Any OpenAI client plugs in right away:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-32b","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
client.chat.completions.create(model="qwen3-32b",
                               messages=[{"role": "user", "content": "Hello"}])
```

---

<a id="installer"></a>

## Install

From source (any platform):

```bash
git clone https://github.com/anticitoyun/anticitoyen-vram.git && cd anticitoyen-vram
./install.sh
```

Or by package, one file attached to each [GitHub release](https://github.com/anticitoyun/anticitoyen-vram/releases/latest):

| Channel | File attached to the release | Command |
|---|---|---|
| Debian / Ubuntu (.deb) | `acvram_<version>_amd64.deb` | `sudo dpkg -i acvram_<version>_amd64.deb` |
| Arch (AUR) | `aur-<version>.tar.gz` (PKGBUILD + .SRCINFO) | `tar xzf aur-<version>.tar.gz && cd acvram && makepkg -si` |
| Fedora / COPR (RPM) | `.rpm` / `.src.rpm` (names generated by `rpmbuild`, not fixed) | `sudo rpm -i acvram-<version>-1.*.noarch.rpm` (or `rpmbuild --rebuild *.src.rpm` from the `.src.rpm`) |
| Flatpak | `acvram-<version>.flatpakref` (OSTree, GitHub Pages) | `flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v<version>/acvram-<version>.flatpakref` |

A `.flatpakref` always installs the latest published version of the repository.

Before installing, verify the downloaded file against the sums attached to the release (`SHA256SUMS`,
published once every other file is present):

```bash
curl -LO https://github.com/anticitoyun/anticitoyen-vram/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing
```

Pip is not published as a package (no wheel built): `pip install -e '.[dev]'` installs from a source clone, same as `./install.sh`.

---

<a id="plan"></a>

## What `acvram plan` says

The planner deserves to be run before any download. It answers the questions that decide whether a model is usable on this machine at all:

```
$ acvram plan ~/models/Llama-3.3-70B --max-model-len 32768 --max-seqs 4

  tier    format      capacity      weights       KV  slice
  cuda:0  nvfp4        30.3 GiB    25.5 GiB   4.5 GiB  layers 0-58
  cuda:1  int4_awq     10.9 GiB     8.7 GiB   1.6 GiB  layers 59-79
  cpu     nvfp4        74.8 GiB     3.4 GiB      0 B   -

  total weights      37.6 GiB
  read per token     35.1 GiB
  KV per token       162.5 KiB  -> 39,843 tokens cached
  MLP on host RAM    55-58

  decode estimate    17.8 tokens/s  (batch of 1)
  prefill estimate   847 tokens/s
```

It explores the configuration space instead of stopping at the first one that fits, and two of its decisions are counter-intuitive enough to spell out:

* **It leaves the 3080 Ti idle** when a model fits on the 5090 alone. A pipeline's stages run in series: adding a 912 GB/s stage to a 1790 GB/s pipeline slows down single-stream decode. Force it with `--gpus all`.
* **It shrinks the KV cache to keep the weights in VRAM.** Every gigabyte given to the cache is a gigabyte of weights pushed onto the PCIe bus, and reading a weight over PCIe costs roughly thirty times what it costs from VRAM. On the 70B above, this one trade-off alone takes decode from 2.3 to 17.8 tokens/s.

---

<a id="optimisations"></a>

## Going fast

Four optimisations, each verified by a proof of equivalence and not just a stopwatch: an optimisation that changes the answer is a bug.

Dense models' NVFP4 linears go through the Marlin layout by default (+57 to +90% throughput at b = 8, TTFT +2 to +4 ms per revue/poste6-piece147-verdict-24-09.md; fallback `ACVRAM_PROJ_MARLIN=0`, see [CHANGELOG.md](../CHANGELOG.md)).

### Speculative decoding (`--speculative`)

Decoding one token with a batch of size 1 is memory-bound: the machine reads every active weight to produce a single token. Checking K proposed tokens reads those same weights **only once**. Two proposers:

* `ngram` (default) — looks for the current suffix earlier in the context and proposes what followed it. Costs nothing, needs no model. Pays off when the output echoes the input: code editing, RAG, summarisation.
* `draft` — a small model on a second device. On this rig that device is the RTX 3080 Ti, which the planner deliberately leaves idle for any model that fits on the 5090.

`mtp` (the model's `nextn` head) and `auto` also exist; not worthwhile as they stand and not enabled by default — see `docs/ARCHITECTURE.md`.

Acceptance is exact, not approximate: a proposal is accepted with probability `min(1, p/q)`, and a rejection resamples from the normalised positive part of `p - q`. Measured over 40,000 draws against a deliberately miscalibrated draft, the emitted distribution stays within 0.002 total variation of the target — speculation buys speed, never a different answer.

```
toy model, greedy, k=4        steps   tokens/step   output
  no speculation                 23          1.00   reference
  n-grams                        13          1.77   identical
  draft (= target)                5          4.60   identical
```

### Prefix cache (on by default)

Blocks are addressed by the *chained* hash of their token slice: two requests sharing a system prompt share its blocks, and the second no longer has to precompute them. Chaining is essential: the same sixteen tokens in a different context do not hold the same keys and values, and hashing the slice alone would serve one sequence's cache to another.

A freed block whose content stays identifiable joins an LRU queue rather than the free-block list: the cache survives across requests this way without ever refusing an allocation it could have served.

### Host-tier compute (`--host-exec`)

A layer whose weights live in RAM can be copied to the GPU or computed in place. Both paths are memory-bound and read the same bytes: the faster one is whichever has the wider bus — PCIe 5.0 x16 gives roughly 54 GB/s, dual-channel DDR5 roughly 70 GB/s — and computing in place additionally leaves the GPU free instead of making it wait on a copy.

This only pays off if the CPU reads the packed 4-bit weights directly. Hence a small C++ kernel with an AVX2 path (`acvram_cpu.cpp`, loaded via ctypes, no Python headers or ninja). Even on its **scalar** fallback branch, it beats `dequantize() @ x` by a factor of 1.44 in INT4 and 3.21 in NVFP4, because the latter first writes a 32-bit copy of the whole matrix.

On Mistral-Large-123B, the planner's estimate goes from 1.35 to 2.42 tokens/s.

### Mixed precision (`--snr-floor`, off by default)

The converter measures the layer-output signal-to-noise ratio for every tensor and can promote to a wider format any that fall below `--snr-floor`, within a cap of 15% of tensors and a price ceiling (`--promotion-cout-max`, in added mebibytes).

The floor is **zero by default**: nothing gets promoted. Decoding is memory-bandwidth-bound, and the measurement on `Huihui-Qwen3.8-27B` settles it — a 25 dB floor costs 13.4% memory and 10.6% throughput (18.50 GiB and 41.8 t/s versus 16.02 and 46.2) for 2.0% perplexity (42.591 versus 43.447, 16,383-token corpus). `--snr-floor 25` restores the old behaviour when quality matters more than speed.

### And `acvram eval`

Signal-to-noise ratio and logit cosine similarity are approximations. `acvram eval REP [REP ...]` measures sliding-window perplexity, so a format choice is settled on evidence:

```
$ acvram eval ~/acv/qwen3-32b-nvfp4 ~/acv/qwen3-32b-int4
  model                    ppl     bpp        size     tokens
  qwen3-32b-nvfp4        6.412    4.51     17.4 GiB      8192
  qwen3-32b-int4         6.583    4.17     16.1 GiB      8192  (+2.7%)
```

---

<a id="http"></a>

## HTTP endpoints

| endpoint | notes |
|---|---|
| `POST /v1/chat/completions` | SSE stream or single response; uses the model's chat template |
| `POST /v1/completions` | prompt as text or as token ids |
| `POST /v1/embeddings` | mean-pooled final hidden states, L2-normalised, `dimensions` honoured |
| `GET /v1/models` | plus an `acvram` block: formats, devices, KV cache capacity |
| `GET /health`, `GET /metrics` | decode throughput, KV block occupancy |

These responses' field names stay in English: it is the OpenAI protocol, and translating them would break every existing client.

---

<a id="chiffres"></a>

## Where the numbers come from

Every value cited above comes from code in this repository and is checked by `pytest`. Measurements taken on CPU with the reference kernels:

| format | bits/weight | weight SNR | logit cosine vs BF16 |
|---|---|---|---|
| BF16 | 16.00 | — | 1.0000 |
| INT8 | 8.19 | 44.6 dB | 0.9998 |
| NVFP4 | 4.50 | 20.4 dB | 0.9664 |
| INT4 | 4.16 | 20.0 dB | 0.9427 |
| INT4 + Hadamard | 4.16 | 21.0 dB | 0.9582 |

Two findings from these measurements changed the defaults:

* **A Hadamard rotation helps INT4 but not NVFP4.** INT4's groups of 128 cannot absorb an isolated outlier channel, so spreading out extreme values is worth an n log n transform per activation. NVFP4's blocks of 16 already carry their own scale. Hence `--hadamard auto`, which only applies it to INT4.
* **INT8 beats FP8 E4M3 for the KV cache**, 44 dB versus 32 dB at equal size, because a per-(token, head) scale already supplies the dynamic range that FP8 spends exponent bits on instead. Both cards therefore use an INT8 KV cache, even though the 5090 could do FP8. A `k8v4` format (INT4 values, −22% cache bytes) exists as an option, **unqualified** — see `docs/ARCHITECTURE.md`.

---

<a id="documentation"></a>

## Documentation

| Document | Content |
|---|---|
| [`REPRISE.md`](../REPRISE.md) | **resuming the project on another machine** (French) |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | how the pieces fit together |
| [`docs/CHOIX-FORMAT-GDN.md`](CHOIX-FORMAT-GDN.md) | pure NVFP4 or attention+GDN in per-channel int8, on a Gated DeltaNet hybrid |
| [`docs/MATERIEL.md`](MATERIEL.md) | tuning this specific machine |
| [`docs/FEUILLE-DE-ROUTE.md`](FEUILLE-DE-ROUTE.md) | **what isn't done yet**, read this first |
| [`CONVENTIONS.md`](../CONVENTIONS.md) | working conventions for the code (language, style, checks before pushing) |

---

<a id="resultats"></a>

## Measured results (2026-09-22, RTX 5090 at 400 W, ≥ 20 s window on the energy meter)

Qwen3-Coder-30B-A3B in NVFP4 (experts) + INT8 (attention, head), same protocol for every engine (`outils/`, one card, `energie.py`):

| | acvram | vLLM 0.29 (`vllm serve`) | llama.cpp (sm_120) |
|---|---|---|---|
| decode, 12 sequences | 1,995.1 t/s ² | 2,027.0 t/s ² | — |
| decode, 1 sequence | 312.3 t/s ³ ⁴ | 284.8 t/s ³ | **329.9 t/s** ⁴ |
| prefill pp2048 | **22,707 tokens/s** | 21,054 | 8,671 (TabbyAPI, retired) |

¹ 2026-09-22 erratum: `serve` speculates by default (`--speculative ngram`, cli.py), the competitors do not; the 380.8 t/s published until now was measured WITH speculation. Without speculation (`--speculative none`, same chain, revue/poste2-piece44-speculation-none-22-09.md): 283.6 t/s — acvram is **third** at b=1, behind llama.cpp and vLLM. On energy it stays ahead of llama.cpp (0.601 vs 0.700 J/token net). At b=12, speculation is never active (guard `lot_max=2`): that cell was already on equal footing.

² 2026-09-23, same session, same HTTP client (`banc-llamacpp-16-09.py` against `acvram serve` and `vllm serve`), `-lgc 2700` explicitly set around each arm, cells alternated A V V A, ≥ 5 batches per arm, a gap reported only beyond 2σ (revue/poste2-piece96-vllm-b12-rejeu-89-23-09.md). acvram 0.6.38 (w13 at decode, unrolled attention reduction): gap −1.6%, **under 2σ: throughput parity**. On J/token, **vLLM stays ahead by 7.0%** (beyond 2σ). With 0.6.37 the same protocol gave −4.7%.

³ Same session and protocol as ², no speculation on either side: acvram 312.3 vs vLLM 284.8 — **acvram ahead by 9.7% in throughput** (beyond 2σ); J/token: **parity** (0.04% gap, under 2σ).

⁴ 2026-09-23, same protocol against llama.cpp (revue/poste2-piece72-llamacpp-b1-23-09.md), acvram 0.6.37 with the rewritten router (+5.6%): acvram 310.8 vs llama.cpp 329.9 t/s — **llama.cpp ahead by 5.8% in throughput, acvram ahead by 13.4% in J/token** (0.598 vs 0.691).

Today's throughput figures (station 1030, eco regime `-lgc 2700`, pipeline in service; greedy sampling captured in the CUDA graph, default since 0.6.35). The b=12 acvram figure is an official sealed cell (median of 6 interleaved windows, clock read per window).

> **Erratum (2026-09-23).** The vLLM comparison published until now (b=12: 1,782 vs 1,634 t/s; b=1: 290.6) compared acvram measured over HTTP against vLLM measured **offline** (`LLM().generate()`), and the 2026-09-22 erratum wrongly claimed the vLLM cell went through `vllm serve`. On 2026-09-23: same HTTP client for both, and `-lgc` set for both (acvram sets its own at startup, `vllm serve` does not: without that precaution vLLM ran at ~2,930 MHz versus ~2,650). Result in note ²: vLLM ahead by 9.1% at b=12.

On the morning of 2026-09-14 acvram stood at 630 t/s and 0.619 J/token on the same cell: the gains come from Blackwell's native FP4 MMA (`mma.sync … kind::mxf4nvf4`, ×7.9 over bf16), grouped-GEMM MoE per batch bucket, single-kernel routing (3,677 → 1,517 launches per step), and a narrow tensor-core GEMM for the projections. Every figure has its note in `acvram-memoire/revue/` with the prediction sealed before the measurement, the instrument, and its regime — a figure without a regime is not published.

Where acvram leads: MLA models (GLM-4.7-Flash) in native sm_120 NVFP4, which vLLM only serves in FP8 (b=1: 165.35 t/s in service); models that don't fit in VRAM. Single-sequence decode is not one of those: without speculation, acvram is ahead of vLLM there by 9.7% (note ³), behind llama.cpp by 5.8% in throughput but ahead of it by 13.4% in energy (note ⁴). At large batch, on a MoE model that fits in VRAM, vLLM is at throughput parity at b=12 (1,995.1 vs 2,027.0 t/s, under 2σ, note ²) but keeps a 7.0% J/token edge; acvram progressed there from 1,540 t/s (0.6.34) to 1,995 (0.6.38).

---

<a id="etat"></a>

## Status

Version 0.6.38. Everything runs on the 5090: CUDA kernels compiled for `sm_120a` (native FP4) and `sm_86`, CUDA graphs, NVFP4/INT8/INT4 quantisation, HTTP server. Guardrails in place: the card is invisible to working sessions (`CUDA_VISIBLE_DEVICES` empty) and only `outils/carte.sh` lends it, under lock, to one measurement at a time; a watcher logs any access outside the lock; an energy measurement spanning more than one card or under 10 s is invalidated; a model loaded in a degraded regime says so and does not enter a duel.

4,107 tests (`pytest --collect-only -q`, one minute on CPU; GPU tests only run under `carte.sh`). Work log: `acvram-memoire/` (rules, directory, notebooks, several hundred review notes).

---

<a id="credits"></a>

## Credits

- **vLLM** — [vllm-project/vllm](https://github.com/vllm-project/vllm) v0.29.0, Apache-2.0 licence: `acvram/kernels/marlin_port/` ports its Marlin kernels (MoE and dense), with full file-by-file attribution in [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).
- **NVIDIA** — CUDA, Blackwell's FP4 tensor cores (`sm_120`), and the libraries this project depends on.
- **PyTorch** — tensor engine and C++/CUDA extensions.

Independent project, not affiliated with ASUS, NVIDIA, or the vLLM project.

---

<a id="licence"></a>

## Licence

[GPL-3.0-or-later](../LICENSE) for the code in this repository. `acvram/kernels/marlin_port/` contains code ported from [vLLM](https://github.com/vllm-project/vllm) v0.29.0 (`marlin_moe_wna16`, `gptq_marlin_repack`, `moe_align_block_size` kernels), under the Apache-2.0 licence: each file keeps its original header, the licence text is in `LICENSE-vllm`, and the file list, origin commit and modifications are in [`acvram/kernels/marlin_port/NOTICE`](../acvram/kernels/marlin_port/NOTICE).

---

<a id="soutien"></a>

## Support the project

acvram is developed on personal hardware. If the project is useful to you:

[![Buy Me a Coffee](https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=☕&slug=anticitoyen&button_colour=FFDD00&font_colour=000000&font_family=Lato&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/anticitoyen)

**https://buymeacoffee.com/anticitoyen**

Translations: [TRADUIRE.md](TRADUIRE.md) (French; the project's contribution guide is not yet translated).
