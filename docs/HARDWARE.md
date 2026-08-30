# Tuning the target rig

i9-14900K · ROG Maximus Z790 Dark Hero · 96 GB DDR5 · RTX 5090 Astral LC OC
32 GB · RTX 3080 Ti 12 GB · Linux Mint 22.3

Everything here should be **verified with `acvram detect`** on the real
machine. The built-in profile carries nameplate values so that planning can
happen before the hardware is reachable; detection always wins over it.

## Software floor

| | minimum | why |
|---|---|---|
| NVIDIA driver | 570 | first with Blackwell / `sm_120` |
| CUDA | **12.8** | `sm_120` cannot be emitted by anything older |
| PyTorch | 2.7+ built for cu128 or cu130 | a cu126 wheel will not run on the 5090 |
| GCC | ≤ 13 for nvcc | newer host compilers are rejected by nvcc |

`acvram doctor` fails loudly on the CUDA-version mismatch, because the symptom
otherwise is a confusing `no kernel image is available for execution`.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## The PCIe asymmetry — check this first

On most Z790 boards the first x16 slot is CPU-attached PCIe 5.0 and the second
is chipset-attached and much narrower. If the 3080 Ti lands in a Gen4 x4 slot
its host link is roughly **7 GB/s against 54 GB/s** for the 5090 — an eightfold
difference that decides which card streams weights from RAM.

```bash
nvidia-smi --query-gpu=index,name,pcie.link.gen.current,pcie.link.width.current \
           --format=csv
acvram bench --what bandwidth      # the number that actually matters
```

The planner reads the live link width and routes streamed layers to the card
with the fastest link. If `bench` disagrees with `detect`, trust `bench` and
open an issue — the profile's pessimistic Gen4 x4 default for GPU 1 is a
guess.

## No peer-to-peer

Neither board has NVLink, and NVIDIA disables PCIe P2P on GeForce. Inter-GPU
tensors therefore stage through pinned host memory. This is fine — the
pipeline crosses once per token and carries one hidden state, a few kilobytes
— but it is why the planner enforces a single crossing rather than
interleaving layers between cards.

## CPU: pin to the P-cores

The 14900K is hybrid: 8 P-cores with SMT plus 16 E-cores. Threads that land
on an E-core run the same work at a fraction of the rate, and for a latency-
sensitive loop that is worse than having fewer threads.

```bash
acvram detect                      # reports the P-core cpuset
taskset -c 0-15 acvram serve ...   # confirm the range on your machine first
export OMP_NUM_THREADS=8
```

Also worth having: microcode `0x12B` or newer, for the Raptor Lake
degradation mitigation. `grep microcode /proc/cpuinfo`.

## Memory

96 GB is almost certainly 2×48 GB. Keep it at two DIMMs — four sticks on Z790
force a lower stable frequency, and the host tier's usefulness is directly
proportional to its bandwidth. Enable XMP/EXPO and verify:

```bash
sudo dmidecode -t memory | grep -E 'Speed|Size'
```

`--host-fraction` (default 0.85) caps how much of `MemAvailable` acvram will
claim. Raise it to 0.95 for a model that only just fits; lower it if the
machine does anything else.

Huge pages measurably help the pinned staging buffers:

```bash
sudo sysctl -w vm.nr_hugepages=8192      # 16 GB of 2 MB pages
```

## Power and thermals

The 5090 Astral LC OC will draw close to 600 W under sustained prefill. A
`nvidia-smi -pl` cap costs less throughput than thermal throttling does, and
decode is memory-bound anyway — it barely notices a power cap.

```bash
sudo nvidia-smi -i 0 -pl 500
sudo nvidia-smi -i 0 --lock-memory-clocks-deferred=  # leave memory clocks alone
```

Do **not** lock memory clocks down on the 5090: decode throughput is
proportional to GDDR7 bandwidth, which is the whole point of the card here.

## What fits

At NVFP4 on the 5090 (4.5 bpw) and INT4 on the 3080 Ti (4.16 bpw), with
32 + 12 GB of VRAM and 96 GB of RAM. Decode figures are the planner's
estimates at batch 1, not measurements.

| model | weights | placement | est. decode |
|---|---|---|---|
| Qwen3-32B | 18 GiB | 5090 alone, 3080 Ti idle | ~93 tok/s |
| Llama-3.3-70B | 38 GiB | both GPUs, 3.4 GiB in RAM | ~18 tok/s |
| Mistral-Large-123B | 61 GiB | both GPUs + RAM | single digits |
| Qwen3-235B-A22B | 121 GiB | needs `--host-fraction 0.95`, ~27k context | ~3.5 tok/s |

The MoE case is the one 96 GB of RAM was bought for: 235B of weights but only
22B active per token, so the machine reads about 10 GiB per token instead of
121 GiB.

## Sanity sequence on first boot

```bash
acvram doctor
acvram detect
acvram bench --what bandwidth        # confirm the two links
acvram bench --what kernels          # confirm the extension built and is fast
acvram plan ~/models/Qwen3-32B       # confirm the plan matches the table above
```

## Deciding where the host tier computes

The one measurement that settles it:

```bash
acvram bench --what bandwidth
```

It prints the host-to-device rate for each GPU and the DDR streaming read
rate, then names the winner:

```
  cuda:0   NVIDIA GeForce RTX 5090       h2d    52.8 GB/s   d2h    51.9 GB/s
  cuda:1   NVIDIA GeForce RTX 3080 Ti    h2d     6.6 GB/s   d2h     6.4 GB/s
  host     DDR streaming read               71.4 GB/s   (16 threads)
           -> compute host-tier layers on the CPU  (--host-exec cpu --host-gb-s 71.4)
```

Feed that number back into planning: `acvram plan MODEL --host-gb-s 71.4`. The
default of 70 GB/s is a nameplate figure for dual-channel DDR5-6000 and should
be replaced with the measurement.

If the DDR figure comes out far below 70 GB/s, check that both DIMMs are in
the right slots and that XMP/EXPO is enabled — a 96 GB kit running at JEDEC
speeds roughly halves the host tier's usefulness.

## Choosing a speculator

```bash
# free, no second model, helps when the output quotes the input
acvram serve DIR --speculative ngram

# the 3080 Ti is idle for anything that fits on the 5090 -- use it
acvram convert ~/models/Qwen3-1.7B -o ~/acv/draft --gpus 1
acvram serve ~/acv/qwen3-32b --speculative draft \
      --draft-model ~/acv/draft --draft-device cuda:1 --spec-k 5
```

Watch `acceptance_rate` and `tokens_per_step` in `/metrics`. Below about 1.3
tokens per step the speculation is not paying for the extra positions; raise
`--spec-k` if acceptance is high, lower it if acceptance is poor.
