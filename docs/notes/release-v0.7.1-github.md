# acvram v0.7.1

Service latency, a working Flatpak, and two measured-and-declined options.

- **`/metrics` no longer blocks the HTTP loop** (piece 268): it cost 343 ms per call inside the event loop. With a
  client polling `/metrics` at 20 Hz, time-to-first-token for 12 concurrent requests on Qwen3-Coder-30B-A3B drops
  **0.78 s → 0.24 s** (0.247 s without any poller); `/metrics` itself **343 → 60 ms**. Output unchanged field by field.
- **Chat template and tokenizer rendered off the event loop**, with a thread-safety fix for the first burst after
  start-up (the Jinja environment was published before its globals were set).
- **TTFT re-assessed** (piece 262): measured without a concurrent `/metrics` reader, TTFT at 12 requests is
  **0.242 s**, versus 0.66 s for llama.cpp `-np 1`. The earlier "2.29× slower" figure was an artefact of that poller.
- **Signed-copy xor on the int8 cuBLAS path, by default** (piece 260x): bit-exact, never slower; measured gain
  −1.98 ms (L=512) / −2.62 ms (L=2047) TTFT, below the announced 2.5 ms threshold at L=512, so not claimed.
- **Opt-in only**: `ACVRAM_I8C_FP8_PREFILL=cublas` (prefill −26.8 %, but wall time only −2.4 % against a KL 9× the
  admitted bound) and `ACVRAM_MARLIN_PAR_LIGNE=1` (no significant difference on 650 paired task questions, but the
  95 % lower bound of the P1/P0 ratio, 93.4 %, is under the 97 % required).
- **Flatpak delivered through a signed OSTree repository** on GitHub Pages: PyTorch 2.14.0+cu130 and its CUDA
  libraries are downloaded at install time (extra-data, checksums verified), since a single bundle cannot carry
  extra-data and would exceed GitHub's 2 GiB asset limit.

## Install

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.1/acvram-0.7.1.flatpakref
```

Other channels (Debian/Ubuntu `.deb`, Fedora/COPR RPM, AUR files) are attached below; verify them with
`sha256sum -c SHA256SUMS --ignore-missing`.

## Full release notes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.1/CHANGELOG.md) — every point with its
source (piece, verdict note, test).
