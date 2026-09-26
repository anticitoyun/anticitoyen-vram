# acvram v0.7.3

Burst admission: a burst of requests now reaches the engine as one prefill step.

- **Admission watch on by default** (`ACVRAM_ADMISSION_GUET=1`, `0` restores the previous behaviour): the admission
  window stays open while another request is still being templated and tokenised. Measured on Qwen3-Coder-30B-A3B
  nvfp4, 12 concurrent requests, 21 rounds per side (ABBA ×3): **21/21 rounds in a single prefill step (was 15/21)**,
  TTFT p50 **247.4 ms (was 251.6)**, per-round p95 and max both lower; single requests unchanged (37.7 vs 38.4 ms, no
  waiting). Not covered: fewer than 12 requests, images, other models.

## Install

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.3/acvram-0.7.3.flatpakref
```

Other channels (Debian/Ubuntu `.deb`, Fedora/COPR RPM, AUR files) are attached below; verify them with
`sha256sum -c SHA256SUMS --ignore-missing`. A `.flatpakref` always installs the latest version published in the
Flatpak repository.

## Full release notes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.3/CHANGELOG.md)
