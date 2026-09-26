# acvram v0.7.4

Safety release: speculative decoding is off by default. **Upgrade recommended for every 0.7.x user of `acvram serve`.**

- **`--speculative` now defaults to `none` for every model** (piece 283, bug 277). Since 0.7.0, `serve` enabled
  n-gram speculation by default. When the engine switched from plain decoding to a speculative step, the in-flight
  pipelined step was not flushed, so **tokens could be repeated**: served output could differ from non-speculative
  decoding, on any model (dense included). The fix to the pipeline is written but not yet qualified on every model,
  so this release does not ship it; `--speculative ngram` stays available on explicit request and prints a warning
  at start-up. N-gram will come back as a default only after a task-level quality check and a bit-exactness test.
  **If you served 0.7.0–0.7.3 without `--speculative none`, outputs from that period may contain repeated tokens.**
- **Admission watch turned off for vision models** (piece 269d): with images, image preparation (16–34 ms) always
  exceeded the 5 ms window, so every round waited the 20 ms ceiling — TTFT p50 +9.6 ms, p95 +25 ms on
  Qwen3-VL-2B, b = 4. Text-only models keep the 0.7.3 default (no measurable cost, one prefill step per burst).
- **Flatpak fixed** (piece 266m): the 0.7.3 Flatpak build failed when updating the OSTree repository seeded from
  GitHub Pages (empty directories are not kept by git), so 0.7.3 has no `.flatpakref`. This release restores it.

## Install

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.4/acvram-0.7.4.flatpakref
```

Other channels (Debian/Ubuntu `.deb`, Fedora/COPR RPM, AUR files) are attached below; verify them with
`sha256sum -c SHA256SUMS --ignore-missing`. A `.flatpakref` always installs the latest version published in the
Flatpak repository.

## Full release notes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.4/CHANGELOG.md)
