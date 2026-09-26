# acvram v0.7.2

Two fixes to `acvram doctor` and a more complete Flatpak.

- **`acvram doctor` exited with code 1 on every channel** (piece 070b): a missing `import subprocess` raised a
  `NameError` on the very last line, after the full report. Fixed; the exit code now reflects the checks.
- **The `vision` extra is a warning, not a failure** (piece 273): without `transformers` / `pillow`, `doctor` now
  reports `vision unavailable … pip install 'acvram[vision]'` and exits 0. Loading a multimodal model without the
  extra still fails with a clear message.
- **The Flatpak ships the `vision` extra** (`transformers`, `pillow`; 44 binary wheels for Python 3.14, no source build).

## Install

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.2/acvram-0.7.2.flatpakref
```

Other channels (Debian/Ubuntu `.deb`, Fedora/COPR RPM, AUR files) are attached below; verify them with
`sha256sum -c SHA256SUMS --ignore-missing`.

## Full release notes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.2/CHANGELOG.md)
