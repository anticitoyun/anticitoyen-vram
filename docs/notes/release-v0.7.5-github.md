# acvram v0.7.5

Speculative decoding bug fixed; speculation stays off by default. Stricter release checks.

- **Bug 277 fixed** (piece 277fix): before a speculative step the decode pipeline is now flushed, so n-gram
  speculation no longer repeats tokens. On Qwen3.8-27B mixte-i8c the n-gram output is bit-identical to
  `--speculative none` (k = 4 and k = 1). On Qwen3-Coder-30B-A3B, 2 differences remain over 5 × 32 tokens; both are
  near-ties (the speculative token is the model's second choice, logit margin 0.015, threshold 0.5 fixed before the
  measurement). The same tests fail on 0.7.4 code (margin 12.19 on Coder).
- **Default unchanged: `--speculative none` for every model.** The fixed n-gram path can take more steps than plain
  decoding (up to 55 steps for 32 tokens), so it stays opt-in; `--speculative ngram` prints a warning saying so.
- **Release checks** (piece 285): `verifier-release.sh` now refuses a release whose notes cite a download that is not
  attached, and a Flatpak check that installed an older version than the one released (GitHub Pages lag).

## Install

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.5/acvram-0.7.5.flatpakref
```

Other channels (Debian/Ubuntu `.deb`, Fedora/COPR RPM, AUR files) are attached below; verify them with
`sha256sum -c SHA256SUMS --ignore-missing`. A `.flatpakref` always installs the latest version published in the
Flatpak repository.

## Full release notes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.5/CHANGELOG.md)
