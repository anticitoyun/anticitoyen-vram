# acvram v0.7.2 — English / français

## English

Two fixes to `acvram doctor` and a more complete Flatpak.

- **`acvram doctor` exited with code 1 on every channel** (piece 070b): a missing `import subprocess` raised a
  `NameError` on the very last line, after the full report. Fixed; the exit code now reflects the checks.
- **The `vision` extra is a warning, not a failure** (piece 273): without `transformers` / `pillow`, `doctor` now
  reports `vision unavailable … pip install 'acvram[vision]'` and exits 0. Loading a multimodal model without the
  extra still fails with a clear message.
- **The Flatpak ships the `vision` extra** (`transformers`, `pillow`; 44 binary wheels for Python 3.14, no source build).

### Install

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.2/acvram-0.7.2.flatpakref
```

Other channels (Debian/Ubuntu `.deb`, Fedora/COPR RPM, AUR files) are attached below; verify them with
`sha256sum -c SHA256SUMS --ignore-missing`.

### Full release notes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.2/CHANGELOG.md)

---

## Français

Deux correctifs de `acvram doctor` et un Flatpak plus complet.

- **`acvram doctor` sortait avec le code 1 sur tous les canaux** (pièce 070b) : un `import subprocess` manquant
  levait un `NameError` en toute dernière ligne, après le rapport complet. Corrigé ; le code de sortie reflète
  désormais les vérifications.
- **L'extra `vision` est un avertissement, pas un échec** (pièce 273) : sans `transformers` / `pillow`, `doctor`
  signale désormais `vision indisponible … pip install 'acvram[vision]'` et sort avec 0. Charger un modèle multimodal
  sans l'extra échoue toujours, avec un message clair.
- **Le Flatpak embarque l'extra `vision`** (`transformers`, `pillow` ; 44 roues binaires pour Python 3.14, aucune
  construction depuis les sources).

### Installation

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.2/acvram-0.7.2.flatpakref
```

Les autres canaux (`.deb` Debian/Ubuntu, RPM Fedora/COPR, fichiers AUR) sont joints ci-dessous ; vérifiez-les avec
`sha256sum -c SHA256SUMS --ignore-missing`.

### Notes complètes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.2/CHANGELOG.md)
