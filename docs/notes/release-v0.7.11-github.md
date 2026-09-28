# acvram v0.7.11 — English / français

## English

Dense models hold long context without offloading.

- **Dense MLP prefill in slices** (piece a5v): at 32k context, the dense prefill reserve (13.41 GiB) pushed 10 of
  Devstral 24B's 40 MLP layers off the GPU, dropping throughput to 7.9 tokens/s. The MLP now runs in slices beyond
  what fits resident (`ACVRAM_MLP_MORCEAU`): **34,816 tokens without offloading, 94.5 tokens/s**. Output is
  bit-identical below the threshold; beyond it, mean KL 2.5e-4 and top-1 agreement 99.28 %.
- **Vision options, off by default** (pieces 276i-k): a CUDA graph for the vision tower (`ACVRAM_TOUR_GRAPHE=1`,
  bit-identical, −10 % batch time but +10–15 % median time-to-first-token) and queued prefill (`ACVRAM_PREFILL_FILE=1`).

### Install

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.11/acvram-0.7.11.flatpakref
```

Other channels (Debian/Ubuntu `.deb`, Fedora/COPR RPM, AUR files) are attached below; verify them with
`sha256sum -c SHA256SUMS --ignore-missing`. A `.flatpakref` always installs the latest version published in the
Flatpak repository.

### Full release notes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.11/CHANGELOG.md)

---

## Français

Les modèles denses tiennent un long contexte sans exil.

- **Préfill du MLP dense par tranches** (pièce a5v) : à 32 k de contexte, la réserve de préfill dense (13,41 Gio)
  sortait du GPU 10 des 40 couches MLP de Devstral 24B, et le débit tombait à 7,9 jetons/s. Le MLP tourne désormais par
  tranches au-delà de ce qui tient résident (`ACVRAM_MLP_MORCEAU`) : **34 816 jetons sans exil, 94,5 jetons/s**. Sortie
  identique au bit sous le seuil ; au-delà, KL moyen 2,5e-4 et accord top-1 99,28 %.
- **Options vision, désactivées par défaut** (pièces 276i-k) : graphe CUDA pour la tour de vision
  (`ACVRAM_TOUR_GRAPHE=1`, au bit, temps du lot −10 % mais délai médian avant le premier jeton +10 à 15 %) et préfill en
  file (`ACVRAM_PREFILL_FILE=1`).

### Installation

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.11/acvram-0.7.11.flatpakref
```

Les autres canaux (`.deb` Debian/Ubuntu, RPM Fedora/COPR, fichiers AUR) sont joints ci-dessous ; vérifiez-les avec
`sha256sum -c SHA256SUMS --ignore-missing`. Un `.flatpakref` installe toujours la dernière version publiée dans le
dépôt Flatpak.

### Notes complètes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.11/CHANGELOG.md)
