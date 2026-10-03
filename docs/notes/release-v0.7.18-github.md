# acvram v0.7.18 — English / français

## English

gpt-oss served, gemma-4-31B at 65,536 tokens, a faster MLA prefill by default, and a 5.5 × faster cold load.

- **gpt-oss (20b, 120b)**: MXFP4 weights converted to NVFP4 (exact except 26 blocks of one 120b tensor, named in the
  manifest); harmony output split into `reasoning_content`, `content` and `tool_calls`. Checked against the Hugging Face
  reference (20b: top-1 98.6 %) and llama.cpp (120b: top-1 96.0 %). Batched MoE path comes next: today 41.8 tok/s at b=1.
- **gemma-4-31B at 65,536 tokens**: ring KV cache for sliding-window layers, per-sequence KV target, embeddings moved to
  host RAM before any MLP. Measured: 65,536 tokens held with no layer offloaded, 31 tok/s (was 3.5).
- **Causal MLA prefill on by default** (`ACVRAM_MLA_CAUSAL=1`, 0 restores the previous path): Kimi-Linear-35B prefill
  27 % faster at 8,192 tokens and 31 % at 12,288. Output is not bit-identical; the decoding perplexity guard held
  (+0.20 % geometric mean, worst slice 0.93 %).
- **Cold load 5.5 × faster** (`ACVRAM_PRELECTURE=1`): model shards are read ahead while loading; no loaded byte changes.
- **Mistral 24B templates fixed**: a missing `strftime_now` made 8 aliases fall back silently to ChatML.
- **Options**: exact bf16 cuBLAS reduction (`ACVRAM_BF16_REDUCTION=exacte`), int8 GEMM for promoted int8 weights at
  prefill (`ACVRAM_INT8_PROMUS=canal`). Both off by default.

### Install

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.18/acvram-0.7.18.flatpakref
```

Other channels (Debian/Ubuntu `.deb`, Fedora/COPR RPM, AUR files) are attached below; verify them with
`sha256sum -c SHA256SUMS --ignore-missing`. A `.flatpakref` always installs the latest version published in the
Flatpak repository.

### Full release notes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.18/CHANGELOG.md)

---

## Français

gpt-oss servi, gemma-4-31B à 65 536 jetons, un préremplissage MLA plus rapide par défaut, et un chargement à froid
5,5 × plus rapide.

- **gpt-oss (20b, 120b)** : poids MXFP4 convertis en NVFP4 (exacts sauf 26 blocs d'un tenseur du 120b, nommés au
  manifeste) ; sortie harmony séparée en `reasoning_content`, `content` et `tool_calls`. Vérifié contre la référence
  Hugging Face (20b : top-1 98,6 %) et llama.cpp (120b : top-1 96,0 %). Le chemin MoE groupé vient ensuite : 41,8 j/s
  à b=1 aujourd'hui.
- **gemma-4-31B à 65 536 jetons** : cache KV en anneau pour les couches à fenêtre glissante, cible KV par séquence,
  plongements en RAM hôte avant tout MLP. Mesuré : 65 536 jetons tenus sans couche exilée, 31 j/s (contre 3,5).
- **Préremplissage MLA causal par défaut** (`ACVRAM_MLA_CAUSAL=1`, 0 rend l'ancien chemin) : préremplissage de
  Kimi-Linear-35B 27 % plus rapide à 8 192 jetons et 31 % à 12 288. Sortie non identique au bit ; la garde de perplexité
  de décodage a tenu (+0,20 % en moyenne géométrique, pire tranche 0,93 %).
- **Chargement à froid 5,5 × plus rapide** (`ACVRAM_PRELECTURE=1`) : les fragments du modèle sont lus en avance pendant
  le chargement ; aucun octet chargé ne change.
- **Gabarits Mistral 24B réparés** : un `strftime_now` manquant faisait retomber 8 alias sur ChatML sans le dire.
- **Options** : réduction bf16 exacte de cuBLAS (`ACVRAM_BF16_REDUCTION=exacte`), GEMM int8 pour les poids int8 promus
  au préremplissage (`ACVRAM_INT8_PROMUS=canal`). Toutes deux désactivées par défaut.

### Installation

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.18/acvram-0.7.18.flatpakref
```

Les autres canaux (`.deb` Debian/Ubuntu, RPM Fedora/COPR, fichiers AUR) sont joints ci-dessous ; vérifiez-les avec
`sha256sum -c SHA256SUMS --ignore-missing`. Un `.flatpakref` installe toujours la dernière version publiée dans le dépôt
Flatpak.

### Notes complètes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.18/CHANGELOG.md)
