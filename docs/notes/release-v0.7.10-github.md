# acvram v0.7.10 — English / français

## English

Long context: 65,536 tokens on the large local models.

- **Prefill in slices, 65,536-token context** (piece d19): the ≈Opus 35B (27,648 → 65,536), Qwen3-Coder-30B and the
  Fable 27B (39,936 → 65,536) and the 80B model now hold 65,536 tokens. Slicing (GDN state carried between slices, MoE)
  engages ONLY beyond what a single-pass prefill holds: **below that threshold, output is bit-identical**. Beyond it —
  prompts that were refused with HTTP 400 until now — output is not identical to a single pass: same perplexity
  (ΔNLL −0.0008 ± 0.0009), mean KL 7.9e-3, top-1 agreement 95.7 %. Served anyway, with the gap stated here;
  `ACVRAM_GDN_MORCEAU=0` / `ACVRAM_MOE_MORCEAU=0` turn slicing off. 35B prefill: 16k 1.04 s, 32k 2.40 s, 61k 5.27 s.
- **Kimi keeps its MCP tools** on local models that serve 64k tokens or more (the user's MCP file is never written).
- **Opt-in n-gram speculation takes fewer steps** (piece 277e); `--speculative none` stays the default.

### Install

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.10/acvram-0.7.10.flatpakref
```

Other channels (Debian/Ubuntu `.deb`, Fedora/COPR RPM, AUR files) are attached below; verify them with
`sha256sum -c SHA256SUMS --ignore-missing`. A `.flatpakref` always installs the latest version published in the
Flatpak repository.

### Full release notes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.10/CHANGELOG.md)

---

## Français

Contexte long : 65 536 jetons sur les grands modèles locaux.

- **Préfill par tranches, 65 536 jetons de contexte** (pièce d19) : le ≈Opus 35B (27 648 → 65 536), Qwen3-Coder-30B et
  le Fable 27B (39 936 → 65 536) et le modèle 80B tiennent désormais 65 536 jetons. Le découpage (état GDN porté d'une
  tranche à l'autre, MoE) ne s'engage QU'au-delà de ce qu'un préfill d'un seul tenant tient : **sous ce seuil, la
  sortie est identique au bit**. Au-delà — invites refusées (HTTP 400) jusqu'ici — la sortie n'est pas identique à un
  seul tenant : même perplexité (ΔNLL −0,0008 ± 0,0009), KL moyen 7,9e-3, accord top-1 95,7 %. Servi quand même, écart
  dit ici ; `ACVRAM_GDN_MORCEAU=0` / `ACVRAM_MOE_MORCEAU=0` coupent le découpage. Préfill du 35B : 16 k 1,04 s,
  32 k 2,40 s, 61 k 5,27 s.
- **Kimi garde ses outils MCP** sur les modèles locaux qui servent 64 k jetons ou plus (le fichier MCP de
  l'utilisateur n'est jamais écrit).
- **La spéculation n-gram optionnelle fait moins de pas** (pièce 277e) ; `--speculative none` reste le défaut.

### Installation

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.10/acvram-0.7.10.flatpakref
```

Les autres canaux (`.deb` Debian/Ubuntu, RPM Fedora/COPR, fichiers AUR) sont joints ci-dessous ; vérifiez-les avec
`sha256sum -c SHA256SUMS --ignore-missing`. Un `.flatpakref` installe toujours la dernière version publiée dans le
dépôt Flatpak.

### Notes complètes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.10/CHANGELOG.md)
