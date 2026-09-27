# acvram v0.7.4 — English / français

## English

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

### Install

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.4/acvram-0.7.4.flatpakref
```

Other channels (Debian/Ubuntu `.deb`, Fedora/COPR RPM, AUR files) are attached below; verify them with
`sha256sum -c SHA256SUMS --ignore-missing`. A `.flatpakref` always installs the latest version published in the
Flatpak repository.

### Full release notes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.4/CHANGELOG.md)

---

## Français

Version de sûreté : la spéculation est désactivée par défaut. **Mise à jour recommandée pour tout utilisateur 0.7.x
de `acvram serve`.**

- **`--speculative` vaut désormais `none` par défaut pour tous les modèles** (pièce 283, bogue 277). Depuis la 0.7.0,
  `serve` activait par défaut la spéculation n-gram. Au passage du décodage simple à un pas spéculatif, le pas déjà
  en vol dans le pipeline n'était pas vidé, si bien que **des jetons pouvaient être répétés** : la sortie servie
  pouvait différer du décodage sans spéculation, sur tout modèle (denses compris). Le correctif du pipeline est écrit
  mais pas encore qualifié sur tous les modèles, donc cette version ne l'embarque pas ; `--speculative ngram` reste
  disponible sur demande explicite et affiche un avertissement au démarrage. Le n-gram ne reviendra par défaut
  qu'après un contrôle de qualité par tâche et un test d'identité au bit.
  **Si vous avez servi une 0.7.0 à 0.7.3 sans `--speculative none`, les sorties de cette période peuvent contenir des
  jetons répétés.**
- **Guet d'admission coupé pour les modèles vision** (pièce 269d) : avec des images, la préparation d'image
  (16-34 ms) dépassait toujours la fenêtre de 5 ms, si bien que chaque tour attendait le plafond de 20 ms — TTFT p50
  +9,6 ms, p95 +25 ms sur Qwen3-VL-2B, b = 4. Les modèles texte gardent le défaut de la 0.7.3 (aucun coût mesurable,
  un seul pas de préfill par rafale).
- **Flatpak réparé** (pièce 266m) : la construction du Flatpak 0.7.3 échouait à la mise à jour du dépôt OSTree
  amorcé depuis GitHub Pages (git ne garde pas les dossiers vides), si bien que la 0.7.3 n'a pas de `.flatpakref`.
  Cette version le rétablit.

### Installation

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.4/acvram-0.7.4.flatpakref
```

Les autres canaux (`.deb` Debian/Ubuntu, RPM Fedora/COPR, fichiers AUR) sont joints ci-dessous ; vérifiez-les avec
`sha256sum -c SHA256SUMS --ignore-missing`. Un `.flatpakref` installe toujours la dernière version publiée dans le
dépôt Flatpak.

### Notes complètes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.4/CHANGELOG.md)
