# acvram v0.7.5 — English / français

## English

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

### Install

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.5/acvram-0.7.5.flatpakref
```

Other channels (Debian/Ubuntu `.deb`, Fedora/COPR RPM, AUR files) are attached below; verify them with
`sha256sum -c SHA256SUMS --ignore-missing`. A `.flatpakref` always installs the latest version published in the
Flatpak repository.

### Full release notes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.5/CHANGELOG.md)

---

## Français

Bogue de la spéculation corrigé ; la spéculation reste désactivée par défaut. Contrôles de sortie plus stricts.

- **Bogue 277 corrigé** (pièce 277fix) : avant un pas spéculatif, le pipeline de décodage est désormais vidé, si
  bien que la spéculation n-gram ne répète plus de jetons. Sur Qwen3.8-27B mixte-i8c, la sortie n-gram est identique
  au bit à `--speculative none` (k = 4 et k = 1). Sur Qwen3-Coder-30B-A3B, 2 écarts subsistent sur 5 × 32 jetons ;
  ce sont deux quasi-égalités (le jeton spéculatif est le second choix du modèle, écart de logit 0,015, seuil de 0,5
  fixé avant la mesure). Les mêmes tests échouent sur le code 0.7.4 (écart 12,19 sur le Coder).
- **Défaut inchangé : `--speculative none` pour tous les modèles.** Le chemin n-gram corrigé peut faire plus de pas
  que le décodage simple (jusqu'à 55 pas pour 32 jetons) ; il reste optionnel, et `--speculative ngram` affiche un
  avertissement qui le dit.
- **Contrôles de sortie** (pièce 285) : `verifier-release.sh` refuse désormais une version dont les notes citent un
  téléchargement non joint, et un contrôle Flatpak qui aurait installé une version plus ancienne que celle publiée
  (délai de GitHub Pages).

### Installation

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.5/acvram-0.7.5.flatpakref
```

Les autres canaux (`.deb` Debian/Ubuntu, RPM Fedora/COPR, fichiers AUR) sont joints ci-dessous ; vérifiez-les avec
`sha256sum -c SHA256SUMS --ignore-missing`. Un `.flatpakref` installe toujours la dernière version publiée dans le
dépôt Flatpak.

### Notes complètes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.5/CHANGELOG.md)
