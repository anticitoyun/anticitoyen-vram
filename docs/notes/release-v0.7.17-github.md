# acvram v0.7.17 — English / français

## English

More room on the GPU for MoE models, more accurate context limits, and fixed model menus.

- **MoE expert stacks at load time**: expert stacks and their Marlin repacking are built before the KV cache, and the
  small tensors left behind are compacted. They used to pin 7.4 GiB of reserved memory for 0.14 GiB in use;
  Qwen3-Coder-30B now serves 29,096 tokens without graphs. Output is bit-identical to the previous layout.
- **Context that fits**: when the KV budget is too small, the engine names the largest context that fits and the
  launcher retries once at that size if the client accepts it, or refuses with the reason. The prefill memory reserve
  now includes attention scores and is checked against the peak measured at warm-up.
- **Model menus** (companion parc package 0.1.11): sorting a column no longer leaves blank rows; the "≈ Opus/Fable" and
  "code Android/Linux" filters work again; "All" resets the filters; a filter that cannot match anything is greyed
  out with the reason.

### Install

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.17/acvram-0.7.17.flatpakref
```

Other channels (Debian/Ubuntu `.deb`, Fedora/COPR RPM, AUR files) are attached below; verify them with
`sha256sum -c SHA256SUMS --ignore-missing`. A `.flatpakref` always installs the latest version published in the
Flatpak repository.

### Full release notes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.17/CHANGELOG.md)

---

## Français

Plus de place sur le GPU pour les MoE, des limites de contexte plus justes, et des menus de modèles réparés.

- **Piles d'experts MoE au chargement** : les piles d'experts et leur reconditionnement Marlin sont construits avant le
  cache KV, et les petits tenseurs restants sont regroupés. Ils immobilisaient 7,4 Gio de mémoire réservée pour 0,14 Gio
  utilisés ; Qwen3-Coder-30B sert désormais 29 096 jetons sans graphes. Sortie identique au bit à l'ancienne disposition.
- **Contexte qui tient** : quand le budget KV ne suffit pas, le moteur nomme le plus grand contexte qui tient, et le
  lanceur relance une fois à cette taille si le client l'accepte, sinon refuse en disant pourquoi. La réserve mémoire du
  préremplissage compte désormais les scores d'attention et se vérifie contre le pic mesuré à la chauffe.
- **Menus de modèles** (paquet parc 0.1.11) : trier une colonne ne laisse plus de lignes vides ; les filtres
  « ≈ Opus/Fable » et « code Android/Linux » fonctionnent de nouveau ; « Tous » remet les filtres à zéro ; un filtre qui
  ne peut rien trouver est grisé et dit pourquoi.

### Installation

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.17/acvram-0.7.17.flatpakref
```

Les autres canaux (`.deb` Debian/Ubuntu, RPM Fedora/COPR, fichiers AUR) sont joints ci-dessous ; vérifiez-les avec
`sha256sum -c SHA256SUMS --ignore-missing`. Un `.flatpakref` installe toujours la dernière version publiée dans le
dépôt Flatpak.

### Notes complètes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.17/CHANGELOG.md)
