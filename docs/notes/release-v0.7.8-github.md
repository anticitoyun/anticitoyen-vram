# acvram v0.7.8 — English / français

## English

Local models work again from the Claude Code and Kimi menus; type-to-search in acvram-gui.

- **Graph capture no longer refuses large models** (piece t5e): the capture guard judged free memory before returning
  PyTorch's cached blocks, and refused the 35B models ("context not held"): 0/2 → 2/2 in the same environment.
- **Model launchers fixed** (parc): `acvram-serveur` died when started from the desktop without
  `CUDA_VISIBLE_DEVICES` (every preload from the menu icons failed), died silently inside a source tree, misread a
  two-line `source=`, and refused to switch model. `claude-modele` now sends 24 k instead of 61 k prompt tokens
  (`--strict-mcp-config`) and sizes its window to the context actually served; `kimi-modele` starts local models
  without MCP servers, leaving the user's own Kimi configuration untouched. Checked on 5 models, Claude and Kimi.
- **acvram-gui: type a model name to filter the list** (piece 9dc): substring, case- and accent-insensitive;
  Esc clears, Ctrl+F focuses; 31 translations.

### Install

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.8/acvram-0.7.8.flatpakref
```

Other channels (Debian/Ubuntu `.deb`, Fedora/COPR RPM, AUR files) are attached below; verify them with
`sha256sum -c SHA256SUMS --ignore-missing`. A `.flatpakref` always installs the latest version published in the
Flatpak repository.

### Full release notes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.8/CHANGELOG.md)

---

## Français

Les modèles locaux marchent de nouveau depuis les menus Claude Code et Kimi ; recherche à la frappe dans acvram-gui.

- **La capture de graphe ne refuse plus les gros modèles** (pièce t5e) : la garde de capture jugeait la mémoire libre
  avant de rendre les blocs en cache de PyTorch, et refusait les modèles 35B (« contexte non tenu ») : 0/2 → 2/2 à
  environnement égal.
- **Lanceurs de modèles corrigés** (parc) : `acvram-serveur` mourait lancé depuis le bureau sans
  `CUDA_VISIBLE_DEVICES` (tout préchargement depuis les icônes échouait), mourait sans message dans un arbre source,
  lisait mal un `source=` sur deux lignes et refusait de changer de modèle. `claude-modele` envoie désormais 24 k
  jetons d'invite au lieu de 61 k (`--strict-mcp-config`) et règle sa fenêtre sur le contexte réellement servi ;
  `kimi-modele` démarre les modèles locaux sans serveurs MCP, sans toucher à la configuration Kimi de l'utilisateur.
  Vérifié sur 5 modèles, avec Claude et Kimi.
- **acvram-gui : taper le nom d'un modèle filtre la liste** (pièce 9dc) : sous-chaîne, sans casse ni accents ;
  Échap vide, Ctrl+F donne le focus ; 31 traductions.

### Installation

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.8/acvram-0.7.8.flatpakref
```

Les autres canaux (`.deb` Debian/Ubuntu, RPM Fedora/COPR, fichiers AUR) sont joints ci-dessous ; vérifiez-les avec
`sha256sum -c SHA256SUMS --ignore-missing`. Un `.flatpakref` installe toujours la dernière version publiée dans le
dépôt Flatpak.

### Notes complètes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.8/CHANGELOG.md)
