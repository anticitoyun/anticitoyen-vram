# acvram v0.7.13 — English / français

## English

Marlin MoE kernels are back in the packages; Claude Code tools work on local models.

- **Upgrade notice for .deb and Flatpak users**: since 0.6.13 the `.deb` did not ship the `*.hpp` headers, so the
  Marlin port silently failed to build and MoE models ran on the slower fallback path; the Flatpak never had Marlin.
  MoE throughput measured from those packages is not the repository's. Both packages now carry Marlin (the Flatpak as
  a precompiled kernel checked by fingerprint and GPU architectures). `acvram doctor` now says "Marlin charge" or
  warns "Marlin indisponible". Replay the model menus after upgrading.
- **Loading fix**: Qwen3-Coder-30B (qkvo-i8c) no longer runs out of memory while preparing Marlin weights (a masked
  index allocated 288 MiB per expert stack).
- **Claude Code tools on local models** (companion parc package 0.1.7): `/v1/messages` handles `tools`, `tool_use`
  and `tool_result`; `claude-modele` picks the tool set from the served context (full from 45,000 tokens, otherwise
  Read, Edit, Bash and Grep without MCP, a prompt of about 10,000 tokens instead of 40,000). End-to-end check: Claude
  Code calls Read on a model served by acvram and returns the value read.

### Install

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.13/acvram-0.7.13.flatpakref
```

Other channels (Debian/Ubuntu `.deb`, Fedora/COPR RPM, AUR files) are attached below; verify them with
`sha256sum -c SHA256SUMS --ignore-missing`. A `.flatpakref` always installs the latest version published in the
Flatpak repository.

### Full release notes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.13/CHANGELOG.md)

---

## Français

Les noyaux MoE Marlin reviennent dans les paquets ; les outils de Claude Code marchent sur les modèles locaux.

- **Avertissement aux utilisateurs du .deb et du Flatpak** : depuis 0.6.13, le `.deb` n'embarquait plus les en-têtes
  `*.hpp` ; le port Marlin échouait sans bruit à la compilation et les modèles MoE tournaient sur le repli, plus lent.
  Le Flatpak n'a jamais eu Marlin. Les débits MoE mesurés depuis ces paquets ne sont pas ceux du dépôt. Les deux
  paquets portent désormais Marlin (le Flatpak en noyau précompilé, vérifié par empreinte et architectures de GPU).
  `acvram doctor` affiche « Marlin charge » ou alerte « Marlin indisponible ». Rejouez les menus de modèles après la
  mise à jour.
- **Chargement corrigé** : Qwen3-Coder-30B (qkvo-i8c) ne manque plus de mémoire en préparant les poids Marlin (un
  index par masque allouait 288 Mio par pile d'experts).
- **Outils de Claude Code sur les modèles locaux** (paquet parc 0.1.7) : `/v1/messages` gère `tools`, `tool_use` et
  `tool_result` ; `claude-modele` choisit le jeu d'outils selon le contexte servi (complet à partir de 45 000 jetons,
  sinon Read, Edit, Bash et Grep sans MCP, une invite d'environ 10 000 jetons au lieu de 40 000). Contrôle de bout en
  bout : Claude Code appelle Read sur un modèle servi par acvram et rend la valeur lue.

### Installation

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.13/acvram-0.7.13.flatpakref
```

Les autres canaux (`.deb` Debian/Ubuntu, RPM Fedora/COPR, fichiers AUR) sont joints ci-dessous ; vérifiez-les avec
`sha256sum -c SHA256SUMS --ignore-missing`. Un `.flatpakref` installe toujours la dernière version publiée dans le
dépôt Flatpak.

### Notes complètes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.13/CHANGELOG.md)
