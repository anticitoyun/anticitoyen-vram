# acvram v0.7.15 — English / français

## English

Long Claude Code prompts no longer run out of memory; model menus know each client's context limits.

- **Attention mask split up front**: a prompt that reuses a cached prefix (as Claude Code does) used to build a dense
  attention mask of up to 2 GiB for 34k tokens and fail with out-of-memory. Masks above 256 MiB
  (`ACVRAM_MASQUE_OCTETS_MAX`) are now built in row blocks of at least 1,024 rows, bit-identical to the single block
  on the RTX 5090 (bf16 and fp32, GQA, 1 to 34k tokens).
- **Model menus** (companion parc package 0.1.9): an alias whose context is below a client's minimum is marked, using
  each launcher's own thresholds (Claude Code: reduced tool set below 45,000 tokens, refused below 15,096; kimi:
  without MCP below 65,536). An alias that failed once is no longer hidden after it passes again.
- **Other engines**: llama.cpp uses both GPUs above 30 GB; ERNIE on vLLM gets its reasoning parser; vLLM 0.29
  reasoning output is read correctly; Gemma 4 26B takes the MMA expert path.
- **Operations**: every server stop is logged with who stopped it; the GPU lock refuses a service from another
  session while one is serving.

### Install

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.15/acvram-0.7.15.flatpakref
```

Other channels (Debian/Ubuntu `.deb`, Fedora/COPR RPM, AUR files) are attached below; verify them with
`sha256sum -c SHA256SUMS --ignore-missing`. A `.flatpakref` always installs the latest version published in the
Flatpak repository.

### Full release notes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.15/CHANGELOG.md)

---

## Français

Les longues invites de Claude Code ne manquent plus de mémoire ; les menus connaissent les limites de contexte de chaque client.

- **Masque d'attention découpé d'emblée** : une invite qui réutilise un préfixe en cache (comme Claude Code) faisait
  construire un masque d'attention dense de 2 Gio pour 34 k jetons, puis échouait en manque de mémoire. Au-delà de
  256 Mio (`ACVRAM_MASQUE_OCTETS_MAX`), le masque est construit par blocs d'au moins 1 024 lignes, identique au bit au
  calcul d'un seul tenant sur la RTX 5090 (bf16 et fp32, GQA, de 1 à 34 k jetons).
- **Menus de modèles** (paquet parc 0.1.9) : un alias dont le contexte est sous le minimum d'un client est marqué,
  avec les seuils propres à chaque lanceur (Claude Code : outils réduits sous 45 000 jetons, refus sous 15 096 ;
  kimi : sans MCP sous 65 536). Un alias une fois en panne n'est plus masqué après avoir repassé.
- **Autres moteurs** : llama.cpp utilise les deux GPU au-delà de 30 Go ; ERNIE sous vLLM reçoit son lecteur de
  raisonnement ; le raisonnement de vLLM 0.29 est lu correctement ; Gemma 4 26B prend le chemin MMA des experts.
- **Exploitation** : chaque arrêt de serveur est journalisé avec son auteur ; le verrou du GPU refuse le service d'une
  autre session tant qu'une première sert.

### Installation

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.15/acvram-0.7.15.flatpakref
```

Les autres canaux (`.deb` Debian/Ubuntu, RPM Fedora/COPR, fichiers AUR) sont joints ci-dessous ; vérifiez-les avec
`sha256sum -c SHA256SUMS --ignore-missing`. Un `.flatpakref` installe toujours la dernière version publiée dans le
dépôt Flatpak.

### Notes complètes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.15/CHANGELOG.md)
