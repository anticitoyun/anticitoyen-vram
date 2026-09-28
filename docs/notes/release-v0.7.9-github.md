# acvram v0.7.9 — English / français

## English

**Upgrade recommended**: the first start of every new install could die on hybrid models.

- **First start fixed** (bd dut): on a cold start, Triton autotuning raised the CUDA stack limit to 12,256 bytes per
  thread (default 1,024) and the driver kept that local memory — 2.74 GiB outside the allocator — so graph capture was
  refused and the server died, on the first start of every new package. The limit is now returned to 1,024 before the
  first capture (`ACVRAM_PILE_RENDUE=1`). No output changes. Qwen3.8-27B mixte, guaranteed cold start: 0.7.7 died 2/2,
  0.7.9 served 2/2.
- **Vision: images are encoded while the request is prepared** (pieces 276g/h): time-to-first-token −20 % on average
  (p50 −30 %, p95 −9 %) with 12 image requests on Qwen3-VL-2B, no regression on any quantile. Image features are
  bit-identical; single-request tokens are identical.

### Install

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.9/acvram-0.7.9.flatpakref
```

Other channels (Debian/Ubuntu `.deb`, Fedora/COPR RPM, AUR files) are attached below; verify them with
`sha256sum -c SHA256SUMS --ignore-missing`. A `.flatpakref` always installs the latest version published in the
Flatpak repository.

### Full release notes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.9/CHANGELOG.md)

---

## Français

**Mise à jour recommandée** : le premier démarrage de chaque installation neuve pouvait mourir sur les modèles hybrides.

- **Premier démarrage corrigé** (bd dut) : à froid, l'autotune Triton faisait monter la limite de pile CUDA à 12 256
  octets par fil (défaut 1 024) et le pilote gardait cette mémoire locale — 2,74 Gio hors de l'allocateur — si bien que
  la capture de graphe était refusée et le serveur mourait, au premier démarrage de chaque paquet neuf. La limite est
  désormais ramenée à 1 024 avant la première capture (`ACVRAM_PILE_RENDUE=1`). Aucune sortie ne change. Qwen3.8-27B
  mixte, démarrage à froid garanti : 0.7.7 mort 2/2, 0.7.9 servi 2/2.
- **Vision : les images sont encodées pendant la préparation de la requête** (pièces 276g/h) : délai avant le premier
  jeton −20 % en moyenne (p50 −30 %, p95 −9 %) avec 12 requêtes à image sur Qwen3-VL-2B, aucune régression sur aucun
  quantile. Traits d'image identiques au bit ; jetons identiques pour une requête seule.

### Installation

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.9/acvram-0.7.9.flatpakref
```

Les autres canaux (`.deb` Debian/Ubuntu, RPM Fedora/COPR, fichiers AUR) sont joints ci-dessous ; vérifiez-les avec
`sha256sum -c SHA256SUMS --ignore-missing`. Un `.flatpakref` installe toujours la dernière version publiée dans le
dépôt Flatpak.

### Notes complètes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.9/CHANGELOG.md)
