# acvram v0.7.3 — English / français

## English

Burst admission: a burst of requests now reaches the engine as one prefill step.

- **Admission watch on by default** (`ACVRAM_ADMISSION_GUET=1`, `0` restores the previous behaviour): the admission
  window stays open while another request is still being templated and tokenised. Measured on Qwen3-Coder-30B-A3B
  nvfp4, 12 concurrent requests, 21 rounds per side (ABBA ×3): **21/21 rounds in a single prefill step (was 15/21)**,
  TTFT p50 **247.4 ms (was 251.6)**, per-round p95 and max both lower; single requests unchanged (37.7 vs 38.4 ms, no
  waiting). Not covered: fewer than 12 requests, images, other models.

### Install

**No Flatpak for this release**: its build failed (OSTree repository, fixed in v0.7.4). Use v0.7.4 or later for the Flatpak.

Other channels (Debian/Ubuntu `.deb`, Fedora/COPR RPM, AUR files) are attached below; verify them with
`sha256sum -c SHA256SUMS --ignore-missing`.

### Full release notes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.3/CHANGELOG.md)

---

## Français

Admission en rafale : une rafale de requêtes atteint désormais le moteur en un seul pas de préfill.

- **Guet d'admission activé par défaut** (`ACVRAM_ADMISSION_GUET=1`, `0` rend l'ancien comportement) : la fenêtre
  d'admission reste ouverte tant qu'une autre requête est encore en cours de gabarit et de découpage en jetons. Mesuré
  sur Qwen3-Coder-30B-A3B nvfp4, 12 requêtes simultanées, 21 tours par côté (ABBA × 3) : **21/21 tours en un seul pas
  de préfill (contre 15/21)**, TTFT p50 **247,4 ms (contre 251,6)**, p95 et maximum par tour tous deux plus bas ;
  requêtes isolées inchangées (37,7 contre 38,4 ms, aucune attente). Non couvert : moins de 12 requêtes, images,
  autres modèles.

### Installation

**Pas de Flatpak pour cette version** : sa construction a échoué (dépôt OSTree, corrigé dans la v0.7.4). Utilisez la
v0.7.4 ou suivante pour le Flatpak.

Les autres canaux (`.deb` Debian/Ubuntu, RPM Fedora/COPR, fichiers AUR) sont joints ci-dessous ; vérifiez-les avec
`sha256sum -c SHA256SUMS --ignore-missing`.

### Notes complètes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.3/CHANGELOG.md)
