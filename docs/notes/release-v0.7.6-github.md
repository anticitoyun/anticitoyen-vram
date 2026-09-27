# acvram v0.7.6 — English / français

## English

Faster first token on hybrid models under load, with bit-identical output; a fixed quality benchmark.

- **Time-to-first-token −54 % on Qwen3.8-27B mixte at 12 requests** (pieces 284 and 284b). On a hybrid model (GDN)
  with prefix caching — the default — any prompt longer than the recurrent-state snapshot boundary made the step fall
  back to prefilling one request at a time, in two passes, redoing weight dequantization for every sequence. Batched
  prefill is now split at the boundary and reordered layer by layer: **TTFT 5.50 → 2.55 s** at 12 requests, throughput
  ×1.75, single request unchanged, output identical to the bit (logits and tokens, test
  `tests/test_prefill_tranches_284.py`). On by default; `ACVRAM_PREFILL_TRANCHES=0` restores the previous path.
- **Quality benchmark fixed** (pieces 275b–d): the MMLU answer filter only recognised "answer is" and captured the text
  after the letter, so reasoning models were scored far too low (5/30 counted right against ≈15–16 right letters).
  New task names `mmlu_v275_*` so that no old score mixes with the new ones.

### Install

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.6/acvram-0.7.6.flatpakref
```

Other channels (Debian/Ubuntu `.deb`, Fedora/COPR RPM, AUR files) are attached below; verify them with
`sha256sum -c SHA256SUMS --ignore-missing`. A `.flatpakref` always installs the latest version published in the
Flatpak repository.

### Full release notes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.6/CHANGELOG.md)

---

## Français

Premier jeton plus rapide sur les modèles hybrides sous charge, sortie identique au bit ; banc de qualité corrigé.

- **Délai avant le premier jeton −54 % sur Qwen3.8-27B mixte à 12 requêtes** (pièces 284 et 284b). Sur un modèle
  hybride (GDN) avec cache de préfixe — le défaut —, toute invite plus longue que la frontière d'instantané de l'état
  récurrent faisait retomber le pas sur un préfill requête par requête, en deux passes, qui refaisait la
  déquantification des poids pour chaque séquence. Le préfill par lot est désormais coupé à la frontière et réordonné
  couche par couche : **TTFT 5,50 → 2,55 s** à 12 requêtes, débit ×1,75, requête isolée inchangée, sortie identique
  au bit (logits et jetons, test `tests/test_prefill_tranches_284.py`). Actif par défaut ; `ACVRAM_PREFILL_TRANCHES=0`
  rend l'ancien chemin.
- **Banc de qualité corrigé** (pièces 275b à 275d) : le filtre de réponse MMLU ne reconnaissait que « answer is » et
  capturait le texte après la lettre, si bien que les modèles à raisonnement étaient très sous-notés (5/30 comptés
  justes contre ≈ 15-16 lettres justes). Nouveaux noms de tâche `mmlu_v275_*`, pour qu'aucun ancien score ne se
  mélange aux nouveaux.

### Installation

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.6/acvram-0.7.6.flatpakref
```

Les autres canaux (`.deb` Debian/Ubuntu, RPM Fedora/COPR, fichiers AUR) sont joints ci-dessous ; vérifiez-les avec
`sha256sum -c SHA256SUMS --ignore-missing`. Un `.flatpakref` installe toujours la dernière version publiée dans le
dépôt Flatpak.

### Notes complètes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.6/CHANGELOG.md)
