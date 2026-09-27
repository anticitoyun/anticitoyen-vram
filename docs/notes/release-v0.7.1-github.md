# acvram v0.7.1 — English / français

## English

Service latency, a working Flatpak, and two measured-and-declined options.

- **`/metrics` no longer blocks the HTTP loop** (piece 268): it cost 343 ms per call inside the event loop. With a
  client polling `/metrics` at 20 Hz, time-to-first-token for 12 concurrent requests on Qwen3-Coder-30B-A3B drops
  **0.78 s → 0.24 s** (0.247 s without any poller); `/metrics` itself **343 → 60 ms**. Output unchanged field by field.
- **Chat template and tokenizer rendered off the event loop**, with a thread-safety fix for the first burst after
  start-up (the Jinja environment was published before its globals were set).
- **TTFT re-assessed** (piece 262): measured without a concurrent `/metrics` reader, TTFT at 12 requests is
  **0.242 s**, versus 0.66 s for llama.cpp `-np 1`. The earlier "2.29× slower" figure was an artefact of that poller.
- **Signed-copy xor on the int8 cuBLAS path, by default** (piece 260x): bit-exact, never slower; measured gain
  −1.98 ms (L=512) / −2.62 ms (L=2047) TTFT, below the announced 2.5 ms threshold at L=512, so not claimed.
- **Opt-in only**: `ACVRAM_I8C_FP8_PREFILL=cublas` (prefill −26.8 %, but wall time only −2.4 % against a KL 9× the
  admitted bound) and `ACVRAM_MARLIN_PAR_LIGNE=1` (no significant difference on 650 paired task questions, but the
  95 % lower bound of the P1/P0 ratio, 93.4 %, is under the 97 % required).
- **Flatpak delivered through a signed OSTree repository** on GitHub Pages: PyTorch 2.14.0+cu130 and its CUDA
  libraries are downloaded at install time (extra-data, checksums verified), since a single bundle cannot carry
  extra-data and would exceed GitHub's 2 GiB asset limit.

### Install

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.1/acvram-0.7.1.flatpakref
```

Other channels (Debian/Ubuntu `.deb`, Fedora/COPR RPM, AUR files) are attached below; verify them with
`sha256sum -c SHA256SUMS --ignore-missing`.

### Full release notes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.1/CHANGELOG.md) — every point with its
source (piece, verdict note, test).

---

## Français

Latence du service, un Flatpak qui fonctionne, et deux options mesurées puis écartées.

- **`/metrics` ne bloque plus la boucle HTTP** (pièce 268) : il coûtait 343 ms par appel dans la boucle d'événements.
  Avec un client qui interroge `/metrics` à 20 Hz, le délai avant le premier jeton pour 12 requêtes simultanées sur
  Qwen3-Coder-30B-A3B passe de **0,78 s à 0,24 s** (0,247 s sans client) ; `/metrics` lui-même **343 → 60 ms**.
  Sortie inchangée champ par champ.
- **Gabarit de conversation et tokenizer exécutés hors de la boucle d'événements**, avec un correctif de sûreté entre
  fils pour la première rafale après le démarrage (l'environnement Jinja était publié avant ses variables globales).
- **TTFT réévalué** (pièce 262) : mesuré sans lecteur `/metrics` concurrent, le délai avant le premier jeton à 12
  requêtes est de **0,242 s**, contre 0,66 s pour llama.cpp `-np 1`. Le chiffre précédent « 2,29 × plus lent » était un
  artefact de ce lecteur.
- **Copie signée xor sur le chemin int8 cuBLAS, par défaut** (pièce 260x) : identique au bit, jamais plus lent ; gain
  mesuré −1,98 ms (L = 512) / −2,62 ms (L = 2047) de TTFT, sous le seuil annoncé de 2,5 ms à L = 512, donc non revendiqué.
- **Optionnels seulement** : `ACVRAM_I8C_FP8_PREFILL=cublas` (préfill −26,8 %, mais temps total −2,4 % seulement,
  pour une KL 9 × la borne admise) et `ACVRAM_MARLIN_PAR_LIGNE=1` (aucune différence significative sur 650 questions
  appariées, mais la borne basse à 95 % du rapport P1/P0, 93,4 %, est sous les 97 % exigés).
- **Flatpak livré par un dépôt OSTree signé** sur GitHub Pages : PyTorch 2.14.0+cu130 et ses bibliothèques CUDA sont
  téléchargés à l'installation (extra-data, sommes de contrôle vérifiées), car un paquet unique ne peut pas porter
  d'extra-data et dépasserait la limite de 2 Gio des fichiers joints GitHub.

### Installation

```
flatpak install --user https://github.com/anticitoyun/anticitoyen-vram/releases/download/v0.7.1/acvram-0.7.1.flatpakref
```

Les autres canaux (`.deb` Debian/Ubuntu, RPM Fedora/COPR, fichiers AUR) sont joints ci-dessous ; vérifiez-les avec
`sha256sum -c SHA256SUMS --ignore-missing`.

### Notes complètes

[`CHANGELOG.md`](https://github.com/anticitoyun/anticitoyen-vram/blob/v0.7.1/CHANGELOG.md) — chaque point avec sa
source (pièce, note de verdict, test).
