# poste7 — GO multimodal, puis le reste dans l'ordre ; organisation pour aller au plus vite (20/09, 13 h 30, horloge machine)

Utilisateur 13 h 28, mot pour mot : « go multimodal puis le reste dans l'ordre, organise le groupe pour être le plus rapide possible ». Ordre des chantiers : **multimodal** (`poste7-acvram-multimodal-20-09`) → référence du chemin B moe_fused → transfert des 48 → `acvram-parc` → lot poste / memtest → reprise b=12 et GLM b=1.

## Principe : le chemin critique est poste1 à sec (3 j) ; tout ce qui n'en dépend pas part maintenant, en parallèle
| qui | maintenant (20/09 après-midi) | ensuite |
|---|---|---|
| **poste1** | multimodal Gemma 4 en **trois sous-agents à sec en parallèle** sur le contrat § 2 : (a) P0 conversion, (b) P2 API, (c) P1 moteur ; elle intègre et écrit les tests cassants | P4 Qwen3-VL (M-RoPE + deepstack) dès Gemma tenu ; puis moe_fused référence de B |
| **poste2** | (1) **cellule témoin llama.cpp `gemma4-31b` + mmproj** : 1 image 896×896, harnais HTTP, `-lgc 2700`, TTFT · J · réponse, 20 images (≤ 30 min) — le scellé du duel a son chiffre avant que le nôtre existe ; (2) **références `transformers` bf16** de gemma-4-31B-it sur les 20 images : embeddings de la tour, logits du 1er jeton, log-prob des 50 légendes (≤ 30 min, fichiers `.pt` scellés par sha256 dans `outils/gpu/mesure/multimodal/`) | conversion gemma-4-31B-it dès P0 fusionné (une prise, ≤ 30 min par tranche reprenable) ; P3 (a)(b)(c) |
| **poste9** | transfert des 48 alias USB → nvme3, alias par alias, **dans les trous** de poste2 (garde `.qui`, `modeles-a-jour --appliquer` après chaque alias) | `acvram-parc` (`poste7-deb-parc-portable-20-09`) : à sec, poste et systemd sont son domaine ; essai en conteneur hors des prises |
| **chef** | lecture seule : quel test juge le chemin B moe_fused contre une référence indépendante (fichier + ligne, ou « aucun ») — décharge poste1 ; fusions ≤ 5 min ; ETAT | idem |
| **poste7** | contrat § 2 ; lit chaque verdict dans les 5 min | — |
Prédit : Gemma 4 tenu **21/09 18 h** (au lieu de 23/09 en série) ; Qwen3-VL **23/09** ; parc et transfert finis en parallèle le 22/09. Faux si un sous-agent d'poste1 rend un code qui ne s'intègre pas au contrat (alors P1 en série, +1 j).

## 2. Contrat d'intégration (pour que trois branches fusionnent sans se voir)
* **Dossier converti** : mêmes safetensors, les tenseurs `model.vision_tower.*`, `model.embed_vision.*` (Gemma) / `model.visual.*` (Qwen) **gardés en bf16 sous leur nom source** ; `processor_config.json`, `preprocessor_config.json`, `chat_template.jinja` copiés ; `config.json` garde `vision_config` et `model_type` source ; manifeste : `vision_bytes`, `vision=oui`. Un alias sans ces tenseurs = texte seul, refus nommé.
* **Moteur** : `ForwardBatch.images: Optional[list[(debut, fin, embeds bf16 [n, h])]]` par séquence ; dispersion **juste après** `model.py:3162` ; masque : plage `[debut, fin)` bidirectionnelle au prefill (Gemma), causal sinon ; la tour de vision est un module `transformers` chargé depuis le dossier, bf16, `torch.no_grad`, hors graphes, sur la carte du plan ; `regime_ligne` : `vision=bf16(eager)`.
* **API → moteur** : `protocol.py` accepte `image_url` (`data:`, `file://` sous `ACVRAM_IMAGES_DIR`, `http(s)://127.0.0.1`) ; `AutoProcessor` rend `input_ids` (jetons image expansés) + `pixel_values` ; la requête interne porte `(tokens, images=[(debut, fin, pixel_values)])` ; la tour tourne dans le runner avant le prefill ; **clé du cache de préfixe = jetons + sha256(pixel_values)** ; N jetons image comptés dans `max_model_len`.
* **Tests cassants, un par branche** : (a) conversion d'un mini-modèle VL factice garde 100 % des tenseurs vision, un alias texte inchangé au bit ; (b) `image_url` refusé sur alias texte (400 nommé), accepté sur alias vision, N jetons comptés ; (c) à sec : dispersion + masque = `transformers` sur formes réduites, deux images ≠ → deux clés de cache.

## 3. Scellés (inchangés, `poste7-acvram-multimodal-20-09` § 3) — rappel des chiffres
Tour : cos ≥ 0,999, max|Δ| ≤ 8 ulp bf16 sur 20 images ; 1er jeton = `transformers` bf16 20/20 (texte bf16) ; log-prob 50 légendes géo ≤ +3 % vs bf16 sous nvfp4 ; duel : TTFT acvram ≤ **le chiffre de poste2 aujourd'hui** (prédit llama.cpp 0,4-0,6 s ; acvram 0,25) ET J ≤ ET top-1 identique ≥ 18/20. Faux si l'un tombe ; issue gênante : +3 % dépassé → recalibration avec embeddings d'image, +1 j, avant tout .deb.

## Ordre
* **poste1** — trois sous-agents à sec sur le contrat § 2 (a)(b)(c), branches `poste1-mm-conversion`, `poste1-mm-api`, `poste1-mm-moteur` ; pointeur à chef par branche ; aucun nvcc/pytest pendant une prise de poste2 (sous `nice` sinon).
* **poste2** — maintenant : témoin llama.cpp gemma4-31b + mmproj (20 images, sept lignes) puis références `transformers` bf16 (fichiers scellés) ; ensuite conversion gemma-4-31B-it sur P0 fusionné ; puis P3.
* **poste9** — transfert des 48 dans les trous dès maintenant ; puis `acvram-parc` à sec.
* **chef** — juge de B moe_fused (lecture) ; fusions ; ETAT avec ce tableau ; à l'utilisateur : memtest86+ = une nuit sans le PC, à caler.
