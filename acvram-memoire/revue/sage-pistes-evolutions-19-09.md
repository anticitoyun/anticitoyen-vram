# Sage — pistes d'amélioration et d'évolution d'acvram (19/09, 17 h 35) : gain attendu, coût, comment réfuter

Réponse à l'utilisateur (« quelles sont les pistes ? »). Chaque chiffre vient d'un verdict nommé ; une piste sans chiffre porte « non mesuré ». Source pour `REPRISE.md` § 10 (Jérôme, après P2 ligne 2).

## 1. Où nous sommes (Coder-30B-A3B, moteurs classés PPL ≤ 1,02, harnais égal)

| cellule | acvram défaut 0.6.13 | acvram éco `-lgc 2100` (19/09) | llama.cpp | reste |
|---|---|---|---|---|
| b=1 t/s · J brut | 363,3 · 0,798 | non mesuré (E1-bis) | 340,1 · 1,151 | rien |
| b=12 t/s · J net | 1 361 · 0,2265 | **1 138 · 0,1739** | 1 066 · 0,2136 | rien à b=12 |
| prefill j/s | 16 426 → **18 850** (P2, opt-in) | — | 15 717 | ×2 vs vLLM W4A4 (35 241, non classé) |
| GLM prefill · b=12 | 5 502 · **vide** | — | vLLM 18 117 · 858 / 0,397 | G1 cette nuit |

## 2. Les pistes, par ordre de valeur

| # | piste | gain attendu | coût | réfutation (mesure qui rend « faux ») | état |
|---|---|---|---|---|---|
| 1 | **Mode éco `-lgc`** publié, puis gouverneur d'horloge par lot (libre b ≤ 2, 2 700 b 3-7, 2 100 b ≥ 8) | −10 % J à débit égal (2700), −25 % J pour −15 % t/s (2100) — mesuré ; gouverneur : les deux sans arbitrage | E1-bis b=1 20 min ; `acvram eco` 1 j ; gouverneur 1-2 j (root → petit service) | b=1 à 2700 < 340,1 t/s ; J/jeton ABAB à lot mixte ≥ libre | **tenu b=12** (`verdict-eco-lgc-b12-19-09`) |
| 2 | **Prefill W4A8 experts** (MMA int8 ou FP8 ; E2M1 × 2 = entiers exacts, échelle E4M3 par bloc 16 → MMA k=16 + remise à l'échelle fp32) | prefill 18 850 → 26 000-30 000 j/s (borne 34 000 : 118 ms × (0,67/2 + 0,33/2)) ; J prefill −30 % | porte A8 cette nuit (20 min) ; noyau 3-5 j (ptxas → ncu → capture → équivalence → PPL) | porte : PPL fausse-quant A8 − 1,0155 > 0,004 ; noyau : j/s ABAB < 22 000 ou PPL > 1,020 | porte en cours (`sage-w4a4-clos-w4a8-porte-19-09`) ; W4A4 **fermé** |
| 3 | **Décodage : instructions par octet** (nos GEMV : 1,3-2,6 instr/octet DRAM, vLLM GEMM 0,18 ; les GEMV saturent seuls 400 W) — projections q/k/v/o à b ≥ 8 par `narrow_gemm`/MMA au lieu de GEMV, ncu M1/M2 enfin faits | −10 à −20 % J à b=12, +5-10 % t/s ; c'est la cause nommée de l'écart ×1,3 en W avec vLLM (14/09) | ncu 1 passe bornée (46 min/passe sous graphes) ; noyau 2-3 j | `instr/octet` inchangé sous ncu après le noyau ; J ABAB ≥ 0,97× | ouvert (`instr-par-octet-14-09`, M1/M2 jamais faits) |
| 4 | **Spéculation exacte** : MTP de GLM-4.7-Flash (tête livrée), n-gram code (taux 1,61 mesuré 13/09, chantier « pas fermé ») | b=1 : +30 à +60 % t/s sur code, J/jeton −20 à −35 % (moins de pas par jeton) ; b=12 : ≈ 0 (compute) | MTP 2-3 j (GLM) ; n-gram : réglage 1 j (`GardeSpeculation` existe) | taux d'acceptation < 1,3 ; un jeton différent du greedy = bogue (invariant REPRISE § 6) | ouvert (`verdict-taux-ngram-code-13-09`) |
| 5 | **GLM : MLA en FP8 au prefill** (q_b, kv_a, o en E4M3 par jeton, `_scaled_mm`, acc fp32) — l'int8 par canal a réfuté (1,0143 > 1,005), le FP8 n'a jamais été mesuré | prefill GLM 5 502 → 9 000-12 000 (MLA bf16 = poste dominant, part **non mesurée**) | porte fausse-quant à sec 1 h + 15 min carte ; noyau : cuBLAS, 1 j | ratio fausse-quant > 1,005 ; profil prefill : MLA < 40 % du pas | non ouvert (REGLES § 9 : jamais NVFP4 sur MLA ; FP8 non couvert par la règle) |
| 6 | **Godets sur `b`** (`bucket_blocks`, capture pour plus de formes) — prérequis `_bind_hybrid` lie `range(godet)` (MECANISMES) | −0,3 à −0,75 ms de coût fixe par pas à b 2-6 (`verdict-courbe-lot-rp-15-09` : J −5 à −11 % à b=2-6) | 2 j (hybrides : état récurrent des créneaux de rembourrage) | PPL décodage hybride ≠ prefill (jetons faux plausibles) ; ms B/A > 1,0 à b=4 | prérequis connu, non fait |
| 7 | **Cache d'experts** (placement par expert + table d'adresses, LRU des experts fréquents) — modélisé par le planificateur, `taux_de_succes` jamais mesuré | modèles > VRAM (Devstral, 119B) servis sans exil par couche ; pour Coder : 0 | 3-5 j ; **suspendu utilisateur** (Devstral/119B) | taux de succès < 60 % sur un corpus réel ; ms/pas exil expert > exil couche (×23 le 14/09 réfuté à b=12) | `sage-cache-experts-13-09`, tables identité posées (`_tables_adresses`) |
| 8 | **Conversion : compensation d'erreur (GPTQ) + Hadamard sur Coder** | PPL 1,0155 → 1,010 : de la marge pour A8 et pour des lots 4 bits plus serrés | reconversion 2 h + reclassification 1 h | ratio 3 tranches ≥ 1,0155 | non engagé (REPRISE § 10 n° 6) |
| 9 | **Cache KV int8** (FP8 KV moins bon, `lm4` réfuté 1,0215) | ×2 séquences concurrentes à VRAM égale ; t/s b=12 : +0 à +5 % (moins d'octets KV lus à ctx long) | 2 j | PPL décodage > +0,004 ; ms/pas ≥ bf16 | non mesuré |
| 10 | **Produit** : `.deb` fiable (CCCL 0.6.14, fait), lanceurs de serveurs refusant sans verrou, `acvram eco`, GUI vedettes, PPL qui emprunte le chemin servi (leçon P2 : `perplexity()` ≠ `generate()`) | opérationnel au sens 1 ; un juge qui mesure ce qu'on sert | 1-2 j au fil de l'eau | `acvram doctor` alerte ; `_chemin` de la PPL ≠ celui du service | en cours |

## 3. Ce qui ne se fera pas, et pourquoi (pour ne pas y revenir)

* **W4A4 experts** : plancher E2M1 ≈ 9 % d'erreur par GEMM, PPL +0,010/+0,013 contre 0,0045 de marge (17/09, 19/09). Seule sortie : porter le lissage dans les poids à la conversion — c'est la piste 8, pas un noyau.
* **Optimiser la lecture des poids au prefill** : le défaut est à 85-95 % du plancher tensor cores bf16 (11-12 TFLOP/pas) ; « lu une fois au lieu de deux » a rendu ±0 % trois fois (MECANISMES).
* **Horloge mémoire, split-K b=1, lm_head int8 à b=12, exil par expert à b=12** : réfutés par mesure, cités avec leurs verdicts dans INDEX.

## 4. Ordre de marche proposé (après les verdicts de la nuit)

1. Éco : E1-bis, cellules publiées, `acvram eco` (20/09 matin) → revendication « devant en vitesse et en énergie » si b=1 tient.
2. Porte A8 → W4A8 (semaine 21) : le seul ×2 du prefill.
3. ncu M1/M2 puis projections MMA à b ≥ 8 (énergie du décodage) ; en parallèle MTP GLM (b=1).
4. GLM FP8-MLA (porte à sec d'abord) ; godets b ; KV int8.
5. Cache d'experts et 119B : sur le oui de l'utilisateur seulement.
