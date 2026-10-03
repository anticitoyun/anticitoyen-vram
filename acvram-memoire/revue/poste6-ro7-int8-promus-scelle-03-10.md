# ro7 — les poids int8 « promus » du parc ne prennent jamais le GEMM int8 du préfill : à sec (cause, parc, options) et scellé AVANT le code et la carte (poste6, 03/10 05 h 1x, ordre chef)

Ordre : chef 03/10 05 h 02 — « d'abord à sec : pourquoi `_i8c_eligible` refuse ces poids, combien de couches par modèle du
parc, ce que coûterait de les servir en int8 ; puis scellé (débit préfill 512-4 096 int8 contre repli, qualité si l'on
re-quantifie) et UNE prise carte.sh si le gain prédit dépasse 5 % ». Arbre : poste6-gemma-anneau après 451d262b5 (main).

## À sec

1. **Pourquoi le refus.** `_i8c_eligible` (`kernels/__init__.py:1075-1085`) n'accepte qu'un poids **symétrique par canal** :
   `group_size == K` et tous les zéros à 128 — c'est la seule forme qu'un GEMM int8 à accumulation int32 sait servir avec
   une échelle par ligne appliquée après coup (`gemm_i8c_cublas`, `:1156`). Les 55 poids int8 de Devstral sont des
   **promotions** nvfp4 → int8 de la conversion (`promoted_from: nvfp4`, `out_snr_db` 53,6 dB) quantifiées **par groupes
   de 128, affines** (`quant/formats.py:154` : `_quantize_int8(weight, 128, symmetric=False)`) : échelle ET point zéro par
   groupe le long de K. Ni l'échelle ni le zéro ne sortent de la somme sur K : un GEMM int8 ne peut pas les porter.
2. **Le parc entier est concerné.** Manifestes des 57 alias de `~/TSV/acvram-chemins.tsv` : **tous** les modèles ont des
   int8 promus g128 affines — Devstral 55, Cydonia 55, Nemo 55, Qwen3-14B 67, gemma4-12B 100-102, les 31B gemma 179,
   Coder-30B 193, GLM-4.7-flash 335, Coder-next-80B 370, kimi-linear 233 (+ 38 par canal), ornith 309… Seules les variantes
   `-i8c` (q/k/v/o par canal, `--attn-qkvo-int8-canal`, `convert.py:443-452`) ont des int8 éligibles (Coder qkvo-i8c 192,
   gemma-4-31B attn-i8c 230, GLM k48-qkvo-i8c 190). **Sur tout modèle non `-i8c`, « prefill_int8=cublas » n'a jamais été le
   chemin pris** pour un seul poids promu — la ligne de régime dit maintenant `repli-bf16×N` (045b7c9e4 / 451d262b5).
3. **Ce que le repli coûte au préfill (Devstral, M = 4 096).** Mesures B4 (03/10) : GEMM d'une couche ≈ 20,8 ms
   (q 0,83 ; k, v 0,22 × 2 ; o 0,81 ; gate, up 6,24 × 2 ; down 6,26) → 833 ms de GEMM sur 1 132 ms de préfill. Part des
   couches int8 en opérations (q 9, k 11, v 6, o 5, gate / up 11, down 13 sur 40 couches) : **19,8 %** des GEMM, ≈ 165 ms,
   plus la déquantification (≈ 4,8 G de paramètres int8 → 9,6 Go de bf16 écrits puis lus, ≈ 15 ms si refaite à chaque
   préfill ; `_w_partage` la garde par forward).
4. **Options.** (A) **Re-quantifier par canal symétrique AU CHARGEMENT** (déquantifier g128 → `_quantize_int8(w, K,
   symmetric=True)`, opt-in `ACVRAM_INT8_PROMUS=canal`) : aucun fichier changé, mesurable tout de suite ; double
   quantification (g128 affine puis canal), qualité à mesurer. (B) **À la conversion** (promotion directement par canal
   symétrique, comme `--attn-qkvo-int8-canal` le fait pour q/k/v/o) : la bonne forme durable, qualité meilleure que (A)
   (une seule quantification), mais une reconversion par modèle. (C) point zéro par groupe dans le GEMM : un noyau W8A8 à
   échelles de groupe n'existe pas ici (ce serait un Marlin-W8 maison) — **écarté** pour cette pièce. L'A8 par jeton du
   chemin cuBLAS (`gemm_i8c_cublas`) s'ajoute : PPL i8c mesurée à **× 1,0094** sur Coder qkvo-i8c (verdict-p2-ppl-19-09,
   seuil poste3 ≤ 1,020) — et ici les poids promus sont précisément les couches sensibles.

## Prédictions (une fenêtre, garde de chaîne, ≈ 6 min)

| # | grandeur | prédit | faux si / seuil |
|---|---|---|---|
| R1 | chrono nu, `gemm_i8c_cublas` (poids par canal symétrique) contre `F.linear` bf16 déquantifié, formes k / v, q, o, gate / up, down, M = 512 … 4 096 | int8 / bf16 = **0,50-0,70** sur les larges (gate / up, down) à M ≥ 2 048 ; 0,7-1,1 à M = 512 et sur k / v (petits N : lancement et A8 dominent) | > 0,9 sur les larges à 4 096 : cuBLASLt int8 ne gagne rien sur sm_120, pièce close sans moteur |
| R2 | débit moteur Devstral, `ACVRAM_INT8_PROMUS=canal` contre défaut (repli), M = 512, 1 024, 2 048, 4 096, ABBA 6 passes | 4 096 : **+6 à +10 %** · 2 048 : +5 à +9 · 1 024 : +3 à +7 · 512 : +1 à +5 | **< 5 % à 4 096 : le gain ne paie pas une reconversion** |
| R3 | ligne de régime sous l'opt-in | `prefill_int8=cublas` (0 repli) ; sous le défaut `repli-bf16×55` | autre |
| R4 | équivalence : sorties du seul tenant, opt-in contre défaut | **ids DIFFÉRENTS attendus** (autre quantification, A8) — HORS BIT par construction, dit ; le témoin reprise borne le bruit | — |
| R5 | qualité : PPL wiki-gptq 2048 (`evaluate.perplexity`, instrument du projet), opt-in / défaut | **× 1,005-1,020** (double quantification des couches sensibles + A8 ; i8c mesuré × 1,0094 sur des couches non promues) | > 1,020 (seuil poste3) : (A) écartée, seule (B) reste à évaluer ; ≤ 1,005 : (A) peut même servir de défaut candidat |
| R6 | mémoire : poids bf16 déquantifiés que `_w_partage` ne fabrique plus | −0,5 à −1,5 Gio de pic au préfill (déquant par tranches bornée par `_DEQUANT_TRANCHE_MAX`) | — (information) |

Décision, fixée ici : (A) proposée au chef comme **opt-in** si R2 ≥ 5 % à 4 096 et R5 ≤ 1,020 ; comme candidate au défaut
seulement si R5 ≤ 1,005 (et alors la garde de qualité complète du modèle avant bascule). Si R1 > 0,9 : rien n'est codé
au moteur au-delà de l'opt-in, la pièce dit que l'int8 cuBLASLt ne paie pas sur cette carte. (B) n'est jugée ici que par
R5 : si (A) passe la qualité, (B) ne peut qu'être meilleure ; si (A) la rate, (B) reste ouverte et demande une reconversion
(ticket). Issues qui me gêneraient : R1 > 0,9 (gain nul au noyau) ; R5 > 1,02 ; un OOM de la re-quantification au
chargement (déquantifier 55 poids d'un coup : par tranches, comme le repli).

Fenêtre prévue : code opt-in (chargeur : re-quantification par canal des int8 promus, test à sec cassant : un poids g128
affine devient éligible, valeurs à ± 1 pas de quantification de la déquantification d'origine ; `prefill_int8_regime`
le dit) → prise : R1 nu (≈ 1 min), R2 débit (≈ 2 min), R3-R4 chaîne S1 A1 sous l'opt-in (1 bras, ≈ 30 s), R5 PPL × 2
(≈ 2 min). Le ticket ro7 reste ouvert jusqu'au verdict.
