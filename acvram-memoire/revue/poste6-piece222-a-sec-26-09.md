# 222 — à sec (poste6, 26/09) : pourquoi Marlin-w13 perd 8 % sur le Coder nvfp4 pur au banc de la 217, et quand il gagne — le préfill MoE change de noyau (W4A4 natif → Marlin W4A16), pas seulement le décodage ; règle au bit possible sans seconde disposition (C17 étendu au préfill)

* instrument : lecture de code (fichier:ligne, origin/main 67fe21d57), chiffres de la 203 (décodage b=8), de la 220 (banc 217 : 2 048 jetons de
  préfill + 256 pas de décodage par lot), de Mesure 1 C17 et du chantier C17 (`chantier-c17-mma2-lit-marlin-19-09.md`). Aucune carte.
* commit : poste6-222 = origin/main 67fe21d57. · régime : sans objet. · scellé : aucun (à sec) ; chaque gain est une borne à prouver (§ 4).
* mesuré (repris) : 220 : a (Marlin-w13, 48 couches) 1 539,8 t/s · 0,1521 J ; b (naturel) 1 667,4 · 0,1300. 203 (i8c, b=8, par couche) : tensor
  Marlin 50,5 µs (aligneur 2,4 + w13 23,1 + down 23,1 + act 1,4 + reduce 0,8) contre `decode_mma` 58,7 (route_pack 7,1 + mma2 26,6 + 21,9 +
  quant_act 2,1 + reduce_trie 1,1). Formes : K 2 048, N 768, 128 experts, top-8, 339,9 Mo d'experts par couche (manifeste), 48 couches.
* verdict : **ERRATUM de ma lecture de la 220** — la 147 L2 (TTFT +27/+35 %) concernait les projections DENSES (PROJ_MARLIN), pas les experts.
  Le mécanisme est ailleurs : § 1-2. Recommandation § 4.
* durée : à sec, ≈ 50 min.

## 1. Quel chemin sert les experts (fichier:ligne)
| phase | naturel (b, d de la 220 ; be837ca1) | Marlin-w13 (a, c ; main + 209 = 1) |
|---|---|---|
| préfill, t > 32 (`_MOE_GROUPED_MAX`, moe.py:1756) → `_forward_prefill_grouped` :783 | branche **`mma`** :813 (`_MOE_MMA` = 1 :1774, `not unique`) : `nvfp4_quant_act` (x → fp4 bloc 16) puis `_gemm_mma` **W4A4** (`nvfp4_gemm_grouped_mma`, :723) gate, up, down, `moe_act`, `moe_reduce_trie` — 8-9 lancements/couche | branche **`marlin`** :945 (`unique` = piles Marlin et naturelle rendue :790) : `aligner_blocs` + `MP.gemm_moe` **W4A16** (bf16 × fp4, port vLLM) gate et up en vues de w13 (:959, au bit du séparé), down, `moe_act`, `moe_reduce_trie` — 8-10 lancements/couche |
| décodage b=8 (t ≥ 5 = `_MOE_DECODE_MMA_MIN_T` :1913) | **`decode_mma`** :1071 (mma2 W4A4, BT 16) — 58,7 µs/couche (203) | **tensor** `gemm_experts_tensor` :2010 (T ≥ 8 = `_MOE_TENSOR_MIN_T` :1952) — 50,5 µs/couche (203) |
| décodage b=1 | GEMV naturel v1 | GEMV Marlin par paire (214 : 12,9 + 8,1 µs) |
Sur le Coder PUR, `_construire_marlin` :490 refuse TOUTE la disposition à `par_ligne=False` (67 477 sous-normales) → colonne de gauche ;
à `par_ligne=True` (209) tout passe à droite. Sur qkvo-i8c, seules 4 couches changent de colonne (157/209) : le 209 y a gagné +5,77 %.
Les deux colonnes diffèrent aussi en ARITHMÉTIQUE : à gauche les activations sont quantifiées en fp4 (W4A4, `nvfp4_quant_act`), à droite
elles restent bf16 (W4A16) — ce n'est pas la même sortie, ni au préfill ni au décodage (209 c : KL 0,324, PPL 12,80 → 12,61).

## 2. Chiffrage par pas (par couche × 48)
| poste | naturel | Marlin-w13 | source |
|---|---|---|---|
| décodage b=8, MoE hors routage | 58,7 µs → **2,82 ms/pas** | 50,5 µs → **2,42 ms/pas** (−0,39, −8 % du pas 5,0 ms) | 203 § 2 |
| décodage b=8, octets par couche | 30,5 distincts × 2,65 Mo = 81 Mo + xq 8 × 2 048 / 2 | 81 Mo + x bf16 8 × 2 048 × 2 | 203 |
| préfill 2 048 jetons (16 384 paires), octets/couche | 340 Mo de poids + x fp4 16,8 Mo + act fp4 6,3 Mo + d 67 Mo | 340 Mo + x bf16 67 Mo (lu 2 ×) + act bf16 25 Mo + d 67 Mo | formes |
| préfill 2 048 jetons, calcul/couche | 154,6 GFLOP sur tensor cores **fp4** (mma block-scaled) | 154,6 GFLOP sur tensor cores **bf16** (Marlin dépaquette dans la tuile) | 2·G·(2KN + NK) |
| préfill 8 × 78 (624 lignes, banc chat), calcul/couche | 47 GFLOP | 47 GFLOP | idem |
| lancements/couche préfill | 8-9 | 8-10 | § 1 |
**Ce que la 220 impose** : par lot, décodage 256 pas × (5,0 − 0,39) = 1,18 s (a) contre 1,28 s (b) ; débit b/a = 1,083 → **P_a − P_b = 0,106 s +
0,083 · P_b** : le préfill Marlin d'un lot de 2 048 jetons coûte **≥ 110-130 ms de plus** que le préfill W4A4 (P_b ≈ 0,15-0,30 s), soit
**+2,3-2,7 ms par couche** — 3 à 5 × le temps d'une GEMM W4A4 à 300-400 TFLOPS (0,4-0,5 ms/couche). Cohérent avec le port : Marlin
(W4A16) vaut 152 TFLOPS en dense à grand M (P1) mais le MoE porté à 128 lignes par expert (bloc 64, `choisir_block_size`) tombe
plausiblement à 50-70 TFLOPS — vLLM lui-même ne l'emploie qu'au décodage et préfille le NVFP4 en cutlass. **Et le J/jeton +16 %** : un
préfill 3 × plus long au plafond de 400 W. Le décodage tensor, lui, GAGNE (203) : c'est le préfill qui perd la cellule. À prouver (§ 4).

## 3. Règle au bit : Marlin au décodage, W4A4 au préfill — sans seconde disposition
* Coexistence des deux dispositions : +339,9 Mo × 48 = **+16,3 Go** ; le Coder pur laisse 10,4 Go libres après chargement (journal 217 A)
  → impossible ; même une moitié ne tient pas avec le KV.
* Déduction Marlin → naturelle par passe (`depaqueter_marlin`, 147 L3') : 340 Mo lus + 340 Mo écrits par couche = 0,4 ms/couche, **≈ 20 ms
  par préfill**, tampon 340 Mo — possible, mais inutile :
* **C17 existe déjà pour cela** : `_gemm_mma_marlin` (moe.py:752) fait tourner le noyau W4A4 (`nvfp4_gemm_grouped_mma`, le même que la
  branche `mma` du préfill) DIRECTEMENT sur les tuiles Marlin, sans copie, échelle globale naturelle à l'épilogue et décalage d'exposant
  `_decal_marlin` :734 ; servi au décodage sous opt-in `ACVRAM_MOE_DECODE_MMA_MARLIN` :2104, scellé C17 (1) « au bit de mma2 sur pile
  naturelle » (à l'octet d'échelle près : S0E5M3 exact pour toute échelle ≥ 2⁻⁶·facteur, ce que le refus 157 garantit pour les piles admises).
  **Règle** : au préfill (t > 32), sous disposition unique, prendre la branche `mma` via `_gemm_mma_marlin` au lieu de `marlin` ; au décodage,
  inchangé (tensor à T ≥ 8, GEMV Marlin en dessous). Coût en code : lever `not unique` :813 sous une variable (`ACVRAM_PREFILL_MOE=mma`),
  brancher `_gemm_mma_marlin` dans les trois GEMM de :940-944, test au bit contre la pile naturelle (jouet, comme test_c17), **≈ 60 lignes**.
* **Mais les piles de la 209 sont par (expert, colonne)** : `_decal_marlin` refuse (:741) — un décalage par pile n'existe plus. Il faut le
  facteur par ligne dans l'épilogue de mma2 (g [E, N] par colonne, geste du 209 (b) sur le Marlin porté : `gs_par_colonne`) : le facteur est
  une puissance de 2 par ligne, fl(fl(acc·f)·g/f) = fl(acc·g) → **au bit** de la naturelle ; **≈ ½ jour** (kernel.h, épilogue, test). Sans
  cela, la règle ne sert que les piles sans sous-normales : Qwen3.8, gemma, GLM — et 44 couches sur 48 du Coder i8c, 0 du pur.
* Résultat attendu de la règle sur le Coder pur (209 = 1 + règle) : préfill = b, décodage = tensor → **≈ b + 8 % ≈ 1 800 t/s, J −8 %** ; sur
  i8c : le 209 gagnait déjà +5,77 % avec 4 couches Marlin au préfill — avec la règle, ces 4 couches préfillent en W4A4 : ≥ +5,77 %.
  Chaque phase reste au bit d'un chemin servi aujourd'hui (préfill = naturel de be837ca1, décodage = tensor de main) ; la combinaison
  n'a jamais été servie : KL et PPL à sceller une fois (b=8, comme 209 c).

## 4. Recommandation (ordre)
1. **Prouver le mécanisme avant tout code, 3 min de carte** : un lot du banc 217 sous `nsys` (ou simplement `ttft` + pas médian par bras
   via `/metrics`) en a et en b : prédiction **préfill a − b ≥ +100 ms par lot de 2 048**, pas de décodage a − b ≤ −0,3 ms. Falsificateur :
   préfill a − b < +40 ms → le mécanisme est ailleurs (épilogue par colonne au décodage, aligneur), et § 3 ne vaut rien.
2. Si tenu : règle § 3 (60 lignes + ½ jour pour les piles par colonne), scellé KL/PPL b=8, ABBA banc 217 sur le pur ET sur i8c ; défaut
   du 209 rediscuté seulement après ce chiffre.
3. Rien à faire sur les modèles sans pile Marlin refusée (Qwen3.8, gemma) : la 217 les a mesurés neutres.
