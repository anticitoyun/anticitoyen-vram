# poste7 — P0 réfuté : la GEMM int8 Triton perd contre déquant + cutlass (+2,0 ms) ; une porte cuBLASLt de 2 min avant de fermer ; la colle Triton se juge sur la dispersion, pas sur le compte de lancements ; le poste prefill reste P1 (18/09)

Entrée : poste3 a04e5c6 `verdict-p0-prefill-a8-colle-18-09` — T 9 811 j/s (196,9 ms GPU) ; A (a8) 9 744 : `_w8a8_kernel` 32,1 ms contre int8_dequant 11,2 + cutlass 19,4 = 30,6 (+2,0, prédiction −17/−23 **réfutée**, pire que l'issue gênante −11) ; B (a8 + colle) 9 977 < 11 000 : `_tri_kernel` 2,4 ms, GPU +0,4, mur −4,8 (lancements 5 017 → 2 857) ; PPL a8 1,0157 (≤ 1,020 tenu, sans objet). Aucun défaut basculé. Préalables tous faits.

## 1. a8 : réfuté sur moi, et fermé pour Triton

J'ai prédit « int8 = moitié de bf16 » sur les tensor cores sans compter ce que Triton en tire à ces formes : 32,1 ms pour ~0,9 TFLOP int8 = 28 TOPS, 3 % de la crête int8 — même famille que B1 (41 TFLOPS) et B1' (58) : **Triton ne sort pas une GEMM compétitive de nos formes, trois fois de suite.** Leçon carnet : un gain « par format » se prédit avec le noyau qui l'exécute, pas avec la crête du format. Le régime `ACVRAM_PREFILL_INT8=a8` reste en témoin, jamais défaut.
**Porte unique avant fermeture, 2 min de carte (poste4 écrit, poste3 lance)** : `torch._int_mm` (cuBLASLt int8) sur les quatre formes q/k/v/o de Coder 2048, activation déjà quantifiée, somme ≤ **16 ms** (contre 30,6 déquant + cutlass ; issue gênante : cuBLASLt int8 n'est pas 2× le bf16 à K = 2 048 sur sm_120 et rend 22-28). ≤ 16 → un seul in situ, bras `a8-cublas`, scellé inchangé ≥ 11 000 ; > 16 → **P0-a8 fermé**, verdict daté, pas de troisième noyau.

## 2. Colle Triton : la règle § 4 ne s'applique pas telle quelle au prefill, et c'est la dispersion qui juge

« Moins de lancements n'est pas un gain » vise le décodage sous graphe (un nœud rejoué coûte 0,5 µs). Le prefill est **eager** : 2 160 lancements de moins × ~2 µs de mise en file hôte ≈ 4,3 ms, et le mur a rendu −4,8 avec GPU +0,4 — le mécanisme est celui-là, compris, sortie identique (ordre/cnt/tuiles, experts vides testés : règle 9 tenue). Ce n'est pas un chiffre qui « ne compte pas », c'est un petit gain (+1,7 %) qui doit prouver qu'il sort du bruit : poste3 publie l'écart-type inter-répétitions (7) de T et de B ; **défaut `ACVRAM_COLLE_MOE=triton` si (T − B) mur ≥ 2 σ, sinon témoin.** Une ligne REGLES § 4 : la règle sur les lancements porte le régime « sous graphe » dans son énoncé.

## 3. Le poste prefill n'a pas bougé : GEMM groupée 82 + `nvfp4_dequant` 51 = 68 % — c'est P1

Rien d'autre à sec n'entame ces 133 ms. P1 (port Marlin, porte 137 TFLOPS, trois bandes) attend le oui de l'utilisateur ; à sa place je n'ouvre rien d'autre sur le prefill.

## Ordre

* poste4 : banc `_int_mm` 4 formes (à sec, 20 min) ; poste3 : 2 min de carte, verdict ≤ / > 16 ms ; in situ `a8-cublas` seulement si ≤.
* poste3 : σ de T et B → colle en défaut ou non (une ligne, pas de carte).
* chef : ETAT — P0-a8 Triton réfuté (porte cuBLASLt en cours), colle selon σ, prefill = P1 en attente du oui ; REGLES § 4 ligne « sous graphe » ; carnet poste7 : deux prédictions réfutées.
