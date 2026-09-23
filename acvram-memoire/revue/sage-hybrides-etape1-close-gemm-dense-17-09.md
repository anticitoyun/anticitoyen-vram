# Sage — Hybrides, étape 1 (fla) CLOSE sur ses gains ; le poste révélé n'est pas hybride, c'est le décodage dense à b > 1 qui relit les poids ligne par ligne — prochain noyau : GEMM W4A16 dense petit M, une lecture des poids par pas (17/09)

Entrée : `verdict-fla-17-09` + addendum (Laure 65b87b0), `verdict-nemotron-srcbf16-17-09` (08f1d26). fla : PPL + 0,0002, lancements 35 818 → 3 274, récurrence 2 % du pas ; b=12 Qwen3.8 93,9 → 128,0 t/s (+36 %), Nemotron 689,8 (+51 %), prefill 2 413 → 2 781 j/s (+15 %), b=1 63,7 → 69,9 (+9,8 %). `NARROW_NVFP4` écarté (51,3 / 85,9 < 128). Mes scellés ≥ 400 et ≥ 5× : **réfutés**, bâtis sur « la récurrence est le poste » — faux, le profil le dit deux fois.

## 1. Étape 1 close, fla en défaut

Équivalence tenue, gains réels sur quatre cellules et deux familles, régime porté par `regime_ligne()` : **`ACVRAM_GDN=fla` (et KDA/Mamba2) passe en défaut**, verdict daté, cellules aux menus (juge géo, source 65b87b0/08f1d26). Pas un tour de plus *sur fla* : il est à 2 % du pas, il n'y a plus rien à y prendre.

## 2. Le poste, lu dans le verdict : « GEMV dense NVFP4 à 12 lignes, 86,6 ms, 20 Go relus ligne par ligne à 0,23 To/s, O(b) »

C'est le fait le plus important de la journée, et il n'est pas propre aux hybrides : **à b > 1, le décodage d'un dense relit les poids une fois par séquence.** Plancher de bande : 20 Go / 1,79 To/s = **11 ms** ; mesuré 86,6. vLLM y est à 19 ms (621 t/s : Marlin lit une fois, M = 12). Toute entrée dense des menus à b > 1 (Qwen3-14B, Coder-14B, Llama-8B, Gemma-4-26B, Kimi-Linear, la partie dense des hybrides) porte le même défaut ; sur Coder MoE il n'apparaît qu'aux projections denses (`_etroit` 11 %). `NARROW_NVFP4` (le noyau C, MMA étroite) est *plus lent* que la boucle : il n'a pas été fait pour N = K = 5 120 à M = 12.

### Chantier suivant (Laurine) : GEMM W4A16 dense, 2 ≤ M ≤ 32, bornée par la bande

* Forme du problème : M = 12 lignes, K × N = 5 120 × 5 120 à 5 120 × 25 600, poids NVFP4 lus **une fois par pas**, déquantifiés en registres, MMA bf16 sur une tuile M = 16 (padding), 0,65 TFLOP par pas — **ce n'est pas un problème de FLOPs, c'est un flux** : le noyau est une GEMV vectorisée qui garde 12 lignes d'activation en registres/SMEM et balaie K. Triton suffit si le micro-banc le dit (pas de dequant fusionnée façon B1' à 58 TFLOPS : ici la crête utile est la bande, pas les tensor cores).
* **Porte micro-banc à sec, avant toute intégration** : `outils/banc-gemm-groupe-17-09.py` étendu à la forme dense M = 12 — octets/temps ≥ **1,3 To/s** (73 % de la bande) sur q/k/v/o et gate/up/down de Qwen3.8 ; faux si < 0,9 To/s (alors Triton n'y arrive pas et c'est un noyau CUDA, décision séparée). Témoins dans le même banc : la boucle GEMV actuelle (attendu ~0,23), `NARROW_NVFP4` (attendu pire).
* Scellés en situ (Laure, 30 min) : Qwen3.8 calibA b=12 **128 → ≥ 500 t/s** (plancher 11 ms ⇒ ≤ 1 090 ; à 73 % ⇒ ~800 ; faux si < 300) ; J/jeton 3,12 → ≤ 1,0 ; b=1 inchangé ± 3 % (M = 1 garde la GEMV) ; PPL et `ppl-decode-kv` identiques ± 0,0005 (mêmes codes, même ordre d'accumulation par tuile : test d'équivalence logits vs boucle GEMV dans le même commit, bras cassant : décaler l'échelle de bloc d'un rang → rouge). Coder b=12 : `_etroit` 1,09 ms → ≤ 0,6 ms, bonus.
* Issue qui me gênerait : le micro-banc donne ≥ 1,3 To/s et l'en situ < 300 — alors le pas a un autre poste O(b) (attention, KV, colle) que le profil de Laure aurait dû montrer, et on relit son top 25 avant d'écrire une ligne.

## 3. Nemotron srcbf16 : 1,0632, pas « calibration si la classe est voulue » d'abord

vLLM NVFP4 *officiel* est à 0,987 : la classe est atteignable sur ce modèle, donc l'écart est dans **ce que nous quantifions et qu'eux ne quantifient pas**. Contrôle à sec (Manon, 20 min) : dtype tenseur par tenseur du checkpoint officiel (sur disque, celui de vLLM) contre le nôtre — prédiction : `in_proj`/`out_proj` des couches Mamba2, ou `dt`/`A_log`/conv, en bf16 chez eux, NVFP4 chez nous (ou expert partagé). Si la liste diffère ⇒ reconversion avec les mêmes exclusions (à sec, 1 h), PPL Laure 20 min, prédiction ≤ 1,020 ; si la liste est identique ⇒ alors seulement la calibration. Pas avant.

## 4. Porte A4 des denses : pas maintenant, et pas sans migration d'échelles

Prefill dense = GEMM cutlass bf16 à 200 TFLOPS mesurés (548 ms pour 110 TFLOP) : **au mur du bf16**, W4A16 n'a plus rien à donner ; vLLM CUTLASS W4A4 8 876 j/s (×3,2) le confirme. Mais : (a) la porte experts a coûté +0,007-0,010 sur gate/up seuls ; sur *toutes* les projections d'un dense, le prior est > +0,012 ; (b) Qwen3.8 est à 1,0253, non classé — l'ouvrir en A4 n'améliore aucune cellule classée. Elle ne se scelle qu'avec la migration d'échelles (`sage-w4a4-porte-fermee` § 2) et sur un dense **classé** (Coder-14B, 1,0143) : une porte, après le GEMM dense, prédiction écrite à ce moment-là.

## 5. Ordre

1. Laurine : micro-banc GEMM dense M = 12 (à sec) → intégration + test d'équivalence → Laure 30 min.
2. Manon : dtype officiel vs nôtre sur Nemotron (à sec), reconversion si écart.
3. Laurine, ensuite : GEMV experts ≥ 85 % (Coder b=12), `sage-lecture-profils` § 2.
4. Katy : fla et Nemotron aux menus (juge géo, sources).
