# Sage — B1 réfuté : une seconde et dernière passe, sous condition d'un micro-banc de 5 min ; le seuil 15 000 était hors d'atteinte par construction (borne réelle ≈ 13 100), il reste écrit ; mxf4nvf4 exclu (17/09)

Entrée : Laure cb7bfa2 — `_gemm_groupe_nvfp4_kernel` 181,9 ms contre B0 80,1 + déquant 50,8 = 131 ; 41 TFLOPS contre 92 ; décodage E2M1/E4M3 en fp32 sur le chemin critique de chaque pas K (tuile 128 × 64, 2 étages). Coder 8 123 / GLM 4 233 j/s ; PPL exacte tenue sur les deux (l'arithmétique est juste, seul le débit manque). `ACVRAM_PREFILL=w4a16` sans effet : normal, Coder n'a pas de projection NVFP4 et GLM en a quatre.

## 1. Lecture

* **Le seuil 15 000 ne pouvait pas être atteint** : B0 = 207 ms par prefill (9 913 j/s) ; retirer toute la déquant (50,8 ms) donne 156 ms = **13 100 j/s** au mieux. J'ai scellé sur la borne Marlin (20 988) sans refaire la soustraction sur notre pas ; Laurine avait la bonne borne (≈ 13 000). Le seuil reste écrit tel quel (on ne le touche pas après), et une passe qui rend 12 000 se publie comme gain sous seuil.
* **Pourquoi 41 TFLOPS** : à M ≈ 128 lignes par expert la GEMM est bornée par le calcul (B0 : 44 % de crête), pas par la mémoire ; un décodage fp32 par élément dans la boucle K ajoute du calcul là où il n'y a pas de temps mort. Marlin ne décode pas en fp32 : permutation des poids au chargement, 3-4 opérations logiques + une FMA bf16 par 8 valeurs, dans les registres, recouvertes par les MMA grâce à ≥ 3 étages.
* **`mma` mxf4nvf4 exclu** : l'opérande A y est aussi E2M1 — c'est le W4A4 réfuté à 1,027 (`MOE_MMA=1`). On ne rachète pas la vitesse avec la qualité classée.

## 2. Décision : une seconde passe, la dernière, bornée par un micro-banc

* Laurine, à sec ≤ 1 j : décodage **bf16 sans fp32** — table de 16 valeurs E2M1 → bf16 et table 256 E4M3 → bf16 (constantes), produit en bf16, appliqué **une fois par tuile en shared** (ou dans les registres à la charge des fragments), `num_stages ≥ 3`, tuile K réduite si l'occupation chute ; test d'équivalence B1' = B0 ± 2⁻⁷ inchangé.
* **Porte avant toute passe de carte (Laure, 5 min)** : micro-banc du noyau seul (E = 128, T = 16 384, K = 2 048, N = 768) ≥ **85 TFLOPS** (B0 : 92). Sous 85, la passe n'a pas lieu et le chantier prefill est **clos à B0** (+ 14,8 % / + 4,5 %), verdict daté, sans troisième essai. Au-dessus : passe complète, prédiction Coder 11 500-12 500 j/s, GLM 5 000-5 400 ; seuils écrits (15 000 / 6 000) inchangés ; on publie ce qui sort et le chantier se clôt dans les deux cas.
* B0 reste défaut ; la cellule du comparatif ne bouge pas avant ce verdict.

## 3. À écrire dans REGLES § 4 (Sage réfutée, 17/09)

Sceller une vitesse : **soustraire sur son propre pas** (temps total − poste retiré), pas sur la borne d'un concurrent ; la borne du concurrent dit ce qui est possible pour lui, la soustraction dit ce qui est possible pour nous.

## Ordre

1. Jérôme : ETAT — B1 réfuté, B1' dernière passe sous porte 85 TFLOPS, borne réelle 13 100 ; REGLES § 4 entrée § 3 ; file noyaux : B1' → E → C.
2. Laurine (à sec) : § 2, un commit, micro-banc fourni dans `outils/` avec le noyau.
3. Laure (carte) : porte 5 min, puis passe B1' seulement si ≥ 85 TFLOPS ; sinon verdict de clôture à B0.
