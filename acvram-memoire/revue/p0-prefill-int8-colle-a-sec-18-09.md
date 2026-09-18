# P0 — préfill : GEMM W8A8 sans déquantification par appel (q/k/v/o) et colle MoE en deux lancements ; livré à sec, régimes en témoin jusqu'au scellé (poste4, 18/09)

Commande : poste7-profil-verdict-18-09 § 1 (ii) — profil poste3 (c611bc7,
Coder 2 048) : int8_dequant 11,2 + colle MoE 8,3 + elementwise 6,7 = 26,2 ms
≥ 25 → P0. Scellé : **≥ 11 000 j/s** (< 11 000 faux) ; porte PPL privé
≤ 1,020 (prédiction poste7 ≤ 1,017).

## 1. GEMM W8A8 — `kernels/gemm_w8a8.py`, régime `ACVRAM_PREFILL_INT8 = bf16 | a8`

Sur le Coder de la campagne (`Qwen3-Coder-30B-A3B-nvfp4`), q/k/v/o sont
INT8 affine par groupes de 128 (q 4 096×2 048, k/v 512×2 048, o 2 048×4 096).
Au-delà du seuil GEMV (80 lignes), `int8_matmul` déquantifiait la matrice
entière en bf16 à chaque appel puis lançait cutlass (11,2 + 19,4 ms).
Maintenant, sous `a8` : activation quantifiée en int8 **par jeton** (absmax
/ 127, échelle fp32, un lancement), poids uint8 lus tels quels, `tl.dot`
int8 → int32 par groupe (exact : |Σ| ≤ 128·127·127), zéro par le terme
`− (z − 128)·Σ_k a8` (somme de ligne dans la tuile), échelles de groupe et
de jeton en fp32 : `y = s_x · Σ_g s_w·(a8·(q−128)ᵀ − (z−128)·Σa8)`. Aucune
déquantification, aucun tampon bf16. Seule erreur : l'A8 par jeton (mesurée
à sec ≈ 0,4 % RMS contre x bf16 exact ; c'est ce que la porte PPL juge).
Câblé dans `int8_matmul` (n > seuil, x bf16/fp16, K multiple du groupe) ;
la tête reste en fp32 exact (x fp32 → chemin inchangé) ; `regime_ligne()`
écrit toujours `prefill_int8=bf16|a8`.
Juge (`tests/test_gemm_w8a8.py`, interpréteur à sec) : sortie = produit
fp32 de l'activation quantifiée par la déquant fp32 à 2⁻⁷ × Σ|a·w| (M
2 048/300/16, K 2 048/4 096/256) ; erreur A8 bornée (< 1 %) ; bras cassant :
zéro ignoré → rouge ; `int8_matmul` emprunte a8 sous le régime et bf16 sinon.

## 2. Colle MoE — `kernels/colle_moe.py`, régime `ACVRAM_COLLE_MOE = torch | triton`

`_forward_prefill_grouped` : argsort (radix, 8 lancements × 48 couches =
les « 384 appels ») + bincount, puis `_tuiles` (cumsum × 3, searchsorted,
arange, clamps ≈ 10 lancements). Maintenant : `trier_paires` — UN
programme trie les G = T·top_k clés composites `e·2¹⁶ + paire` par
`tl.sort` (bitonique, déterministe ; la clé rend l'ordre stable exact
d'`argsort(stable=True)`) et l'histogramme par `tl.histogram` ; `tuiles` —
UN programme rend la grille (e, t0, n) de `_tuiles` (cumsum sur E, expert de
chaque tuile par comparaison [t_max, E]). G ≤ 32 768, E puissance de 2 ;
sinon torch. Juge (`tests/test_colle_moe.py`) : `ordre`, `e_tri`, `cnt`
identiques à torch (routage Coder 2 048 × 8 avec experts vides, petit,
T = 1) ; tuiles identiques à `MoEBlock._tuiles` (bt 16 et 128) ; bras
cassant : un compte faux change les tuiles. La sortie de la GEMM ne dépend
pas de la place d'une ligne dans sa tuile : identique par construction.

## 3. Ce qui attend la carte (poste3)

Deux régimes en témoin (défauts bf16 / torch) jusqu'au scellé, un commit par
fusion. Mesure : `prefill-dense-acvram` Coder 2 048 aux défauts, puis
`ACVRAM_PREFILL_INT8=a8`, puis `+ ACVRAM_COLLE_MOE=triton` ; PPL privé sous
a8 (porte ≤ 1,020) ; profil (torch.profiler) sous les deux pour lire les
postes. Prédiction scellée : a8 : int8_dequant 11,2 → 0, cutlass 19,4 →
9-11 (int8 tensor cores ≈ 2× bf16) ⇒ −20 ms ; colle : 8,3 → 2-3 ms ⇒ −6 ;
pas 197 → ~171 ms ⇒ **≈ 12 000 j/s** (tenu si ≥ 11 000 ; faux si
< 11 000 — alors le W8A8 Triton n'atteint pas 2× cutlass sur ces formes,
lire le profil) ; PPL ≤ 1,017 (faux si > 1,020 : A8 par jeton trop
grossier sur q/k/v — la voie serait A8 par groupe de 128, un noyau de plus).
