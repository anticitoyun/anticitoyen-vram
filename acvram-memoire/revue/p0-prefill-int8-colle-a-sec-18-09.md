# P0 — préfill : GEMM W8A8 sans déquantification par appel (q/k/v/o) et colle MoE en deux lancements ; livré à sec, régimes en témoin jusqu'au scellé (Laurine, 18/09)

Commande : sage-profil-verdict-18-09 § 1 (ii) — profil Laure (c611bc7,
Coder 2 048) : int8_dequant 11,2 + colle MoE 8,3 + elementwise 6,7 = 26,2 ms
≥ 25 → P0. Scellé : **≥ 11 000 j/s** (< 11 000 faux) ; porte PPL privé
≤ 1,020 (prédiction Sage ≤ 1,017).

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

## 3. Ce qui attend la carte (Laure) — trois bras, trois prédictions (Sage)

Deux régimes en témoin (défauts bf16 / torch) jusqu'au scellé, un commit par
fusion. Avant toute mesure, dans la fenêtre de Laure (je n'ai pas de carte) :
(a) `pytest tests/test_gemm_w8a8.py tests/test_colle_moe.py` sur carte — la
suite GPU des deux noyaux Triton neufs (REGLES § 7), jamais exécutée sur
carte à ce jour ; (b) cache Triton vidé (`rm -rf ~/.triton/cache`, REGLES
§ 6) puis un préfill de chauffe hors mesure (le JIT des deux noyaux se fait
là, pas dans le chrono). Instrument : `prefill-dense-acvram` Coder 2 048,
`regime_ligne()` en tête (elle porte `prefill_int8=` et `ACVRAM_COLLE_MOE`
hors défaut), profil torch.profiler par bras pour lire les postes.

| bras | régime | prédiction (pas GPU 197,0 ms ; 9 844 j/s) |
|---|---|---|
| T témoin | défauts (bf16, torch) | 197 ± 2 ms — 9 750-9 950 j/s |
| A a8 seul | `ACVRAM_PREFILL_INT8=a8` | int8_dequant 11,2 → 0 ; cutlass 19,4 → 9-11 ⇒ **−20 ms** (fourchette −17 à −23) ⇒ 174-180 ms, 11 400-11 800 j/s |
| B a8 + colle | `+ ACVRAM_COLLE_MOE=triton` | colle 8,3 → 2-3 ⇒ **−6 ms** de plus (−5 à −7) ⇒ 168-174 ms, 11 800-12 200 j/s |

Verdicts, écrits avant : scellé P0 sur B ≥ 11 000 j/s (< 11 000 faux : lire
le profil — le W8A8 Triton n'a pas 2× cutlass sur ces formes, ou la colle
n'a pas bougé). La colle passe en défaut seulement si **B − A ≥ 3 ms** ;
sous 3 ms « moins de lancements n'est pas un gain » (REGLES § 4), elle reste
témoin. a8 passe en défaut si A − T ≥ 15 ms ET PPL privé ≤ 1,020
(prédiction ≤ 1,017 ; > 1,020 faux : A8 par groupe de 128, un noyau de plus,
décision séparée). Issue qui me gênerait : A − T ≈ −11 (la déquant seule,
le GEMM int8 pas plus rapide que cutlass bf16 : alors le tl.dot int8 de
Triton n'atteint pas les tensor cores int8 à 2×, et c'est un profil avant
toute ligne).

## 4. Verdict (Laure, verdict-p0-prefill-a8-colle-18-09) : FAUX — et la dernière porte

a8 Triton : `_w8a8_kernel` 32,1 ms contre déquant 11,2 + cutlass 19,4 = 30,6
(A − T = +2,0 ms GPU, prédit −17 à −23 ; ≈ 28 TOPS : troisième GEMM Triton
perdante après B1/B1' — le `tl.dot` int8 → int32 par groupe avec la remise
à l'échelle par jeton ne rend pas le 2× de l'INT8). Régime `a8` : témoin.
Colle : lancements 5 017 → 2 857 tenus, GPU +0,4 ms (le tri en un
lancement coûte ce que valaient les sept petits noyaux), mur −4,8 ms
(−2,3 %) — B − A = +233 j/s > 2σ (≈ 208) : Sage tranche.

Dernière porte P0-a8 (Sage) : `outils/banc-int-mm-a8-18-09.py` — la voie
cuBLASLt int8 (`torch._int_mm`) sur les quatre formes q/k/v/o Coder 2 048,
× 48 : **≤ 16 ms ⇒ in situ a8-cublas (2 min), > 16 ms ⇒ P0-a8 fermé, pas de
troisième noyau.** Ce que le banc mesure, et ce qu'il faut savoir avant de
lire le chiffre : `_int_mm` fait UN produit int32 sur K entier — il ne sert
qu'avec des poids int8 **symétriques par canal** (une échelle par ligne de
sortie), pas avec notre INT8 affine par groupes de 128 ; la voie (b)
mesurée suppose donc une **requantification** des q/k/v/o (chargement ou
conversion : porte PPL à part, et +1 o/poids en VRAM ≈ 1 Go sur Coder si le
décodage garde la forme par groupes). La voie (a) « par groupe » (NG
`_int_mm` de K = 128 + NG épilogues) respecte nos poids et est mesurée
aussi ; attendue hors porte. Arithmétique des deux voies contrôlée à sec
contre une référence par groupes.

Prédiction scellée (banc, × 48) : (b) quant A8 + `_int_mm` + épilogue ≈
0,25-0,35 ms/couche ⇒ **12-17 ms** — la porte 16 au bord haut de la
fourchette ; cuBLASLt int8 sur 5090 ≈ 350-450 TOPS sur q/o, moins sur k/v
(N = 512, sous-occupés) ; (a) 60-120 ms (hors porte) ; témoin bf16 ≈ 19-21
ms (le 19,4 en situ). Faux si (b) > 16 : fermé, comme écrit ; si (b) ≤ 16,
le gain in situ = 30,6 − (b) ≈ 14-18 ms ⇒ 10 600-10 800 j/s — **sous le
scellé 11 000** même si la porte ouvre : je l'écris avant la mesure, Sage
décide si l'in situ vaut ses 2 minutes.
