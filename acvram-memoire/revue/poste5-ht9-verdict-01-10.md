# ht9 verdict — préfill MoE servi de Kimi-Linear (8 k) : PAS DE LEVIER (Marlin à 12,1 % du préfill et à 64 % de sa borne de calcul ; prédiction FAUSSE) ; le préfill se perd dans le cœur MLA en fp32 (41 % en GEMM seules) (poste5, mesureuse, 01/10)

* scellé : `poste5-ht9-borne-01-10.md` (ce248cefc, avant toute trace). Borne MoE : 24,1 TFLOP ≈ 110 ms (calcul),
  13,6 ms (octets). Prédit : Marlin MoE 25-50 % du préfill, 15-35 % de la borne ; colle 3-10 %.
* commit : poste5-ht9 ce248cefc = a77970f45 (origin/main) + la note. .so précompilé 535cd9a984565f86 (source de main).
* instrument : `outils/gpu/mesure/nsys-kda-p81.sh` inchangé, plage `p81:prefill8k` (une séquence de 8 192 jetons,
  `nsys stats nvtx_kern_sum`). Classes de noyaux écrites depuis le source avant la trace (lanceur et `classes_ht9.py`
  hors dépôt, scratchpad de session). Sorties : `scratchpad/poste5-ht9-01-10/` du worktree (classes.txt, non commitées).
* régime : Kimi-Linear-35B-kda-nvfp4, NOMINAL (graphes on, 0 exilée), mla_core par défaut (tf32 ≤ 2 048 clés, fp32
  au-delà), 5090 seule. nvidia-smi au début (14:50:01) et à la fin (14:51:12) : seul llama-server 4436, sur la 3080 Ti.
  Pause e50.2 posée à 14:45:37 ; ligne « === pause … attente 14:49:45 » postérieure vérifiée ; retirée à 14:51:33.
* contre-épreuve : la trace nsys-A de la prise ddw (84123994d, même instrument, 3 h plus tôt) donne 1 394,3 ms, Marlin
  12,0 %, GEMM fp32 50,1 % — à ± 1,5 % de cette prise, classe par classe.

## Mesuré (p81:prefill8k, 1 415,9 ms GPU)

| classe | ms | part | lancements |
|---|---|---|---|
| GEMM cutlass hors MoE | 714,7 | 50,5 % | 1 138 |
| — dont sgemm fp32 SIMT tn + nn (cœur MLA, morceaux > 2 048 clés) | 491,3 | 34,7 % | 168 + 168 |
| — dont s1688gemm tf32 tn + nn (cœur MLA, morceaux ≤ 2 048 clés) | 90,4 | 6,4 % | 56 + 56 |
| élémentaire et colle torch (masque, échelle, softmax, copies) | 310,2 | 21,9 % | 5 259 |
| **Marlin MoE routé** | **171,1** | **12,1 %** | 156 = 26 couches × 6 |
| attention MLA (noyaux nommés) | 76,2 | 5,4 % | 224 |
| KDA | 72,6 | 5,1 % | 320 |
| int8 (expert partagé, etc.) | 27,7 | 2,0 % | 502 |
| colle MoE | 21,9 | 1,5 % | 497 |
| autre (dont conv depthwise fp32 20,3) | 21,5 | 1,5 % | 405 |

* Marlin MoE contre la borne de calcul de 110 ms : **rendement 64 %** (141 TFLOPS effectifs).
* Attribution des GEMM fp32 et tf32 au cœur MLA, par le code ET par les comptes : `mla.py:571-581`, morceaux de 256
  requêtes. À 8 192 jetons, 24 morceaux voient plus de 2 048 clés (fp32) et 8 au plus (tf32, règle des 2 048 clés,
  `mla.py:114`) ; 7 couches MLA donnent 168 et 56 lancements par produit. C'est exactement ce que compte nsys.

## Verdict

* **P1 FAUX** (Marlin à 12,1 % du préfill, < 15 %) et **P2 FAUX** (rendement 64 %, > 50 %). Seuil de pièce non atteint
  (ni ≥ 15 % à ≤ 50 % de la borne, ni colle ≥ 5 %) : **PAS DE LEVIER sur le préfill MoE servi.** Issues (i) et (ii)
  réalisées : Marlin est déjà près de sa borne, et le préfill se perd ailleurs.
* **Où il se perd** : le cœur d'attention MLA du préfill. Ses GEMM seules coûtent 581,7 ms (41 % du préfill), en grande
  partie en fp32 SIMT, sans cœurs tensoriels, à cause de la règle des 2 048 clés (verdict-regime-2048-cles-19-09 : LA
  RÈGLE RESTE). S'y ajoute une part non attribuée finement des 310 ms d'élémentaire (masque, échelle, softmax : 276 et
  264 lancements à 76 ms chacun).
* Fait de code, non mesuré séparément : chaque morceau calcule ses scores contre TOUTES les clés (`mla.py:574` :
  `einsum(q_eff[d0:d1], C32)` sur `total` lignes), puis masque celles qui sont au-delà de `passe + d1` (`:578`).
  À 8 k jetons, les morceaux fp32 n'ont besoin que de 64 % des clés calculées (Σ d1 = 125 952 sur 24 × 8 192), et les
  morceaux tf32 de 14 %.

## Suite proposée (décision chef, pièce nouvelle hors ht9)

Troncature causale du cœur MLA au préfill : les scores et le produit par V limités à `C32[:passe + d1]`. Même règle des
2 048 clés et mêmes dtypes. Gain arithmétique à 8 k : −36 % sur les sgemm fp32 (≈ −177 ms), −86 % sur les tf32
(≈ −78 ms), et l'élémentaire en proportion. Soit −250 à −345 ms, **−18 à −24 % du préfill de 8 k**, davantage à 32 k.
Sortie : le produit par V tronqué ne retire que des termes nuls (exp(−inf) = 0, au bit si cuBLAS garde le même noyau
sans split-K), mais la longueur de ligne du softmax change l'ordre de sa somme. Donc au bit à vérifier, sinon tolérance
dérivée de l'ordre de sommation fp32 (REGLES § 9). Scellé à écrire avant le code.
