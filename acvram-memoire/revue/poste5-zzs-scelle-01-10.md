# zzs — cœur MLA du préfill : troncature causale des clés (poste5, 01/10) — borne, prédiction et critère d'équivalence SCELLÉS avant le code

Ordre de chef (bd zzs, suite de `poste5-ht9-verdict-01-10.md`). Base : dac452617 (origin/main), branche poste5-mla-causal.

## Ce qui change

`mla.py:571-581` : chaque morceau de 256 requêtes [d0, d1) calcule `einsum(q_eff[d0:d1], C32)` sur les `total` clés, puis
masque celles qui sont au-delà de `passe + d1` (`:578`), puis fait softmax et produit par V sur `total`. La pièce limite
les deux produits, le masque et le softmax à `C32[:passe + d1]`. La règle des 2 048 clés (`cles = passe + d1`), les
dtypes, la taille des morceaux et le chemin flash ne changent pas. `ACVRAM_MLA_CAUSAL=0` rend l'ancien chemin (témoin).

## Borne (Kimi-Linear-35B, 7 couches MLA, 32 têtes, rang 512 + rope 64)

Travail du cœur proportionnel au nombre de clés traitées, S, par morceau : 2 × 256 × 32 × (576 + 512) × S FLOP, et
256 × 32 × S éléments pour le masque, l'échelle et le softmax.

| t | morceaux fp32 (> 2 048 clés) : Σ S tronqué / complet | morceaux tf32 (≤ 2 048) | ensemble |
|---|---|---|---|
| 8 192 | 125 952 / 196 608 = **64,1 %** | 9 216 / 65 536 = 14,1 % | 51,6 % |
| 32 768 | 2 104 320 / 3 932 160 = **53,5 %** | 9 216 / 262 144 = 3,5 % | 50,4 % |

Mesuré au ht9 (p81:prefill8k, 1 415,9 ms GPU) : sgemm fp32 491,3 ms, tf32 90,4 ms, masque 76,0, échelle 75,9 et softmax
76,2 ms (224 lancements = 7 × 32 morceaux). Borne du gain à 8 k, à rendement de noyau égal :
−176,6 (fp32) − 77,7 (tf32) − 110,4 (élémentaire) = **−364 ms, soit −25,7 % du GPU du préfill**.
À 32 k (extrapolation : cœur quadratique, le reste linéaire), préfill ≈ 15,7 s dont cœur fp32 ≈ 9,8 s et élémentaire
≈ 3,6 s ; borne du gain ≈ −6,7 s, soit **−43 %**.

## Prédiction (scellée)

* 8 k, nsys p81 (`p81:prefill8k`, GPU) : **−280 à −365 ms, soit −20 à −26 %** (le bas de la fourchette paie la perte de
  rendement des GEMM courtes : les morceaux tf32 voient 256 à 2 048 clés).
* 8 k, mur du préfill (instrument ci-dessous) : **−18 à −26 %**.
* 32 k, mur du préfill : **−35 à −48 %**.
* Décodage b=1 et b=12 : inchangés (le chemin t = 1 n'est pas touché), 0 ± 1 %.
* **FAUX si** le gain du GPU à 8 k est inférieur à 150 ms (−10,6 %), ou si le gain au mur à 32 k est inférieur à 20 %.
* Issues nommées : (i) cuBLAS choisit, pour les S courts, des noyaux moins efficaces ou un découpage en K, ce qui ronge
  le gain des morceaux tf32 (le bas de la fourchette) ; (ii) le softmax PyTorch change d'algorithme ou de taille de bloc
  selon la longueur de ligne (non vérifié dans le source de la version installée), ce qui changerait l'ordre de
  somme pour les morceaux courts et coûterait le « au bit » ; (iii) la gênante : le mur à 32 k est dominé par autre
  chose que je n'ai pas mesuré (mémoire des scores de 1 Go par morceau, allocateur), et le gain au mur est loin de la
  borne GPU.

## Critère d'équivalence (scellé, dans cet ordre)

* **E1 — au bit, visé.** (a) Test sur carte : la sortie du préfill MLA (y, [t, nh·dv]) est `torch.equal` entre
  ACVRAM_MLA_CAUSAL=1 et =0, pour t = 8 192 sans passé et pour t = 1 000 avec passe = 3 000 (préfill par morceaux),
  au régime servi (tf32 ≤ 2 048 clés, fp32 au-delà). (b) Bout en bout : ids gloutons de 64 jetons après une invite de
  8 192 et une de 32 768, A contre B IDENTIQUES, avec les témoins A1 = A2 et B1 = B2.
* **E2 — si E1 tombe : tolérance dérivée de l'ordre de somme, sous témoin (comme lic, 2 × le témoin).** Le témoin T est
  l'écart de l'ANCIEN chemin à lui-même sous un autre ordre de somme légitime : morceaux de 128 au lieu de 256
  (`ACVRAM_MLA_MORCEAU`, diagnostic), ce qui change la longueur des GEMM et des softmax par morceau, comme la
  troncature. Exigence : max |y_B − y_A| ≤ 2 × max |y_A128 − y_A| par couche dans le test sur carte, et pour le bout en
  bout, |Δ logprob| des 64 jetons gloutons ≤ 2 × le témoin, les ids pouvant diverger seulement après le premier jeton
  où le témoin lui-même diverge.
* Le verdict dit lequel de E1 ou E2 tient. Si ni l'un ni l'autre : FAUX en équivalence, rien n'est servi.

## Tests cassants (même commit que le code)

* Sans carte (CPU, CUDA_VISIBLE_DEVICES="" et TRITON_INTERPRET) : la forme. Chaque morceau ne traite que `passe + d1`
  clés (einsum espionné). Le test casse si l'on revient à `total`, ou si un décalage de un fait perdre la diagonale.
  Écart au chemin complet ≤ 2 × celui du témoin à morceaux de 128, contre une référence fp64.
* Sur carte : E1(a) au bit, ou E2 selon le verdict, avec le bras cassant « troncature à `passe + d1 − 1` » qui doit
  rendre faux.

## Amendement du 01/10 (~15 h 30), AVANT toute mesure sur carte : le témoin de E2

Le témoin « morceaux de 128 » était dégénéré. Dans le chemin complet, chaque produit et chaque softmax porte sur
`total` clés quelle que soit la taille des morceaux. Changer 256 en 128 ne change donc aucune longueur de somme : sur
le processeur, A128 = A au bit (mesuré à sec, 300 + 700 jetons). Un témoin nul ne peut rien borner. Il est remplacé par
**l'erreur d'arrondi du chemin servi contre une référence fp64** (même module, même entrée, tout en fp64) :

* E2, test : max |y_B − y_A| ≤ 2 × max |y_A − y_ref| ET max |y_B − y_ref| ≤ 2 × max |y_A − y_ref|. À sec en fp32
  (forme réduite) et sur carte en fp32, aux formes de Kimi avec un passé (3 000 + 1 000), au régime servi.
* E2, bout en bout : inchangé pour les ids (64 jetons gloutons après 8 k et 32 k, témoins A1 = A2 et B1 = B2). Une
  divergence est publiée avec son premier jeton, et la décision revient au chef ; aucune tolérance n'est inventée après coup.
* `ACVRAM_MLA_MORCEAU` est retiré : il ne servait que ce témoin.
* Déjà constaté à sec : sur le processeur, B n'est PAS au bit de A (|B − A| = 1,0e-7 pour max |A| = 0,28), et
  |B − ref| = |A − ref| = 1,4e-7 : la troncature est aussi juste que le chemin complet. E1 se joue sur la carte, où
  cuBLAS et le softmax de PyTorch décident.
