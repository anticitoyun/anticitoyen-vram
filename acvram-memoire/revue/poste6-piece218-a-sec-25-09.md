# 218 — à sec (poste6, 25/09) : un noyau persistant gate·up → act → down par expert, chiffré ; gain au bit 0,10-0,17 ms/pas b=1 (3-6 %) pour ≈ 2 jours — NON recommandé avant un banc « fantôme » d'une minute qui fixerait la seule inconnue

* instrument : lecture de code (fichier:ligne) et des chiffres de la 214 (`poste6-piece214-verdict-25-09.md`, banc synthétique L2 froide),
  de la 206 (trous réels sous graphe), de la 141 (PDL en service b=1) et de la 194 a-sec (poste1, mégakernel/PDL). Aucune carte.
* commit : poste6-218 = origin/main dc2cef612. · régime : sans objet. · scellé : aucun (à sec) ; chaque gain ci-dessous est une borne.
* mesuré (repris) : par couche MoE du Coder à b=1 — gate·up 12,9 µs (7,9 de transfert au pic, 1,60 To/s marginal), down 8,1 (3,95),
  `moe_reduce` 0,9, deux trous de 0,19 ; **≈ 22,3 µs dont 11,85 de transfert** ; L2 tiède 7,3 / 5,25 (la structure n'est pas de la HBM).
* verdict : fusion faisable et **au bit** ; ce qu'elle retire est borné par ce que la 141 a mesuré nul (PDL : −0,008 ms/pas) — recommandation § 4.
* durée : à sec, ≈ 40 min.

## 1. Le noyau (esquisse, `acvram_kernels.cu:2191` réutilisé tel quel comme corps de phase)
Grille persistante = blocs résidents (63 registres × 256 fils → 4 par SM = **680**) ; file d'items par compteur atomique, dans l'ordre :
phase 1 = tuiles gate·up de chaque paire (12 colonnes × S = 4 → 384 items, corps actuel de `nvfp4_gemv_marlin_kernel<bf16,2>`, dépôt
d'act [G, 768] fp32) ; phase 2 = tuiles down (32 × S = 2 → 512 items, corps `<float,1>`) qui attendent le compteur « 12 tuiles finies »
de leur paire (`__nanosleep`) ; phase 3 = `moe_reduce` par (jeton, tranche) après « 32 tuiles down × 8 paires ». Sans interblocage : un
item de phase 2 n'est pris qu'après que tous ceux de phase 1 sont assignés à des blocs qui tournent. Compteurs remis à zéro par le
dernier bloc (schéma du split-K, :2318) → capturable sous graphe sans memset ; b = 2-7 : mêmes items × G, S auto inchangé.

## 2. Les six chiffres demandés
| poste | valeur | d'où |
|---|---|---|
| rampes économisées | 2 par couche (down, reduce) : montée + vidange d'un noyau de 512 blocs ≈ **1-1,5 µs** chacune, borne haute ; la 141 a mesuré que recouvrir cette montée par PDL rend **−0,08 µs/frontière** en service | 214 § 1, 141 verdict |
| octets de l'intermédiaire évités | act 8 × 768 × 4 = 24,6 Ko + d 8 × 2 048 × 4 = 65,5 Ko écrits puis relus en L2 : **90 Ko/couche, 4,3 Mo/pas = 0,3 % des 1,4 Go du pas** — rien ; ils restent d'ailleurs en global (L2) dans la fusion, seule la dépendance change | formes Coder |
| risque d'occupation | 680 résidents pour 384 + 512 + 8 items : phase 1 tient en une vague, les 296 blocs restants tournent à vide (`__nanosleep`, pas de bande) jusqu'aux paires finies ; mémoire partagée max(12, 5) Kio ; **aucune seconde vague, aucun interblocage** ; à b ≥ 4 (G ≥ 32, S = 1) : 384 + 1 024 items, file dynamique | :2136, :2448 |
| au bit | **oui si** chaque item garde exactement la découpe du noyau d'aujourd'hui (mêmes tuiles k par warp, même ordre z du dernier bloc, même `acv_act`) ; toute redistribution du travail (S = 8, tuiles plus fines pour équilibrer) change l'ordre des sommes → hors bit, scellé KL comme le split-K du 19/09 (PPL +0,0042, non tranché) | :2244-2330 |
| coût en code | ≈ 350 lignes CUDA (corps de phase factorisés en `__device__`, file, compteurs), lanceur + opt-in `moe.py:1358`, test au bit contre la chaîne à 3 noyaux (attendu 0 différence), banc ; **2 jours**, 3-4 recompilations sous verrou (2,5 min chacune) | 214 |
| gain par pas b=1 | Coder-30B (48 couches) : **0,10-0,17 ms/pas (3-6 % de 2,94)** = 2-3,5 µs/couche (vidange gate·up recouverte 0,5-1, montée down 0,5-1, reduce 0,6, deux trous 0,4) ; Qwen3.5-35B-A3B (40 couches MoE, 256 experts, N 512 : gate·up 9,4 Mo à S = 8 auto, down 4,7 Mo) : mêmes coûts fixes par noyau → **0,08-0,14 ms/pas** ; pas de chiffre de la 215 (poste2) dans le dépôt | 203, 206, 214 |

## 3. Pourquoi la borne est basse
Le noyau est borné mémoire : les 384 blocs partent ensemble et se partagent la HBM ; un SM à 3 blocs n'est pas plus lent qu'un SM à 2
(pas de queue de déséquilibre à récupérer — la fusion ne rééquilibre rien). Ce que la fusion recouvre, c'est la vidange d'un noyau
(derniers épilogues, HBM oisive) et la montée du suivant : le PDL de la 141 recouvrait exactement cela et n'a rien rendu en service
(−0,26 % de débit). Les 4-5 µs de « structure » de la 214 sont surtout INTERNES à l'item (remplissage de x + barrière, épilogue
split-K : dépôt, fence, atomique, relecture) et survivent à la fusion. Un mégakernel de couche (194 a-sec § a) porte la même borne.

## 4. Recommandation
**Ne pas coder.** Avant toute décision, un banc d'UNE minute de carte, sans code produit (`banc-gemv-214.py` + jeux d'experts à −1 :
les blocs sortent après un store, :2205) donne montée + vidange d'un noyau de 384/512 blocs à ± 0,1 µs. Si ≥ 2 µs par noyau, la
fusion vaut 0,2-0,3 ms/pas et se justifie ; si ≤ 1 µs (ce que la 141 laisse prévoir), la pièce se ferme. Je le lance sur ton feu.
