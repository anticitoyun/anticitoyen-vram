# Verdict — pièce 208 (à sec) : les 4 couches MoE du Coder-30B hors Marlin (157) — un facteur PAR LIGNE D'EXPERT les rend toutes exactes (0 bloc écrasé, 0 ligne impossible) ; il faut l'échelle globale par colonne dans l'épilogue du Marlin MoE, comme le dense l'a déjà (101 étape 2) (poste6, 25/09)

* **instrument** : `scratchpad/poste6-p208-25-09/compter-sous-normales.py` (processeur, safetensors : pour chaque (couche, matrice)
  des 48 × 3 piles d'experts, échelles de bloc E4M3 décodées ; écrasées par la conversion Marlin (`echelles_ecrasees` :
  half(s)·f·2⁷ < 2) avec le facteur de la PILE (servi, `moe.py:488`), PAR EXPERT, PAR LIGNE ; lignes dont max/min > 448·2⁷/2 ;
  énergie Σ(code·bs)² des blocs écrasés / totale sur les experts touchés) ; résultat `sous-normales.json`. Aucune carte.
* **commit** : 07444b306 (poste6-208 = origin/main) + ce fichier.
* **régime** : à sec ; alias Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c (128 experts, K 2048, I 768, 48 couches).
* **scellé** : aucun (à sec, ordre chef (1)-(3)).
* **mesuré** : 8 matrices touchées sur 144 — gate et up des couches 0, 1, 2, 4 (jamais down) ; par pile : 69 059 / 67 492 blocs
  écrasés (0,55 % de 12,6 M, 43 experts) en couche 0, 202 / 203 (1 expert) en 1, 174 / 268 (2) en 2, 128 / 128 (1) en 4 ;
  **par expert : identique** (157 confirmée) ; **par ligne : 0 partout, 0 ligne impossible** ; min non nul 2⁻⁹ (le plus petit
  sous-normal e4m3), max 448, facteur de pile 1 ; énergie des blocs écrasés < 10⁻⁶ des experts touchés.
* **verdict** : (1) les piles sont inexactes parce qu'un expert (43 en couche 0, 1-2 ailleurs) porte à la fois une échelle 448
  et des échelles 2⁻⁹ (rapport 2^17,8 > 2^14,8 de S0E5M3) — par expert rien ne change ; (2) **un facteur par (expert, ligne
  N) rend les 4 piles exactes**, donc au bit du dépaquetage de référence ; il exige que le noyau MoE porté applique une
  échelle globale PAR COLONNE ([E, N]) au lieu de `global_scale_ptr[expert_id]` — le dense l'a déjà (`quantization/marlin/
  marlin_template.h:1657-1670`, pièce 101 étape 2) ; (3) sans cela, servir ces 4 couches en tensor avec les blocs à zéro
  coûterait une énergie < 10⁻⁶ : KL attendue ≈ 0 (à mesurer), mais ce serait un poids faux servi, contre la règle de la 157.
* **durée** : à sec, 45 min (lecture 157 + code + comptage 6 min).

## (1) Pourquoi, en fichier:ligne
`moe.py:488-492` : `MP.echelles_ecrasees(bs)` sur la pile entière → refus nommé « up_proj : N échelles sous-normales » ; le
facteur de `preparer_pile` (`marlin_port/__init__.py:288`, `facteur_nvfp4` sur `[E, N, K/16]`) est UN scalaire par pile, calé
sur le max 448 ; `traiter_echelles_nvfp4` (:240) met à 0 tout half(s)·f·2⁷ < 2. Les experts fautifs (43/128 en couche 0)
contiennent eux-mêmes 448 et 2⁻⁹ : le facteur par expert est donc 1 aussi (mesuré : mêmes comptes). Le rapport est un
artefact de la conversion : un bloc à 2⁻⁹ pèse 2^-17,8 du plus gros — énergie perdue < 10⁻⁶.

## (2) Le facteur par ligne : exact, et ce qu'il coûte à coder
Par ligne d'expert (`bs.max(2)` par (e, n)), 0 bloc écrasé et aucune ligne dont l'étendue dépasse S0E5M3 : la préparation
`bs' = bs · f[e, n]` (puissance de 2, e4m3 exact) et `g'[e, n] = g[e] / f[e, n]` reproduit exactement chaque poids. Pièces :
| pièce | fichier | lignes |
|---|---|---|
| échelle globale [E, N] dans l'épilogue MoE (`c0 *= gs[expert_id·N + col]`, aussi dans la branche `mul_topk_weights`, :527) | `moe/marlin_moe_wna16/marlin_template.h:551, 1803-1804` | ≈ 20 (copie de :1657-1670 du dense) |
| `ops.cu` : accepter `global_scale` [E] ou [E, N] | `moe/marlin_moe_wna16/ops.cu:784-790` | ≈ 10 |
| `preparer_pile` : facteur par (expert, ligne) seulement si `echelles_ecrasees(bs)` (les 44 autres piles gardent AU BIT leur préparation) | `marlin_port/__init__.py:266-290` | ≈ 25 (modèle : `preparer_dense` :540-565) |
| w13 (gate·up fusionné, :516-528) : g par colonne = gg/f_gate ‖ gu/f_up, et `moe_act` à rapport 1 pour ces piles | `moe.py:516-528`, `moe_act` | ≈ 20 |
| dépaquetage témoin `PAR_COLONNE` pour E > 1 (`depaqueter_marlin` :739) — test au bit contre `nvfp4_dequant` | `marlin_port/__init__.py:730-748` | ≈ 10 |
| tests : au bit des poids sur les 4 piles réelles + cassant (facteur retiré → rouge), capture godets | tests/ | ≈ 80 |
Une recompilation du port (25 s, empreinte). Effort ≈ une journée. Hors bit du chemin SERVI de ces 4 couches (aujourd'hui
`decode_mma` W4A4 à b=8, GEMV naturel à b=1, « groupe » au préfill) — mais au bit des poids : scellé KL comme pour tout
changement de noyau (témoins à échantillon égal).

## Chiffrage
* b=8 : 4 couches 82 → 50,5 µs = **−0,128 ms/pas (2,6 %)** (206 § 3) ; b=1 : GEMV naturel 22,8 → Marlin 19,2 µs : −0,014 ms (0,5 %) ;
  préfill de ces 4 couches (naturel « groupe » → Marlin) : non chiffré ici — la 157 a mesuré le prix des 4 piles naturelles à
  +0,6 % (b=1) / +1,7 % (b=8, certifie-b12, préfill compris) : c'est la borne du gain total, **≈ +1,5 à +2 % à b=8**.
* (3) alternative « blocs à zéro » : gain identique, énergie perdue < 10⁻⁶ (KL ≈ 0 attendue), mais poids inexact servi et
  contraire à la 157 : je ne la recommande pas.

## Recommandation
Faire (2) : facteur par (expert, ligne) + échelle globale par colonne dans le Marlin MoE, restreint aux piles qui en ont besoin,
test au bit des poids à sec (les 4 piles réelles) avant toute carte, puis scellé KL + ABBA b=1/b=8 (prédit +1,5 à +2 % à b=8).
Code seulement sur feu.
