# Protocole — PPL du reconverti -k48 en régime duel (prefill W4A16, décodage MMA=1 MIN_T=5)

poste3, 16/09/2026, avant mesure. Ordre : chef, poste7 § 8 (`poste7-glm-pile-correctif-16-09`,
main ce71723). Arbre : **travail/poste3 figé au commit de ce protocole** (code
`acvram/` = main ce71723, noyau par ligne 877169c inclus) ; extension recompilée
hors verrou (ninja, 86 s, sources identiques à poste3-qa, empreinte contrôlée).

## Montage
`scratchpad/reppl-eval-16-09.py` (ligne de poste2 + régime lu), **`ACVRAM_MOE_MMA=0`**
(prefill W4A16 : c'est le régime du duel ; la PPL est un prefill, le chemin
MMA du décodage n'y passe pas), `ACVRAM_QA_COMPTE=1` (attendu : aucun bloc
quantifié, preuve que le W4A4 n'est pas pris), deux passes (chaud/froid).
Converti `GLM-4.7-Flash-srcbf16-nvfp4-k48` (poste2, nvfp4 seul, 08:08).
Référence bf16 8,1427.

## Scellé (poste7 § 8.2)
**≤ 1,005** ; réfuté **> 1,010** → `-avant-noawq-experts` (0,998 en W4A16) prend
sa place au duel, même régime, MTP vérifié (une PPL de plus, même ligne).

## Ma prédiction (scellée)
**K W4A16 = 1,000-1,004** : le livrable AWQ partout rend 0,998 en W4A16 avec
les 124 tables int8 ; poste1 a mesuré que ces tables valent < 10 % d'une erreur
de 0,3 % (`verdict-glm-awq-int8`, 0,982/0,983) → leur retrait coûte ≤ +0,002 ;
le W4A4 valait +0,018 sur ce converti (1,0183 − ≈1,000). Réfuté si > 1,005
(alors la recette nvfp4-seul coûte, pas le W4A4 : le -k48 ne prend pas le
duel) ou < 0,999 (le retrait des tables aiderait, à expliquer). Compteurs :
0 bloc quantifié. Durée : 2 × ~7 s + chargements, unité `ppl-k48-w4a16-poste3`.
