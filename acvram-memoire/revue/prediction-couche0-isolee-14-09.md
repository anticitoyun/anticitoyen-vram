# Prédiction scellée — couche 0 isolée (MLA sans MoE)

poste1, 14/09/2026, avant mesure. Item (2) de chef : isoler la couche 0
(dense + MLA) pour savoir si la MLA est en cause indépendamment du routage.

Contexte : le correctif routeur fp32 (revue/prediction-routeur-fp32-14-09.md)
a été mesuré RÉFUTÉ — même après le correctif, les 4 mêmes positions (0,1,3,5)
divergent en topi, avec des deltas de logits presque inchangés (1,87→1,82 ;
5,11→5,08). L'arithmétique du routeur n'est donc PAS la cause dominante ; il
reste à savoir si la source est amont (MLA, couche 0) ou dans la couche MoE
elle-même (poids gathered, expert partagé, échelle).

## Prédiction

L'état caché en sortie de couche 0 (entrée de couche 1, la ligne du dernier
jeton, capturée aux 16 longueurs de préfixe) a un cosinus ≥ 0,999999 avec HF
à TOUTES les positions, y compris 0, 1, 3, 5 — une couche dense sans
branchement top-k ne peut diverger que par arrondi numérique diffus, pas par
un écart de cet ordre.

**Seuil de réfutation** : si le cosinus tombe sous 0,999999 à une ou
plusieurs positions — et *a fortiori* si c'est aux positions 0,1,3,5
précisément — la MLA (nope=192/v_head_dim=256, jamais testée avant ce modèle)
est impliquée, pas seulement la couche MoE.
