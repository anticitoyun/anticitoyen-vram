# Prédiction — porte KL étape 1 acvram_mojo (écrite avant mesure, ordre chef)

* instrument : dump HF bf16 (`transformers`, poids source non convertis) sur les 8 premiers jetons gloutons
  des 5 invites, KL@20 teacher-forcée via `/v1/completions echo+logprobs` contre acvram serve et MAX serve
  (même source bf16, mêmes invites que l'étape 1)
* prédiction chiffrée : acvram KL_max ≤ 0,10 (moteur déjà scellé à ce niveau sur d'autres portes bf16 du
  projet) ; MAX KL_max entre 0,05 et 0,74 (noyaux différents mais même dtype source, pas de quantification —
  la divergence au bit vue à l'étape 1 vient de l'ordre de sommation/attention, pas d'un format de poids
  différent, donc écart attendu petit, pas nul)
* seuil : **0,74 chacun** (contrat §3). Si MAX > 0,74 : la question (b) répond « les noyaux Mojo de MAX
  divergent trop de la référence bf16 pour cette porte » — pas d'incrimination du langage en soi, à creuser
  (ordre de réduction dans l'attention/MoE, format interne des activations)
* ce que ça rendrait si l'hypothèse (MAX proche de HF) est fausse : KL_max(MAX) > 0,74 → porte NON tenue,
  cellule HTTP (débit/énergie) repoussée jusqu'à diagnostic ; KL_max(acvram) > 0,74 serait une alarme sur
  notre PROPRE moteur bf16 (jamais vu jusqu'ici), pas sur Mojo — à remonter immédiatement si observé
