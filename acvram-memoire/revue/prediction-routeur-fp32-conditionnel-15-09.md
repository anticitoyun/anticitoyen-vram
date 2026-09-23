# Prédiction scellée — fp32 du routeur conditionné à sigmoid+biais

Océane, 15/09/2026, avant mesure. Régression trouvée par Laure
(bissection ABAB, 15/09) : mon correctif 7f3f422 (`_router_logits`
toujours fp32) coûte +0,187 ms/pas à b=1 sur Coder-30B (233→223 t/s),
qui route en softmax sans biais — aucun besoin du fp32, jamais eu de
positions rouges dans l'équivalence.

## Correctif

`_router_logits` (model.py) : fp32 seulement si
`self.scoring == "sigmoid" and self.score_bias is not None` (GLM-4.7-
Flash), `x.dtype` (bf16 en service) sinon.

## Prédiction

1. `test_routeur_fp32_seulement_si_sigmoid_biais` (nouveau) : un
   `MoEBlock` softmax sans biais rend des logits dans `x.dtype` (bf16) ;
   un `MoEBlock` sigmoid avec biais rend des logits en fp32.
2. L'équivalence GLM (`mini-acvram4` vs HF, 16 jetons, seuils de Manon)
   reste à 15/16 positions — le chemin sigmoid+biais est inchangé pour
   GLM, seul le chemin softmax change.

**Seuil de réfutation** : si le test de dtype échoue, OU si l'équivalence
GLM retombe sous 15/16 (position autre que 0 qui casse), le correctif est
mal ciblé.

Non mesuré ici (hors carte, GPU requis) : le compte de lancements/pas et
le débit b=1 de Coder-30B — laissé à la remesure de Laure (seuil
4,30 ± 0,02 ms) après fusion.
