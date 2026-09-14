# Prédiction scellée — biais de correction en fp32 à la conversion

poste1, 14/09/2026, avant mesure.

## Cause trouvée (mesurée, pas supposée)

Contrôle décisif fp32/fp32 (revue/prediction-fp32-decisif-14-09.md) :
positions 0 et 5 corrigées (bruit bf16 diffus, comme prévu), mais 1 et 3
restent divergentes MÊME les deux bras en fp32 — un vrai bogue indépendant
de la précision de calcul.

Capture des scores bruts (`sel = sigmoid(logits) + e_score_correction_bias`)
aux positions 1 et 3 : les 64 valeurs du biais diffèrent entre acvram et
HF, toujours par un multiple de ~0,03 (ex. 9,0625 côté acvram vs
9,038032531738281 côté HF) — exactement le pas de bf16 à cette magnitude
(~9). `e_score_correction_bias` est dans `SENSITIVE_SUFFIXES`
(acvram/quant/convert.py:216) mais cette liste protège seulement de la
quantification agressive (int4/nvfp4) en promouvant à "16 bits" — bf16 ou
fp16 selon `--format` — jamais fp32. Le pas bf16 (~0,03 à cette magnitude)
est du même ordre que la correction par expert qui décide le rang 4/5 :
la conversion efface l'information qui tranche les ex-aequo.

## Correctif

`convert.py::TensorRouter.format_for` : `e_score_correction_bias` retourne
"fp32" inconditionnellement, avant même `keep_sensitive_16bit`. Écriture
correspondante dans la boucle du convertisseur (`tensor.to(torch.float32)`
au lieu de bf16/fp16). Le chargeur lisait déjà `.float()` — il ne pouvait
pas récupérer une précision jamais écrite.

## Prédiction

Reconvertir `mini-hf` avec ce correctif (nouveau `mini-acvram3`) et
rejouer l'équivalence 16 jetons ORIGINALE (bf16 des deux côtés, seuils de
poste2) : le topi de couche 1 est identique à HF sur 16/16 positions, pire
cosinus ≥ 0,999.

**Seuil de réfutation** : toute position encore divergente en topi après
ce correctif réfute l'hypothèse — il resterait une troisième source (poids
gathered, expert partagé, ou une MLA plus subtile que ne l'a montré le
contrôle couche-0 diffus).
