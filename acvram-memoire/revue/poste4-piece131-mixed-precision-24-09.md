# Pièce 131 — `acvram --passage-direct` refuse unsloth/Qwen3.8-27B-NVFP4 (à sec)

## Constat (en-têtes safetensors seuls, `model.safetensors.index.json`, aucun poids chargé)

Checkpoint réel : `Qwen3_5ForConditionalGeneration` (VL, `model.language_model.*` + `model.visual.*`),
64 couches, attention hybride (`linear_attention`×48 / `full_attention`×16, motif 3+1), MLP **dense** partout
(aucune couche `moe` — l'hypothèse de poste2 dans `scelle-102bis-source.md` est réfutée par les en-têtes).

`quantization_config` : **`format: "mixed-precision"`**, DEUX `config_groups` :
* `group_1` (`format: "nvfp4-pack-quantized"`) : cible `mlp.(gate|up|down)_proj` — couches 0-55
* `group_0` (`format: "float-quantized"`, fp8 dynamique) : cible `self_attn.(q|k|v|o)_proj`,
  `linear_attn.(in_proj_qkv|in_proj_z|out_proj)`, `lm_head`, **et `layers.(56-63).mlp.(gate|up|down)_proj`**
  — les 8 dernières couches ont un MLP fp8, pas nvfp4.

## Cause exacte (pas une table de noms)

`acvram/quant/hfquant.py::is_hfquant()` teste `quantization_config["format"] in ("pack-quantized",
"nvfp4-pack-quantized")` — un champ de **premier niveau**. Sur ce point de contrôle, ce champ vaut
`"mixed-precision"` ; les formats réels vivent dans `config_groups.*.format`, jamais lus par `is_hfquant`.
**`is_hfquant()` rend donc `False`**, `_iter_checkpoint` (convert.py:1162) saute `HFQuantCheckpoint` en
entier et tombe sur la lecture générique du safetensors — chaque tenseur est rendu TEL QUEL, sans
dépaquetage : les 56 `mlp.gate_proj.weight_packed` (couches 0-55, nvfp4) restent sous ce nom, jamais
renommés en `.weight` → 56 manquants, le premier étant `model.layers.0.mlp.gate_proj.weight`. Compte
exact : 56 couches nvfp4 × 1 tenseur vérifié par couche (`attendus` ne teste que `gate_proj`, pas
up/down) — colle au chiffre de l'erreur.

Les tenseurs fp8 (self_attn, linear_attn, lm_head, mlp des couches 56-63) ne manquent PAS : le format
fp8 ne pakke pas (1 octet/valeur), leur clé `.weight` existe déjà telle quelle sur disque — ce qui masque
la vraie nature du problème et laisse croire à un souci de nommage isolé sur les 56 couches nvfp4.

## Second défaut, plus grave, sous le premier

`HFQuantCheckpoint.iter_tensors` n'a **aucun dispatch par groupe** : `self.method`/`self.format` sont des
valeurs UNIQUES lues au niveau du checkpoint (`hfquant.py:75-76`), pas par tenseur. Même si `is_hfquant()`
était corrigé pour accepter `"mixed-precision"`, la branche `self.method == "compressed-tensors"` déciderait
`_ct_nvfp4` vs `_ct_int4` sur la base du SEUL `self.format` global (`"mixed-precision"`, qui ne correspond
à aucun des deux) — et pour les tenseurs fp8 (`weight` + `weight_scale`, sans `weight_packed`), cette même
branche ne les reconnaît PAS du tout (aucun `elif` pour `float-quantized` sous `compressed-tensors`) : ils
tomberaient dans le repli générique `yield key, fh.get_tensor(key)`, c'est-à-dire **non déquantifiés**,
silencieusement — les poids fp8 (attention entière, lm_head, 8 couches de MLP) sortiraient faux sans lever
d'erreur.

## Verdict (ordre chef)

**Format de quantification non géré**, pas une simple table de noms : `compressed-tensors
mixed-precision` (plusieurs `config_groups` à formats différents dans le MÊME checkpoint) n'a pas de
dispatch par groupe dans `hfquant.py` — ni pour la détection (`is_hfquant`), ni pour le déquantification
par tenseur (`iter_tensors`). Corriger `is_hfquant()` seul ferait *pire* que le refus actuel : conversion
qui se termine, poids fp8 faux et silencieux, sans lever d'erreur pour le dire. Je n'écris rien — c'est
une fonctionnalité manquante (résolution par groupe via les motifs `targets` de `config_groups`), pas un
correctif de nommage. Rien lancé sous `carte.sh` (inutile : le défaut est dans le code, pas la carte).

## Reste (si l'utilisateur tranche pour l'implémenter)

Résoudre le groupe par tenseur via les motifs `re:` de `config_groups.*.targets` (déjà des regex Python
valides une fois le préfixe `re:` retiré), router `weight_packed`→`_ct_nvfp4`/`_ct_int4` selon le format du
groupe trouvé, ajouter un chemin fp8 sous `compressed-tensors` (actuellement seulement sous
`self.method == "fp8"`, mutuellement exclusif). Test qui casserait : un point de contrôle synthétique à
deux groupes (nvfp4-pack-quantized + float-quantized) vérifiant que les DEUX ressortent déquantifiés
correctement, pas seulement présents sous le bon nom.
