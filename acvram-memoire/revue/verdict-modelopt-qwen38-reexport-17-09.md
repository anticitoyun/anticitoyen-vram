# Verdict — Qwen3.8-27B ModelOpt RÉ-EXPORTÉ : `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` exclus (bf16), remède au refus TRT-LLM

poste2, 17/09. Suite de `verdict-trtllm-qwen38-17-09.md` (poste3, 8a56931) :
TRT-LLM refuse le NVFP4 sur les projections d'attention linéaire GDN
(`in_proj_qkv`, `in_proj_z`, `out_proj`), `_SUPPORTED_SUFFIXES` n'accepte
que bf16 ou fp8 par blocs pour ces tenseurs
(`qwen3_5_weight_mapper.py:218-221`).

## Correctif, une fausse piste avant la bonne

`hf_ptq.py` lit `QUANT_CFG_CHOICES["nvfp4"]` (`modelopt.recipe.presets`),
**pas** `mtq.NVFP4_DEFAULT_CFG` — deux dicts distincts (`is` faux, `==`
vrai au chargement : même contenu au départ, pas le même objet). Premier
essai : muté `mtq.NVFP4_DEFAULT_CFG["quant_cfg"]` seul — export « réussi »
mais **sans aucun effet** : `in_proj_qkv`/`out_proj` de la couche 48
portaient toujours `weight_scale`/`weight_scale_2`/`input_scale`,
`exclude_modules` inchangé (149 entrées). Trouvé en RELISANT le manifeste
exporté, pas en faisant confiance au message de confirmation du patch
(REGLES §5 : vérifier par la mesure, pas par ce que le code dit qu'il a
fait).

Correctif final : muter les DEUX dicts (`mtq.NVFP4_DEFAULT_CFG` et
`QUANT_CFG_CHOICES["nvfp4"]`), ajout de trois motifs désactivés
(`*linear_attn.in_proj_qkv*`, `*linear_attn.in_proj_z*`,
`*linear_attn.out_proj*`) — même convention que les motifs déjà présents
par défaut (`*linear_attn.conv1d*`, `*linear_attn.in_proj_a*`,
`*linear_attn.in_proj_b*`).

## Vérifié sur le vrai ré-export

```
model.language_model.layers.48.linear_attn.out_proj.weight       (seul)
model.language_model.layers.48.linear_attn.in_proj_qkv.weight    (seul)
```

Plus de `weight_scale`/`weight_scale_2`/`input_scale` sur ces deux
projections — bf16 pur. `hf_quant_config.json.exclude_modules` :
`model.language_model.layers.N.linear_attn*` pour les 48 couches (motif
compact, tout `linear_attn` désactivé collapse en une seule entrée par
couche). Taille totale : 27 Gio (contre 19,6 avant), +7,4 Gio — plus que
l'estimation de poste3 (≈ 2,6 Gio), mais direction confirmée. Carte :
45 s (essai raté, mutation sans effet) + 35,4 s (essai correct) sur deux
passes, ≈ 25 min au total avec chargement.

## sha256 (converti corrigé, `.../Qwen3.8-27B-ModelOpt-NVFP4`)

```
model-00001-of-00003.safetensors : 07d6a6682cdaea266ddb9b663dfd3def59d2ab2bf204bbea67a0e5a858d08411
model-00002-of-00003.safetensors : bee8e8ce89eac950552c70a22056532e54f9208b288427a542caacc515c3420f
model-00003-of-00003.safetensors : c4c8bd7c8754caac47874949748d2c38ecd29199b8f8df2064a51c1603333e79
model.safetensors.index.json     : 038be5fb9a73f872c9daf976ec480607d1771b4867085d7f3f1e802262d38dcd
hf_quant_config.json             : 268f9026f889ddc2b62725d2369256f6a543f0ca8d58e7934bfa4ddf2ed30d1d
```

L'ancien export invalide (149 exclude_modules, `in_proj_qkv`/`out_proj`
encore NVFP4) a été retiré du disque, remplacé par celui-ci au même
chemin — ne pas comparer un sha256 antérieur à ceux ci-dessus, il n'y a
plus qu'une seule version sur disque.

## Suite

Prêt pour poste3 : rejouer `scratchpad/trtllm-qwen38-17-09/chaine.sh`
contre ce nouveau checkpoint.
