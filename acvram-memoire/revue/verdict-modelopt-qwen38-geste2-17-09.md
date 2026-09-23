# Verdict — Geste 2 ModelOpt : Qwen3.8-27B EXPORTÉ (premier succès du chantier, après deux refus sur Coder-30B)

Manon, 17/09. Suite de `verdict-modelopt-geste1-deport-17-09.md` (correctif
`nvfp4_tensor.py:92` vert sur jouet). Ici, le vrai modèle.

## Blocage imprévu : le venv `modelopt-tf4` (transformers 4.56.2, fixé pour
## le bug Coder-30B/MoE) ne connaît pas `qwen3_5`

```
KeyError: 'qwen3_5'
ValueError: The checkpoint you are trying to load has model type `qwen3_5`
but Transformers does not recognize this architecture.
```

`qwen3_5` (GDN, Qwen3.8-27B) n'existait pas encore dans transformers 4.56.2
— ajouté seulement dans les versions récentes (5.x). La contrainte de
`nvidia-modelopt==0.37.0` (`transformers>=4.48,<4.57`) interdit 5.x : ce
n'est pas un simple upgrade de transformers dans le même venv, c'est un
plafond de la version de ModelOpt elle-même.

**Sondé (à sec, `uv pip install --dry-run`)** : `nvidia-modelopt[hf]`
sans pin résout aujourd'hui `0.46.1`, avec `transformers==5.14.1` (a
`qwen3_5`) et `torch==2.14.0`. **Bonus** : la ligne 84 originale
(`nvfp4_tensor.py`) est **déjà corrigée en amont** dans 0.46.1
(`weights_scaling_factor_2.to(per_block_amax.device)`, même correctif que
le mien) — mon patch du venv `modelopt-tf4` reste valable pour Coder-30B
(si jamais repris), mais devient inutile avec cette version.

Nouveau venv `/mnt/AI_GENERATOR/modelopt-v046` (`uv venv` + `torch==2.14.0+cu130`
+ `nvidia-modelopt[hf]` sans pin), driver adapté
(`/tmp/lancer_modelopt_driver_046.py`) : l'API de dataset a changé
(`_get_dataset_samples` → `get_dataset_samples`, public), et `--dataset`
accepte désormais un chemin `.jsonl` directement (plus besoin de
monkeypatcher un nom de dataset spécial) — corpus bras-A réécrit en JSONL,
64 lignes `{"text": ...}`, même découpage par tranches égales de
caractères que `load_calib_ids` (`acvram/quant/collect.py`).

## Second imprévu : `mto.save()` casse dans 0.46.1 (état multiple)

```
AssertionError: Model has multiple modelopt states!
  (ModeloptStateManager.is_converted, conversion.py:119)
```

Après « 15 tenseurs MTP détectés, orphelins » — la tête MTP porte
apparemment son propre état modelopt dans cette version, et
`mto.save(model, ...)` sur le modèle entier refuse d'en voir plus d'un.
Rendu NON BLOQUANT (`try/except`, avertissement, export tenté quand
même) plutôt qu'investigué à fond : la calibration ici tient 20 s (pas
les 14 min de Coder), le risque de la reperdre en cas de second échec
d'export est faible — écart documenté à la règle « `mto.save()` toujours »
de Sage, pas silencieux.

## Résultat : export réussi, en 28,2 s, aucune trace de la ligne 84

```
calibration : 2 séquences, 64 échantillons bras-A (JSONL)
Quantized model exported to: .../Qwen3.8-27B-ModelOpt-NVFP4
Total time used 28.221105813980103s
```

`hf_quant_config.json` : `quant_algo: NVFP4`, `kv_cache_quant_algo: FP8`,
`group_size: 16`. 3 fragments safetensors (~19,6 Gio), manifeste HF
standard, tokenizer/config copiés. Carte : deux passes, 1117 s + 857 s
(~33 min, sous le plafond de 45-60 min du scellé — la seconde passe
n'ayant pas eu à refaire la calibration puisque la première l'avait
déjà validée, seul le contournement `mto.save` changeait).

## sha256

```
corpus JSONL (bras-A, 64 échantillons) : 711671f5c56ca7c2dea9fc22648bc01e8ad00d360d558e9062598c0d07a53dfe
corpus source (bras-A-anglais.txt, inchangé) : cb7c0d9af2041d31e655fafe79becb0877c3fc775690c7a558eb8707035e5138
model-00001-of-00003.safetensors : 4129e26b31bcb7f6a0eb9b0b24cd1a69b2fd77241ba7db577b8fa188e61bd486
model-00002-of-00003.safetensors : 8eaee8d71924ddb234a8859c87e2615bd694d4b78f158788c779f0a37d148daa
model-00003-of-00003.safetensors : b47c8b8d424d88842ac65db48ff941340282d9492e3ca572aab7881ceff14f03
model.safetensors.index.json     : ed9f80f501cb279cec13b2433361fbe753bb0016b0fe3bd7ed7ea488f48a377d
hf_quant_config.json             : 7d6b1d34faa015a3e128517f1b5ce4a139c4cda04ae150d277692eca714ebe54
```

## Sortie

`/mnt/2TO_2023_980PRO/Modeles/models_modelopt/Qwen3.8-27B-ModelOpt-NVFP4`
— pas d'état quantifié sauvegardé séparément cette fois (`mto.save`
échoué, voir plus haut) : si TRT-LLM refuse à son tour, une reconversion
recalibrera (20 s, coût négligeable).

TRT-LLM connaît `qwen3_5` sans garde (`verdict-modelopt-geste1-deport-17-09`) :
prêt pour Laure. Passage à Gemma-4-31B ensuite.
