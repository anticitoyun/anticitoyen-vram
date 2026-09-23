# Verdict — Geste 2 ModelOpt : Gemma-4-31B-it EXPORTÉ (second succès, même pipeline que Qwen3.8-27B)

Manon, 17/09. Suite de `verdict-modelopt-qwen38-geste2-17-09.md` — même
venv (`modelopt-v046`, `nvidia-modelopt==0.46.1`, `transformers==5.14.1`),
même driver (`/tmp/lancer_modelopt_driver_046.py`), même corpus bras-A
(JSONL, 64 échantillons, sha256 `711671f5c56ca7c2dea9fc22648bc01e8ad00d360d558e9062598c0d07a53dfe`).

## Contrôle préalable (Sage, à sec, avant la carte)

`load_model_spec` (acvram) sur le vrai checkpoint
`/mnt/4TO_SATACMR_2022/Modeles/models/gemma-4-31B-it-bf16` (59 Gio,
`google/gemma-4-31B-it`, PAS le dérivé heretic local) :

```
model_type gemma4_text · architecture llama (alias)
hidden_size 5376 · num_layers 60
num_attention_heads 32 · num_key_value_heads 16
layer_types {full_attention, sliding_attention}
```

`config.py:499-500` (`if "text_config" in cfg and "hidden_size" not in
cfg: cfg = {**cfg, **cfg["text_config"]}`) déplie correctement le
`text_config` (top niveau : seulement `architectures`, `model_type`
`gemma4`, `text_config`, `vision_config`, `audio_config`…, sans
`hidden_size` propre). Index safetensors : préfixes réels
`{embed_vision, language_model, vision_tower}` — même enrobage
multimodal que Qwen3.8-27B (`model.language_model.*`), aucun tenseur
`audio_*` malgré `audio_config` présent dans `config.json` (branche
inutilisée sur cette sortie). `load_model_spec` charge sans erreur,
sans avertissement.

## Conversion réelle

```
outils/carte.sh env CUDA_VISIBLE_DEVICES=0 timeout 3600 \
  modelopt-v046/.venv/bin/python lancer_modelopt_driver_046.py hf_ptq.py \
  --pyt_ckpt_path .../gemma-4-31B-it-bf16 --qformat nvfp4 --calib_size 64 \
  --dataset bras-a-calib-64.jsonl --export_path .../gemma-4-31B-it-ModelOpt-NVFP4 \
  --trust_remote_code --use_seq_device_map
```

Même avertissement non bloquant que Qwen3.8 : `mto.save()` échoue
(« Model has multiple modelopt states! »), export tenté quand même —
réussi :

```
Quantized model exported to: .../gemma-4-31B-it-ModelOpt-NVFP4
Total time used 29.49 s
```

`hf_quant_config.json` : `quant_algo: NVFP4`, `kv_cache_quant_algo: FP8`,
`group_size: 16`, `exclude_modules: [lm_head, model.embed_vision*,
model.vision_tower*]` — la tour visuelle et sa projection sont
correctement exclues par ModelOpt lui-même (confirmé aussi pour
Qwen3.8, indépendamment de l'angle mort `embed_vision` de
`acvram/quant/collect.py`, hors de ce chantier). 3 fragments
safetensors (~19,4 Gio, le 3ᵉ minuscule : 4,7 Kio, probablement la tête
de sortie/normes finales isolées par le découpage HF). Carte : une
seule passe, 967 s (≈ 16 min, sous le plafond de 45-60 min).

## sha256

```
model-00001-of-00003.safetensors : d0bfc932dff018505ca7dbe98b9ab643a468fa10026bf05e3f6922612fa649e6
model-00002-of-00003.safetensors : f6603d8d643654c490e9cedd4077ff3efdfe44c37ed63b91110a6e3522ceccce
model-00003-of-00003.safetensors : 14c9ac8148b74001e394b3a6f42fa610a94298ce437ce1e18527c6a9f973578a
model.safetensors.index.json     : ad09443566508ec2a088005a67686f1e9ffc4005b4f295fdcf41414560f10d09
hf_quant_config.json             : 4667fbd839ec5cbc5d43061de0a60eae318cfe387ed7aa34bc6fdcb2f220694b
```

## Sortie et suite

`/mnt/2TO_2023_980PRO/Modeles/models_modelopt/gemma-4-31B-it-ModelOpt-NVFP4`
— nouvelle entrée classable (`gemma-4-31B-it-nvfp4`, bras A, ModelOpt
propre), à distinguer du dérivé heretic local déjà présent côté
acvram/llama.cpp. Aucun état quantifié séparé sauvegardé (même limite
que Qwen3.8, `mto.save` cassé dans 0.46.1). TRT-LLM connaît `gemma4`
sous garde transformers≥5.5.0 satisfaite
(`verdict-modelopt-geste1-deport-17-09`) : prêt pour Laure, les deux
conversions du geste 2 sont faites.
