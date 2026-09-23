# Verdict — GLM-4.7-Flash-NVFP4-a16 : montage par liens symboliques + config modifié, pour basculer vLLM en Marlin W4A16

Manon, 17/09. Ordre Sage (`sage-passage-direct-verdict-17-09.md`) : sur
les mêmes poids GadflyII/GLM-4.7-Flash-NVFP4, vLLM sert en W4A4 par
construction (`compressed_tensors.py:735-743`, lu par Sage : `input_quant
None → Marlin W4A16, sinon W4A4`) — comparer acvram (W4A16) à vLLM sous
le MÊME régime W4A4/W4A16 exige de faire basculer vLLM sans reconvertir
ni retoucher les poids.

## Montage

`/mnt/4TO_SATACMR_2022/Modeles/models_vllm/GLM-4.7-Flash-NVFP4-a16/` :
liens symboliques (relatifs) vers chaque fichier de
`GLM-4.7-Flash-NVFP4/` — safetensors, tokenizer, `chat_template.jinja`,
`calibration_amax.json`, `README.md`, `generation_config.json`,
`model.safetensors.index.json` — **sauf** `config.json`, écrit en dur
(pas un lien) avec un seul champ modifié :

```
quantization_config.config_groups.group_0.input_activations : {…4 bits dynamic…} → null
```

`diff` contre l'original : un seul bloc changé (les 8 lignes de
`input_activations` remplacées par `null`), rien d'autre — vérifié avant
publication.

```
sha256 (config.json de GLM-4.7-Flash-NVFP4-a16) :
d574e72b6ae894b993534b7dc7dcfb49a338992a8e88256807baf6459af6717f
```

## Portée

Aucun poids copié ni modifié — seuls les octets réellement écrits sont
ceux du `config.json` (1,8 Kio). Les deux répertoires partagent les mêmes
fichiers de poids par le système de fichiers ; une modification des
safetensors originaux (aucune prévue) affecterait les deux.

## Suite

Pas de mesure de ma part (à sec, pas de carte) : à Laure, servir vLLM sur
`GLM-4.7-Flash-NVFP4-a16` et vérifier qu'il bascule effectivement en
Marlin W4A16 (log de démarrage vLLM, ou lecture du schéma choisi), puis
PPL sous ce régime pour la comparaison à acvram W4A16.
