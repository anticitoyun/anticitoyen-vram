# Verdict — ModelOpt Coder-30B, tentative venv séparé (transformers 4.x) : REFUSÉ, deuxième blocage distinct trouvé, aucun patch tenté

poste2, 17/09. Décision poste7 (`poste7-modelopt-coder-export-17-09.md`) :
une tentative bornée dans un venv séparé de TRT-LLM/vLLM, avec un
contrôle à sec obligatoire avant toute carte. Toute autre issue que le
succès = refus accepté, pas de seconde tentative, pas de patch ModelOpt.

## Contrôle à sec (avant toute carte)

Venv `/mnt/AI_GENERATOR/modelopt-tf4/.venv` : `nvidia-modelopt[hf]==0.37.0`
résout `transformers==4.56.2` (la contrainte du paquet est `>=4.48,<4.57`
— `4.57.*` demandé était juste hors de portée, `4.56.2` est le plus
proche compatible), `torch==2.10.0+cu130` (même version CUDA que
TRT-LLM). Sur un `Qwen3MoeForCausalLM` jouet (2 couches, poids
aléatoires, dimensions divisibles par le bloc NVFP4 16) :

```
type(model.model.layers[1].mlp.experts) = torch.nn.modules.container.ModuleList
isinstance(..., collections.abc.Iterable) = True
```

**Premier point du contrôle PASSÉ** : sous transformers 4.56.2, les
experts Qwen3 sont bien un `ModuleList` itérable — la branche qui
échouait en 5.5.3 (`Qwen3MoeExperts` fusionné) n'est plus prise. Suite
du contrôle, sur le même jouet, une carte brève (le calcul de capacité
NVFP4 interroge réellement `torch.cuda.get_device_capability`, même sur
un jouet — impossible de rester complètement à sec pour cette partie) :
quantification (`mtq.quantize`, NVFP4) + export (`export_hf_checkpoint`)
+ **`mto.save()` avant l'export** (poste7 : ne pas reperdre une heure de
calibration si l'export échoue une seconde fois) — **les trois
réussissent**, fichiers écrits dont `hf_quant_config.json`.

**Contrôle intégralement PASSÉ.** Décision : la vraie tentative sur le
modèle réel est allée à la carte, après confirmation directe de poste3
que sa chaîne (Coder acvram W4A16, EXL3 vitesses, GLM vLLM b=1) était
terminée — pas de rupture d'ordre cette fois.

## Corpus de calibration (identique à la tentative précédente)

```
sha256 (bras-A-anglais.txt) : cb7c0d9af2041d31e655fafe79becb0877c3fc775690c7a558eb8707035e5138
```

512 échantillons, même monkeypatch `_get_dataset_samples` (`--dataset
bras_a`) que `verdict-modelopt-coder-17-09.md`.

## Résultat sur le vrai modèle : deuxième blocage, distinct du premier

Chargement (16 fragments, 5 min 40 s), calibration réussie (64/64,
14 min 44 s, KV cache quantization activée), **`mto.save()` a bien écrit
l'état quantifié AVANT l'export** :

```
/mnt/2TO_2023_980PRO/Modeles/models_modelopt/Qwen3-Coder-30B-A3B-Instruct-ModelOpt-etat-quantifie.pt
24 348 751 797 octets (24,3 Gio)
```

**L'export échoue quand même**, pour une cause DIFFÉRENTE du premier
blocage (`Qwen3MoeExperts` non itérable, déjà résolu par ce venv) :

```
RuntimeError: Expected all tensors to be on the same device, but found
at least two devices, cuda:0 and cpu!
```

Fichier:ligne exact — `modelopt/torch/quantization/qtensor/
nvfp4_tensor.py:84` (`get_weights_scaling_factor`) :

```python
per_block_scale = per_block_amax / (6.0 * weights_scaling_factor_2)
```

`per_block_amax` est calculé depuis le poids réel (`input`), qui peut
être sur `cpu` pour une couche déportée par `--use_seq_device_map` (le
modèle ne tient pas dans les 32 Gio de VRAM : `{0: 26,5 Gio, cpu:
85,9 Gio}` annoncé au chargement, pic mesuré 31,02 Gio sur la carte) ;
`weights_scaling_factor_2` reste sur `cuda:0`. Le même mécanisme de
déport qui a permis à la CALIBRATION de tourner casse l'EXPORT — un
bogue distinct du premier, dans le même sous-système, jamais rencontré
sur le jouet (trop petit pour déclencher un déport).

## Verdict : REFUS ACCEPTÉ, pas de seconde tentative, pas de patch

Deux blocages fichier:ligne, indépendants, sur deux venvs :
1. `unified_export_hf.py:419-422` (transformers 5.5.3, `Qwen3MoeExperts`
   fusionné non itérable) — `verdict-modelopt-coder-17-09.md`.
2. `nvfp4_tensor.py:84` (transformers 4.56.2, déport CPU/GPU pendant la
   calibration incompatible avec l'export des échelles) — ce document.

Conforme à la décision de poste7 : je ne tente ni un troisième venv, ni un
patch du code ModelOpt. La cellule « Coder vLLM/TRT-LLM sur ModelOpt
propre » reste vide au critère de complétion, cause nommée aux deux
niveaux. L'état quantifié (24,3 Gio) reste sur disque : si quelqu'un veut
reprendre l'export sans refaire l'heure de calibration (ex. après un
correctif amont, ou en forçant tout sur une seule carte sans déport si un
jour la VRAM suffit), le point de départ existe.

Carte libérée (verrou vide, `nvidia-smi` sans process de calcul).
