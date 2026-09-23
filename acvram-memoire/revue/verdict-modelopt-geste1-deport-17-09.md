# Verdict — Geste 1 ModelOpt (sage-modelopt-priorite-manon-17-09) : export corrigé pour le déport, VERT ; TRT-LLM connaît les deux architectures

Manon, 17/09, à sec puis quelques secondes de carte pour le contrôle
(GPU requis même pour un jouet, comme le contrôle du 16/09). Suite de
`verdict-modelopt-coder-tf4-17-09` : export cassé à `nvfp4_tensor.py:84`
quand des couches sont déportées CPU par `--use_seq_device_map`.

## Correctif (fichier:ligne)

`/mnt/AI_GENERATOR/modelopt-tf4/.venv/lib/python3.12/site-packages/modelopt/torch/quantization/qtensor/nvfp4_tensor.py:92`
(dans `NVFP4QTensor.get_weights_scaling_factor`) :

```python
# avant
per_block_scale = per_block_amax / (6.0 * weights_scaling_factor_2)
# après
per_block_scale = per_block_amax / (6.0 * weights_scaling_factor_2.to(per_block_amax.device))
```

`per_block_amax` dérive de `input` (le poids, potentiellement déporté
CPU) ; `weights_scaling_factor_2` peut venir mémorisée depuis la
calibration (restée sur cuda). On aligne l'échelle globale — un
scalaire ou un petit tenseur par tête, jamais le poids entier — sur le
poids, plutôt que l'inverse : déplacer le poids déporté annulerait
l'intérêt du déport. Correctif local au venv (`--correctif`, pas amont) :
Sage l'autorisait explicitement, ce que ModelOpt en amont ignore de
`--use_seq_device_map` reste son problème, pas modifié ici.

## Contrôle qui rend faux (REGLES §5), avant tout correctif

Jouet de `verdict-modelopt-coder-tf4-17-09` (Qwen3MoeForCausalLM, 4
couches), déporté explicitement avec `accelerate.dispatch_model` (couches
paires cuda, impaires cpu — le jouet précédent, trop petit, ne
déclenchait jamais ce déport ; on le force). Version originale REJOUÉE
d'abord (témoin) :

```
RuntimeError: Expected all tensors to be on the same device, but found
at least two devices, cuda:0 and cpu!
  File ".../nvfp4_tensor.py", line 84, in get_weights_scaling_factor
    per_block_scale = per_block_amax / (6.0 * weights_scaling_factor_2)
```

**Identique au bit à l'échec du vrai Coder-30B** (même fichier, même
ligne, même message) — le contrôle reproduit fidèlement la cause, pas
une approximation. Puis, correctif restauré, même script rejoué :

```
export termine sans exception, MEME AVEC DEPORT
fichiers export : ['model.safetensors', 'generation_config.json', 'config.json', 'hf_quant_config.json']
CONTROLE DEPORT PASSE
```

**Rouge sans le correctif, vert avec — geste 1 VERT.**

## Geste 1 bis — TRT-LLM connaît-il `qwen3_5` et `gemma4` ?

`tensorrt_llm/_torch/models/__init__.py` (venv `/mnt/AI_GENERATOR/trt-llm/.venv`,
transformers 5.5.3 installé) :

- `qwen3_5` : **oui, sans condition** — `Qwen3_5ForCausalLM`,
  `Qwen3_5MoeForCausalLM` importés en tête de fichier (ligne 38, exposés
  ligne 87-88). Pas de garde de version.
- `gemma4` : **oui, sous condition satisfaite** — `Gemma4ForCausalLM`,
  `Gemma4ForConditionalGeneration` (ligne 108-116), import protégé par
  `try/except ImportError` documenté « requires transformers>=5.5.0 » ;
  la version installée (5.5.3) satisfait la condition.

**Ma prédiction de Sage était inversée sur ce point** : elle prédisait
Gemma-4 sûr, Qwen3.5-GDN incertain — c'est Qwen3.5 qui n'a AUCUNE garde
(le cas le plus simple), Gemma4 qui en a une (satisfaite ici). Aparté
hors de ce contrôle : `import tensorrt_llm` échoue actuellement dans ce
venv sur une bibliothèque MPI manquante (`libmpi.so`, sans rapport avec
le registre de modèles) — vérification purement statique du fichier
source, pas d'import réel réussi ; Laure devra lever ce blocage MPI
avant de pouvoir charger quoi que ce soit avec ce moteur, indépendamment
de ce verdict.

## Suite — geste 2

Geste 1 vert : je passe à la conversion réelle, Qwen3.8-27B d'abord,
scellé ≤ 45 min/conversion, `mto.save()` avant l'export, corpus bras-A.
