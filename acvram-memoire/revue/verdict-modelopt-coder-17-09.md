# Verdict — ModelOpt propre Coder-30B : quantification faite, export BLOQUÉ (Qwen3MoeExperts non supporté)

Manon, 17/09. Ordre Sage (`sage-plan-completion-comparatif-17-09.md` § 1/3,
§6 rouvert pour Coder seul) : ModelOpt 0.37 NVFP4 depuis `srcbf16`,
calibration `bras-A-anglais.txt`, 512 échantillons.

## Corpus de calibration

Même fichier que le bras A du chantier calibration (REGLES §3 : aucun
`revue/*.md`, aucun wikitext) :

```
sha256 (bras-A-anglais.txt) : cb7c0d9af2041d31e655fafe79becb0877c3fc775690c7a558eb8707035e5138
```

`get_dataset_dataloader` de ModelOpt (`modelopt/torch/utils/dataset_utils.py`)
n'accepte qu'un nom de jeu de données parmi une liste figée
(`cnn_dailymail`, `pile`, `pg19`, `wikipedia`, `c4`, ...) — aucun fichier
local. `_get_dataset_samples` monkeypatché (`/tmp/lancer-modelopt-
driver.py`, gardé, pas dans le dépôt car outil ponctuel) pour reconnaître
`--dataset bras_a` : découpe le fichier en 512 tranches égales par
caractères, même principe que `load_calib_ids` (`acvram/quant/
collect.py:69-71`). 512 échantillons produits, vérifié par assertion.

## En-tête de mesure

`nvidia-modelopt` 0.37.0 (`/mnt/AI_GENERATOR/trt-llm/.venv`), script
officiel `examples/llm_ptq/hf_ptq.py` cloné depuis `github.com/NVIDIA/
TensorRT-Model-Optimizer` au tag exact `0.37.0`. `outils/carte.sh`, une
carte + hôte (le modèle ne tient pas en VRAM seule : `--use_seq_device_map`,
limite appliquée automatiquement `{0: 26,5 Gio, cpu: 87,9 Gio}`).
`--qformat nvfp4` (= `mtq.NVFP4_DEFAULT_CFG`).

## Deux bogues d'environnement corrigés en route (avertis par le paquet lui-même : « transformers 5.5.3 incompatible avec nvidia-modelopt »)

1. `hf_ptq.py` importe le module frère `example_utils` du même dossier —
   cassé sous `runpy.run_path` (le monkeypatch exige d'exécuter le script,
   pas de l'importer tel quel, `main(args)` tourne au niveau module sans
   garde `if __name__`). Corrigé : `sys.path.insert(0, dirname(hf_ptq.py))`
   avant `runpy.run_path`.
2. `get_dataset_dataloader` (`dataset_utils.py:206`) appelle
   `tokenizer.batch_encode_plus(...)`, retiré de `Qwen2Tokenizer` dans
   transformers 5.5.3 (`AttributeError`). Corrigé : alias
   `PreTrainedTokenizerBase.batch_encode_plus = PreTrainedTokenizerBase.__call__`
   quand absent (même signature : texte, `return_tensors`, `padding`,
   `truncation`, `max_length`) — n'affecte que ce process, pas le venv
   partagé.

## Résultat : quantification FAITE, export BLOQUÉ

Chargement des 531 fragments (5 min 53 s), calibration exécutée avec
succès sur les 512 échantillons `bras_a` — le journal publie un exemple
avant/après PTQ confirmant que la quantification a réellement changé la
sortie du modèle (texte différent sur le même passage). **L'export du
checkpoint quantifié échoue** :

```
NotImplementedError: MoE model with experts type 'Qwen3MoeExperts' is not
supported in export. Please file an issue or add support for this model
architecture.
```

Vérifié avant de conclure (REGLES §7) : `unified_export_hf.py:386-420` —
le chemin d'export ne reconnaît que deux représentations d'experts MoE
fusionnés/non-itérables par leur nom de classe exact, `Llama4TextExperts`
et `GptOssExperts` (`:302`, `:386`, `:468`) ; toute AUTRE classe non
itérable (dont `Qwen3MoeExperts`, la représentation fusionnée du Qwen3
MoE dans transformers 5.5.3) tombe dans le `else` et lève l'erreur
ci-dessus (`:419-422`). Ce n'est pas un défaut de configuration de ma
part : ModelOpt 0.37.0 sait QUANTIFIER Qwen3MoeExperts (la calibration a
tourné), mais ne sait pas encore l'EXPORTER vers un format de checkpoint
portable. Aucune version plus récente vérifiée pour un correctif éventuel
— je n'ai pas cherché à mettre à jour `nvidia-modelopt` dans le venv
partagé (Laure y sert TRT-LLM) sans autorisation.

Aucun fichier écrit dans `models_modelopt/` (l'export échoue avant toute
écriture) ; répertoire vide supprimé.

## Suite

Rendu à Jérôme/Sage : la cellule « Coder vLLM/TRT-LLM sur ModelOpt propre »
du plan de complétion (§1) reste vide, cause nommée fichier:ligne — même
statut qu'EXL3 GLM (limite d'un outil tiers, pas un échec de méthode).
Options possibles côté Sage : mettre à jour `nvidia-modelopt` (risque sur
le venv partagé, à peser), ou accepter la cellule vide au critère de
complétion (ii). Carte libérée pour Laure (vLLM Coder W4A4/W4A16 sur les
poids ModelOpt communautaires déjà connus, ou la suite de sa chaîne).
