# poste7 — convertisseur de formats : ce qui existe, ce qui manque, ce qui est nécessaire (16/09)

Question de l'utilisateur : intégrer un convertisseur gguf / nvfp4 / exl3 / awq / bf16 / instruct dans acvram — possible, nécessaire ? Le format acvram est-il spécifique au projet ?

## 1. Réponse courte

* **Le convertisseur existe déjà : `acvram convert`** (`cli.py:814`, `quant/convert.py:1011`). Il lit cinq familles de sources et écrit un seul format de sortie.
* **Oui, le format acvram est spécifique au projet** : fragments `acvram-NNNNN.safetensors` + `acvram_manifest.json` (`convert.py:749-780`, lu par `loader.py:195`). Aucun autre moteur ne le lit. L'arithmétique NVFP4 (E2M1 + E4M3/16 + FP32 global, `nvfp4.py:1-25`) est celle de NVIDIA/OCP ; le nommage, le manifeste, `int4_awq`, `int8` et `q3n` sont à nous.
* **« instruct » n'est pas un format** : c'est un fine-tune, mêmes tenseurs. Le gabarit de dialogue est copié tel quel (`convert.py:1893-1901` : `tokenizer_config.json`, `chat_template.jinja`).
* **L'export (acvram → gguf/exl3/awq) n'existe pas et n'est pas nécessaire** à l'objectif (battre les concurrents en t/s et J/jeton). Il ne servirait qu'à distribuer nos poids vers d'autres moteurs.

## 2. Ce que `_iter_checkpoint` (`convert.py:711-742`) accepte aujourd'hui

| source | module | portée | limite |
|---|---|---|---|
| safetensors bf16/fp16 HF | `convert.py:730-742` | tout modèle HF, instruct compris | — |
| GGUF | `quant/gguf.py` (52 Ko) | 13 types ggml en numpy (F32/F16/BF16, Q4_0/1, Q5_0/1, Q8_0, Q4_K/Q5_K/Q6_K, IQ4_XS) ; types à grille (IQ1-3, TQ) via l'exécutable llama.cpp (`gguf.py:196`) | déquantifié puis **requantifié** : deux erreurs s'additionnent (dit en tête du module) ; RoPE IMROPE liste blanche (`gguf.py:82`) |
| EXL3 | `quant/exl3.py` | décodage QTIP délégué à `exllamav3` (dépendance optionnelle, GPU requis) | même requantification |
| HF déjà quantifié | `quant/hfquant.py` | AWQ gemm, compressed-tensors `pack-quantized` et `nvfp4-pack-quantized`, modelopt NVFP4 (`hfquant.py:49-54`) | **déquantifié en bf16 puis requantifié** (`hfquant.py:1-5`) |
| GPTQ, MXFP4 (gpt-oss), bitsandbytes, EXL2 | — | **absents** (`hfquant.py:49-54` ne les détecte pas) | refus |

Tests : `test_gguf.py` 3, `test_exl3.py` 3, `test_source_au_manifeste.py` 4 — aucun test ne vérifie `hfquant.py` (0 fichier de tests le nomme ; contrôle : `grep -rln hfquant tests/`).

## 3. Ce qui manque et compte pour l'objectif — trois trous, par ordre

1. **Passage direct d'un NVFP4 modelopt / compressed-tensors, sans requantification.** Le duel (`8d5edb5`) compare acvram à vLLM sur « le même » checkpoint, mais nous le déquantifions puis relançons une recherche AWQ (`--no-awq` absent par défaut) : les poids ne sont plus ceux de vLLM, la colonne PPL du duel compare deux quantifications, pas deux moteurs. Coût : ~200 lignes (mapper `weight` u8 / `weight_scale` e4m3 / `weight_scale_2` f32 vers `NVFP4Tensor`, `nvfp4.py:112`), à sec. **Gain attendu : aucun en t/s ; il rend la colonne PPL du duel valide.** Réfutation : `dequantize_nvfp4(passage direct)` vs déquantification `hfquant.py` du même tenseur — égalité bit à bit exigée ; puis PPL acvram vs vLLM sur le checkpoint, écart ≤ 0,004 (bruit d'instrument, ETAT) sur 3 tranches ; > 0,004 → l'écart est dans le noyau, pas le format, et c'est un résultat.
2. **Lecteur MXFP4** (E2M1 + échelle E8M0 par 32, format HF de gpt-oss-20b/120b). Sans lui, la décision utilisateur en suspens sur gpt-oss-120b (`poste7.md`) n'a pas de chemin de conversion. Coût : ~80 lignes dans `hfquant.py`, mêmes conventions ; à sec. Réfutation : PPL du converti vs PPL vLLM sur le même checkpoint, ≤ 0,02 privé (seuil du comparatif, REGLES § 3).
3. **Lecteur GPTQ** : même empaquetage que AWQ gemm à l'ordre des quartets près (`hfquant.py:37`, `_AWQ_ORDER`) ; ~40 lignes. Utile parce que l'étalon de la courbe du quota est un GPTQ (`courbe-du-quota-deux-points`). Réfutation : dequant maison vs `auto_gptq` sur 3 tenseurs, égalité bit à bit.

Ne vaut pas le coût : types GGUF Q2_K/Q3_K en numpy (requantifier un 3 bits en NVFP4 additionne deux erreurs, `gguf.py:7-9`), export vers GGUF/EXL3, lecteur EXL2.

## 4. Doute nommé

Si la requantification NVFP4 → NVFP4 avec `--no-awq` reproduit les codes modelopt bit à bit (même formule amax/6 par bloc, même global amax/(448·6)), le trou 1 se réduit à une option de ligne de commande et un test — c'est l'issue qui me gênerait, elle est à mesurer en premier (10 min à sec sur 3 tenseurs), avant d'écrire le passage direct.

## Ordre

1. poste1 (à sec, 10 min) : 3 tenseurs d'un NVFP4 modelopt → `hfquant` déquant → `quantize_nvfp4(no_awq)` → codes et échelles comparés bit à bit aux originaux. Verdict : `verdict-requant-nvfp4-bitabit`. Identiques → trou 1 = `--no-awq` + garde manifeste ; différents → poste4 écrit le passage direct (§ 3.1) après le correctif MLA.
2. poste2, seulement si l'utilisateur retient gpt-oss-120b : lecteur MXFP4 (§ 3.2), scellé PPL ≤ 0,02 privé.
3. Personne : export, EXL2, Q2_K/Q3_K numpy — écartés.
4. chef porte à l'utilisateur : convertisseur import = existant ; format acvram = propre au projet, par choix (noyaux maison) ; aucun export prévu.
