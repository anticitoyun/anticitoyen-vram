# Sage — convertisseur de formats : ce qui existe, ce qui manque, ce qui est nécessaire (16/09)

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
2. **Lecteur MXFP4** (E2M1 + échelle E8M0 par 32, format HF de gpt-oss-20b/120b). Sans lui, la décision utilisateur en suspens sur gpt-oss-120b (`sage.md`) n'a pas de chemin de conversion. Coût : ~80 lignes dans `hfquant.py`, mêmes conventions ; à sec. Réfutation : PPL du converti vs PPL vLLM sur le même checkpoint, ≤ 0,02 privé (seuil du comparatif, REGLES § 3).
3. **Lecteur GPTQ** : même empaquetage que AWQ gemm à l'ordre des quartets près (`hfquant.py:37`, `_AWQ_ORDER`) ; ~40 lignes. Utile parce que l'étalon de la courbe du quota est un GPTQ (`courbe-du-quota-deux-points`). Réfutation : dequant maison vs `auto_gptq` sur 3 tenseurs, égalité bit à bit.

Ne vaut pas le coût : types GGUF Q2_K/Q3_K en numpy (requantifier un 3 bits en NVFP4 additionne deux erreurs, `gguf.py:7-9`), export vers GGUF/EXL3, lecteur EXL2.

## 4. Doute nommé

Si la requantification NVFP4 → NVFP4 avec `--no-awq` reproduit les codes modelopt bit à bit (même formule amax/6 par bloc, même global amax/(448·6)), le trou 1 se réduit à une option de ligne de commande et un test — c'est l'issue qui me gênerait, elle est à mesurer en premier (10 min à sec sur 3 tenseurs), avant d'écrire le passage direct.

## 5. Conversion depuis la GUI — question 2 de l'utilisateur (« donc nécessaire »)

**Oui pour le produit, non pour l'objectif de mesure.** Fait : la fenêtre ne liste que les dossiers portant `acvram_manifest.json` (`packaging/acvram-gui:83-84`), et le serveur n'a aucune route de conversion (`server/app.py:208-935` : console, parc, moteurs, v1/*). Un utilisateur du `.deb` qui possède un GGUF, un EXL3 ou un HF n'a que la ligne de commande — le `.deb` ne sert donc que ce que nous avons converti. Le format étant le nôtre (§ 1), la conversion est l'entrée du produit, pas une option.

Contraintes qui décident de la forme, toutes issues de REGLES § 2 :
* la conversion alloue sur la carte (`--calib-device cuda:0`, `--quant-device auto`, `cli.py`) → **le sous-processus prend `carte.sh` lui-même** et écrit `CONVERSION` dans le verrou ; verrou tenu par une mesure → refus affiché, pas d'attente ;
* serveur en marche et conversion partagent la VRAM → soit serveur arrêté d'abord (la fenêtre sait déjà le faire, `acvram-gui:174`), soit conversion à sec (`cpu`/`cpu`) — durée CPU à mesurer avant de la proposer par défaut ;
* forme minimale : `POST /convertir` (source, format, sortie, `--no-awq`) → sous-processus `acvram convert`, journal streamé comme `page_attente` le fait pour le serveur (`acvram-gui:146-149`) ; `is_gguf/is_exl3/is_hfquant` appelés AVANT le lancement pour dire « lisible / refusé (GPTQ, MXFP4) » ; ~150 lignes `app.py`, ~60 `console.py`, 0 ligne dans `convert.py`.

Prédiction et réfutation : fragments et manifeste produits par la GUI **identiques (sha256) à ceux de la CLI** aux mêmes options sur un petit modèle ; lancement pendant une mesure → **refus** (ce contrôle doit rendre « faux » : le tester verrou tenu). Coût : une session à sec, ~½ journée. Quand : **après le duel** (file en un bloc, REGLES § 1) et **après le point 1 de l'ordre** — sinon la GUI industrialise la requantification en défaut.

## 6. Retour du point 1 (Océane 98d1e87, main 2964f07) — trou 1 confirmé, mécanisme non établi

Mesuré : `global_scale` identique au bit ; `block_scale` et codes divergent sur ~50 % des blocs, écart médian 33 %, jusqu'à 84 %. Le raccourci `--no-awq` est **réfuté** : le passage direct est un vrai chantier.

La lecture « ModelOpt calibre son échelle de bloc » est une hypothèse, pas un mécanisme (REGLES § 4 bis). Un E4M3 arrondit à ≤ 6,25 % ; 33 % de médiane a deux causes possibles, qui n'appellent pas le même remède :
* (a) échelle de bloc **cherchée** (MSE/optimale) et non amax/6 → ModelOpt produit de meilleurs poids que nous à octets égaux : c'est un levier de qualité pour NOTRE convertisseur, pas seulement une question d'équité ;
* (b) **désalignement** de la comparaison (blocs de 16 pris sur le mauvais axe, échelles swizzlées, transposition [out, in/16]) → les poids sont les mêmes, l'instrument compare deux découpages.

Contrôle qui rend « faux », 10 min à sec : sous la règle amax/6 chaque bloc de 16 porte au moins un code de magnitude 6 (à l'arrondi près). Sur les blocs divergents de ModelOpt, compter la part de blocs **sans** code à 6. ≈ 0 % → (b), reprendre l'instrument avant d'écrire une ligne ; nettement > 0 % (scellé : > 20 %) → (a), et le passage direct s'accompagne d'une recherche d'échelle de bloc dans `quantize_nvfp4` (`nvfp4.py:216`), scellé : PPL ≤ celle de la conversion actuelle − 0,004 sur 3 tranches.

Priorité : **à sec, sans carte, après la capture Hadamard de Laurine (ne pas l'interrompre) et AVANT toute colonne PPL GLM publiée dans le comparatif** — sans lui, la colonne compare deux quantifications. Jérôme en fait la porte du comparatif, pas une urgence de carte.

## Ordre

0. GUI : Manon, à sec, après le duel et après le point 1 — route `/convertir` + verrou pris par le sous-processus ; scellé : sha256 GUI = CLI, refus sous verrou tenu. Pas avant.
1. FAIT (Océane 98d1e87) : divergent → passage direct à écrire. Suite :
   1a. Océane (à sec, 10 min) : sur les blocs divergents ModelOpt, part de blocs sans code de magnitude 6 ; scellé > 20 % → (a) échelle cherchée, ≈ 0 % → (b) instrument désaligné. Verdict : `verdict-modelopt-bloc-amax`.
   1b. Laurine, après la capture Hadamard, à sec : passage direct `weight/weight_scale/weight_scale_2 → NVFP4Tensor` sans déquant ; scellé : déquant bit à bit égale à `hfquant`, PPL acvram vs vLLM même checkpoint ≤ 0,004 sur 3 tranches. Si 1a rend (a) : ajouter la recherche d'échelle de bloc dans `quantize_nvfp4`, scellé PPL ≤ actuelle − 0,004.
   1c. Jérôme : aucune colonne PPL GLM dans le comparatif avant 1b.
2. Manon, seulement si l'utilisateur retient gpt-oss-120b : lecteur MXFP4 (§ 3.2), scellé PPL ≤ 0,02 privé.
3. Personne : export, EXL2, Q2_K/Q3_K numpy — écartés.
4. Jérôme porte à l'utilisateur : convertisseur import = existant ; format acvram = propre au projet, par choix (noyaux maison) ; aucun export prévu.
