# Sage — Passage direct : vLLM est W4A4 par construction (lu, pas déduit) ; on publie, avec un contrôle de 20 min qui peut me donner tort ; EXL3 GLM = case « non supporté » (17/09)

Entrées : `verdict-passage-direct-17-09` (Laure, 80a7c2d, main 500a998) ; `verdict-exl3-coder-glm-17-09` (Manon, 280487b, main 1029706). Scellé S2 (`sage-convertisseur-formats-16-09` § 3.1, ≤ 0,004) réfuté : 0,959 / 0,950 — l'issue « > 0,004 → l'écart est dans le noyau » était nommée d'avance, c'est elle.

## 1. Ce que Laure déduisait, le code de vLLM le dit

Le checkpoint `GadflyII/GLM-4.7-Flash-NVFP4` est **compressed-tensors** (`config.json` : `quant_method: compressed-tensors`, `format: nvfp4-pack-quantized`, `config_groups.group_0.input_activations = {num_bits 4, float, tensor_group 16, dynamic true}`), pas ModelOpt. Dans vLLM (`/opt/ia/vLLM/.venv/.../vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:735-743`) : `input_quant is None → CompressedTensorsW4A4Fp4(use_a16=True)` (Marlin, activations bf16), sinon `CompressedTensorsW4A4Fp4()` — **W4A4 : blocs de 16 d'activation quantifiés en FP4 à l'exécution, sous une échelle globale par tenseur fixée à la calibration** (`schemes/compressed_tensors_w4a4_nvfp4.py:87-128`, `input_global_scale`). Le côté ModelOpt fait pareil (`modelopt.py:1040-1045`, `:1421-1430` pour le MoE). Donc la lecture de Laure est juste, et elle se cite désormais avec ces lignes, pas comme une déduction. Reste non expliqué : notre W4A4 (1,040) est encore 0,03 sous vLLM (1,072) — candidat : l'échelle globale **statique** de vLLM (amax de calibration, écrête tout ce qui dépasse) contre notre échelle dynamique ; non tranché, pas prioritaire.

## 2. Décision : (a), et un contrôle de 20 min avant la table

* Publier : à poids strictement identiques, acvram W4A16 1,028 / 1,021, vLLM 1,072 / 1,075 ; la réserve du duel (« vLLM 1,056 non apparié ») devient « même poids, la perte est l'arithmétique W4A4 du moteur ». `-vllm-direct` reste **non classé** (1,028 > 1,02) : à moteur égal, les poids ModelOpt/CT communautaires valent 1,028 là où les nôtres (`-k48-calibA`) valent 1,015 — ligne à publier telle quelle, sans l'arrondir en victoire.
* Contrôle (Manon à sec 10 min, Laure carte 20 min) : copie du dossier GadflyII par liens symboliques, `config.json` avec `input_activations: null` → vLLM en Marlin W4A16 sur les mêmes poids, PPL privé + public, 3 tranches. **Prédiction : 1,028 ± 0,006** (= notre W4A16 ; Marlin déquantifie exactement). Si ≈ 1,07 : la perte n'est pas l'activation, ma lecture est fausse et la cause se cherche ailleurs (routage sigmoid + biais en bf16 chez vLLM, `sage-refutation-glm-routage-14-09`). Si vLLM refuse `input_activations: null` sur le MoE : « impossible », publié tel quel, § 1 tient sur le code seul.
* Conséquence pour la file : **ModelOpt propre depuis srcbf16 (`sage-comparatif` § 6) est suspendu** tant que ce contrôle n'est pas rendu — un checkpoint recalibré servi en W4A4 resterait au-dessus de 1,02 par le même mécanisme ; s'il rend 1,028, la colonne vLLM classable est « W4A16 Marlin, régime non défaut », vitesses à remesurer dans ce régime (Laure, 2 × 20 min), et le ModelOpt propre n'a plus d'objet.
* v1 (attention MLA en NVFP4 : 1,33 × bf16) : va dans REGLES § 9 en une ligne — « ne jamais quantifier les projections MLA en NVFP4 ».

## 3. EXL3 GLM : case « non supporté », pas une décision utilisateur

exllamav3 (1.4.8 et master) n'implémente pas MLA — Manon cite le fichier et la ligne où `Glm4MoeForCausalLM` est déclaré sans `q_lora_rank`/`kv_lora_rank`. Un échec est un résultat : la case GLM/TabbyAPI de la table porte « MLA non implémenté dans exllamav3 » avec la référence ; l'aliasser aurait mesuré une attention fausse, et c'était le bon refus. Ajouter MLA à un outil externe n'est pas notre objectif ; Jérôme le porte à l'utilisateur en une ligne d'information, pas en question. Coder EXL3 continue (Laure, PPL 20 min).

## Ordre

1. Jérôme : ETAT — réserve du duel remplacée par « même poids, W4A4 moteur (fichier:ligne § 1) » ; ModelOpt propre suspendu ; case GLM/TabbyAPI « MLA non implémenté » ; une ligne à l'utilisateur (information). REGLES § 9 : ligne MLA/NVFP4.
2. Manon (à sec, 10 min) : dossier `GLM-4.7-Flash-NVFP4-a16` (liens symboliques + `config.json` modifié, `input_activations: null`), sha256 du config dans le verdict ; puis citer fichier:ligne d'exllamav3 dans `verdict-exl3-coder-glm-17-09`.
3. Laure (carte, 2 × 20 min, dans l'ordre) : PPL Coder EXL3 ; puis vLLM W4A16 Marlin sur `-a16`, scellé § 2 (1,028 ± 0,006), verdict `verdict-vllm-a16-17-09`. Si ≈ 1,028 : vitesses vLLM b=1/b=12 dans ce régime, même en-tête que le duel.
4. Laurine : inchangé (`sage-calibration-verdict-17-09` § 3, puis rien de nouveau ici).
