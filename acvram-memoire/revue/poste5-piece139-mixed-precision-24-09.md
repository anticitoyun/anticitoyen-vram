# Pièce 139 — étape 1 (à sec) : servir unsloth/Qwen3.8-27B-NVFP4 « mixed-precision » par dispatch PAR GROUPE (poste5, 24/09)

Ordre chef. Source : `/mnt/AI_GENERATOR/ninfer/sources/Qwen3.8-27B-NVFP4` (la copie que NInfer convertit, scellé
102 bis). Aucune carte ; en-têtes safetensors et un calcul processeur (`scratchpad/poste5-p139-24-09/`).

## 1. Le point de contrôle (en-têtes, pas le config seul)

`quant_method: compressed-tensors`, `format: mixed-precision`, deux groupes (config.json) :
* group_1 `nvfp4-pack-quantized` : `re:.*mlp\.(gate|up|down)_proj$` → **168 tenseurs** (MLP des couches 0-55) :
  `weight_packed` u8 [out, in/2], `weight_scale` e4m3 [out, in/16], `weight_global_scale` f32 [1].
* group_0 `float-quantized`, poids **par canal** (`strategy: channel`), activations dynamiques par jeton :
  `self_attn.(q|k|v|o)`, `linear_attn.(in_proj_qkv|in_proj_z|out_proj)`, `lm_head`, `layers.(56-63).mlp.*` →
  **233 tenseurs** : `weight` F8_E4M3 [out, in], `weight_scale` **BF16 [out, 1]**.
* Les couches 56-63 correspondent AUX DEUX motifs : c'est le contenu (F8, sans `weight_packed`) qui tranche pour le
  groupe 0. La résolution doit donc croiser les motifs ET la preuve du tenseur, pas prendre le premier motif.
* En plus : `k_scale`/`v_scale` (×16, `kv_cache_scheme` fp8 statique par tenseur) ; `ignore` = 303 modules de la tour
  visuelle (restent en clair).

Paramètres : nvfp4 14,97 G ; fp8 10,62 G (linear_attn 5,54, self_attn 1,68, MLP 56-63 2,14, lm_head 1,27) ;
plongements 1,27 G bf16. Pour comparaison, la copie locale `models_vllm/Qwen3.8-27B-NVFP4` est celle de **NVIDIA**
(modelopt, `MIXED_PRECISION`, MLP des 64 couches ET `lm_head` en nvfp4, fp8 par tenseur sur l'attention seule) :
autre point de contrôle, pas celui de NInfer.

## 2. Points de dispatch (fichier:ligne, branche poste5 après fusion de main)

| lieu | aujourd'hui | à changer (opt-in) |
|---|---|---|
| `acvram/quant/hfquant.py:55-70` `is_hfquant` | accepte `mixed-precision` (131 bis) | rien |
| `hfquant.py:78-99` `__init__` | refus nommé si > 1 format de groupe | reste le DÉFAUT ; sous opt-in : table des groupes (motifs `re:` compilés, format, `ignore`) |
| `hfquant.py:100-103` | refuse bits ≠ 4 (le groupe 0 a 8 bits) | ne pas lire bits/format sur `g0` sous opt-in |
| `hfquant.py:244-247` liste des suffixes ignorés | sans `k_scale`/`v_scale` : ils sortiraient comme tenseurs | les ignorer, et le dire (le KV d'acvram a son propre format) |
| `hfquant.py:248-266` `weight_packed` | décidé par `self.format` GLOBAL | format du groupe résolu pour CE tenseur |
| `hfquant.py:297` repli | un `weight` F8 de compressed-tensors sort BRUT, sans échelle (faux et muet) | branche fp8 par canal |
| `hfquant.py:181-185` `_fp8` | échelle scalaire seule (`reshape(())` échoue sur [out, 1]) | fp8 par canal, en fp32 (exact : 4 × 8 bits de mantisse < 24) |
| `acvram/quant/convert.py:1164` `_iter_checkpoint` | appelle `iter_tensors(direct_nvfp4)` | rien |
| `convert.py:1728-1745` | `NVFP4Tensor` → octets de la source | rien : les 168 nvfp4 passent tels quels (mêmes octets que NInfer) |
| `convert.py:1747-1755` | passage direct : tenseur bf16 déquantifié → forcé « clair » bf16 | tenseurs d'origine fp8 → int8 par canal (§ 3, option B) |
| `convert.py:1841-1855` `attn_canal` | int8 symétrique par canal pour q/k/v/o seulement | étendu aux tenseurs d'origine fp8 |

Chemins fp8 réutilisables : `HFQuantCheckpoint._fp8` (scalaire, Devstral) → généralisé au canal ; l'int8 symétrique
par canal (`formats.py:219` `_quantize_int8(symmetric=True)`, `group_size` = largeur) et son noyau servi existent
(tête int8 et projections q/k/v/o en int8 par canal). **acvram n'a aucun format fp8 servi** (`convert.py:483`).

## 3. Que faire des 10,62 G paramètres fp8 — trois options chiffrées

Octets de poids lus par pas de décodage (b=1, hors plongements) :
* NInfer (fp8 W8A16, recette 102 bis `recette_nvfp4_a16.py:30`) : 8,42 (nvfp4 à 4,5 bpw) + 10,62 = **19,0 Go**.
* **(A) bf16 clair** (le passage direct d'aujourd'hui) : 8,42 + 21,25 = **29,7 Go**, + 2,5 Go de plongements bf16 = 32,2
  Go > 32 Go de la 5090 → exil de couches, régime dégradé : **non servable, écarté**.
* **(B) int8 par canal ré-encodé depuis le fp8** : 8,42 + 10,63 = **19,1 Go** (+0,3 % contre NInfer). Poids pas au bit
  de NInfer. Mesure à sec (`erreur-fp8-int8.txt`, 4 tenseurs, erreur relative de Frobenius) :

  | tenseur | fp8 contre bf16 d'origine | int8c(fp8) contre fp8 | int8c(fp8) contre bf16 d'origine | int8g128(fp8) contre fp8 |
  |---|---|---|---|---|
  | L3 self_attn.q_proj | 2,66 % | 1,03 % | 2,87 % | 0,61 % |
  | L0 linear_attn.in_proj_qkv | 2,66 % | 0,94 % | 2,84 % | 0,60 % |
  | L60 mlp.down_proj | 2,66 % | 1,12 % | 2,90 % | 0,60 % |
  | lm_head (32 768 lignes) | 2,66 % | 0,95 % | 2,83 % | 0,60 % |

  Le ré-encodage ajoute ~1 % en quadrature à l'erreur du fp8 lui-même (2,66 → 2,85 % contre le bf16 d'origine). À
  noter : le fp8 d'unsloth est 2,8 × moins fidèle que notre int8 par canal pris sur le bf16 (0,94-1,12 %).
* **(C) fp8 servi nativement** (GEMV e4m3 × échelle par canal) : mêmes octets ET mêmes poids que NInfer, 19,0 Go.
  Demande un noyau neuf (poste moteur), hors de l'étape 2 processeur.

**Étape 2 codée en (B)**, derrière une variable d'opt-in. (C) est la seule option au bit. Elle se décide sur la KL de (B).

## 4. Étape 2 — forme et prédiction (écrite avant le code)

Opt-in `ACVRAM_HFQUANT_PAR_GROUPE=1`, imprimée par la conversion (ligne de régime). Sans elle : refus nommé inchangé.
Dispatch : pour chaque module, groupes dont un motif correspond, hors `ignore` ; preuve du tenseur (`weight_packed` →
format pack, `weight` F8 + `weight_scale` → float-quantized) ; on garde le groupe correspondant dont le format concorde.
**Refus nommé** (module, groupes candidats, preuve) si aucun ne concorde, ou si un tenseur quantifié ne correspond à
aucun groupe. fp8 → fp32 exact `w · s` ; convert force int8 symétrique par canal sur ces noms (manifeste :
`"origine": "fp8"`).

Tests processeur (`tests/test_hfquant_par_groupe_139.py`, point de contrôle synthétique à deux groupes, même
disposition que les en-têtes réels, motifs recopiés) — prédiction : tous verts ; chacun casse si l'on retire la pièce
qu'il garde :
1. sans opt-in : `NotImplementedError` inchangée ;
2. nvfp4 : `direct_nvfp4` → `NVFP4Tensor` aux MÊMES octets (qweight, block_scale, global = 1/weight_global_scale) ;
   sans direct → égal à `_ct_nvfp4` au bit ;
3. fp8 par canal : sortie fp32 == `weight.float() * weight_scale.float()` au bit (tenseur [out, 1]) ;
4. double motif (couche 60) : résolu en fp8 par la preuve ; un tenseur `weight_packed` sur un module visé par le seul
   groupe fp8 → refus nommé ;
5. `k_scale`/`v_scale`/`input_global_scale` jamais rendus ;
6. convert : un nom d'origine fp8 sort en int8 symétrique `group_size` = largeur, jamais bf16 clair ni nvfp4.

Carte (après feu, scellé à écrire avant) : conversion du 27B (octets manifeste prédits nvfp4 8,42 + int8 10,63 +
plongements 2,54 Go), jeton décodé au godet 1, KL b=1 contre la référence fp8 déquantifiée.

## 5. Défaut voisin, trouvé en route (non corrigé, décision demandée)

`models_vllm/Qwen3.8-27B-NVFP4` (NVIDIA, modelopt `MIXED_PRECISION`) passe `__init__` sans refus (les groupes modelopt
n'ont pas de champ `format`) ; ses 208 `weight` F8 (sans `weight_scale_2`) tombent au repli `hfquant.py:297` : poids
fp8 BRUTS, échelle perdue, **faux et muets** — même classe que le défaut de la 131. Remède d'une ligne dans la branche
modelopt (F8 + `weight_scale` scalaire → `_fp8` existant), ou refus nommé. Change la sortie du chemin par défaut (de
faux à juste) : à trancher par le chef.

## 6. Incohérence trouvée à la carte (notée, NON traitée — bead ouverte par le chef)

Une conversion Qwen3_5 neuve (`Qwen3_5ForConditionalGeneration`, source VL) GARDE la tour : `convert.py` VISION_PREFIXES
→ 333 tenseurs `model.visual.*`, manifeste `vision: oui`. Or le chargement REFUSE ensuite toute tour de cette famille :
`vision.py:260-268` `masque_images_famille` ne connaît que Gemma3, Gemma4, Qwen2VL, Qwen2_5_VL, Qwen3VL →
`MasqueImageInconnu` (1re prise (a), 06:55, `scratchpad/poste5-p139-24-09/prise-ab-1.txt`). La conversion produit donc
un alias qui ne se charge pas. L'alias 102 `Qwen3.8-27B-nvfp4` (16/09) n'a pas de tour : il a été converti avant
que le convertisseur la garde. Contournement pour la 139 (choix du chef) : `convert --sans-vision`, alias texte seul.
