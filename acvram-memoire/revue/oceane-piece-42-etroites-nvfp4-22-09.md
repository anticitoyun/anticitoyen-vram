# Pièce 42 — servir les projections d attention en NVFP4 au décodage : note de décision (22/09, Océane, à sec)

* sources : verdict 41 (service : qkv [80×4] 10,37 µs, o [32×11] 12,02 µs, **1,075 ms/pas à deux**), `verdict-nsys-trtllm-serve` (leurs projections FP8 : 0,498 ms/pas), `verdict-gemm-dense-palier2-banc-17-09` (dense NVFP4 Triton à M=12, **par forme** : q 0,71 · kv 0,31 · o 0,81 · gate_up 1,00 · down 1,10 To/s), `verdict-gemm-dense-palier1-situ-17-09` (palier 1 = défaut `ACVRAM_DENSE_NVFP4=triton`, 2 ≤ M ≤ 32), `verdict-c1-noyau-19-09` (C1 W4A8 **FAUX**), `verdict-porte-a8-19-09` (activation int8 par jeton : +0,0006 de ratio PPL, porte OUVERTE), `sage-c17-faux-ferme-20-09` (W4A4 experts fermé sur la qualité)
* correction de la prémisse : trtllm ne lit pas « moins d octets parce qu ils sont en 4 bits » — leurs projections sont en **FP8 W8A8** (`verdict-cellules-trtllm` § 5 : KvCacheConfig FP8, point de contrôle FP4-hub = experts FP4, attention FP8). 18,8 Mo/couche en 10,4 µs = **1,81 To/s** : même volume d octets que nos 18,8 Mo (int8), **2,7 × notre débit**. Leur avance est le noyau, pas le format. Une conversion NVFP4 de nos projections réduirait les octets de 1,82 × ; à débit égal au nôtre elle les mettrait à 5,9 + 4,7 = **0,51 ms/pas** — soit leur chiffre, mais par un autre chemin.

## 1. Quel noyau servirait M = 12, K 2048/4096, N 5120/2048 en NVFP4 ?
| chemin | où | statut à ces formes |
|---|---|---|
| `gemm_dense_etroit` (Triton, palier 1, **défaut** pour 2 ≤ M ≤ 32) | `kernels/gemm_dense_etroit.py` via `nvfp4_matmul` | **existe et sert déjà** les projections NVFP4 des modèles convertis sans `-qkvo-i8c` ; débit mesuré au banc 17/09 : **q 0,71, kv 0,31, o 0,81 To/s** |
| `narrow_gemm` (CUDA, M ≤ 16) | `acvram_kernels.cu` | opt-in `_NARROW_GEMM`, jamais devenu défaut sur ces formes |
| `nvfp4_gemv` (dense, M ≤ 32) | `.cu` | témoin (`DENSE_NVFP4=gemv`), poids relus par ligne |
| Marlin (`nvfp4_gemv_marlin*`) | `.cu` + `marlin_port` | **experts seulement** : la disposition Marlin est construite par pile d experts (`preparer_pile` [E, …]), pas pour une projection dense unique — il faudrait une pile à E = 1 et un repack au chargement : faisable, jamais fait |
| C1 W4A8 (`narrow_gemm` + activation int8) | | **FAUX au 19/09** : pas au bit à `ks=128` aux formes du modèle, 1,8 × plus lent que Marlin sur ses propres noyaux, × 21 sur le pas (quantification A8 en torch élémentaire) — **ne s applique pas ici** : c était un noyau W4A8 pour les EXPERTS au préfill, et son échec était le noyau, pas le format |
**Le chemin existe déjà** : convertir avec `-qkvo` en NVFP4 au lieu d int8 par canal, et `nvfp4_matmul` prend le `gemm_dense_etroit` par défaut. Aucun noyau à écrire — ce qui change la nature de la pièce : c est une **conversion + une mesure**, pas un chantier de noyau.

## 2. Octets et prédiction
| | int8 (servi) | nvfp4 | µs à 0,8 To/s (le débit mesuré du dense étroit sur ces formes) | µs à 1,55 |
|---|---|---|---|---|
| qkv [5120, 2048] | 10,73 Mo | **5,90 Mo** | 7,4 | 3,8 |
| o [2048, 4096] | 8,59 Mo | **4,72 Mo** | 5,9 | 3,0 |
| deux, × 48 couches | 1,075 ms (mesuré) | — | **0,64 ms** | 0,33 ms |
**Prédit** : à 0,71-0,81 To/s (banc 17/09, formes proches), **0,62-0,70 ms/pas → −0,38 à −0,46 ms (−5,2 à −6,3 %)**, t/s 1 700 → **1 790-1 815**. La prédiction de la Maîtresse (« 5 µs chacun, −0,55 ms ») suppose 1,0-1,2 To/s sur ces formes : le banc dit 0,71-0,81, et `kv` [512, 2048] seule tombe à 0,31 — **q, k, v séparées seraient pires que la pile qkv actuelle** ; il faut que la conversion garde la pile (`qkv_proj` empilée, `SEUIL_FUSION`), sinon le gain s inverse. Réfuté si la cellule rend < −0,2 ms.

## 3. Coût de conversion et risque qualité
* Conversion : `--attn-qkvo-int8-canal` absent → les projections passent en NVFP4 par le routeur de format existant (`format_for`), avec recherche AWQ par canal comme les experts (`search_channel_scales`, stats de calibration déjà collectées) : **aucun code**, un alias de plus (~20 min de carte pour le 30B), et un manifeste qui dit `attn_int8: absent`.
* Qualité : NVFP4 W4A16 sur q/k/v/o est exactement ce que les alias sans `-qkvo-i8c` servent déjà (c est le défaut du dépôt) ; le risque n est pas neuf. Mesure : KL sur les 5 dumps (seuil 1,0, référence 0,519) **et** PPL A/B (`acvram eval` corrigé BOS + budget) à 2 SE. **Alarme nommée d avance** : `q_proj` et `k_proj` sont sûrs en NVFP4, la perte connue vient de `v_proj`, `o_proj` et `lm_head` (mémoire des variantes A/C) — si KL > 1,0, le bras suivant est « q/k en nvfp4, v/o en int8 », pas l abandon.
* Ce qui ne change pas : `lm_head` reste int8 (verdict 41 : 0,203 ms, déjà à 1,52 To/s) ; les experts restent Marlin W4A16.

## 4. Go / no-go
**GO**, avec ce cadrage : (1) pas de noyau à écrire, pas d opt-in `ACVRAM_ETROITES_FORMAT` — le format est une propriété du CONVERTI, pas du moteur (un drapeau de moteur qui change le format des poids servis n a pas de sens : les octets sont sur le disque) ; le témoin est l alias int8 actuel, le bras est un alias nvfp4, les deux coexistent ; (2) ordre : convertir (Manon, ≤ 30 min), `familles-noyaux --detail proj_etroites_int8` sur le nouvel alias pour lire les µs réels, puis ABBA t/s et KL + PPL ; (3) **je ne code rien** tant que la cellule n a pas montré ≥ −0,2 ms : s il faut ensuite un noyau (Marlin dense à E = 1, ou `narrow_gemm` promu), ce sera une pièce à part, chiffrée sur les µs mesurés.
