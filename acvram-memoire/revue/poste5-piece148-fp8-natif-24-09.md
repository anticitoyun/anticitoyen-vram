# Pièce 148 (à sec) : servir les couches fp8 de l'alias mixte autrement qu'en int8 W8A16 (poste5, 24/09)

Ordre chef. Aucun code, aucune carte. Alias `Qwen3.8-27B-unsloth-mixte-i8c` : 233 tenseurs d'origine fp8, **10,62 G
paramètres**, poids fp8 e4m3 de la source avec une échelle bf16 par ligne (`hfquant.py:_fp8_canal`).

## 0. Ce qu'on attaque, et ce qu'on sait déjà

| forme (N × K) | tenseurs | poids | famille |
|---|---|---|---|
| in_proj_qkv 10240 × 5120 | 48 | 2,52 G (24 %) | large, K = 5120 |
| in_proj_z 6144 × 5120 | 48 | 1,51 G (14 %) | large |
| out_proj 5120 × 6144 | 48 | 1,51 G (14 %) | « o », K > N |
| lm_head 248320 × 5120 | 1 | 1,27 G (12 %) | tête |
| q_proj 12288 × 5120 | 16 | 1,01 G (9 %) | large |
| gate/up 17408 × 5120, down 5120 × 17408 (L56-63) | 24 | 2,13 G (20 %) | large / « o » |
| o_proj 5120 × 6144, k/v 1024 × 5120 | 48 | 0,66 G (6 %) | « o » / petit |

* Aujourd'hui : **int8 symétrique par canal, W8A16**, 1 octet + 2 o par ligne ≈ **10,63 Go lus par pas**. Coût ≈ **5,8 ms/pas
  à b=8** (décomposition du scellé 139 bis, confirmée à 5 % près), ≈ 9,7 ms à b=1 (10,63 Go à ~1,1 To/s).
* **Déjà mesuré, décisif** (`revue/verdict-banc-etroites-noyaux-fp8-22-09.md`, pièce 43, poste2) : sur nos formes
  étroites, le **W8A8 fp8 par `torch._scaled_mm`** (cuBLASLt) plafonne à **0,61 To/s (qkv) et 0,30 To/s (o)**, et n'est
  pas au bit. Le meilleur 8 bits d'acvram est l'int8 Triton : **1,19 / 0,82 To/s**. Aucun noyau n'y dépassait 1,0 To/s
  sur la forme « o ».
* Aucune MMA fp8 dans acvram : seule `kind::mxf4nvf4` e2m1 (`acvram_kernels.cu:3497-3526`). Le commentaire :3507 note
  que `mxf8f6f4` n'est pas un repli.

## 1. Les variantes

Les octets ne changent dans AUCUNE variante (1 octet par poids plus une échelle par ligne) : **tout gain vient de
l'efficacité du noyau par octet**. Gain par pas = 10,63 Go × (1/E_actuel − 1/E_nouveau).

| variante | octets/pas | sortie | noyaux réutilisables (fichier:ligne) |
|---|---|---|---|
| **A. fp8 W8A16, déquant e4m3 dans le GEMV** (poids de la source tels quels, format « fp8 » neuf) | 10,63 Go | poids **au bit du fp8 source** (e4m3 × s exact en fp32) ; sortie ± ulp de l'ordre des sommes ; retire le ~1 % de ré-encodage int8 (erreur contre bf16 2,85 → 2,66 %) | `acvram_kernels.cu:72` `e4m3_to_float` (déjà dans les GEMV nvfp4 pour les échelles) ; Marlin porté : `dequant.h:357-378` (bf16 × `kFE4M3fn`), `marlin.cu:825` accepte `kFE4M3fn` — instanciation absente (`generate_kernels.py:65`, fp4 seul) |
| **B. fp8 W8A8 comme NInfer** (activation e4m3 par jeton) | 10,63 Go | **change la sortie** au-delà de l'ulp (activations quantifiées) : rend une partie des −5,67 % de PPL gagnés contre NInfer (139 b) | `fp4_gemm.py:257-282` (`torch._scaled_mm` fp8×fp8 par ligne : utilisable tel quel avec les poids source) ; aucune MMA fp8 maison |
| **C. int8 inchangé, noyau Marlin 8 bits** (`kU8B128` = notre int8 zéro 128, échelle par canal) | 10,63 Go | mêmes poids ; sortie ± ulp (ordre des sommes, comme PROJ_MARLIN : PPL +0,004 % sur la 142) | Marlin porté : `dequant.h:227-271` (`kU8B128`), `marlin.cu:823` ; repack générique en bits (`gptq_marlin_repack.cu:14-20`) ; disposition et GEMV v2 de la 129/130 (`ACVRAM_GEMV_MARLIN_V2`, nvfp4 seulement aujourd'hui) |

**Gain prédit** (E en To/s ; actuel b=1 ≈ 1,1 moyen, b=8 ≈ 0,8-1,1 selon la forme) :
* **B** : décodage **plus lent** — E mesuré 0,30-0,61 → b=8 **+4 à +26 ms/pas**. Au préfill seulement (M grand, calcul
  borné), `_scaled_mm` fp8 est plausible, mais il change la sortie : hors sujet ici.
* **A et C, b=1** : un GEMV M=1 dépend de la lecture, pas du type. Même noyau à 1,1 → 0 ; à E = 1,4 (78 % du pic HBM de
  1,79 To/s) → **−2,1 ms/pas (+14 %)**. Borne haute à 1,6 : −3,0 ms (+21 %).
* **A et C, b=8** : E actuel 0,8-1,1 → 9,7-13,3 ms. À E = 1,2-1,5 → 7,1-8,9 ms : **−0,8 à −6,2 ms/pas (+3 à +30 %)**. Le
  gros du gain possible est sur les formes « o » (K > N, 26 % des octets), à 0,82 aujourd'hui.
* A n'a aucun avantage de vitesse sur C : la conversion e4m3 → bf16 (`cvt` matériel sur sm_89+, 2 éléments par
  instruction) coûte le même ordre que int8 → bf16. A gagne en exactitude, pas en temps.

## 2. Recommandation

1. **Ne pas faire B** pour le décodage. Déjà réfuté au banc (0,30-0,61 To/s, pas au bit) ; et il changerait la sortie
   dans le sens de NInfer, là où la 139 b nous donne l'avantage.
2. **Le levier est le noyau 8 bits, pas le format.** Avant tout port, un **banc de 5 min** (après feu, comme la p100
   d'poste1) : sur les 4 formes réelles (10240×5120, 6144×5120, 5120×6144, 5120×17408), à M = 1 et M = 8, L2 froid.
   Candidats : int8 servi (témoin), Marlin W8A16 `uint8b128` et Marlin fp8 (`apply_fp8_marlin_linear`), tous deux dans
   le venv vLLM **sans les porter**. Seuil scellé : un candidat ≥ 1,35 To/s à M = 8 sur les formes « o » ET larges (gain
   ≥ 2 ms/pas à b=8) → port ; sinon fermé, et l'écart avec NInfer vient d'ailleurs.
3. Si le banc désigne un noyau : **C d'abord**. Mêmes poids et octets, aucun format neuf : il suffit d'étendre
   `generate_kernels.py:65` à `kU8B128` (group_blocks −1) et de brancher la disposition de la 129 sur l'int8. **A
   ensuite, seulement si** la qualité le justifie : ~1 % d'erreur de poids en moins sur 40 % des paramètres, KL à mesurer.
   A demande un format « fp8 » neuf (formats, convert, loader, dispatch).
4. Porte de sortie pour C : ± ulp, KL b=1 ≤ 0,74 et PPL appariée à ± 0,5 % contre l'int8 actuel (règle de la 142).
   Pour A : même porte, plus la PPL attendue en baisse (à sceller, prédiction −0,1 à −0,5 %).
