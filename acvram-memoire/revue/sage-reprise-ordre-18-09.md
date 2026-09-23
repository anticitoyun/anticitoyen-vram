# Sage — Reprise : restes 3/3 soldés ; hybrides/dense clos sur le profil (un noyau à 75 %, CUDA = parité → ligne utilisateur) ; prochain poste = les deux cellules classées encore perdues, toutes deux contre llama.cpp (prefill −37 %, b=1 −16 %) : un profil de 40 min avant tout noyau (18/09)

Entrée : Jérôme e812c5a ; Laure 16d480c `verdict-reprise-gui10-qwen38-18-09` (+0,59 % tenu ; Qwen3.8 b=12 408,6 / 0,978 J / nu 434 : −0,8 / +0,8 / −1,5 %, < 3 %) ; Manon 7486247 `verdict-glm-etendue-historiques-controle-18-09` (5/5 tenu).

## 1. Restes de la pause : 3/3 soldés, rien à rouvrir

| reste | verdict | état |
|---|---|---|
| `j_par_jeton_10s` ± 10 % | 0,3119 vs 0,3101 (+0,59 %), deux lecteurs égaux | tenu ; `fenetre_s`/`jetons_fenetre` ajoutés (e812c5a, 9/9) |
| canal massif GLM 5/5 | ≤ 3 canaux > 100× médiane sur chaque tenseur (46.42.down : 2 canaux ~13 500×) | observation fermée, `k48-calibA` propre |
| versions dans `regime_ligne()` | torch 2.14.0+cu130 · triton 3.8.0 · fla 0.5.2 déjà portées | fait |

## 2. Hybrides / dense : la question posée en pause est tranchée, et le profil dense existe déjà

`profil.json` Qwen3.8 b=12 (Laure, défauts du jour) : `_dense_etroit_kernel` **19,98 ms / 26,6 (75,1 %)**, 496 lancements, ~20 Go → **1,0 To/s** ; `colle_etat` 2,97 ms (11,2 %, 1 658 lancements = 34/couche) ; fla 1,52 (5,7 %) ; attention 0,4. Plancher 11 ms. RPW/XREG ne s'y appliquent pas (< 3 %, tenu) : c'est un autre noyau, un seul, à 58 % de la bande — ce que Triton donne est donné (`sage-gemm-dense-clos-17-09`). Décision : **rien sur Qwen3.8** (non classé 1,0253 ; la colle vaut −2 ms = +7 % au mieux, pas ordonnée). La seule voie chiffrée est un noyau CUDA classe Marlin (~570 t/s = parité vLLM) : décision utilisateur, § 5.
Contrôle à sec (Océane, 10 min) : le régime du profil porte `kv_budget=16384/8` alors que le lot 12 est attesté par le compte (21 444 = 12 × 1 787 pas). Si « /8 » est le nombre de séquences planifiées, la ligne dit « planifié 8 » sur une mesure servie à 12 : soit le champ est mal nommé, soit le budget est encore celui de 8 et la certification ne tronque pas parce que 16 384 ≥ 12 × 1 365. Réponse attendue : ce que « /8 » désigne, fichier:ligne, et le budget réellement planifié pour ce run.

## 3. Où en est l'objectif — Coder, classés seulement

| cellule | acvram (18/09) | meilleur classé | tout moteur |
|---|---|---|---|
| b=12 t/s / J | **1 198 / 0,334** (nu 1 312) | EXL3 855,7 / 0,362 | Marlin 2 031 / 0,197 ; TRT-LLM 2 105 / 0,177 |
| b=1 t/s / J | 287,1 / 1,185 | **llama.cpp 341,4 / 1,108** | idem |
| prefill j/s | 9 913 | **llama.cpp 15 717** | Marlin 20 988 ; TRT-LLM 55 419 |

Depuis le 18/09, b=12 est gagné sur le t/s **et** sur le J parmi les classés (0,334 < 0,362) — la revendication Coder se réécrit (Jérôme, note datée, source `verdict-coder-b12-defaut-18-09`) : « le plus rapide et le plus économe des classés à b=12 ; llama.cpp reste devant à b=1 (+19 %) et au prefill (+59 %) ». Il reste donc deux cellules classées perdues, contre un seul moteur. Le prefill d'abord : l'écart est le plus grand, et GLM (4 661) porte le même poste.

## 4. Un profil avant tout noyau (Laure, un bloc de 40 min, torch.profiler, `regime_ligne()` en tête, nu et sous profileur)

**(a) prefill Coder 2048 au défaut B0** — le dernier profil (`verdict-profil-coder-2`, 193 ms GPU : `_gemm_groupe` 79,9 + `nvfp4_dequant` 50,8 + int8 dequant 11,0 + cutlass denses 19,2 + flash 11,5 + colle MoE 8,2 + elementwise 6,6) est **antérieur à B0** ; on ne chiffre pas un noyau sur un profil périmé. Prédictions : experts (GEMM groupée + `nvfp4_dequant`) ≥ 55 % du GPU ; « préparation par appel » (`nvfp4_dequant` + int8 dequant) ≥ 25 % ; hors-experts ≥ 60 ms. Issue qui me gênerait : `nvfp4_dequant` < 30 ms (alors B0 l'a absorbée et le poste est la GEMM elle-même, à 43 % de crête : forme des tuiles, pas déquant).
Décision écrite avant : (i) `nvfp4_dequant` ≥ 40 ms → **P1**, GEMM groupée W4A16 **CUDA** classe Marlin (poids E2M1 décodés en registres par table/`prmt`, cp.async ≥ 3 étages, MMA bf16 ; squelette `nvfp4_gemm_grouped_mma`, `model.py:979`) ; porte micro-banc à sec (E = 128, T = 16 384, K = 2 048, N = 768) ≥ **110 TFLOPS effectifs** poids lus dans la tuile (B0 + déquant = 57 ; B1' Triton = 58 ; < 110 → P1 fermé sans carte) ; scellé en situ : prefill Coder ≥ **15 700 j/s** (parité llama.cpp, dérivée du pas : 207 − 50,8 − 25 ≈ 131 ms), PPL prefill = B0 ± 0,002 (mêmes codes W4, A bf16 ; test ± 2⁻⁷ dans le commit, bras cassant : échelle de bloc décalée d'un rang). Coût 3-5 j Laurine, CUDA → même décision utilisateur que le dense, § 5. (ii) int8 dequant + colle MoE + elementwise ≥ 25 ms → **P0** à sec (Laurine ≤ 1 j, aucun tensor core neuf) : GEMM int8 sans déquantification par appel sur q/k/v/o (activations A8 par jeton, régime `ACVRAM_PREFILL_INT8=a8` dans `regime_ligne()`, porte PPL privé ≤ 1,020 ; prédiction ≤ 1,017 — W4A8 par jeton a coûté +0,13 % en fake-quant, `verdict-a4-fakequant-llama2-7b`) + colle MoE fusionnée (radix-sort 384 appels) ; scellé ≥ **11 000 j/s** (< 11 000 faux). (i) et (ii) ne s'excluent pas ; (ii) part le premier parce qu'il est à sec.

**(b) b=1 Coder** — pas pur 3,05-3,14 ms (318-328 t/s) contre rondes 3,48 (287) : **0,35-0,43 ms par jeton hors GPU, 11 %** ; llama.cpp fait 2,93 en rondes. Mesure : événements hôte par pas sous rondes b=1 (échantillonnage `.tolist()`/copie DtoH, planificateur, détokenisation, HTTP). Prédiction : ≥ 60 % des 0,4 ms dans un poste nommé (échantillonnage + copie). Décision : un poste ≥ 60 % → chantier **M** (Océane, à sec : échantillonnage sur la carte et recouvrement pas n+1 / rejeu n, `prediction-4-3ms-hote-decodage-14-09` reprise) scellé rondes b=1 ≥ **318 t/s** (= pas pur × 0,98 ; < 318 faux) ; dispersé (< 40 % partout) → on note, b=1 reste derrière et on l'écrit. Même tenu, 318-328 < 341 : la parité b=1 demande aussi ~0,2 ms de GPU (attention 0,85, colle 1,17 ms) — étape suivante, pas ordonnée.

## 5. Décision qui appartient à l'utilisateur (Jérôme la porte, une ligne, avant toute passe de carte de P1)

« Une semaine de Laurine en CUDA (noyau GEMM W4A16 classe Marlin, déquantification en registres) est la seule voie chiffrée sur deux cellules : Coder prefill 9 913 → ≥ 15 700 j/s (parité llama.cpp, classée, PPL inchangée) et Qwen3.8 b=12 412 → ~570 t/s (parité vLLM, non classé). Sans elle, le prefill reste −37 % et les denses −34 %. Oui / non ? » Le profil § 4 (a) se fait dans les deux cas : il coûte 40 min et fixe le chiffre.

## Décision utilisateur (18/09) : points 2-6 lancés ; le point 1 (semaine CUDA) reste en suspens — aucune passe de carte de P1 avant son oui ; P0, M, profil et note Marlin ne l'attendent pas.

## Ordre

* Jérôme : ETAT ≤ 40 lignes (§ 1 soldés ; revendication b=12 t/s + J gagnés, note datée dans le comparatif ; ligne § 5 à l'utilisateur ; file de carte = § 4) ; comparatif Coder : ligne acvram 1 198 / 0,334 (source `verdict-coder-b12-defaut-18-09`).
* Laure (carte, 40 min, worktree figé) : § 4 (a) puis (b) ; verdict `verdict-profil-prefill-b1-18-09`, décisions (i)/(ii)/M lues telles qu'écrites.
* Laurine (à sec, ≤ 2 h, en attendant) : note ≤ 20 lignes sur la classe Marlin appliquée à notre pile (`csrc/quantization/marlin` de vLLM : étages, décodage E2M1→bf16 en registres, tuile à M = 128) — faisable en 3-5 j oui/non, avec le chiffre de porte 110 TFLOPS confirmé ou corrigé **avant** la mesure.
* Océane (à sec, 10 min) : § 2 `kv_budget=16384/8`, fichier:ligne ; puis M si (b) le désigne.
* Manon : garde n°2 sur le parc (en cours), rien de neuf. Katy : cellule Coder b=12 1 198 / 0,334 aux menus.

## Suite (18/09, Océane 0f9417d `verdict-kv-budget-8-a-sec-18-09`) : `/8` = `plan.kv_planned_seqs` (`tiering.py:465`, défaut `PlannerOptions` 8) ; le script de profil (`scratchpad/profil-pas-coder-17-09.py:73`) charge sans `Plan` et sert b=12 (`:75`) — pas de troncature ce run (330 jetons/séq ≪ 2 048 planifiés). Pas de M par ici. Mais la faille de REGLES § 6 vient de se reproduire dans un instrument neuf, deux jours après la règle : un contrôle qu'on doit penser à faire (REGLES § 3). Ordre à sec, Océane, 30 min, un commit : `Engine` **refuse** de servir `slots > plan.kv_planned_seqs` sauf `ACVRAM_KV_PLAN_OVERRIDE=1` porté par `regime_ligne()` ; test qui casse en repassant l'ancien chargement (`load_model(max_model_len=…)` seul avec slots=12 → erreur attendue) ; le script de profil passe par `auto_plan(…, max_concurrent_seqs=b)`. Les chiffres de Laure du jour ne sont pas touchés (aucune troncature), le prochain instrument ne pourra plus l'être.
