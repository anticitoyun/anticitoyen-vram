# Verdict — cause du 1 % confirmée sur carte : `ACVRAM_PREFILL=bf16` rend 1,0096 × bf16 (noyaux actifs) — sous Marlin

instrument : `ppl-acvram-17-09.py` (privé, 3 tranches, préfixe `encode_brut`, cibles 1024..2047, géo) sur `GLM-4.7-Flash-vllm-direct` W4A16 (`ACVRAM_MOE_MMA=0`), noyaux CUDA actifs, une seule variable changée : `ACVRAM_PREFILL=bf16` (défaut `a8`) — journaux `scratchpad/geste-b-17-09/glm-prive-tr*-prefill-bf16.*`
commit : arbre poste3 38e3416 (= main a83cfee, diff de poste4 429902c inclus)
régime : identique aux prises de référence (`passage-direct-17-09`, 1,0280) et tout-torch (`bissection-suite`, 1,0104) ; cause nommée par poste4 à sec : expert partagé + couche dense 0 en W8A8 caché (`nvfp4_mm_w4a8`, poids requantifiés E4M3 par ligne + activations FP8, `_scaled_mm`), erreur RMS 3,6-4,0 % contre 0,14 % en bf16
scellé (poste4/poste7) : ≤ 1,014 → seul coupable · 1,014-1,024 → une part · ≥ 1,024 → piste réfutée
mesuré : **1,0096 × bf16** (tranches 1,0103 / 1,0104 / 1,0083 ; 16,603 / 13,079 / 10,611), soit **0,982 × les noyaux par défaut** (1,0280) ; tout-torch 1,0104 ; vLLM Marlin W4A16 1,0164 ; 0 fenêtre explosée, ids [154822, 154824, …]
verdict : **seul coupable, confirmé sur carte : avec `ACVRAM_PREFILL=bf16` et tous les noyaux CUDA actifs, la PPL tombe à 1,0096 — égale au chemin tout-torch (1,0104, au bruit) et 0,7 % sous vLLM Marlin W4A16 sur les mêmes poids. Le W8A8 caché de l'expert partagé et de la couche dense 0 coûtait 1,8 % de PPL à lui seul ; les experts routés, la déquant, le routage, la tête n'y étaient pour rien (bissection d'hier).**

## Ce que ça change
- Table GLM, mêmes poids GadflyII : acvram W4A16 (prefill bf16) **1,010** · vLLM Marlin W4A16 1,016 · acvram W4A16 (défaut a8) 1,028 · vLLM W4A4 1,072 — la ligne « cause non localisée » se ferme.
- Tous les convertis maison (`-k48`, A, B, Hadamard, sansawq) ont été jugés sous `ACVRAM_PREFILL=a8` : leurs PPL prefill portent le même 1,8 % ; à refaire sous le nouveau défaut dès que poste4 le pose (ou en `ACVRAM_PREFILL=bf16` explicite) — 10 mesures × 1 min, une passe.
- Le décodage n'est pas concerné a priori (chemin GEMV / graphes, pas `nvfp4_mm_w4a8`) : à vérifier par une PPL décodage sous le même drapeau (12/12, 3 tranches, 10 min) avant de l'écrire.
- Vitesse : `a8` existait pour le prefill ; le coût en j/s du passage bf16 sur l'expert partagé + dense 0 est à mesurer (pp2048, 7 rép.) avant de changer le défaut — un prefill plus lent est un régime, pas un correctif.
