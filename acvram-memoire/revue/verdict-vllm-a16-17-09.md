# Verdict — vLLM forcé en Marlin W4A16 (`-a16`, poste2) : 1,016 / 1,013 × bf16 — sous notre W4A16 sur les mêmes poids (1,028 / 1,021)

instrument : `ppl-vllm-17-09.py` (préfixe `encode_brut`, cibles 1024..2047, géo + médiane, 4 ids, KV auto) sur `GLM-4.7-Flash-NVFP4-a16` ; références : bf16 HF (`ppl-refonte`), vLLM W4A4 KV bf16 et acvram `-vllm-direct` W4A16 (`passage-direct-17-09`) — journaux `scratchpad/vllm-a16-17-09/`
commit : arbre poste3 d67596a (= main) ; config `-a16` de poste2 (d574e72 : `input_activations: null`, mêmes `ignore`)
régime : vLLM choisit `MarlinNvFp4LinearKernel for NVFP4 GEMM` (journal) = W4A16 ; **shards réécrits sans les 9 170 tenseurs `*.input_global_scale`** (le schéma W4A16 ne les déclare pas → `KeyError glm4_moe_lite.py:512` au chargement ; poids inchangés, 28 043 tenseurs, index refait ; l'original `GLM-4.7-Flash-NVFP4` intact) ; corpus privé 5909d27 et public, 3 tranches
scellé (poste7) : 1,028 ± 0,006 sur le privé (= notre W4A16 sur les mêmes poids) → si tenu, ModelOpt propre sans objet et vitesses vLLM à remesurer dans ce régime
mesuré : **privé 1,0164** (géo ; méd 1,0136 ; tranches 1,0166 / 1,0168 / 1,0159), **public 1,0133** (méd 1,0093) ; vLLM W4A4 : 1,0717 / 1,0751 ; acvram W4A16 mêmes poids : 1,0280 / 1,0213 ; **a16 / acvram = 0,9888 / 0,9922** ; 0 fenêtre explosée, ids [154822, 154824, …]
verdict : **scellé réfuté par le bas : en W4A16, vLLM (Marlin) rend 1,016 / 1,013 là où notre W4A16 rend 1,028 / 1,021 sur les mêmes fichiers — notre noyau W4A16 perd 1,1 % (privé) / 0,8 % (public) de PPL de plus que Marlin à poids strictement identiques ; l'écart de 5,5 % mesuré hier contre vLLM était son W4A4 (1,072). Deux résultats : (1) vLLM à sa meilleure qualité sur ce checkpoint est 1,016, pas 1,072 — sa colonne PPL du comparatif change de régime ; (2) il reste ≈ 1 % à trouver dans notre chemin W4A16 (déquantification, accumulation, échelle globale — poste4), 2 à 3 fois le bruit, sur les deux corpus, toutes tranches.**

## Table GLM, mêmes poids GadflyII (× bf16, géo)
    moteur / régime            privé    public
    vLLM W4A4 (défaut)         1,0717   1,0751
    vLLM Marlin W4A16 (-a16)   1,0164   1,0133
    acvram W4A16 (direct v2)   1,0280   1,0213
    acvram W4A4 (direct v2)    1,0398   1,0299

## Bornes
- Le retrait des `input_global_scale` ne touche pas la valeur des poids : ce sont des échelles d'activation, inutilisées en W4A16 ; leur présence faisait échouer le chargeur, pas le calcul.
- Vitesses vLLM b=1 / b=12 dans ce régime : non mesurées (le scellé conditionnel est réfuté) — à faire sur ordre : le comparatif de temps de vLLM change aussi de régime si sa colonne qualité est celle-ci.
