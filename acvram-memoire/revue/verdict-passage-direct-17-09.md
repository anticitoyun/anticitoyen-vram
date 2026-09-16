# Verdict — passage direct NVFP4 (scellé 2) : sur les MÊMES poids que vLLM, acvram W4A16 fait 4 à 5 % de PPL de MOINS que vLLM

instrument : `ppl-acvram-17-09.py` (préfixe `encode_brut`, cibles 1024..2047, géo + médiane, 4 ids) sur `GLM-4.7-Flash-vllm-direct` (v2, 8164622) en W4A16 (`ACVRAM_MOE_MMA=0`) et W4A4 (`=1`) ; `ppl-vllm-17-09.py` sur la source `GadflyII/GLM-4.7-Flash-NVFP4`, **KV bf16** (auto), mêmes ids ; bf16 HF de `ppl-refonte-17-09` — journaux `scratchpad/passage-direct-17-09/`
commit : conversion v2 sur f508159 (= main a987c30, 8164622 inclus), PPL sur 1b53568 (= main) ; protocole `protocole-passage-direct-17-09.md`
régime : converti v2 = experts 9 024 nvfp4 et MLP dense 144 nvfp4 copiés tels quels, attention 384 / gate 47 / lm_head / embed bf16 (la liste `ignore` de vLLM honorée) ; corpus privé 5909d27 et public wiki-gptq, 3 tranches × 12 fenêtres ; v1 provisoire (attention et lm_head nvfp4 par le plan, avant 8164622) gardé sous `-vllm-direct-v1-attention-nvfp4`
scellé : S2 |acvram W4A16 / vLLM − 1| ≤ 0,004 par corpus · S2' W4A4 − vLLM ≤ +0,015 et W4A4 ≠ W4A16 de > 0,004 · témoin : ids identiques dans les trois bras
mesuré (moyenne géométrique des 3 tranches) :
    corpus   vLLM (KV bf16) ×bf16   acvram W4A16 ×bf16   acvram W4A4 ×bf16   W4A16/vLLM          W4A4/vLLM   W4A4/W4A16
    privé    1,0717                 **1,0280**           1,0398              **0,9592** (méd 0,951)   0,9702      1,0115
    public   1,0751                 **1,0213**           1,0299              **0,9499** (méd 0,955)   0,9579      1,0085
    par tranche W4A16/vLLM : privé 0,972 / 0,956 / 0,949 ; public 0,950 / 0,932 / 0,968 — toutes sous 1. 0 fenêtre explosée ; ids [154822, 154824, …] dans les 18 JSON. vLLM KV fp8 (refonte) : 1,0734 / 1,0740 — le KV ne compte pas en prefill.
verdict : **S2 réfuté, dans le sens inattendu : à poids strictement identiques, acvram W4A16 rend 1,028 / 1,021 × bf16 là où vLLM rend 1,072 / 1,075 — vLLM perd 4 à 5 % de PPL de plus que nous sur son propre checkpoint. L'écart n'est ni dans le format ni dans les poids : il est dans le noyau de vLLM (NVFP4 sur sm_120 : activations quantifiées en FP4 par échelle globale, W4A4 de fait), alors que notre W4A16 garde l'activation en bf16. Notre W4A4 (+1,15 % / +0,85 % sur notre W4A16) reste 3 à 4 % sous vLLM. S2' tenu (W4A4 − vLLM = −0,03 ; W4A4 ≠ W4A16). La colonne PPL du duel/comparatif GLM devient comparable : la qualité de vLLM (1,043 sur 4 fenêtres, 1,07 ici) n'était pas le checkpoint communautaire, c'était vLLM.**

## Ce que ça déplace
- Comparatif GLM : la ligne vLLM GadflyII (1,074 privé) se lit désormais « même poids que notre `-vllm-direct` à 1,028 » ; `-vllm-direct` W4A16 est **classé** au juge géo (≤ 1,02 : non, 1,028 — non classé de peu ; public 1,021 idem) — mais meilleur que tous nos convertis maison sur le public (k48 1,004 ? non : le k48 est à 1,004 public, 1,016 privé). Table : k48 1,016/1,004 · A 1,015/1,028 · direct 1,028/1,021 · vLLM 1,072/1,075.
- Le v1 provisoire (attention nvfp4 par le plan) : 1,334 / 1,353 × bf16 sur le privé — la quantification NVFP4 des projections MLA (q_a, q_b, kv_a, o) coûte +30 % à elle seule : ne jamais quantifier l'attention MLA en nvfp4 (le plan le faisait par défaut sur une source déjà NVFP4 ; 8164622 le corrige pour le passage direct, la recette maison garde int8/bf16).

## Bornes
- La lecture « vLLM = W4A4 » est déduite (vLLM compressed-tensors NVFP4 sur Blackwell quantifie l'activation en FP4 avec `input_global_scale`) : pas vérifiée dans le code de vLLM ce soir ; ce qui est mesuré, c'est l'écart. Un vLLM forcé en « W4A16 » (s'il en a un chemin sur sm_120) trancherait.
- Les deux moteurs lisent les mêmes fichiers ; la seule différence de poids restante est nulle par construction (8164622, formats du manifeste ci-dessus). Bruit ± 0,004 par tranche : l'écart 0,04-0,05 est dix fois au-dessus.
