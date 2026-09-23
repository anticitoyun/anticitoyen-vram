# Protocole — GEMV groupée des experts v2 (Laurine c904806 : paires triées par expert, poids relus une fois pour ≤ 4 jetons du même expert ; `ACVRAM_MOE_GEMV=v1|v2`)

instrument porte : `pytest tests/test_gemv_experts_v2.py` (compilation réelle) puis `outils/carte.sh python outils/banc-gemv-experts-18-09.py` (synthèse Coder : E=128, top-8, b=12, K=2048, I=768, 48 couches, 20 routages, gate/up + down, rejeu sous graphe) ; arbre laure = main 8a75b48 ; `scratchpad/gemv-experts-v2-18-09/`
scellé porte (Sage) : v2 ≤ 5,3 ms/pas ET identique au bit à v1 → ouvre ; > 5,3 ou un bit d'écart → faux. Prédiction Laurine : v1 6,5-7 ms (1,2 To/s), v2 4,6-5,3 ms (1,55-1,8 To/s).
scellé in situ (si la porte ouvre) : `ACVRAM_MOE_GEMV=v2`, Coder b=12 nu ≥ 1 300 t/s tenu, < 1 200 faux (montage certifie-b12 ×2 comme gemm-dense-17-09, bridé + nu par pas tracés) ; `ppl-decode-kv` (Coder 256+1024, témoin 5,544) identique au bit à v1.
