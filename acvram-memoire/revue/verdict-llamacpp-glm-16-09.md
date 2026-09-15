# Verdict — llama.cpp officiel sur GLM-4.7-Flash Q4_K_M : non classé (1,043× bf16), b=12 702 t/s

instrument : `scratchpad/banc-llamacpp-16-09.py` (decode flux SSE + `ignore_eos`, fenêtre ≥ 20 s, 7 passes courtes ; prefill `llama-bench -p 2048 -n 0 -r 7`) ; `ppl-llamacpp-16-09.py` (cadrage min-ctx 1023) ; étalons `glm-ppl-bf16-hf-16-09.py` (HF bf16, script de poste2 paramétré), acvram `reppl-eval-16-09.py` (`-k48`, `ACVRAM_MOE_MMA=0` = régime du duel), vLLM `ppl-vllm-glm-16-09.py` (GadflyII, KV fp8) — journaux `scratchpad/llamacpp-glm-16-09/`
commit : arbre poste3 537ea38 ; binaire llama.cpp 4c9233c ; protocole : celui de Coder (`protocole-llamacpp-coder-16-09.md`), sans prédiction chiffrée propre à GLM (pas de référence llama.cpp GLM avant ce soir)
régime : `unsloth/GLM-4.7-Flash-GGUF` `GLM-4.7-Flash-Q4_K_M.gguf` (18,31 Go, imatrix), `-ngl 999 -np b -c 2304×b --no-cache-idle-slots --no-jinja --reasoning-format none`, `ignore_eos` (sans lui : 108 lots de ~180 jetons à b=1, `decode-sans-ignore-eos.log`) ; une carte, plafond 400 W
scellé : Q1 PPL ≤ 1,02× bf16 (même critère que Coder) ; I1 retokenisation identique des tranches ; pas de scellé de temps
mesuré : Q1 **réfutée** : 4 fenêtres 8,3498 / bf16 8,0305 = 1,040 ; 3 tranches 1,0369 / 1,0512 / 1,0404, moyenne géométrique **1,0428×**, max 1,051 · I1 tenue (24 577 = 24 577 ×3) · temps : b=1 **241,4 t/s / 1,408 J**, b=12 **701,6 t/s / 0,4947 J** (bridage puissance, 2 erreurs finales du parseur sur 24 594 jetons comptés), prefill 2048 **11 293 j/s** médian (181,3 ms, meilleure 11 900)
verdict : **llama.cpp Q4_K_M sur GLM n'est pas classé (1,043× bf16 sur 36 864 jetons, toutes les tranches > 1,02) ; en temps il fait 702 t/s à b=12 (×1,37 acvram 514, 0,88× vLLM 796) et 241 t/s à b=1 (×2,25 acvram 107, ×1,57 vLLM 154), prefill 11,3 k j/s (×2,6 acvram 4 422, 0,42× vLLM 26 732).** Seul acvram `-k48` est classé sur GLM (0,9956×) — et sous bf16 sur deux tranches (0,9856 sur la tranche 1), ce qui appelle une question, pas un trophée (voir bornes).

## Tableau GLM-4.7-Flash (temps acvram/vLLM : duel prise B 52c019c ; PPL au cadrage min-ctx 1023)
    moteur / converti              b=1 t/s  b=1 J   b=12 t/s  b=12 J   prefill 2048   PPL 4 fen. (×bf16)   PPL 3 tranches (×bf16)
    llama.cpp Q4_K_M (unsloth)      241,4   1,408     701,6    0,4947   11 293 j/s     8,3498 (1,040)       1,0428 [1,037-1,051]
    acvram -k48 (W4A16, prise B)    107,3   2,263     514,0    0,774     4 422 j/s     7,9856 (0,994)       0,9956 [0,986-1,001]
    vLLM GadflyII NVFP4 (KV fp8)    153,7   1,97      796,3    0,445    26 732 j/s     8,3725 (1,043)       — (non mesuré : hors classe à 4 fenêtres, 1,043 > 1,02)
    TRT-LLM                          refuse (verdict-trtllm-glm-16-09)
    bf16 HF (étalon)                  —       —          —       —          —            8,0305               7,7414 (moy. géo.)

## Bornes
- acvram sous bf16 (0,9856 sur une tranche de 12 288 jetons, bruit ±0,004) : un converti NVFP4/AWQ ne devrait pas battre sa source ; l'explication la plus simple est que la calibration AWQ a vu `wiki-gptq.txt` (nom du corpus : « gptq » = corpus de calibration). Le classement « ≤ 1,02× » d'acvram sur GLM et sur Coder (1,018) n'est donc pas comparable à celui de llama.cpp (Q4_K_M imatrix unsloth, calibré ailleurs) tant que le corpus de calibration d'acvram n'est pas écrit ; à poste7/poste2. Un corpus d'évaluation disjoint de toute calibration trancherait (poste1 : croisement PPL/octets).
- Temps acvram/vLLM repris du duel prise B (lots de 1 024, même invite 256, mêmes scripts de dénominateur) : comparables au jeton près ; llama.cpp b=12 sur 35 s (2 lots de 17,5 s).
- `erreurs_finales_parseur: 2` à b=12 : deux séquences dont le message final a été un 500 ; leurs jetons ont été comptés au fil du flux (24 594 ≈ 24 × 1 024).
- Sans `ignore_eos`, GLM (modèle qui pense) finissait ses lots sur EOS : 192,9 t/s à b=1 était un chiffre de prefill déguisé (108 × 256 jetons d'invite dans la fenêtre) — jeté, journal gardé.
- Prefill : énergie sans valeur (fenêtre llama-bench avec chargement).

## Reste
Colonne llama.cpp close (Coder classé, GLM non). TabbyAPI (EXL3) : convertis à vérifier / conversion de poste2 ; vLLM remesuré une fois sur le ModelOpt sain de poste2 (Coder) — ordre de poste7.
