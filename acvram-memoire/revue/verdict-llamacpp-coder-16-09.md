# Verdict — llama.cpp officiel sur Coder-30B Q4_K_M : décodage, prefill, PPL (classé 1,015×)

instrument : `scratchpad/banc-llamacpp-16-09.py` (decode : llama-server + flux SSE, fenêtre ≥ 20 s + 7 passes courtes, `energie.py` ; prefill : `llama-bench -p 2048 -n 0 -r 7`, `puissance_nvml`) ; `scratchpad/ppl-llamacpp-16-09.py` (`llama-perplexity -c 2048`, cadrage min-ctx 1023) ; étalons bf16 HF (`coder-ppl-bf16-hf-16-09.py`) et acvram (`reppl-eval-16-09.py`) au même cadrage — journaux `scratchpad/llamacpp-coder-16-09/`
commit : arbre laure 537ea38 (bancs) ; binaire llama.cpp 4c9233c (CUDA 13.4, sm_120a, 15/09) ; protocole `protocole-llamacpp-coder-16-09.md` (4bb25e5)
régime : `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` (4,5 bpw), `-ngl 999 -np b -c 2304×b --no-cache-idle-slots --no-jinja --reasoning-format none`, `-b/-ub` par défaut, FA auto, KV f16 ; lots de 1 024 jetons/séquence, invite 256, `ignore_eos` absent sur Coder (aucun lot n'a fini avant 1 024 : 7 175 jetons / 7 lots à b=1) ; une carte, plafond 400 W
scellé : D1 b=12 t/s [675 ; 825], J [0,40 ; 0,49] · D2 b=1 t/s [290 ; 350], J [1,05 ; 1,30] · P1 prefill [2 500 ; 6 000] j/s · Q1 PPL ≤ 1,02× bf16 (70 %) · I1 retokenisation 2 049/2 049
mesuré : D1 t/s **709,2** tenue, J **0,5225** brut **réfutée** (au-dessus) · D2 **341,4 t/s**, **1,108 J** tenues · P1 **15 717 j/s** médian (130,3 ms ; meilleure 16 836) **réfutée** — prédiction 3× trop basse · Q1 **tenue** : 3 tranches 1,0168 / 1,0180 / 1,0091, moyenne géométrique **1,0146×**, max 1,018 · I1 tenue (4 fenêtres et 3 tranches retokenisent à l'identique)
verdict : **llama.cpp Q4_K_M est classé (1,015× bf16 sur 36 864 jetons, aucune tranche > 1,02) ; à b=12 il fait 709 t/s / 0,52 J — un tiers de TRT-LLM (2 105 t/s), la moitié de vLLM 14/09 (1 438), ×1,12 acvram 14/09 (631) ; à b=1 il mène (341 t/s, 1,11 J) ; prefill 2048 à 15,7 k j/s, 3,5× moins que TRT-LLM (55,4 k).**

## Tableau Coder-30B (mêmes dénominateurs ; PPL au cadrage min-ctx 1023, 4 fenêtres = 4 096 notés ; tranches = 3 × 12 288)
    moteur / converti           b=1 t/s  b=1 J   b=12 t/s  b=12 J   prefill 2048   PPL 4 fen. (×bf16)   PPL 3 tranches (×bf16)
    llama.cpp Q4_K_M (4c9233c)   341,4   1,108     709,2    0,5225   15 717 j/s     8,8092 (1,0166)      1,0146 [1,009-1,018]
    acvram nvfp4 (14/09 temps)   233,0   1,430     630,6    0,619    —              8,8101 (1,0167)      1,0180 [1,017-1,019]
    vLLM ModelOpt (14/09 temps)  196,7   1,373   1 437,9    0,272    —              9,7283 (1,123)       — (hors classe)
    TRT-LLM ModelOpt             234,9   1,481   2 104,7    0,177    55 419 j/s     10,0907 (1,164)      — (hors classe)
    bf16 HF (étalon)               —       —         —        —      —              8,6656               7,9915 (moy. géo.)
    Sur 4 fenêtres à min-ctx 256 (7 164 notés) : bf16 9,1747, acvram 9,2833 (1,012), vLLM 10,262 (1,119), TRT-LLM 10,617 (1,157) ; llama.cpp non mesurable à ce cadrage (voir bornes).

## Bornes et pièges (tous consignés dans les scripts)
- Cadrage PPL : `llama-perplexity` v1 note les cibles 1024..2047 de chaque chunk de 2048 ; son mode `--ppl-stride` (v2) exige ≥ 2×n_ctx jetons par fichier, la reconstruction fenêtre par fenêtre est impossible → tous les moteurs remesurés à `PPL_MIN_CTX=1023` (4 096 notés sur 4 fenêtres : bruit plus grand qu'à 7 164, d'où les 3 tranches). Les tranches sont des fichiers texte décodés du tokenizer HF, retokenisés à l'identique (24 577 = 24 577, trois fois) ; pas de BOS (Qwen).
- acvram 1,018× ici contre 1,012× à min-ctx 256 : les deux cadrages ne notent pas les mêmes positions ; on ne compare que dans une colonne.
- llama-server 4c9233c : `/completion` passe la sortie par le parseur de chat même en `--no-jinja` et rend 500 « Content-only format » sur certaines sorties (invites de jetons tirés, sortie finissant vraisemblablement sur un octet UTF-8 incomplet — non prouvé) : le banc lit le flux SSE et compte les jetons reçus (`erreurs_finales_parseur` = 0 sur Coder). `-b/-ub 2048` avec 12 slots : `CUDA error: an illegal memory access` dans `set_rows_cuda` → valeurs par défaut.
- b=12 : `bridage pendant la fenêtre : puissance` (370 W moyens) ; carte bridée par choix de l'utilisateur, publié tel quel. Fenêtre b=12 = 30 s (2 lots de 15 s) ; la meilleure passe courte (872 t/s, lots de 128) est plus rapide que la fenêtre (709, lots de 1 024 : contexte qui s'allonge).
- Le 14/09 (binaire de Katy, lots de 2 000) donnait 747,8 t/s / 0,4436 J : −5 % / +18 % ici — lots plus courts (prefill 20 % du lot) et binaire différent, pas départagés.
- Énergie prefill : fenêtre `llama-bench` incluant le chargement → watts sans valeur ; seul le temps compte.

## Reste
GLM-4.7-Flash Q4_K_M (unsloth, 18,31 Go, téléchargé) : décodage rejoué avec `ignore_eos` (GLM finissait ses lots sur EOS : 108 lots de ~180 jetons à b=1), PPL 1023 + 3 tranches contre bf16 GLM et acvram `-k48` — en cours, verdict séparé.
