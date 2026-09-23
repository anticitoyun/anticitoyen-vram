# Verdict — PPL EXL3 Coder-30B (converti Manon 4,25 bpw) : 1,0006 × bf16 privé, 1,0004 public — le meilleur bras de la table, de loin

instrument : `scratchpad/ppl-exl3-17-09.py` (exllamav3 du dépôt `/tmp/exllamav3-repo` dans le venv TabbyAPI, forward `flash_attn_nc`, logits fp32 ; tokenizer HF de la source pour les mêmes ids ; cadrage `ppl_commun_17_09` : fenêtres 2048, cibles 1024..2047, géo + médiane) — journaux `scratchpad/ppl-exl3-17-09/`
commit : arbre laure 8645fee (= main c34aed5) ; converti `models_exl3/Qwen3-Coder-30B-A3B-Instruct-EXL3-4.25bpw` (Manon, exl3 1.4.8, `bits 4.25, head_bits 6, calibration rows 250 × cols 2048`, 16 Gio)
régime : sans préfixe (Qwen), ids [2, 2823, 329, 220] ; corpus privé 5909d27 (bf16 HF par fenêtre de `ppl-refonte`) et public wiki-gptq (bf16 HF par tranche de `llamacpp-coder-16-09`) ; 3 tranches × 12 fenêtres chacun ; bpw mesuré par le modèle : 4,283 (couches) / 6,008 (tête)
scellé : ≤ 1,02 × bf16 (comparatif) ; ma prédiction avant lecture : [1,005 ; 1,015] (entre llama.cpp 1,010 et « mieux »)
mesuré : **privé 1,0006** (géo ; méd 1,0032 ; tranches 0,9987 / 1,0011 / 1,0021), **public 1,0004** (géo ; tranches 1,0003 / 1,0000 / 1,0010) ; 0 fenêtre explosée ; ma prédiction réfutée par le bas
verdict : **EXL3 4,25 bpw est classé, et quasi sans perte : +0,06 % privé, +0,04 % public — contre llama.cpp Q4_K_M 1,010 / 1,015, acvram nvfp4 1,027 / 1,018, vLLM/TRT-LLM ModelOpt 1,16-1,23. À bits égaux (4,25-4,5), la quantification par treillis calibrée sur 512 k jetons (250 × 2 048) laisse les trois autres formats loin ; l'écart privé/public est nul (0,0002).**

## Table Coder-30B, PPL × bf16 (géo, 3 tranches), classement ≤ 1,02 sur le privé
    bras                               bpw        privé     public    classé
    EXL3 4,25 (Manon)                  4,28/6,0   1,0006    1,0004    oui
    llama.cpp Q4_K_M (unsloth)         4,5        1,0103    1,0146    oui
    acvram nvfp4 (aucune calibration)  4,5 nom.   1,0270    1,0180    non
    vLLM ModelOpt communautaire        4,25       1,1554    1,123     non
    TRT-LLM même ModelOpt (KV fp8)     4,25       1,2313    1,164     non
    (public acvram/vLLM/TRT-LLM sur 4 fenêtres du 16/09, les autres sur 3 tranches)

## Bornes
- Calibration EXL3 : 250 lignes × 2 048 du jeu interne d'exllamav3 (contenu non publié par nous ici ; à écrire avec son sha256 si Sage l'exige — Manon) ; le privé est disjoint par construction, et il est à 1,0006 : le résultat ne dépend pas d'un recoupement.
- bf16 public par tranche = PPL sur les 12 288 cibles de la tranche (moyenne géométrique des fenêtres) : comparable au `ppl_geo` d'EXL3 ; pas de médiane publique (pas de bf16 par fenêtre sur le public Coder).
- Temps TabbyAPI : non mesuré (hors de ce verdict ; le serveur TabbyAPI reste à qualifier pour le comparatif de temps).
