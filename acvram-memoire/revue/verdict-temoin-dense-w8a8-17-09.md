# Verdict — témoin dense `Qwen2.5-Coder-14B-pur-nvfp4` (337 projections NVFP4 non groupées) : `w8a8` perd 1,4 % de PPL ET 36 % de prefill contre `bf16`

instrument : `ppl-acvram-17-09.py` (privé Coder, 3 tranches, cibles 1024..2047, géo + médiane, `regime_ligne()` en en-tête) ; `scratchpad/prefill-dense-acvram-17-09.py` (pp2048, generate(max_tokens=1), 2 chauffes, 7 rép., médian ; le banc MoE `prefill-glm-acvram-15-09.py` refuse un dense : « piles_ok non vérifié ») — journaux `scratchpad/table-glm-bf16-17-09/{dense-*,prefill-dense-*}`
commit : arbre poste3 fc04517 (= main, défaut `ACVRAM_PREFILL=bf16`, 0971c90) ; régime lu : `regime_ligne()` → `ACVRAM_PREFILL` = `w8a8` / `bf16` selon le bras, tout le reste au défaut, MOE sans objet (dense), une carte
scellé (poste7, amendement) : PPL `bf16` meilleure de 2-5 % ; j/s `bf16` −25 à −45 % ; chantier « GEMM W4A16 à déquantification fusionnée » si > 25 % de j/s perdus ET ≥ 1 % de PPL gagné
mesuré : **PPL w8a8 / bf16 = 1,0143** (géo ; médiane 1,0116 ; tranches 13,021/12,827 · 11,825/11,663 · 12,229/12,060) · **prefill pp2048 : bf16 5 732 j/s (σ 15), w8a8 3 681 j/s (σ 1)** → bf16 **+56 %** (w8a8 = 0,64 × bf16)
verdict : **le mécanisme est confirmé sur le témoin où il est massif : le W8A8 tacite coûte 1,4 % de PPL (borne basse de la prédiction 2-5 %)… et il est aussi 36 % plus LENT que le chemin bf16 à L = 2048 — la prédiction de vitesse (−25 à −45 % pour bf16) est réfutée dans l'autre sens. Le scellé du chantier GEMM fusionnée n'est pas atteint (aucun j/s perdu) : le nouveau défaut `bf16` gagne sur les deux colonnes ; le chantier est sans objet tant qu'un cas où w8a8 est plus rapide n'est pas montré (L court ? à mesurer si quelqu'un le revendique).**

## Compléments du même passage
- GLM `-k48-calibA` (4 projections concernées seulement, poste4) : prefill bf16 4 464 j/s vs w8a8 4 453 (égal, comme prédit) ; PPL sous le défaut bf16 : privé **1,0143** (a8 : 1,0150), public 1,0281 (a8 : 1,0284) — neutre, ≤ 0,001, aucune conclusion sur le W8A8 (mécanisme quasi absent de ce converti) ; c'est la cellule de la table dans son régime réel.
- Ligne « mêmes poids » (GadflyII) : a8 1,028 / **bf16 1,010** / Marlin W4A16 1,016 / vLLM W4A4 1,072 — acquise (`verdict-prefill-bf16-17-09`), régimes nommés.

## Table GLM — PPL × bf16 (géo, 3 tranches, préfixe), régime prefill nommé
    bras / converti (calibration)                       régime          privé    public   classé (privé ≤ 1,02)
    acvram -k48-calibA (anglais, 2 048 j.)              W4A16 bf16      1,0143   1,0281   oui
    acvram -k48 (défaut 6 phrases)                      W4A16 a8*       1,0156   1,0040   oui  (*4 proj. concernées : ≈ bf16)
    acvram -vllm-direct (poids GadflyII)                W4A16 bf16      1,0096   —        oui
    acvram -vllm-direct                                 W4A16 a8        1,0280   1,0213   non
    vLLM GadflyII                                       Marlin W4A16    1,0164   1,0133   oui
    vLLM GadflyII                                       W4A4 (défaut)   1,0717   1,0751   non
    llama.cpp Q4_K_M (unsloth, imatrix)                 —               1,0248   1,0397   non
    acvram -hadamard512 / -k48-calibB / -sansawq        W4A16 a8*       1,029 / 1,024 / 1,017   non / non / oui
    Temps b=12 (t/s, J/j) : acvram -k48 514 / 0,774 · vLLM W4A16 858 / 0,397 · vLLM W4A4 796 / 0,445 · llama.cpp 702 / 0,495 · prefill pp2048 : acvram 4 46x · vLLM W4A16 18 117 · W4A4 26 732 · llama.cpp 11 293.

## Bornes
- Le témoin dense est un Qwen2.5 : la tranche privée est retokenisée par son tokenizer (mêmes textes, ids [2, 2823, 329, …] identiques à Qwen3 sur le début) ; la comparaison w8a8/bf16 est interne au converti, sans étalon bf16 HF du 14B (non nécessaire pour le scellé).
- Vitesse dense : σ 15 et 1 j/s sur 7 répétitions ; l'écart 56 % est vingt fois hors bruit. Régime : GEMM bf16 après déquant contre `_scaled_mm` FP8 avec requantification des poids par ligne à chaque appel (poste4 429902c) — le coût de requantification à chaque prefill est la cause probable de la lenteur du w8a8.
