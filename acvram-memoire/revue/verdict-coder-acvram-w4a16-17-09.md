# Verdict — cellule Coder-30B × acvram W4A16 : PPL 1,0144 privé (classé), b=12 730 t/s / 0,546 J, prefill 8 633 j/s

instrument : `ppl-acvram-17-09.py` (privé + public, 3 tranches, cibles 1024..2047, géo + médiane, `regime_ligne()` dans chaque JSON) ; `certifie-b12-15-09.py` (rondes, 2 passes par lot) ; `prefill-glm-acvram-15-09.py w4a16 2048` (7 rép., médian, compteurs de chemin) — journaux `scratchpad/coder-acvram-w4a16-17-09/`
commit : arbre laure fd1e204 (PPL) et cecfb18 (rondes, prefill) = main ≥ 0971c90 (défaut `ACVRAM_PREFILL=bf16`) ; converti `Qwen3-Coder-30B-A3B-nvfp4` (aucune calibration, awq=False ; « non concerné » par le W8A8 tacite selon l'outil de Laurine)
régime (`regime_ligne()`) : `ACVRAM_PREFILL=bf16`, `ACVRAM_MOE_MMA=0`, `ACVRAM_MOE_DECODE_MMA=0` (MIN_T=5), `ACVRAM_NARROW_GEMM=1`, SLOTS=b, graphes, NOMINAL 0 exilé ; prefill : `fpg=336, pile=1008, gemm=0, mma=0` (déquant + grouped_mm, comme prévu à L=2048) ; une carte, plafond 400 W ; sans préfixe (Qwen)
scellé (Sage) : PPL privé 1,016-1,020 (classé) · b=12 780-850 t/s · J 0,50-0,53
mesuré : **PPL privé 1,0144** (géo ; méd 1,0168 ; sous la fourchette de peu, classé), **public 1,0099** (tranches 1,0113 / 1,0082 / 1,0103) · **b=1 : 4,283 / 4,267 ms, 233,5 / 234,4 t/s, 1,364 / 1,365 J** · **b=12 : 14,981 / 14,992 ms, 730,7 / 730,1 t/s, 0,546 / 0,547 J, 399 W** (bridage puissance) · **prefill pp2048 : 8 633 j/s** (σ 44, 237 ms) ; 0 fenêtre explosée, ids [2, 2823, 329, 220]
verdict : **cellule remplie. PPL : classé (1,014 privé, 1,010 public), un cran sous la prédiction ; vitesse : b=12 réfuté par le bas (730 t/s contre 780-850 ; 0,546 J contre 0,50-0,53) — le W4A16 (GEMV) coûte 14 % de t/s par rapport au W4A4 du 14/09 (630 t/s → non : 14/09 était 630,6 t/s en… régime différent, voir bornes) ; b=1 identique au 14/09 (233 t/s). Prefill 8 633 j/s : 2,2× le GLM (le MoE Coder à 128 experts déquantifie moins par jeton), 0,48× llama.cpp (15 717), 0,16× TRT-LLM (55 419).**

## Table Coder-30B (mise à jour de cette cellule)
    moteur / régime                         PPL privé  PPL public  b=1 t/s  b=1 J   b=12 t/s  b=12 J   prefill j/s
    acvram W4A16 (nvfp4, bf16 prefill)      1,0144     1,0099      233,9    1,365   730,4     0,546    8 633
    EXL3 4,25 bpw (exllamav3)               1,0006     1,0004      168,0    1,546   855,7     0,362    10 656   ← verdict-coder-exl3-vitesses
    llama.cpp Q4_K_M                        1,0103     1,0146      341,4    1,108   709,2     0,523    15 717
    vLLM ModelOpt communautaire (W4A4)      1,1554     1,123       196,7*   1,373*  1 437,9*  0,272*   —          (*14/09)
    TRT-LLM même ModelOpt (KV fp8)          1,2313     1,164       234,9    1,481   2 104,7   0,177    55 419

## Bornes
- Le 630,6 t/s du 14/09 (acvram b=12) était mesuré avant SLOTS=b et en régime MMA par défaut : pas comparable ; la cellule d'aujourd'hui est la première en W4A16 déclaré, SLOTS=12, graphes, préfill bf16.
- b=12 sous bridage puissance (399 W sur 400) : le t/s est celui de la carte bridée, choix de l'utilisateur.
- PPL public bf16 par tranche (pas de médiane publique) : géo seulement.
