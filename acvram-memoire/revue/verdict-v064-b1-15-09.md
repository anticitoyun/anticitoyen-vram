# Verdict — v0.6.4 (routeur fp32 conditionné) : le chiffre b=1 est retrouvé

poste3, 15/09/2026, 18:09-18:14, une prise de carte (v064-poste3). Ordre :
chef, seuil **4,30 ± 0,02 ms / 232,7 t/s** (le good c652947 de la
bissection, [`verdict-gemvmax-bissect-15-09.md`](verdict-gemvmax-bissect-15-09.md)).
Moteur : main f2273d7 (v0.6.4 = da17b01 d'poste1, fp32 seulement sigmoid + biais, + une note de revue), relevé au journal de l'unité.

## En-tête (REGLES §3)

    instrument   energie.py, compteur NVML ; cartes [0] ; fenêtre 24 s (rondes
                 ctx 2048, invite 256), repos 30 s ; -pl 400, horloge libre
                 (2 960 MHz) ; température 37-39 °C avant, 47-51 °C pendant
    bras         témoin int8-promu (models_acvram/Qwen3-Coder-30B-A3B-nvfp4),
                 ACVRAM_MOE_DECODE_MMA=0, arbre main, chemin d'import au JSON
    carte        vide au début et à la fin ; aucune alarme « processus »
    données      scratchpad/v064-b1-15-09/certifie-b1-A{1..4}.json

## Résultat (4 passes)

    passe   pas ms   t/s      J/jeton   W
    A1      4,296    232,76   1,4601    339,9
    A2      4,297    232,72   1,4607    339,9
    A3      4,297    232,75   1,4629    340,5
    A4      4,296    232,76   1,4656    341,1

**CONFORME** : 4,296-4,297 ms, écart au good c652947 (4,297) ≤ 0,001 ms ;
232,7-232,8 t/s. La régression de 7f3f422 (+0,187 ms) est effacée par
v0.6.4 pour le routeur softmax de Coder-30B. Chiffre officiel b=1 MMA=0 :
**4,297 ms / 232,7 t/s / 1,46 J** (une carte, ctx 2048, 47-51 °C).

Ce que je ne conclus pas : l'équivalence GLM (15/16) sous v0.6.4 — c'est le
test d'poste1, pas ce chiffre.
