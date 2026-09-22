# nsys + familles-noyaux, graphe b=12 (leviers 1+2 posés) — TENU sur Σ — 22/09 (poste2)

* instrument : `nsys profile -t cuda --cuda-graph-trace=node`, nsys DANS la prise `carte.sh` (pas l'inverse — 1er essai bloqué 9 min en post-traitement par le reparentage `setsid` de `carte.sh` autour de nsys, tué ; 2e essai sans `--cuda-graph-trace=node` a donné une granularité fausse — 3770 lancements/pas, mur 140 ms, `attention`=0 — écarté, non publié) ; `outils/gpu/mesure/familles-noyaux.py` sur `nsys stats --report cuda_gpu_trace`
* commit : main 92657459+, `ACVRAM_RAPATRIEMENT_EPINGLE=1` posé (leviers 1+2, sampler=graphe par défaut)
* régime : `NOMINAL, graphes=on(hybrides≤12), repli_eager=0`, Coder nvfp4, b=12, 60 pas demandés (105 pas jugés dans la fenêtre nsys)
* scellé (poste1) : Σ familles 6,6-6,9 ms attendu ; 700-760 lancements/pas (prédiction corrigée) ; alarme si Σ > 7,0 (horloge pas à 2 700) ou lancements très hors fourchette
* mesuré : **Σ = 6,749 ms/pas** (mur 7,083, trou 0,334) — **654 lancements/pas** (14 % sous la fourchette 700-760, pas une alarme franche) :

| famille | ms/pas | % | lancements |
|---|---|---|---|
| experts_marlin | 3,937 | 58,3 | 96 |
| proj_etroites_int8 | 1,335 | 19,8 | 145 |
| autres (attention probable, cf. suite) | 0,459 | 6,8 | 48 |
| tête | 0,132 | 2,0 | 48 |
| routage | 0,310 | 4,6 | 48 |
| normes | 0,241 | 3,6 | 97 |
| rope_kv | 0,234 | 3,5 | 96 |
| glue_torch | 0,044 | 0,7 | 17 |
| experts_glue | 0,052 | 0,8 | 48 |
| copies | 0,005 | 0,1 | 11 |
| attention (classée) | 0,000 | 0,0 | 0 |

* verdict : **TENU sur Σ** (6,749 dans 6,6-6,9), pas d'alarme horloge. Deux écarts nommés, pas diagnostiqués (hors mon domaine, à poste1, `familles-noyaux.py`) : (1) `attention` = 0 lancement mais `autres` (0,459 ms) tombe exactement dans la fourchette prédite pour l'attention (0,39-0,45) — classification probablement fausse, le noyau d'attention tombe dans « autres » ; (2) `glue_torch` = 0,044 ms / 17 lancements, très en dessous de toute prédiction (0,90-1,10 ou ≤ 10 ms selon la version) — signe encourageant si réel : les leviers 1+2 auraient déjà éliminé l'essentiel de la glue torch par pas, cohérent avec le gain ABBA mesuré (+5 % en moyenne) au-delà de la seule frontière `trou_gpu`.
* durée : 3e essai ~1 min de carte (les 2 précédents : 1er tué après 9 min de blocage, 2e ~2-3 min gaspillées sur une granularité inexploitable)

## Suite
Faire nommer par poste1 la classification `autres`/`attention` et confirmer si `glue_torch` à 0,044 ms est réel (les leviers auraient déjà consommé la marge du levier 3, à revoir avant de le coder). PPL relative A/B 31B ensuite.
