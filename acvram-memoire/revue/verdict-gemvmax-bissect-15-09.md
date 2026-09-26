# Verdict — marche 1 (GEMV_MAX=8) RÉFUTÉE ; régression b=1 = 7f3f422 (routeur fp32)

poste3, 15/09/2026, 17:28-17:51, deux prises de carte (bissect-poste3 17:28-17:36,
bissect2-poste3 17:37-17:51). Ordre : poste7
[`poste7-1aj-decodage-15-09.md`](poste7-1aj-decodage-15-09.md). Protocole scellé :
[`protocole-gemvmax-bissect-15-09.md`](protocole-gemvmax-bissect-15-09.md)
(poste3 e99d5f6).

## En-tête de mesure (REGLES §3)

    instrument    energie.py, compteur NVML ; cartes [0] ; fenêtres 24-31 s
                  (rondes ctx 2048, invite 256), repos 30 s ; -pl 400,
                  horloge libre ; température 35-40 °C avant, 45-54 °C pendant
    arbres        un worktree par commit (travail/bissect/<sha>), chemin
                  d'import écrit au JSON (`preuve.acvram`) ; jamais l'arbre main
    incident      première prise : les cellules des commits d'avant v0.6.1
                  plantaient sur ma preuve (`model._MOE_DECODE_MMA` absent) ;
                  arrêtée à 17:36, preuve passée en `getattr`, relancée en file
                  — la marche 1 (faite dans la première prise) n'a pas été
                  rejouée
    carte         vide à chaque début/fin ; aucune alarme « processus »
    données       scratchpad/gemvmax-bissect-15-09/{m1-b12-*,bis-b1-*}.json

## (1) Marche 1 — D à b=12, `ACVRAM_NVFP4_GEMV_MAX` 32 → 8 : RÉFUTÉ

    GEMV_MAX   pas ms (A1/A2 · B1/B2)   t/s     J/jeton      W
    32 (déf.)  17,928 / 17,929          610,5   0,615/0,620  376-378
    8          19,609 / 19,607          558,2   0,684/0,687  382-384
    B/A        +9,4 %                   −8,6 %  +11 %

Seuil poste7 (≤ 16,4 ms, réfuté > 16,9) : **RÉFUTÉ** ; ma fourchette
16,3-16,8 : réfutée. À M=12 le chemin tensorcore/w4a8
(`kernels/__init__.py:555-560`) est plus lent que `nvfp4_gemv`, lui-même
plus lent que `int8_gemv` par tranches (16,25 au témoin). Aucun chemin
NVFP4 existant ne sert les projections étroites à M=12 : la marche 2 de
poste4 (GEMV NVFP4 par tranches) est la seule voie, il n'y a pas de
réglage.

## (2) Bissection b=1 (MMA=0, témoin int8) : 7f3f422 explique tout

    arbre                 pas ms (A1/A2 ou B1/B2)   t/s
    c652947  good         4,297 / 4,297             232,7
    0c90017  bad (main)   4,484 / 4,484             223,0      +0,187 ms
    97f2313  parent       4,297 / 4,297             232,7
    7f3f422  suspect 1    4,484 / 4,483             223,0      +0,187 ms

Coupable si ≥ 0,13 ms : **7f3f422, +0,187 ms = 100 % de l'écart good/bad**.
Prédiction poste7 (≥ 0,2 ms, réfuté < 0,1) tenue à 0,013 près ; la mienne
(0,10-0,20) tenue. Les autres suspects (v0.6.1/2/3, 6-pré, carte.sh)
n'ont rien à expliquer — non mesurés, et rien ne l'impose.

Lu dans le diff (`git show 7f3f422 -- acvram/engine/model.py`) : le
routeur passe de « poids dans le dtype de l'entrée » à
`F.linear(x.to(torch.float32), w_fp32)` puis sortie fp32, à chaque
couche MoE, pour TOUS les modèles — motivé par GLM-4.7-Flash (4/16 jetons
d'équivalence basculaient sur un arrondi bf16 de la sortie du routeur,
`prediction-routeur-fp32-14-09.md`). Coût mesuré : 0,187 ms/pas à b=1 =
3,9 µs × 48 couches — un cast de x (1×2048) + un GEMM fp32 128×2048 + le
retour, soit 2-3 lancements supplémentaires par couche à M=1 : c'est du
lancement, pas du calcul (poste7 : 0,2-0,4 prédit sur le même
raisonnement). À b=12 le même surcoût est noyé (16,25 ms de pas).

**Correctif (poste7, à poste1)** : fp32 seulement là où l'équivalence GLM
l'exige (sigmoid + biais de correction, ex-aequo du top-k), bf16 pour les
routeurs softmax comme Coder-30B ; ou fp32 gardé mais fusionné dans un
seul lancement. Chiffre officiel b=1 MMA=0 à retrouver après correctif :
**4,297 ms / 232,7 t/s** (good, reproduit aujourd'hui à 45-50 °C — le
« 223 vs 233 » n'a jamais été thermique).

## Trois états

    marche 1 (GEMV_MAX=8)   RÉFUTÉ   +9,4 % au lieu de −8 %
    bissection b=1          TROUVÉ   7f3f422, +0,187 ms, 100 % de la régression
    autres suspects         SANS OBJET (rien à expliquer)
