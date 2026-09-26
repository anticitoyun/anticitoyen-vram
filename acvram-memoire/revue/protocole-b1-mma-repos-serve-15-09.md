# Protocole — chiffre officiel b=1 sous MMA=1 et repos sous `acvram serve`

poste3, 15/09/2026, avant mesure. Ordre : poste7
[`poste7-reprise-15-09-b.md`](poste7-reprise-15-09-b.md) (main 02a33fa), transmis
par chef. Code : main 02a33fa (défaut `ACVRAM_MOE_DECODE_MMA=1`,
`model.py:1294`).

## 1. Chiffre officiel b=1 — ABAB × ≥ 20 s, MMA=0 / MMA=1

Même montage que c6377d5 (`certifie-b12-15-09.py`, argument B=1) : rondes
ctx 2048, invite 256, repos 30 s, compteur `energie.py`, une carte, -pl 400,
bras prouvé par relecture de `model._MOE_DECODE_MMA` dans le processus.

Seuils de poste7 : **ms −4 à −9 %, J −6 à −12 %** ; réfuté si J(B) ≥ 1,40 ou
ms(B) > 1,02 × ms(A) → MMA=1 conditionné au lot b ≥ 4 (poste4).

Ma prédiction : à b=1, `cnt` ≤ 1 par expert, la tuile BT=16 travaille pour
1 jeton : la MMA n'a rien à grouper. Je prédis **ms −1 à +2 %, J −2 à +2 %**
— pas de gain, pas de perte nette ; réfuté (dans le sens de poste7) si ms
≤ −4 % ET J ≤ −6 %. Alarme d'avance : A doit reproduire 233 t/s / 1,42-1,45 J
(14/09) à ±2 % ; sinon le régime a changé et je le dis avant de comparer.

## 2. Repos sous `acvram serve`

Serveur chargé (`--max-batch 4 --max-model-len 2048 --speculative none`,
port 8099, jamais 8081-8083), une requête de 64 jetons, puis aucune ;
échantillons toutes les 5 s pendant 90 s (W, P-state, MHz), puis compteur
sur 30 s (t = +90 à +120 s). Tué par port (PID lu sur `ss -tlnp`).

Seuil de poste7 : **≤ 25 W** à +60 s ; réfuté si ≥ 60 W (attente active).

Ma prédiction, lue dans le code : la boucle de service dort par
`time.sleep(0.002)` quand `engine.idle` (`server/app.py:102-104`) — 500
réveils/s CPU, aucune commande CUDA ; `idle` ne lit que deux listes Python
(`runner.py:1472`). Je prédis **P8, 16-20 W à +60 s**, comme le processus
sans serveur (089ec1f). Réfuté si ≥ 60 W ou P0/P1 tenu au-delà de +30 s —
alors quelque chose d'autre que la boucle touche la carte (capteurs ?
`server/app.py:325` appelle `nvidia-smi`, qui ne réveille pas le P-state).
