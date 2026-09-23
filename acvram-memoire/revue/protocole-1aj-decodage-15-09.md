# Protocole — 1aj variante D au décodage (q,k,v,o NVFP4, lm_head int8) contre le témoin

Laure, 15/09/2026, avant mesure. Ordre : Jérôme, seuils de Sage
[`sage-point0-ordre-14-09.md`](sage-point0-ordre-14-09.md) § 4 ; PPL de D
tenue par Manon ([`verdict-1aj-variantesDE-15-09.md`](verdict-1aj-variantesDE-15-09.md),
ratio 0,987). Attend la carte après Manon (conversion GLM) et Laurine (3 min).

## Bras (vérifiés dans les manifestes, pas supposés)

    témoin  /mnt/2TO_2023_980PRO/Modeles/models_acvram/Qwen3-Coder-30B-A3B-nvfp4
            q/k/v/o int8 ×48, lm_head int8, experts nvfp4 ×18 432, embed bf16
            (snr_floor 25, awq off) — le modèle de toutes mes mesures du 14-15/09
    D       /mnt/4TO_SATACMR_2022/Modeles/models_acvram_hdd/Qwen3-Coder-30B-A3B-srcbf16-1ajvarD
            q/k/v/o nvfp4 ×48, lm_head int8, experts nvfp4, embed bf16
            (source bf16 Qwen3-Coder-30B-A3B-Instruct, snr_floor 25, awq off)

Réserve d'avance : le témoin est le dossier du parc, pas une reconversion
« même source bf16 » datée d'aujourd'hui ; si son manifeste ne porte pas
la même source, je le dis dans le verdict (le manifeste du parc n'a pas
de champ `source`, celui de D en a un).

## Montage

`certifie-b12-15-09.py` (modèle par `ACVRAM_MODELE_MESURE`), b=1 et b=12,
ABAB par lot (8 cellules), rondes ctx 2048, invite 256, ≥ 20 s, repos
30 s, compteur `energie.py`, une carte, -pl 400, horloge libre,
**défaut du moteur** (v0.6.3 attendu sur main : MMA au godet ≥ 12 —
version relevée au lancement et écrite au journal), température GPU dans
l'en-tête (avant/min/max).

## Seuils (Sage) et prédictions

Sage, b=1 : **ms ≤ 3,6** (contre 4,18 ; réfuté ≥ 4,0), **J ≤ 1,10 ×**
(réfuté ≥ 1,25) ; b=12 : rendre l'écart.

Mes prédictions : les 4 projections int8 → NVFP4 retirent ≈ 0,43 Go/jeton
sur 2,30 (Sage : attention int8 0,93 → ≈ 0,50) ; à b=1 le pas est borné par
les octets (1 017 Go/s) → **ms 4,48 → 3,7-3,9 (−13/−17 %)**, **J 1,51 →
1,28-1,36 (−10/−15 %)**. Réfuté (mon bord) si ms > 4,1 ou J > 1,42.
Alarme d'avance : mon témoin b=1 est 4,48 ms (rondes ctx 2048), pas 4,18
(montage de Manon) — le seuil absolu 3,6 est dans un autre montage ; je
rends l'absolu ET le relatif, et je ne conclus que sur le relatif si le
témoin de ma session s'écarte de 4,18 de plus de 5 %. b=12 : les
projections pèsent moins (MoE domine, plafond 400 W) → **ms −3/−6 %, J
−3/−6 %** ; réfuté si l'écart est nul (±1 %) — alors les octets d'attention
ne comptent pas à b=12.
