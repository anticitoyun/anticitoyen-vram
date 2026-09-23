# Verdict — 1aj variante D au décodage : −4 % à b=1, +10 % à b=12

Laure, 15/09/2026, 17:06-17:17, une prise de carte (1aj-laure, 623 s, après
397 s derrière la conversion GLM de Manon). Ordre : Jérôme ; seuils Sage
[`sage-point0-ordre-14-09.md`](sage-point0-ordre-14-09.md) § 4 ; PPL de D
tenue ([`verdict-1aj-variantesDE-15-09.md`](verdict-1aj-variantesDE-15-09.md)).
Protocole scellé : [`protocole-1aj-decodage-15-09.md`](protocole-1aj-decodage-15-09.md)
(laure ce93dc9). Moteur : main 8d24a83 (v0.6.3, MMA au godet ≥ 12), défaut.

## En-tête de mesure (REGLES §3)

    instrument    energie.py, compteur NVML TotalEnergyConsumption
    cartes        [0] seule ; fenêtre 24-31 s (rondes ctx 2048, invite 256),
                  repos 30 s ; -pl 400, horloge libre
    température   35-40 °C avant, 44-55 °C pendant (en-tête des JSON)
    bras (manifestes lus)
      témoin      models_acvram/Qwen3-Coder-30B-A3B-nvfp4 : q/k/v/o int8 ×48,
                  lm_head int8, experts nvfp4, snr_floor 25 (manifeste sans
                  champ `source` — parc, pas une reconversion datée)
      D           models_acvram_hdd/Qwen3-Coder-30B-A3B-srcbf16-1ajvarD :
                  q/k/v/o nvfp4 ×48, lm_head int8, experts nvfp4, source
                  bf16 Qwen3-Coder-30B-A3B-Instruct, snr_floor 25
    carte         vide au début et à la fin ; aucune alarme « processus »
    données       scratchpad/1aj-decodage-15-09/certifie-b{1,12}-20s-{A1,B1,A2,B2}.json

## Résultats (ABAB, écart entre passes ≤ 0,002 ms)

    b    témoin ms  témoin J  W    | D ms     D J      W    | ms D/A   J D/A   t/s
    1    4,484      1,495     333  | 4,297    1,397    325  | −4,2 %   −6,6 %  223,0 → 232,7
    12   16,249     0,560     377  | 17,924   0,620    378  | +10,3 %  +10,7 % 673,6 → 610,7

## Contre les seuils

**Sage, b=1** : ms ≤ 3,6 (réfuté ≥ 4,0) → **RÉFUTÉ** (4,30) ; J ≤ 1,10
(réfuté ≥ 1,25) → **RÉFUTÉ** (1,40). En relatif, D gagne 4 % de pas et
7 % de J — loin des −19 % d'octets (0,43 Go/jeton sur 2,30) que 1aj devait
convertir en temps. **Ma prédiction** (ms −13/−17 %, J −10/−15 %) :
réfutée. Alarme d'avance déclenchée : mon témoin est 4,48 ms, pas 4,18 —
relatif seulement, et le relatif dit −4 %.

**b=12** : D est **plus lent de 10 %** et coûte 11 % de J de plus. Ma
prédiction (−3/−6 %) réfutée dans l'autre sens.

## Cause, lue dans le code

Les projections d'attention passent par `kernels.matmul` (`layers.py:485`).
En NVFP4, `nvfp4_matmul` prend le noyau **`nvfp4_gemv`** tant que
M ≤ `_NVFP4_GEMV_MAX` = 32 (`kernels/__init__.py:497`, `:522-533`) — un
noyau de GEMV, une ligne à la fois, qui relit W pour chaque ligne du lot ;
en int8, `int8_matmul` a son `int8_gemv` **par tranches** (seuil 80,
`:622`, « tranche de 8 », `:636`) qui lit W une fois pour plusieurs
lignes. À M=1 les deux lisent W une fois : D gagne ses octets (−0,43 Go)
mais seulement −4 %, parce que le GEMV NVFP4 de ces formes (q 2048×4096,
k/v 2048×512, o 4096×2048) n'atteint pas le débit du GEMV int8 sur des
matrices étroites — le −19 % d'octets ne se traduit pas en −19 % de temps
(`notre-gemv-nvfp4-contre-cublas` : le pas entier, pas le noyau). À M=12
le GEMV NVFP4 relit W douze fois : les projections coûtent plus qu'en
int8 par tranches, +1,7 ms/pas.

Ce que je ne conclus pas : rien sur `nvfp4_mm_tensorcore` / `w4a8`
(`:555-560`) pour ces formes à M=12 — non mesuré, c'est la voie à essayer
si 1aj doit tenir à b=12 (abaisser `ACVRAM_NVFP4_GEMV_MAX` sous 12 est un
essai d'une cellule).

## Trois états

    b=1   RÉFUTÉ (Sage 3,6 / 1,10) — relatif −4 % ms, −7 % J ; point 0 non repris (1,40 vs 1,17-1,39)
    b=12  écart rendu : +10 % ms, +11 % J — D perd
    cause noyau (nvfp4_gemv M ≤ 32, relit W par ligne), pas octets
