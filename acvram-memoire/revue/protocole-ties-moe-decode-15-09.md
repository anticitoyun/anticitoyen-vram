# Protocole — ties du chemin MoE-MMA au décodage et chiffre certifié b=12

Laure, 15/09/2026, avant mesure. Ordre : Sage
[`sage-laure-15-09.md`](sage-laure-15-09.md) (main b516eb4), transmis par
Jérôme. Code mesuré : main du jour (b516eb4 ≥ 92beffe), chemin
`_forward_grouped_mma` (`model.py:1174`), constantes `model.py:1290-1291`.

## Seuils de Sage (repris tels quels)

1. Ties : à la première divergence de chaque séquence, écart top-1/top-2 du
   bras B ≤ 0,30 ; ≤ 4/12 séquences divergentes avant 64 jetons. Réfuté si
   un seul écart > 0,30 ou > 4/12.
2. Témoin `ACVRAM_MOE_DECODE_MMA_BT=8` : doit rendre « réfuté » (ou
   planter avant de diverger = le contrôle ne garde rien, à dire avant tout
   chiffre).
3. Certifié ABAB × 20 s, b=12, moyen : A 15,9 ± 0,5 ms, B 13,75 ± 0,5 ms /
   0,43 ± 0,03 J ; réfuté si gain B < 8 % ms ou < 12 % J.

## Montage

    ties       12 invites × 128 jetons (`invite(k)` d'Océane), 256 jetons
               greedy chacune, EOS neutralisé, b=12 tenu de bout en bout
               (lot_min = lot_max = 12 vérifié), graphes, ctx 1024,
               ACVRAM_REPIN=0 ; un PROCESSUS par bras (constante de module
               lue à l'import) ; bras prouvé par relecture de
               `model._MOE_DECODE_MMA`/`_BT` dans le processus, écrit au JSON
    capture    à chaque `_emit`, par ligne : jeton, top-2 (indices, valeurs
               f32) ; comparaison hors ligne (`compare-ties-15-09.py`)
    certifié   `certifie-b12-15-09.py`, rondes ≥ 20 s, ctx 2048, invite 256,
               repos 30 s, compteur `energie.py`, une carte, -pl 400,
               en-tête §3 ; A1 B1 A2 B2 = 4 processus
    verrou     un seul carte.sh (nouveau : carte exposée dans le verrou seul)

## Mes prédictions ajoutées (scellées)

* Ties : **conforme** — 1 à 3 séquences divergent avant 64 jetons, toutes
  avec écart B ≤ 0,15 (≤ 1,2 ulp) ; le témoin bt8 : **réfuté** par écart
  franc (> 1) dès les premiers jetons (tuile plus petite que `cnt` = jetons
  perdus, pas un tie) OU plantage ; si le témoin rend « conforme », mon
  contrôle est aveugle et je le dis avant tout chiffre.
* Certifié : A 18,5-19,5 ms (le 14/09 en rondes ctx 2048 : 630,7 t/s =
  19,0 ms/pas, PAS 15,9 — la fenêtre de Laurine était un autre montage ;
  alarme publiée : si A sort de 15,9 ± 0,5 je ne compare que B/A relatif),
  B −12 à −16 % ms, −16 à −22 % J. Réfuté si gain < 8 % ms ou < 12 % J.
