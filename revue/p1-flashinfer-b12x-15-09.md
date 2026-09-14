# P1 — étalon FlashInfer b12x (SM120 W4A4 fused MoE) sur nos formes (15/09, poste4)

Ordre de poste7 (`poste7-lancement-14-09.md` § 3) : le banc `bench_b12x_mxfp4_moe.py`
de flashinfer 0.6.18.post1 (`/opt/ia/flashinfer`, venv séparé : torch
2.14+cu130, cutlass-dsl 4.7.1) sur la forme Coder-30B (2048/768/128/top-8)
et GLM-4.7-Flash (2048/1536/64/top-4), 1/8/12/16 jetons, 20 s, sous
carte.sh, compteur NVML.

## Ce que le banc d'origine mesure — deux écarts corrigés dans la copie

`outils/banc_flashinfer_b12x.py` (copie Apache-2.0 avec ajouts) :

1. **Routage** : l'original route `(t·k + j) % E` → à 12 jetons top-8, 96
   experts DISTINCTS (255 Mo de poids par couche Coder) là où le modèle en
   touche ~30 (`bras experts`, 14/09 : 30,7 par couche, 80 Mo). Le seuil
   de poste7 (80-100 µs/couche à 12 jetons) suppose ~30 experts ; sous le
   routage d'origine il est inatteignable (255 Mo à 1 To/s = 255 µs).
   `--experts-pool 30` (Coder) / `--experts-pool 34` (GLM, calcul de poste7)
   : chaque jeton tire top-k experts distincts dans les N premiers.
2. **L2** : `cold_l2_cache=False` en dur → 80 Mo par couche tiennent dans
   les 96 Mo de L2, la boucle mesure une relecture L2 (P7). `--cold-l2`
   purge entre deux mesures ; les deux sont publiés, le froid est le
   chiffre du pas (48 couches différentes par pas).

Plus l'énergie : `energie.py` (compteur) sur la fenêtre `repeat-ms`, repos
10 s avant, W brut/net et horloge par ligne.

## Prédictions scellées (avant la carte)

| forme, jetons, routage | poste7 | moi (L2 froid) | moi (L2 chaud, banc d'origine) |
|---|---|---|---|
| Coder, 12, pool 30 (80 Mo) | 80-100 µs (réfuté > 110 ou < 70) | 85-110 µs | 40-60 µs (le L2 sert) |
| Coder, 12, 96 distincts (255 Mo) | — | 250-300 µs | 220-280 (ne tient pas en L2) |
| Coder, 1, 8 experts (21 Mo) | 30-40 µs | 30-45 µs (plancher de lancement) | 20-30 |
| GLM, 12, pool 34 (160 Mo) | 170-210 µs | 165-220 µs | 150-200 (ne tient pas) |
| W pendant la boucle | — | ≤ 345 W brut (pas de plafond : 0,18 inst/oct) | — |

Ce que ça achète : notre chemin MMA décodage (B) fait le MoE de Coder en
≈ 3,6 ms + glue par pas = **75 µs de GEMM par couche à 12 jetons** (3
lancements + 2 quant_act + moe_act + reduce ≈ 100-110 µs tout compris). Si
b12x fait 80-100 µs tout compris (une seule fusion), le levier de la
fusion vaut ≤ 10-30 µs/couche = 0,5-1,4 ms/pas ; si b12x est à 40-60 µs
à L2 froid, notre noyau a un défaut de 2× à chercher.
