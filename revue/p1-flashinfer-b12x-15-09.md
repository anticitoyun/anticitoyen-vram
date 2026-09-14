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

## Mesures (15/09, 13h17-13h30, carte exclusive)

En-tête (REGLES §3) : 5090 seule (`CUDA_VISIBLE_DEVICES=0` posé par carte.sh,
`[]` hors verrou), plafond 400 W (relevé avant/après), horloge libre (3 135
max), 35 °C au départ, `nvidia-smi --query-compute-apps` vide avant et
après, venv séparé `/opt/ia/flashinfer/.venv` (torch 2.14+cu130, flashinfer
0.6.18.post1, cutlass-dsl 4.7.1), backend `static` (choisi par
`select_sm120_moe_backend`), NVFP4 (une ligne MXFP4 témoin), fenêtres
20 s + 2 s de chauffe, repos 8 s au compteur avant chaque ligne (72-82 W,
carte chaude), médiane des rejeux de graphe.

**Le `--cold-l2` de FlashInfer est inopérant ici** (chaud = froid = 31,7 µs
au bit) : sa rotation copie les ARGUMENTS de `fn` (tenseurs passés), or
`run()` les capture par fermeture, et elle compte les octets alloués
(604 Mo) et non les 74 Mo touchés. Le froid ci-dessous = 4 (Coder) ou 2
(GLM) pools d'experts DISJOINTS dans un même graphe (297 / 282 Mo > 96 Mo
de L2), temps de rejeu ÷ appels — ce qu'un pas fait sur 48 couches.

| forme, jetons | experts distincts | L2 | µs/couche | Go/s effectif (poids) | W brut | MHz |
|---|---:|---|---:|---:|---:|---:|
| Coder, 1 | 8 (21 Mo) | chaud | 22,5 | 940 | 243 | 3 000 |
| Coder, 1 | 8 ×4 pools | froid | **26,8** | 790 | 253 | 2 977 |
| Coder, 12, pool 30 | 28 (74 Mo) | chaud | 31,7 | (2 340 : L2) | 398 | 2 872 |
| Coder, 12, pool 30 | 28 ×4 pools | **froid** | **62,7** | **1 180** | **401** | 2 910 |
| Coder, 16, pool 30 | 30 ×4 pools | froid | 81,1 | 980 | 399 | 2 887 |
| Coder, 12, routage d'origine | 96 (255 Mo) | (> L2) | 174,7 | 1 460 | 400 | 2 512 |
| Coder, 12, pool 30, **MXFP4** | 28 ×4 pools | froid | 60,5 | 1 220 | 400 | 2 872 |
| GLM, 1 | 4 (19 Mo) | chaud | 22,5 | 840 | 244 | 2 985 |
| GLM, 12, pool 34 | 23 (108 Mo) | chaud (> L2 ?) | 89,0 | 1 210 | 400 | 2 580 |
| GLM, 12, pool 30 | 25 ×2 pools (118 Mo) | froid | **112,2** | **1 050** | 400 | 2 752 |
| GLM, 16, pool 30 | 26 ×2 pools | froid | 125,4 | 970 | 399 | 2 767 |

(E=64 ne permet pas 2 pools disjoints de 34 : 30 → 25 distincts ; à 34
experts, 160 Mo, extrapolation à 1,05 To/s ≈ 150 µs.)

## Contre les seuils scellés

| | poste7 | moi | mesuré | verdict |
|---|---|---|---|---|
| Coder 12 jetons, ~30 experts | 80-100 µs (réfuté < 70 : « ma borne d'octets est fausse ») | 85-110 froid, 40-60 chaud | **62,7 froid, 31,7 chaud** | **réfuté par le bas, les deux** : b12x tient 1,18 To/s sur 74 Mo (1,46 sur 255) — la borne de 1 050 Go/s est celle de NOS noyaux, pas de la carte (REGLES §9, déjà annoté) |
| Coder 1 jeton | 30-40 | 30-45 | 26,8 froid / 22,5 chaud | sous le seuil : plancher de lancement ~20 µs, pas 30 |
| GLM 12 jetons, ~34 experts | 170-210 | 165-220 | 112 à 25 experts → ≈ 150 extrapolé à 34 | sous les deux (≈ −12 %) |
| W en boucle | — | ≤ 345 (0,18 inst/oct) | **400, au plafond** (2 900 MHz) | réfuté : même à 0,18 inst/octet, 1,2 To/s de DRAM + tensor cores saturent 400 W ; ce qui distingue b12x de nos GEMV n'est pas la puissance mais l'horloge tenue (2 900 contre 1 600-1 800) et le temps |

## Ce que ça ordonne

Notre chemin B (MMA décodage) : 3 GEMM ≈ 75 µs + glue (2 quant_act,
moe_act, reduce, argsort/gather) ≈ 105-110 µs par couche à 12 jetons ;
b12x fusionné à L2 froid : **62,7 µs**, même octets, même W. Écart ≈ 45
µs/couche = **2,1 ms par pas de 13,75 (−15 %)** et, à W égal, −15 % de
J/jeton. Deux voies : (a) brancher b12x (venv séparé : cutlass-dsl 4.7.1 +
torch 2.14 — notre venv est en torch 2.13 ; une passerelle .so ou une
mise à niveau, décision poste7) ; (b) fusionner nous-mêmes gate·up + down
+ activation en un noyau (la glue est 30 µs, les 3 lancements ~10 µs) —
prédiction : −25 à −35 µs/couche, soit la moitié de l'écart, sans la
dépendance. À b=1 le fusionné ne rend que 26,8 µs contre nos ≈ 4,18 ms /
48 = 87 µs par couche tout compris (GEMV 21 Mo) — l'écart à b=1 est
ailleurs (projections int8, lm_head : 1aj).
