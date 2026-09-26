# Pièce 255 — FP8 natif sm_120, micro-banc : VERDICT (poste5 26/09 05 h)

Scellé `poste5-piece255-scelle-26-09.md` (a860051dd + addendum f6b551a90, poussés avant mesure). Instrument
`scratchpad/poste5-p255-26-09/banc255.py` ; prise sur f6b551a90, 04:55:16-04:58:01 (2 min 45), -lgc 2700, cpu-safe 100 ;
carte 0 : aucun PID hors prise avant/après (llama-server 4219 sur l'autre carte). torch 2.14.0+cu130, capacité (12, 0).
Résultats bruts : `banc255.json`. µs par appel ; x aléatoire N(0 ; 0,5) en bf16 (pas des activations réelles — voir réserves).

## Noyau lancé (P6) : TENU
F lance `cutlass::device_kernel<… enable_3x_kernel_for_sm10_or_later …>` (CUTLASS 3.x, sm100+), jamais `sm89` ni cuBLASLt,
à n = 8, 78 et 624 sur les quatre formes. Autour : 6 noyaux élémentaires de la quantification de l'activation (abs, amax,
division, conversion e4m3, copie) — NON fusionnés.

## Mesuré (n = 8, 78, 624 ; T_seul = servi hors portée, T_part = portée partagée de la 243)

| forme | n | T_seul | T_part | F | F_mm | I | F/T de réf. |
|---|---|---|---|---|---|---|---|
| qkv 10240×5120 | 8 | 40,1 | — | 57,3 | 43,5 | — | 1,43 |
| | 78 | 645,4 (GEMV) | 174,2 | 62,5 | 44,5 | 69,3 | **0,36** |
| | 624 | 1 030,8 | 446,3 | 244,2 | 201,5 | 234,8 | 0,24 |
| gate 6144×5120 | 8 | 33,2 | — | 54,1 | 37,4 | — | 1,63 |
| | 78 | 421,1 | 102,4 | 56,7 | 37,6 | 63,6 | **0,55** |
| | 624 | 599,0 | 246,3 | 173,8 | 134,4 | 153,5 | 0,29 |
| out 5120×6144 | 8 | 26,1 | — | 57,9 | 43,0 | — | 2,22 |
| | 78 | 374,8 | 92,7 | 64,4 | 43,5 | 63,5 | **0,69** |
| | 624 | 608,3 | 256,6 | 206,7 | 161,0 | 169,5 | 0,34 |
| down 5120×17408 | 8 | 61,6 | — | 123,9 | 110,7 | — | 2,01 |
| | 78 | 1 020,9 | 281,0 | 133,6 | 111,4 | 103,8 | **0,48** |
| | 624 | 1 756,4 | 750,0 | 551,9 | 453,6 | 434,1 | 0,31 |

(Réf. : T_seul à n = 8 et 624, T_part à n = 78.) n = 1, 64, 128 dans le JSON, même lecture.

Justesse (erreur relative RMS contre x · W_fp8 exact, fp64), identique à 0,1 pt près sur toutes les formes et tous les n :
**T 0,94-1,13 %** ; **I 1,29-1,45 %** ; **F 2,26-2,74 %**. Écart des poids int8 servis contre le fp8 source : 0,94-1,08 %.

## Contre le scellé
| | prédit | mesuré | issue |
|---|---|---|---|
| P1 décodage n=8 | F/T 1,2-2,5 (plus lent) ; seuil F ≤ 0,90 T sur les 4 formes | 1,43 / 1,63 / 2,22 / 2,01 | tenu : **décodage fermé** |
| P1 T n=1 / n=8 qkv | 33-37 / 36-40 | 39,6 / 40,1 | n=1 FAUX de 7 % ; n=8 tenu |
| P2 n=78 T_part | 160-185 | 174,2 (qkv) | tenu |
| P2 n=78 T_seul | 700-760 | 645,4 | **FAUX : ma prédiction prenait la déquant ; à n = 78 ≤ 80 hors portée le servi est le GEMV** (erreur d'écriture) |
| P2 n=78 F | 40-70 | 62,5 | tenu |
| P2 F/T_part | 0,22-0,45 | qkv 0,36, down 0,48, gate 0,55, out 0,69 | qkv tenu ; **gate, out, down FAUX** |
| P2 I/F | 0,8-1,3 | 1,11 / 1,12 / 0,99 / 0,78 | tenu sauf down (0,78, I plus rapide) |
| P3 n=64/128 F/T_part | 0,2-0,5 | qkv 0,40/0,36 ; gate 0,73/0,56 ; out 0,85/0,71 ; down 0,51/0,49 | qkv tenu, autres FAUX |
| P4 n=624 F / F/T_seul | 170-300 / 0,15-0,3 | qkv 244 / 0,24 ; gate 0,29 ; out 0,34 ; down 0,31 | tenu (qkv, gate), out/down légèrement au-dessus |
| P5 justesse | T 0,8-1,5 ; F 3-4,5 ; I 1,5-3 | T 0,94-1,13 ; F 2,26-2,74 ; I 1,29-1,45 | T tenu ; F et I **meilleurs** que prédit ; F reste 2,5 × l'erreur de T |
| P6 noyau | CUTLASS sm100/120 | `enable_3x_kernel_for_sm10_or_later` | tenu |

**Seuil de l'étape moteur : NON TENU.** Il fallait F ≤ 0,50 × T_part à n = 78 sur qkv ET out : qkv 0,36, **out 0,69**.
Le volet n = 624 (≤ 0,50 × T_seul sur les quatre formes) est tenu (0,24-0,34). Par la lettre du scellé : **pas d'étape moteur
FP8 en l'état.**

## Lecture
1. **Décodage** : le FP8 W8A8 est 1,4-2,2 × plus lent que le GEMV int8 servi (F_mm seul plafonne à ≈ 43 µs dès n = 1 :
   1,2 To/s sur qkv, 0,73 sur out). Confirme 148 et p43 sur les formes réelles. Fermé.
2. **Préfill** : la MMA e4m3 est rapide (F_mm qkv à n = 78 : 44,5 µs, quasi au plancher de lecture de 52 Mo), mais
   **la quantification de l'activation, non fusionnée, coûte 13-22 µs par appel à n ≤ 78, 39-98 à n = 624** (F − F_mm) : c'est elle qui fait perdre gate
   et out à n = 78, là où la 243 a déjà rendu T_part bon marché (92,7 µs sur out). **F_mm seul passerait le seuil** (qkv 0,26,
   gate 0,37, out 0,47, down 0,40) — non scellé, donc pas un verdict, une piste.
3. **L'int8 W8A8 (`_int_mm`) égale le FP8** (I/F 0,78-1,12) avec **deux fois moins d'erreur** (1,3-1,45 % contre 2,3-2,7 %) et
   sans format neuf : issue (ii) du scellé. La question n'est pas « fp8 ou int8 » mais « quantifier l'activation au préfill
   ou non ». Le FP8 n'apporte que l'exactitude des POIDS (≈ 1 % gagné), et la reperd trois fois par l'activation e4m3.
4. Ordre de grandeur moteur (estimation, NON mesurée) : au préfill b=8 × 78, W8A8 avec quantification fusionnée ôterait
   ≈ 70 ms (GDN, 48 couches × 8 séquences) + ≈ 50 ms (attention et mlp 56-63 à n = 624) sur 410 ms par lot.

## Réserves
x est aléatoire ; les activations réelles portent des valeurs aberrantes par canal qui dégradent une quantification par
jeton (e4m3 comme int8) — l'erreur réelle de F et I peut être plus haute. La justesse ne remplace pas la KL (2b).
I suppose les poids int8 SIGNÉS en mémoire ; le servi les garde en uint8 à zéro 128 : signé = `(q ^ 0x80).view(int8)`, une
passe d'un octet par poids (copie transitoire de la 201) ou un stockage signé avec XOR dans le GEMV.

## Proposition à chef (pièces, non jouées)
(a) quantification e4m3/int8 de l'activation par jeton en UN noyau Triton (amax + échelle + conversion), puis rejouer ce
banc tel quel (même seuil) ; (b) si (a) passe : le bras I (int8 W8A8, poids servis inchangés) est le candidat de l'étape
moteur avant le FP8 — KL au protocole 2b, prédite plus proche du témoin que F (erreur 1,3 % contre 2,7 %) ; (c) format fp8
unique : FERMÉ pour la vitesse (décodage plus lent, préfill = I) ; reste seulement un argument de qualité des poids, à
juger à la PPL si un jour il compte.
