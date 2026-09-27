# 270 — étape 1 : décomposition du préfill du Coder, solo (p1) et 12 invites (p12) (poste1, 26/09)

* instrument : `scratchpad/poste1-p270-26-09/` — `profil-270.py` (Engine comme serve, NVTX p1/p12, 5 de chaque), nsys `-t cuda,nvtx` ;
  `fenetres-270.py` (attribution par FENÊTRE DE TEMPS, voir alarme) ; `familles-270.py` (famille moe en tête)
* commit : poste1-270 cc16e4937 (prise), analyse ultérieure sur le même rapport (sha256 bc1c2d2a1eeca724…, hors dépôt)
* régime : Qwen3-Coder-30B-A3B-Instruct-srcQ4_K_M-nvfp4, -lgc 2700, ACVRAM_ECO=off, max_batch 16, graphes on, sous nsys
* scellé : `scratchpad/poste1-p270-26-09/scelle.md` (b8553df89, avant le nsys)
* mesuré : p1 = 1 invite du duel (412-518 jetons), p12 = 12 invites distinctes (5 251-5 476 jetons), un pas de préfill chacun
* verdict : MoE ≥ 50 % de p1 TENU (55 %) ; MoE ≤ 35 % de p12 FAUX (59 %) ; condition FAUX 1 (MoE < 35 % de p1) non atteinte ;
  plafond du levier MoE ≈ 7 ms en p1, SOUS le gain prédit (−8 à −12) — à trancher avant tout code
* durée : prévu ≤ 10 min ; tenu 194 s (`carte.sh` journal `tenue=`) ; analyses sous verrou, processeur seul

## Alarme scellée déclenchée, puis levée par un second instrument
`--filter-nvtx` ne garde que les noyaux LANCÉS par le fil qui ouvre la plage : 6,2 ms de noyaux pour 50,8 ms de mur en p1
(12 %), 60 lancements MoE pour 48 couches — le moteur lance aussi depuis d'autres fils. Refait par fenêtre de temps (tout
noyau dont le début GPU tombe dans la plage, close par un synchronize) : p1 34,5 ms de noyaux, GPU occupé 65 % du mur sous
nsys ; hors nsys le mur p1 vaut 36 ms (262) — le préfill solo est borné par le GPU, pas par l'hôte. Mur p1 sous nsys 50,8 ms
(hors de 30-45 ms) : nsys gonfle le chemin hôte ; les parts ci-dessous sont des parts de NOYAUX.

## Familles (ms de noyaux par préfill, médiane de 5)
| famille | p1 | part | p12 | part |
|---|---|---|---|---|
| moe | 18,97 | 55,0 % | 136,08 | 59,1 % |
| gemm_bf16 (projections d'attention, après déquant) | 6,31 | 18,3 % | 51,72 | 22,5 % |
| glue | 2,75 | 8,0 % | 7,21 | 3,1 % |
| attention | 2,39 | 6,9 % | 24,64 | 10,7 % |
| copies | 1,67 | 4,8 % | 2,84 | 1,2 % |
| dequant nvfp4 | 1,02 | 3,0 % | 1,05 | 0,5 % |
| Σ noyaux | 34,51 | | 230,06 | |

Dans le MoE de p1 : `marlin_moe_wna16` 15,84 ms (126 lancements, 125,7 µs) ; `nvfp4_gemm_grouped_mma2` 1,78 ms (18 lanc.,
98,9 µs) ; act 0,51, route 0,33. En p12 : Marlin 112,4 ms (891,9 µs par lancement : ×7,1 pour ×12 jetons).
126 + 18 = 144 = 48 × 3 : trois GEMM groupés par couche — porte et montée SÉPARÉES au préfill, par choix (`moe.py:965-982`,
pièce 82 ter : une GEMM 2N découpe K selon prob_n, pas au bit ; KL de fin de préfill 0,41 → 0,93). Bloc M choisi = 32 au solo (`choisir_block_size`,
`marlin_port/__init__.py:552` : ≈ 28 jetons par expert) — pas de gaspillage de tuile (Q20 point 2 écarté).

## Plancher et plafond du levier (p1)
* Tous les experts sont touchés au solo (~3 640 paires jeton-expert / 128) : lire leurs poids ≈ 14,5 Go / 1,8 To/s ≈ 8 ms ; calcul
  ≈ 3 640 × 3 × 768 × 2 048 × 2 × 48 ≈ 1,65 TFLOP ≈ 8 ms en bf16. Marlin MoE à 15,8 ms ≈ 2 × le plancher : **gain maximal ≈ 7 ms**,
  réaliste 3-5 ms (Q20 : 1,2-1,5 × contre un groupé déjà compétent — c'est notre cas, trois lancements groupés par couche).
* Projections (déquant + cutlass bf16) : 7,3 ms pour ≈ 0,82 TFLOP ≈ 130 TFLOPS — déjà borné par le CALCUL bf16 ; un W4A16 Marlin
  ne ferait pas mieux ; seules des activations 8 ou 4 bits (hors bit, 260 de poste5) le réduiraient.

## Proposé (chef tranche)
* La fusion porte + montée (Q20 point 4) est DÉJÀ jugée hors bit (82 ter) : exclue d'un levier au bit. Reste au bit : la grille
  Marlin (blocs par SM, ordre des tuiles) SANS toucher à la découpe de K — gain non chiffrable à sec, plafond 7 ms.
1. **Proposé : clore la 270 ici** — le solo (36 ms) est à ≈ 2 × le plancher, le seul levier au bit restant n'a pas de gain
   prédit ≥ 3 ms étayé ; condition FAUX 2 non jouée faute de levier.
2. Hors bit (mode « ± 1 ulp », REGLES § 1, KL) : fusion 2N (82 ter) et activations fp8 aux projections (260) — à ouvrir par
   pièce propre si le TTFT solo devient un objectif.
