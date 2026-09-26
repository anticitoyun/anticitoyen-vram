# Pièce 118 — cellule d'attribution : les experts (AWQ) portent l'essentiel des ~20 %, pas le format des projections (poste2, 23/09, ordre chef)

instrument : `scratchpad/poste2-piece118-attribution-23-09/bloc.sh <b>`, palindrome à 8 bras
  (i8c, S1b, S8, S4, S4, S8, S1b, i8c, n=10/alias), `--speculative none`, `-lgc 2700`, même banc
  HTTP (méthode 89/101/94 ter)
commit : 64182bec (main, worktree poste2-w-21-09)
régime : -lgc 2700 posé par le moteur, RAS aux journaux (aucune invalidation)
scellé : `scratchpad/poste2-piece118-attribution-23-09/scelle.md` (avant la prise) ; écart déclaré
  si |diff| > 2σ_diff (Welch, n=10/alias = 2×5)
mesuré (b=12, ms/pas ≈ 1000×12/t/s) :

| alias | t/s | J/jeton | Δ contre i8c (t/s) |
|---|---:|---:|---:|
| i8c (servi) | 1 995,0 (σ 62,0) | 0,1358 | — |
| **S1b** (experts A + reste i8c) | 1 645,6 (σ 24,7) | 0,1918 | **−17,51 %** |
| S8 (A, projections nvfp4 sans AWQ) | 1 541,9 (σ 19,2) | 0,2047 | −22,71 % |
| S4 (A, v/o+tête int8) | 1 538,1 (σ 13,2) | 0,2046 | −22,90 % |

Tous au-delà du seuil 2σ (40-42 t/s).

b=1 (régime GEMV, non tranché par un seuil scellé) : i8c 318,1 ; S1b 315,0 (−0,97 %) ; S8 346,3
(**+8,87 %**, plus rapide qu'i8c) ; S4 291,1 (−8,50 %). Motif différent de b=12 — GEMV et non GEMM
tensor-core, pas la même famille d'effet.

verdict : **les experts (application de l'échelle AWQ) portent l'essentiel de l'écart à b=12** —
  S1b (SEULE différence avec i8c : experts nvfp4 AWQ au lieu de nvfp4 sans AWQ, projections et tête
  identiques à i8c) perd déjà **17,5 points sur les ~20-23 % observés entre i8c et A** (pièce 101 :
  A à 1 544-1 613 t/s). Prédiction tenue (+15 à +25 % de ms/pas ↔ −13 à −20 % de t/s, mesuré −17,5 %).
  **Ma lecture de la 107 § 4 — « les projections nvfp4 portent l'écart » — concernait la QUALITÉ
  (KL sur l'invite 11), pas la VITESSE : ce sont deux attributions différentes, à ne pas confondre.**
  Le format nvfp4 des projections (S8, sans AWQ) et le retour de v/o+tête en int8 (S4) n'apportent
  QUASI RIEN de plus par rapport à S1b seul en pourcentage absolu du côté « pire » — S8 et S4 sont
  même un peu PIRES que S1b (−22,7/−22,9 % contre −17,5 %), ce qui isole un second contributeur,
  plus petit : passer q/k (et/ou toutes les projections) en nvfp4 (même sans AWQ) coûte environ
  **5 points supplémentaires**, indépendamment de l'AWQ — c'est un coût de FORMAT (le noyau nvfp4
  dense contre le noyau int8 Marlin), pas de calibration. v/o et tête ne changent presque rien
  seuls (S4 ≈ S8 malgré v/o+tête revenus en int8).
  **Réponse à la question du chef** : experts (AWQ) ≈ 17,5 pts, projections q/k (format nvfp4) ≈
  5 pts, v/o et tête négligeables. Les deux causes s'additionnent à peu près (17,5+5 ≈ 22,5,
  cohérent avec S8/S4 et avec A elle-même).
  À b=1, motif renversé (S8 devant i8c) : régime GEMV, hors périmètre de cette attribution scellée
  pour b=12 — à creuser séparément si utile.
suite : chef — le levier le plus net pour fermer l'écart de vitesse est le coût d'application de
  l'AWQ sur les experts (S1b), pas le format des projections ; pièce 115 (balayage d'horloge SM)
  ensuite, puis pièce 102 (NInfer) après poste1.
