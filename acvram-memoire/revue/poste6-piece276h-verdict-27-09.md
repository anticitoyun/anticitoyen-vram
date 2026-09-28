# Pièce 276 h — deux flux annexes (deux tours en vol) : au bit 12/12, mur −5 % et TTFT moyen +9 % contre la 276 g — scellé NON tenu, opt-in, verdict

poste6, 27/09/2026 17 h 5x. Scellé : `poste6-piece276h-a-sec-27-09.md` (avant le code, inchangé). Code : origin/poste6-276h
(606d20adc mesuré ; défaut ramené à 1 après la mesure, ce verdict). Sous carte.sh (ACVRAM_NOM=poste6-276h, 17:46:44 → 17:50:50,
premier essai), arbre importé contrôlé (`ARBRE`), charge hôte par cellule, échelle **M G H H G M** : M = main a6268af2e, G = poste6-276g
5206ee47f, H = cet arbre (`ACVRAM_TOUR_FLUX=2`). Qwen3-VL-2B, une image 448×448, b = 12 × 7 tours, solo × 5, équivalence b = 4/12, cellule
b = 1 (5 invites, M1 et les deux H). Traces (sha256 en tête, M1 G2 H3 H4 G5 M6) : f0703179 32632198 43db5043 7883a5f8 60efa8bf 95ebc6fe.
Journaux : M « tour (_admit) » 112/107, G et H « tour (préparation) » 107/112/112 — chaque bras sur son chemin.

## Ce qui change (606d20adc)
`VerrouCapture` lecteurs/rédacteur dans graphs.py (N tours en vol = lecteurs ; la capture = rédacteur, prioritaire) ; réserve de
`ACVRAM_TOUR_FLUX` flux CUDA annexes (`Engine._reserve_flux_tour`, queue) prise/rendue par `encoder_images` ; chaque tour reste UNE
image, même appel. 3 tests à sec (deux lecteurs coexistent, rédacteur seul et prioritaire ; deux `encoder_images` en vol ensemble et au
bit ; régime déclaré).

## 1. Identité
* Deux fils sur deux flux contre série, 12 images distinctes, deux passages : **AU BIT 12/12 et 12/12** (série = série, deux-fils =
  deux-fils 12/12). Durée des 12 images à chaud : série 104 ms, deux fils **81 ms** (−22 % seulement : la tour eager est bornée par
  ses lancements Python — deux fils se partagent le GIL — pas par la carte).
* Jetons b = 1 (24, temperature 0) : H = M **9/10** (H3 ≠ M sur l'invite 0, H4 = M 5/5, H3 ≠ H4 sur cette invite). MAIS main lui-même :
  M1 de cette prise ≠ M1 de la prise 276 g sur l'invite 1 (1/5), même arbre, même serveur neuf, même invite seule. **Le critère
  « jetons identiques à b = 1 » n'est pas non plus un contrôle stable** : un jeton bascule parfois à b = 1 entre deux serveurs neufs
  de main. Le 5/5 de la 276 g tenait ; ici 9/10 avec une bascule de même nature que celle de main. Cause ouverte (préfixe froid/chaud ?
  noyau non déterministe au préfill ?) — à poser à poste4/duck.ai, hors pièce. b = 4/12 : publiés (G 2-3/4, H 2-3/4, M6 4/4 ;
  b = 12 tous 10-11/12), inapplicables (composition).

## 2. Tableau (b = 12, 84 requêtes par bras ; ms)
| bras | pas de préfill par tour | TTFT **moyen** | p50 | p95 | **mur** max (méd) | solo |
|---|---|---|---|---|---|---|
| M1 main | [3, 3, 3, 3, 3, 3, 3] | **218,4** | 276 | 290 | **293,2** (284,1) | 30,9 |
| G2 276 g | [12, 7, 6, 8, 7, 5, 8] | **176,9** | 176 | 265 | 287,2 (262,8) | 30,4 |
| H3 276 h | [4, 3, 4, 4, 4, 4, 5] | **192,0** | 206 | 251 | **267,7** (247,9) | 30,2 |
| H4 276 h | [5, 4, 4, 4, 4, 4, 4] | **193,7** | 208 | 252 | **283,2** (249,9) | 31,1 |
| G5 276 g | [12, 7, 6, 7, 7, 8, 5] | **175,8** | 176 | 263 | 279,7 (260,1) | 30,0 |
| M6 main | [3, 3, 3, 3, 3, 3, 3] | **215,4** | 274 | 283 | **293,5** (278,7) | 30,9 |

## 3. Scellé, point par point
* **Mur H ≤ 0,85 M** : max 293,4 → 275,5 (**0,94 : FAUX**) ; médiane 281,4 → 248,9 (0,88, faux aussi) ; contre G : −12 ms (méd), −4 %.
  Prédit −40 à −80 ms : obtenu −18 (max) / −32 (méd). Deux tours ne se recouvrent qu'à 22 % (identité : 104 → 81 ms).
* **TTFT moyen H ≤ 0,80 M** : 216,9 → 192,9 (**0,89 : FAUX**) ; **H ≤ G** : 176,4 → 192,9, **+9 % : FAUX**. p50 176 → 207.
* **Issue nommée survenue** : le TTFT moyen remonte alors que le mur baisse. Cause lue dans les pas par tour : avec deux tours en vol
  les requêtes sortent de préparation par paires rapprochées, la fenêtre d'admission les groupe en 4 préfills de 3 au lieu de 7-12
  petits — chacune attend davantage ses voisines. Le recouvrement des tours (gain) est mangé par le regroupement des préfills (perte).
* `_admit` ≤ 10 ms : 0,1 (tenu, inchangé). Solo : 30,2/31,1 contre G 30,4/30,0 : ± 1 (tenu).
* Si l'hypothèse était fausse, « mur H = G ± 3 % » : −4 % sur le max, −5 % sur la médiane — à la frontière ; le gain est réel mais
  petit, et payé ailleurs.

## 4. Décision
Scellé non tenu → **`ACVRAM_TOUR_FLUX` défaut 1 (opt-in 2)**, code et régime le disent (commit de ce verdict). Le VerrouCapture
lecteurs/rédacteur reste (à 1 flux il se comporte comme le verrou de la 276 g ; test à sec). Ce que la série 276 g/h établit : à
b = 12 images, le mur (≈ 280 ms) est désormais dominé par les 12 tours eager en série (≈ 105 ms, bornées par les lancements, pas par
la carte) puis les préfills ; le levier qui reste est une tour à lancements réduits (graphe CUDA de la tour à forme fixe 448×448, ou
torch.compile) — pas plus de fils. Prédit : 12 tours 105 → ≤ 40 ms, mur −60 ms, TTFT moyen −40. À chef. Hors pièce : la
non-reproductibilité de main à b = 1 entre deux serveurs neufs (1/5) — question pour poste4/duck.ai.
