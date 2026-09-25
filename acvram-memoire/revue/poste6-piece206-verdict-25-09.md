# Verdict — pièce 206 (à sec) : les trous entre noyaux du Coder-30B sous graphe — ERRATUM de la 203 (poste 2 surévalué ×3), trous réels 0,20-0,25 ms/pas, candidats chiffrés, aucun second flux possible sur cette architecture (poste6, 25/09)

* **instrument** : traces nsys de la 203 (b1 49fd64e8d69da5a8…, b8 8a9ecc53aa250b43…, hors git), `scratchpad/poste6-p206-25-09/
  trous-206.py` (trou entre noyaux consécutifs par paire, fenêtres de couche du lot plein) et `trous-pas-206.py` (comptabilité
  d'un PAS entier de route_fusee(0) à route_fusee(0) : pas, Σ noyaux, trous, 14 plus gros trous, noyaux hors couches) ;
  résultats `trous-b{1,8}.txt`, `trous-pas-b{1,8}.txt`. Lecture de code : `couches.py:460`, `attention.py:395`, `moe.py:1540-1566`,
  `regime.py:214-217`. Aucune carte.
* **commit** : 78ea82fc2 (poste6-206 = origin/main) + ce fichier.
* **régime** : celui de la 203 (graphes on, lot plein 1 / 8, Coder qkvo-i8c).
* **scellé** : aucun (à sec, ordre de chef : (1) paires et indépendance, (2) chiffrage, (3) code sur feu seulement).
* **mesuré** (pas médian) : b=1 2 935 µs, noyaux 2 734 (605 lancements), **trous 202 µs (6,9 %)** ; b=8 5 043 µs, noyaux 4 797
  (761 lancements), **trous 246 µs (4,9 %)** ; dans une fenêtre de couche, CHAQUE trou vaut 0,19 µs (médiane = p90), 0 chevauchement.
* **verdict** : (1) ERRATUM 203 : les « 0,43 / 0,52 ms de trous » étaient pas − 48 × médiane des noyaux de couche : ils
  contenaient la tête (190 / 201 µs), les 4 couches hors tensor (b=8 : +0,13 ms, pas +0,03 — second erratum) et l'échantillonneur ;
  les trous réels sont 0,20 / 0,25 ms ; (2) un seul gros trou par pas : 67 / 80 µs à la frontière hôte (memcpy H2D des jetons →
  premier noyau du pas suivant), le reste = 0,19 µs par nœud de graphe × 600-760 ; (3) **aucune paire indépendante à
  paralléliser** sur le Coder (la chaîne d'une couche est strictement séquentielle, § 2) : les leviers sont des fusions de
  lancements et la frontière hôte, pas un second flux ; le plus gros levier chiffré n'est pas un trou : les 4 couches hors tensor.
* **durée** : à sec, 55 min (deux scripts, deux traces de 190 Mo).

## 1. Où sont les trous (pas médian)
| | b=1 | b=8 |
|---|---|---|
| frontière hôte (memcpy H2D → 1er noyau) | 67 µs | 80 µs |
| trous ≥ 1 µs (memcpy/memset d'entrée, échantillonneur), 12-13 par pas | 23 | 24 |
| trous < 1 µs entre nœuds de graphe (0,19 µs médian) | 112 (591 nœuds) | 142 (748 nœuds) |
| **total** | **202 (6,9 %)** | **246 (4,9 %)** |
Le « 0,85 µs par lancement » de la 203 était faux : c'est 0,19 µs par nœud sous graphe (`--cuda-graph-trace=node`) — la
latence de lancement est déjà cachée par le rejeu ; ce qui reste est le coût de dépendance entre nœuds, incompressible sauf en
supprimant des nœuds. La 141 (PDL sur les GEMV b=1 : −0,008 ms) l'avait déjà montré : rien à gagner sur le lancement lui-même.

## 2. La chaîne d'une couche (b=8, 15 lancements) et ce qui dépend de quoi
`route_fusee → aligneur → marlin w13 → moe_act → marlin down → moe_reduce → rmsnorm → qkv (etroit_reduit) → rope → kv_write →
attention → o (etroit_reduit) → rmsnorm → routeur GEMM (cutlass) → splitK_reduce`. Chaque maillon lit la sortie du précédent :
aligneur ← top-k ; w13 ← alignement ; act ← w13 ; down ← act ; reduce ← down ; norme ← résidu ; qkv ← norme ; rope ← qkv ;
kv_write ← rope ; attention ← cache écrit ; o ← attention ; norme ← résidu ; routeur ← norme ; top-k ← logits. À la différence
de Qwen3.8 (194 : α‖β lit la même entrée que la pile qkv‖gate et ne nourrit que la récurrence), le Coder n'a **aucune branche
latérale** : pas d'expert partagé, normes q/k dans rope, routeur seul consommateur de la norme avant les experts. **Le levier
« second flux » n'existe pas ici.** À b=1 (12 lancements) : même chaîne sans aligneur ni act, GEMV par paire ×2.

## 3. Candidats chiffrés (b=8 sauf mention ; pas 5,04 ms)
| # | candidat | gain max ms/pas | % | au bit ? | état |
|---|---|---|---|---|---|
| A | **4 couches hors tensor** (157 : up_proj à échelles sous-normales → `decode_mma` W4A4 : 82 µs/couche contre 50,5 en tensor) | **0,13** | 2,6 | non (même passage que les 44 autres couches, 62-A4 KL 5/5) | pièce 157 bis (poste5/poste1) : représenter ces échelles en Marlin ou les reconditionner |
| B | frontière hôte 67-80 µs/pas (H2D → 1er noyau ; `trou_gpu` par événements = 24 µs : le reste est la latence du rejeu après la copie) | ≤ 0,05 | 1,0 | oui | frontière de pas (poste1, `frontiere-pas.py`) |
| C4 | aligneur (1 bloc, 2,43 µs) dans `route_fusee` (top-k + comptage + tri en un noyau) | 0,07-0,12 | 1,4-2,3 | **oui** (entiers) | à coder sur feu, banc 5 min |
| C3 | logits du routeur ([8×2048]·[2048×128] : cutlass 2,56 + splitK 0,93) DANS `route_fusee` (0,5 Mo en L2, ≈ 2 µs) | 0,08 | 1,6 | non (ordre des sommes fp32 ; top-k stable hors égalité) | avec C4 : un noyau « logits + top-k + alignement » ≈ −4 µs/couche = **0,19 ms (3,8 %)** |
| C5 | `moe_act` dans l'épilogue w13 (1,38 µs + nœud) | 0,075 | 1,5 | non (act sur accumulateurs fp32 au lieu de g/u bf16) | port Marlin, coût moyen |
| C1 | norme d'entrée dans le GEMM qkv (3b, `ACVRAM_NORME_FUSEE`) | — | — | — | **RÉFUTÉ** 6dbb1bb (+0,31 ms : norme recalculée par bloc), témoin gardé |
| C2 | rope + kv_write (3a, `ACVRAM_ROPE_KV`) | 0,09 si refait | 1,8 | oui si mêmes codes | **RÉFUTÉ** a3f1b7e (corrompt sous graphe) ; à ne rouvrir qu'avec un test de capture |
| C6 | `moe_reduce` dans l'épilogue down | — | — | non, et non déterministe (atomiques) | rejeté |
| b=1 | C3 seul applicable (cublas_gemvx 2 µs) ; tête 190 µs = 6,5 % du pas à 91,5 % du pic (116 bis) | 0,05 | 1,7 | non | — |
Somme au bit réalisable : C4 + B ≈ 0,12-0,17 ms (2,4-3,4 %) ; hors bit : A + C3 + C5 ≈ 0,28 ms (5,6 %).

## Suite (à chef, code sur feu seulement)
Ordre proposé : **A** (157 bis, le plus gros, hors de ma pièce), puis **C4 + C3 en un noyau** « logits + top-k + alignement »
(banc 5 min : `route_fusee` + cutlass + aligneur = 7,7 µs/couche aujourd'hui contre ≤ 3,5 visé ; scellé KL comme 195 puisque
les logits changent d'ordre de somme), puis B avec la frontière. Je ne propose ni second flux ni PDL ici (§ 1-2, 141).
