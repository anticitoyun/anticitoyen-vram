# Sage — reprise immédiate (utilisateur 05 h 56 : « on poursuit avec reprise tant que les tokens sont disponibles ») : la fin de carte de 08 h 00 tombe, la file continue ; ordre des chantiers par valeur mesurée, scellés déjà écrits, un membre = une file (20/09, 05 h 56, horloge machine)

Source : `sage-bilan-nuit-20-09` § 6 ; scellés de `sage-c1-ferme-c15-prefill` § 2 (C15-prefill), `sage-glm-b1-noeuds-c15-niveau3` § 2 (niveau 3), `sage-cloture-nuit-0540` addendum 05 h 45 (niveau 2 biais), `sage-fiches-c5b-c13c-c14b` § 3 (C14-b), `sage-c13c-forme1-faux-forme2-tf32` addendum 01 h 25 (C13-c structure), `sage-parc-c5b-poste-e-niveau2` § 1 (gemma, Ornith), `sage-c15-niveau3-coder-faux` (sélection ≤ 3 µs). Règles inchangées : ETAT seule entrée, pointeurs, un scellé par fenêtre, `date` avant toute heure.

## 1. Ce qui change
La clôture de 08 h 00 (bilan, dernier push) **n'a plus d'objet** : le bilan de 05 h 51 tient lieu de point d'étape, Jérôme continue fusions et ETAT au fil ; la question 119B reste posée (§ 6 du bilan), **C9 reste en pause** jusqu'à la réponse ; l'équipe garde son rythme (une fusion par verdict, pas de « fin de nuit »). Rangs restants de `sage-cloture-nuit-0540` (ligne de régime → 0.6.25, gemma repli annoncé, vLLM KV fp8, C3 MTP, tri des worktrees) : **ils passent en tête** de chaque file avant ce qui suit.

## 2. Files (par valeur mesurée sur le comparatif ; scellés = ceux des notes citées)
| rang | Océane (code, à sec puis pointeur) | Manon (carte) | gain visé (mesuré, pas promis) |
|---|---|---|---|
| 1 | **niveau 2 GLM, biais** : moyenne signée noyau − torch sur q_eff / k_new couche 5 (10 min) ; si ≠ 0 → correctif d'arrondi (`_rn`), test qui casse sur l'ancien code | rejeu niveau 2 : `=2` EAGER contre `=1` EAGER ≤ 1 ulp bf16 par couche à p = 8 192 ET PPL ± 2 SE aux deux longueurs, capture 5/5, cellule GLM b=1 | GLM b=1 : noyaux −1,0 ms/pas (6,1 → 5,1) |
| 2 | **gemma-4-26B-A4B** : capture des godets avant la réservation KV (ou bassin réservé au plan) + repli annoncé (déjà ordonné) ; **Ornith-35B** OOM Triton godet 1 | capture 4/4 des deux, un jeton décodé par alias (règle S2) | deux modèles du catalogue servis comme les autres |
| 3 | **C15-prefill** (norm + résidu + cast en prologue, glue routeur/permutations) | prefill Coder servi **≥ 20 500 j/s**, noyaux ≤ 83 ms, PPL **au bit** 3 tranches, capture | 17 784 → ≥ 20 500 (parité vLLM 20 824) |
| 4 | **C15 niveau 3 GLM** (≤ 700 nœuds : routeur un nœud, division AWQ + `y+shared` dans les experts, C14-b, C10 pour les 13 xreg ; niveau 2 préalable) | espaces ≤ 0,7 ms ET pas b=1 servi ≤ 7,0 ms, juges du niveau (jetons ou « B ≤ témoin »), PPL avec SE, capture | GLM b=1 122,5 → ≥ 140 |
| 5 | **C14-b** (k_b en prologue, v_b dans le combine, `mla_prep_batch` par ncu) | GEMM + reduce 1,05 → ≤ 0,35 ms, `ppl-decode-kv` au lot de 12 ± 0,002, capture | GLM b=12 −0,7 ms/pas |
| 6 | **C13-c réécrit** : structure « q par tranches BK en registres, un accumulateur [BM, rank] » ; **sonde de temps à bras bf16 ≤ 2 ms/appel d'abord**, exactitude ensuite (TF32 ≤ 2 048 clés, juge contre l'einsum TF32) | cœur ≤ 70 ms à L=2 047, prefill ≥ 9 500 j/s, PPL |Δ| ≤ 0,0005, cellule L=8 192 | GLM prefill 7 268 → ≥ 9 500 |
| 7 | **C15-3d sélection ≤ 3 µs** (7,46 : grille, un warp par ligne trop sériel) puis **bande de Marlin** (1,07 → ≥ 1,25 To/s : étages `cp.async`, blocs en vol — noyau, plusieurs jours) | Coder b=12 servi contre témoin de fenêtre : t/s ≥ A + 3 % à J ≤ 1,005 × A ; ncu `dram__throughput` | Coder b=12 vers vLLM 1 626 (l'écart de J reste structurel : Marlin à 400 W) |
| — | hygiène au fil : C5-b nsys d'un pas B (quel noyau porte 3,7 ms) ; déterminisme ON/OFF de Coder b=12 (bissection par famille, nommer l'op) ; C3 MTP ; C10 (a) ; PPL longue tf32 2 000 paires | fenêtres de 10-40 min dans les trous | — |

Un scellé réfuté ferme son rang et passe au suivant, sans note de Sage sauf prédiction réfutée ou changement d'ordre ; une prédiction chiffrée avant chaque fenêtre, avec l'issue qui gênerait son auteur.

## Ordre
* **Océane** — rangs restants de la clôture (ligne de régime, gemma repli annoncé, C3 estimation) puis 1 → 2 → 3 → 4 → 5 → 6 → 7 ; sous-agents autorisés (un par chantier, worktree, chaîne carte livrée avec le scellé en tête).
* **Manon** — vLLM KV fp8 (en cours), bras éco de 0.6.25, puis les fenêtres dans l'ordre des pointeurs d'Océane ; trous : hygiène (C5-b nsys 10 min, déterminisme ON/OFF 20 min, PPL longue 40 min).
* **Jérôme** — tri des worktrees, 0.6.25, fusions à leurs verdicts, un .deb par changement de défaut (bras éco à chaque fois) ; ETAT et INDEX au fil ; **question 119B à l'utilisateur** (§ 6 du bilan) : C9 attend sa réponse.
