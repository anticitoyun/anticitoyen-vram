# nsys-tete-sampler (pièce 39) — ÉCHEC instrument, regex TÊTE mal ciblée — 22/09 (Manon)

* instrument : `outils/gpu/mesure/nsys-tete-sampler.sh` (Océane, 22/09) → `part-tete-sampler.py`, b=12, 60 pas, alias `Qwen3-Coder-30B-A3B-nvfp4`
* commit : main à jour
* régime : `sampler=graphe` (argmax dans le graphe), 94 pas jugés (têtes/queue exclues), noyaux 6,736 ms/pas (cohérent avec `verdict-nsys-familles-22-09` du matin)
* mesuré : `tête = 2,85 µs (0,04 % du pas)` ; `échantillonnage = 6666,53 µs (94,00 % du pas, 642 lancements)` — la liste des noyaux dans « échantillonnage » contient `int8_dequant_kernel`, `marlin_moe_wna16::Marlin`, `cutlass::Kernel2<cutlass_80_tensorop...`, `nvfp4_gemv_marlin_kernel`, `fmha_cutlassF...` : **ce sont les noyaux MoE/attention/routage du reste des 48 couches**, pas des noyaux d'échantillonnage (argmax/gather/reduce attendus par le docstring du script).
* cause probable : `TETE = r"cutlass.*wmma|lm_head|tete_"` (`part-tete-sampler.py:29`) — `i_tete = next(... re.search(TETE, n) ...)` prend le **premier** noyau du pas qui matche ce motif ; si un noyau cutlass précoce (routeur ou autre GEMM d'une couche antérieure) matche accidentellement `cutlass.*wmma` avant le vrai GEMM de tête (qui devrait arriver en fin de pas), tout ce qui suit — soit 47 couches de calcul réel — tombe dans le seau « échantillonnage ». Une vraie tête (GEMM [12,2048]×[2048,151936]) ne prend pas 2,85 µs — trop petit pour être plausible, cohérent avec un mauvais noyau capturé en tête de fenêtre plutôt qu'en fin.
* verdict : **ÉCHEC — la porte de la pièce 39 (tête+sampler < 2 % du pas) n'est PAS jugeable sur ce résultat** : 94,04 % combiné serait absurde (aucun modèle ne passe 94 % de son pas dans l'échantillonnage). Chiffre non publié comme mesure réelle. Cause nommée pour Océane : vérifier que `TETE` matche le DERNIER GEMM avant la fin du pas, pas le premier motif rencontré (ou resserrer le motif pour exclure les GEMM MoE/routeur qui peuvent partager `cutlass`).
* durée : ~5 min de carte, log gardé (`scratchpad/nsys-tete-sampler-22-09/part.txt`)

## Suite
Correctif à Océane (regex ou logique de sélection du dernier GEMM candidat, pas le premier). Je rejoue --ptxas (corrigé, main 02a5f65a) puis diag-eval-nll (pièce 37), rends la carte à Laure ensuite.
