# M-UVA — NON CONCLUANT, chronométrage brut inexploitable — 22/09 (Manon)

* instrument : `pytest tests/test_placement_par_expert.py::test_chemin_groupe_table_bit_identique_au_chemin_groupe_pile -v --durations=5`, sous `carte.sh`, worktree manon-w-21-09, « 0 code »
* commit : main à jour
* régime : test bit-identique existant, pas un banc de débit dédié
* scellé (Océane, § 2) : zéro-copie ≥ 17 Go/s = C9 vaut la peine ; < 13 Go/s = arrêt
* mesuré : test PASSED, durée `call` = **0,60 s** — mais ce temps couvre **deux** forward passes complètes (chemin pile ET chemin table), la construction de `construire_table` sur 3 tenseurs (gate/up/down), et un JIT/warmup CUDA potentiel au premier appel — PAS un GEMV UVA isolé et répété. En dérivant naïvement 21,2 Mo (8 experts × 2,65 Mo, un seul passage) / 0,60 s = **≈ 0,035 Go/s**, très en dessous de tout seuil — mais ce chiffre mesure l'overhead du test, pas le débit du noyau.
* verdict : **NON CONCLUANT** — je ne publie pas ce 0,035 Go/s comme verdict d'arrêt C9, ce serait un chiffre reconstruit qui masque le vrai trou : ce test unitaire (une exécution, deux chemins comparés, aucune boucle) n'isole pas le débit demandé. Il faudrait une boucle dédiée (200 pas comme prévu au § 0) chronométrant SEULEMENT le chemin table après un warmup séparé — hors du « 0 code » de cette pièce.
* durée : < 1 min de carte

## Suite
M-UVA à refaire avec un vrai harnais de boucle (code minimal, warmup + N répétitions du chemin table seul) avant de trancher § 2 (a). Je n'ai pas ce script sous la main — à écrire ou à pointer par Océane. PPL relative A/B 31B ensuite.
