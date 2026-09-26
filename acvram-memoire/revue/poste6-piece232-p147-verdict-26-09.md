# 232 / p147 — les 2 rouges `test_depaqueter_cuda_p147` : DÉFAUT PRÉEXISTANT (209 a, 25/09), pas la branche 232 ; corrigé en 250

instrument : pytest sous `outils/carte.sh` (prise poste6-p147-ab, ACVRAM_DUREE_MAX 600), deux arbres dans la même prise
commit : main 5d86a3bf5 (worktree travail/poste6-p147-main) puis poste6-232 fa93bcc4d (travail/poste6-232) ; correctif poste6-250
régime : à sec côté modèle (tests unitaires, CUDA_VISIBLE_DEVICES=0, load 1,93 au départ) ; carte : llama-server 4219 seul
scellé : rouge sur les deux arbres = préexistant ; rouge sur fa93bcc4d seul = régression 232 (PAR_LIGNE=0, 213 b ou 232)
mesuré : main 2 failed (0,40 s) ; fa93bcc4d 2 failed (0,36 s) — même `ValueError: depaqueter_marlin : échelle globale par colonne
  d'une pile (E > 1) : noyau triton ou torch`, sur les deux tests, les deux arbres
verdict : PRÉEXISTANT — la 232 n'y est pour rien. Cause : `git log -S` → a256676f8 (209 a, 25/09 16:02, poste6) a remplacé la
  condition `E == 1 and g.numel() == N` par `par_colonne = g.numel() == E * N` puis ajouté `if par_colonne and noyau == "cuda": raise`
  SANS `E > 1` (marlin_port/__init__.py:405) : le dense par colonne (134/147, E = 1), que l'extension sert au bit, est refusé dès qu'on
  nomme le noyau. Les deux tests p147 le nomment (`noyau="cuda"`, lignes 60 et 89). Le service passe par `auto` (résolu APRÈS la
  garde, __init__.py:413) et ne dépaquette que des denses (kernels/__init__.py:645 et :1135) : non touché, cohérent avec 226/228.
  Trou voisin : `auto` → cuda sur une pile E > 1 à g [E, N] aurait lu g[e] comme scalaire (`_depaqueter_cuda`:747, par_colonne = E == 1)
  sans erreur — jamais atteint en service aujourd'hui, fermé quand même.
durée : prévu ≤ 10 min / tenu 04:07:32 → 04:07:35 dans la prise (attente 550 s)

Correctif (poste6-250) : garde `par_colonne and E > 1 and noyau == "cuda"` ; `auto` n'élit pas cuda si `par_colonne and E > 1` ;
test cassant `test_pile_par_colonne_refusee_au_cuda_et_auto_evite_cuda` (refus explicite + auto = torch au bit + compteur cuda
inchangé) ; CHANGELOG. À prouver sous verrou : `tests/test_depaqueter_cuda_p147.py` (8 tests) + `tests/test_marlin_pile_par_ligne_209.py`.
La « suite fantôme » à 11 rouges p147 sur un autre arbre (chef) est à relire à cette lumière : même message → même cause.
