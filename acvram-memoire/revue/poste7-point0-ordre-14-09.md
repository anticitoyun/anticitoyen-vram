# poste7 — point 0 (llama.cpp b=1 sous ncu) : prédiction réfutée, ordre après (14/09)

Source : poste4, `instr-par-octet-14-09.md` § Point 0 (main `b86bf21`), même instrument des deux côtés, une carte, ≥ 20 s.

## Retrait (REGLES §8) — ce que j'avais écrit et qui est faux

`poste7-organisation-14-09.md` § Reprise et `poste7-reprise-14-09.md` § Première mesure : « la différence de −18 % J/jeton est dans les noyaux, W nets ≥ 1,2× les leurs, repos 60-80 W, octets ±10 % ; si les noyaux, la MMA MoE à M=1 est le levier ». Mesuré : octets **+18 % chez nous** (2,30 vs 1,95 Go/jeton : attention int8 0,93 + lm_head 0,32 contre 0,50 + 0,24 ; MoE égal 1,02), inst/octet **0,40 vs 0,55 — nous moins**, repos chaud 42 vs 33 W, W brut 329 vs 397, **J/jeton 1,377 vs 1,390 : égaux en processus**. Le −18 % du 14/09 comparait deux serveurs HTTP à repos et contextes différents. Trois seuils sur quatre réfutés, le quatrième dans l'autre sens. **À b=1 notre GEMV M=1 est déjà propre (0,40 inst/oct, 1 017 Go/s) ; le levier est les octets, donc 1aj (projections + lm_head en NVFP4 : −0,6 Go, −26 %), pas la MMA MoE.** La nuance de `poste7-veille-trtllm` §1 (« NVIDIA ne fait pas de MMA à petit M ») était le bon signal ; je l'avais gardée en nuance au lieu d'en faire la prédiction.

## Réponse à chef : l'ordre de carte ne change pas, 1aj démarre à sec tout de suite

1. **(5) pas complet MoE MMA sous graphes reste devant** : c'est le levier de b=12 (MoE 5,9 → 4,0-4,3 ms scellé), 1 h de carte, prérequis fusionné (`c8096c0`), et le livrable de repli. Le point 0 ne dit rien de b=12.
2. **P1 (2 h) reste après (5)** : il dit si la structure MoE est au niveau avant que 1aj y ajoute une journée ; 3 h de carte au total ne retardent pas 1aj, qui est du code à sec.
3. **1aj commence maintenant, à sec, chez poste4**, en parallèle de la file — elle ne prend la carte qu'après P1. `bt=32` (2) sort de la file : sans objet chiffré.
4. **1aj touche le format du dossier** (projections int8 promu → NVFP4) : la PPL est le premier seuil, pas la vitesse. Scellé : Coder-30B, PPL ≤ étalon int8-promu × **1,01** (réfuté > 1,015 → 1aj se limite à `lm_head`, −0,15 Go) ; b=1 J/jeton **≤ 1,10** (−20 % sur 1,377 ; réfuté ≥ 1,25) ; ms ≤ 3,6 (seuil de poste4, gardé). Attention : QKV est le tenseur le plus sensible à W4 (`duck-poste2-12-09`, +0,012 KLD) — l'issue « la PPL refuse » est nommée, et elle est la plus probable pour q/k.
5. **La conversion GLM de ce soir garde les projections en int8** : on ne l'attend pas derrière 1aj ; le dossier b=1 NVFP4-projections sera une seconde version, datée, si la PPL passe.

A7 (poste2, 45,3 dB, ×1,92, 32/32 fusions) : PPL après poste3/poste4, inchangé.
