# banc-etroites-occupation --ptxas rejoué (correctif poste1 6c005b5a) — CONFIRMÉ, registres réels lus — 22/09 (poste2)

* instrument : `outils/carte.sh env CUDA_VISIBLE_DEVICES=0 .venv/bin/python outils/gpu/mesure/banc-etroites-occupation.py --ptxas`, sha `6c005b5a`
* mesuré : qkv/4w2s = **139 registres** (identique au point de contrôle manuel), qkv/2w2s-2w3s = 144-146 registres mais **7 blocs/SM (14 warps actifs)** contre 3 blocs/SM (12 warps) pour 4w/8w — occupation théorique meilleure sur 2w, comme prédit par le groupe après ma lecture.
* verdict : **CONFIRMÉ, correctif tenu**. Tension nommée sans trancher : l'occupation théorique favorise (2,·) (7 blocs/SM), mais le **chronométrage réel déjà publié** (`verdict-balayage-warps-etages-22-09.md`) montre (2,·) plus LENT que (4,2) sur la forme `o` (11,2-11,5 µs contre 9,28 µs) — l'occupation calculée par ptxas ne prédit pas le classement mesuré ici (latence dominée par autre chose : lancement, dépendances, bande mémoire réelle plutôt que blocs/SM théoriques). Le verdict RÉFUTÉ du balayage chronométré reste valide, il n'était pas affecté par le bug ptxas (chemins de mesure indépendants).
* durée : <1 min de carte

## Suite
Enchaîne diag-eval-nll (pièce 37).
