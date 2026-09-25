# Verdict — 176b : tuiles N de 32 pour gate‖up int8 (1,07 vague) — 25/09 (poste1) — FAUX, code retiré

* **harnais** (`scratchpad/poste1-p176b-25-09/bn32.py`, poste1-p176b-bn32, avant tout code) : BN 32 au bit de BN 64 sur
  3 formes × M 8, 16 ; gate‖up 34 816 × 5 120 appel ISOLÉ 165,5 → 151,9 µs (M = 8) ; qkv +29 %, down +20 % à BN 32.
* **code** : règle `bn_pour` (32 si une tranche et 510 < tuiles ≤ 765), commit 95a1dfe0 ; **retiré** (098d6c70).
* **prise** (poste1-p176b-abba, scellé avant ; A = 4850303c, B = 95a1dfe0 ; certifie-b12, CTX 8 192, 5 lots par bras) :
  mixte **b=8 A 19,664 ms, B 19,690 ms, B/A 1,0013** (prédit 0,990-0,999 ; FAUX si > 1,000) → **FAUX** ; b=1 1,0001 (tenu).
* **lecture** : le gain de l'appel isolé (−13,6 µs × 8 appels) ne se retrouve pas dans le pas servi, qui perd ≈ 3 µs par
  appel. Le harnais mesurait gate‖up seul, dans une carte vide ; dans le pas, le noyau suit et précède d'autres noyaux.
  Hypothèses non mesurées : la queue de la 2e vague à BN 64 était déjà recouverte par le noyau suivant, et 1 088 petits
  blocs coûtent plus en ordonnancement. Leçon : un harnais d'appel isolé ne prédit pas un gain de queue de grille.
* **faute d'instrument** : le cassant (appel `bn_pour` remplacé par BN dans gemm_etroit) est resté VERT (7 passed) ; le test
  de règle interrogeait la fonction `bn_pour`, pas le routage. Un test qui ne garde pas le chemin servi ne garde rien.
