# Pièce 116 quater — micro-banc ROWS_PER_BLOCK : hypothèse des vagues RÉFUTÉE

- instrument : `scratchpad/poste3-piece116quater-rows-per-block-24-09/{banc_rows.cu,banc.py}`,
  extension isolée JIT (`torch.utils.cpp_extension.load`, aucun fichier du dépôt modifié),
  copie du noyau `int8_gemv_kernel` (acvram_kernels.cu:569, helpers 99-256), dispatch runtime
  sur `ROWS∈{2,4,8,16}`, `k_splits=1` fixe, sous graphe CUDA, 1000 répétitions/mesure,
  sous `outils/carte.sh`
- commit : `9db0ad03` (base 116 ter) + ce fichier
- régime : b=1 (N=1), k_splits=1, mêmes formes que la trace réelle (QKV M=5120 K=2048 ;
  O M=2048 K=4096), `--use_fast_math` (comme la production, `kernels/__init__.py:381`)
- scellé (chef) : prédiction +139,5 µs/pas au meilleur réglage ; FAUX si < 30 µs
- mesuré :

  | forme | réf. (ROWS=4, appel isolé) | ROWS=2 | ROWS=4 | ROWS=8 | ROWS=16 |
  |---|---|---|---|---|---|
  | QKV | 6,16 µs | 4,102 µs | 4,105 µs | 4,102 µs | 4,101 µs |
  | O | 6,14 µs | 4,099 µs | 4,101 µs | 4,100 µs | 4,101 µs |

  Écart max entre ROWS (même extension, même compilation) : **0,006 µs** sur QKV, **0,002 µs**
  sur O — dans le bruit de la mesure (bornes 1000 répétitions).
- verdict : **FAUX — hypothèse des vagues RÉFUTÉE**, exactement au seuil que chef a posé
  (< 30 µs). Faire varier `ROWS_PER_BLOCK` de 2 à 16 (facteur 8 sur le nombre de blocs, de
  2560/1024 blocs à 320/128) ne change RIEN à la durée du noyau, en isolation — si l'effet de
  vagues (dernière vague sous-remplie faute de blocs) dominait, un facteur 8 sur le compte de
  blocs l'aurait fait bouger. Ce n'est pas le mécanisme. Gain total mesuré : 0,006 µs/pas,
  très en dessous des 139,5 µs prédits et du seuil de réfutation à 30 µs.
- **anomalie à consigner, pas cachée** : l'égalité au bit contre le noyau réel
  (`ext.int8_gemv` de l'extension de production) n'est PAS atteinte, y compris à ROWS=4 — qui
  est pourtant la MÊME configuration que la production. Hypothèse la plus probable : deux
  compilations séparées du même code source (deux `.so` distincts) sous `--use_fast_math` ne
  garantissent pas la même contraction FMA / le même ordonnancement des instructions
  flottantes — un écart de compilateur, pas un écart d'algorithme. NON ÉLUCIDÉ ici (pas
  d'écart absolu mesuré entre `y` et `y_reel`, à faire si la piste est rouverte). **La
  comparaison ROWS=2/4/8/16 entre elles n'est PAS affectée** : toutes mesurées dans la MÊME
  extension compilée une seule fois — seule la comparaison ABSOLUE contre le noyau de
  production (`reel : 6,15 µs` contre `~4,10 µs` en isolation) est douteuse, et pour une
  raison identifiée séparément : un appel isolé (une seule paire poids/activation rejouée
  1000× sous graphe) profite d'une localité de cache que la production, qui tourne 48 couches
  différentes à chaque pas, n'a pas — donc plus rapide en isolation, sans rapport avec
  `ROWS_PER_BLOCK`.
- conséquence pour la 116 ter : la piste « vagues / occupation » est morte. Les 327,9 µs
  (QKV) et 265,1 µs (O, recalculé 116 ter : en réalité 81,3 µs visés à 85 % — voir tableau
  116 ter) restent inexpliqués par ce mécanisme ; soit c'est un plancher réel d'accès du motif
  GEMV à bloc court sur cette carte (aucun code ne le change), soit une autre cause reste à
  trouver — mais pas `ROWS_PER_BLOCK`.
- durée : ~25 min (2 échecs de compilation avant la prise réelle : `--use_fast_math=false`
  invalide en nvcc, puis une erreur de déballage de tuple dans le banc — corrigés avant
  la mesure qui compte ; carte attendue ~1 100 s avant obtention, fenêtre chargée)
