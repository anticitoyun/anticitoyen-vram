# Pièce 213 / 213b — verdict (poste5, 26/09)

**213 (frontière hôte)** : leviers a (H2D épinglé) et b (prép n+1 pendant n) DÉJÀ en place (graphs.py:930-974,
pipeline.py:138) → 0. H1 (exec en vol bloquant) RÉFUTÉE par P0 (1,4 µs) ; exec jumeau opt-in sans effet, non fusionné.
nsys : Qwen3.8 nvfp4 b=1 cudaGraphLaunch retient l'hôte 12,15 ms, trou 194 µs (1 924 noyaux, 1 flux) ; mixte b=8 non
bloquant, trou 8,7 µs (2 flux). H2 (fourche) scellée, micro3 non joué (priorité donnée à la contamination).

**Contamination servie (trouvée par l'A/A' du témoin au bit)** : dans un même moteur, un lot servi APRÈS un premier
lot ne rend pas les mêmes jetons que servi en premier (contam2, 4 processus : 700-2 j15, 700-5 j3, reproductible).
Pas de fuite de contenu (contam3 : Y après X = après Z = après X,X). Défaut ancien (0.6.38 cb4089950, be837ca1b).
Cause (bissect3 + trace) : `RotaryEmbedding` principal construit sans dtype (loader.py:313 → fp32) ; `_ensure`
(layers.py:721) bâtissait la table au dtype du 1er appelant et ne la reconstruisait que sur la longueur. Lot 1 :
`tables32` ← `rope_fusee` (préfill) → fp32 [1024] NON arrondi ; puis `reserver` ← 1re capture → bf16 [1153] ; lots
suivants : bf16. Écartés par mesure : créneaux GDN (bissect T2/T3), compteurs split-K (tous nuls, bissect2 T2),
adresses (decal), fourche β‖α (abflux 0/1), exec jumeau.

**Correctif 213b** (layers.py, loader.py:313) : table au dtype FIXE du module ; forward convertit au dtype de
l'appelant ; reconstruction sur dtype ; tables dérivées (cos32, demi-tables) indexées par génération ; RoPE principal
au dtype du service. collect.py : pièce 235 (poste2).
Preuves : processeur correctif 5 passed + 1 skipped (le test carte) ; carte (b4fcf07df) lot joué deux fois au bit ;
CASSANT sur main : 4 rouges processeur + test carte rouge (le 5e, chemin processeur = carte simulé en bf16, est un
contrôle de non-régression, vert sur main par construction) ; lots 2+ au bit contre main (mixte b=8 ET Coder b=8) ;
1er lot corrigé = lot 2 de main au bit. Les chiffres servis en régime établi ne changent pas ; seul le 1er lot d'un
processus change (il rejoint les suivants).
