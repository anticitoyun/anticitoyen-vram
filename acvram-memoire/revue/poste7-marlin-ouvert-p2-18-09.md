# poste7 — Porte Marlin : X = 152 TFLOPS, bande haute, intégration ouverte (scellé ≥ 15 700) ; la donnée cuBLASLt (7,86 ms) ne rouvre pas P0 mais ouvre P2 après P1 : q/k/v/o en int8 par canal, prefill par `_int_mm`, converti préparé à sec dès maintenant (18/09)

Entrée : poste3 350bdbb `verdict-marlin-p1-porte-18-09` — X = **152,1 TFLOPS** (48,8 ms/prefill contre 133,3 en situ), 0 valeur hors 2⁻⁷, .so du cache, régime en tête ; GLM 19 184 / 16 157 / 13 955 selon part fixe. cuBLASLt (tournée avant l'ordre de sauter, publiée comme donnée) : `torch._int_mm` q/k/v/o = **7,86 ms** contre bf16 20,4 (+ déquant 11,2) — ma prédiction « ≥ 16 » réfutée vers le bas.

## 1. P1 : ouvert, bande haute

poste4 intègre (`poste7-p1-porte-marlin` § 4) : `preparer_pile` au chargement, double disposition comptée par le `Plan`, `experts_layout=double`, `ACVRAM_PREFILL_GROUPED=marlin` (témoin `groupe`), test ± 2⁻⁷ + bras cassant dans le commit ; scellé in situ Coder **≥ 15 700 j/s** (formule 16 405), PPL prefill = B0 ± 0,002 ; GLM même passe, prédiction de poste4 rebasée sur T. Réserve notée : le témoin du banc (50 TFLOPS) n'est pas B0 ; le scellé en situ est le seul juge.

## 2. cuBLASLt : P0 reste fermé, P2 s'ouvre — après P1, pas en parallèle sur la carte

P0 était « à sec, sans changer les poids » : 7,86 ms exige des poids int8 **symétriques par canal**, les nôtres sont affines par groupes de 128 — c'est une conversion, donc un autre chantier. Et sa valeur change avec P1 : aujourd'hui −22 ms sur 197 (+12 %, ≈ 11 000, au bord) ; **après P1**, −22 ms sur ≈ 125 → pas ≈ 0,102 s ⇒ **≈ 20 000 j/s**, à 5 % de Marlin (20 988). C'est le geste qui suit P1, et il ne coûte qu'un converti.
* **poste2, à sec, maintenant (1 h)** : `Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c` depuis la source bf16 (pas depuis le converti : une double quantification group→canal serait un confondant) — q/k/v/o int8 symétrique par canal, tout le reste identique au classé ; manifeste, sha256, `regime_ligne()` porte `attn_int8=canal`. Prédiction PPL privé : **1,015-1,018** (int8 par canal coûte ≤ +0,3 % sur des projections d'attention ; classé si ≤ 1,020 ; > 1,020 → P2 fermé, converti gardé comme pièce).
* **poste4, après P1** : chemin prefill `_int_mm` (activation A8 par jeton, scellé déjà écrit, `ACVRAM_PREFILL_INT8=cublas`), décodage inchangé sur ces poids (le GEMV int8 lit une échelle par canal au lieu d'une par groupe — test d'équivalence contre la référence torch du converti, pas contre l'ancien converti). Scellé P2 in situ (poste3, une passe) : prefill Coder ≥ **0,95 × (2 048 / (pas_P1 − 0,0227))** avec pas_P1 mesuré dans la passe P1 ; b=1 et b=12 dans ± 2 % des cellules P1 ; PPL décodage = prefill du même converti ± 0,002.
* Réfuté sur moi (carnet) : cuBLASLt int8 à 7,86 ms = 2,6× le bf16 cutlass, j'avais écrit « pas 2× à K = 2 048 » comme issue gênante — c'est l'inverse qui est vrai.

## Ordre

* poste4 : P1 intégration (§ 1), rien d'autre.
* poste2 : converti `-qkvo-i8c` à sec (§ 2), verdict conversion → poste3 PPL 20 min dans la fenêtre P1.
* poste3 : harnais égal (en cours) ; puis passe P1 quand poste4 livre ; PPL du converti poste2 dans la même fenêtre.
* chef : ETAT — P1 ouvert bande haute (152), P0 fermé, P2 nommé « après P1 » avec ses scellés ; ligne utilisateur : « prefill Coder visé 15 700 (P1) puis ≈ 20 000 (P2), PPL classée conservée ».
