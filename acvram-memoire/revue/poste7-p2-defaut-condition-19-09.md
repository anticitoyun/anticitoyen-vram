# poste7 — P2 : quatre lignes tenues (PPL 1,0094) ; « au défaut » sous une condition que personne n'a encore mesurée : le décodage du converti i8c (19/09, 18 h 05)

Source : `verdict-p2-ppl-19-09` addendum 17 h 42 (poste2, 77ddd7d) ; `verdict-p2-moteur-19-09` ; `poste7-vibe-elimine-p2-top1-19-09` § 2.

## 1. Tenu, tel que scellé

PPL i8c `cublas` 3 tranches 1,0042 / 1,0112 / 1,0128, géo **1,0094** ≤ 1,020 (prédiction poste7 1,0094 : au 10⁻⁴ ; défaut 1,0155 ; hors-moteur 1,0096) ; équivalence B ≤ 2A + 2 ; prefill 18 850 j/s (+14 %) ; J prefill 0,89-0,93 × A. La réserve de poste2 (|déc − pré| tranche 1 : 0,156 > 2 × 0,134 sur 256 positions, somme des trois tranches tenue) est publiée telle quelle : le scellé ne disait pas « par tranche », on ne le resserre pas après ; la ligne 2 est jugée par la PPL, tenue de 0,011.

## 2. Ce que « P2 au défaut » changerait, et qui n'est pas mesuré

Le défaut servi deviendrait le converti **`-qkvo-i8c`** (q/k/v/o int8 symétrique par canal) avec `PREFILL_INT8=cublas`. Au décodage, ces quatre projections sont alors lues par `int8_gemv` : **8 bits par poids au lieu de 4** — q/k/v/o = 37,7 M paramètres × 48 couches = 1,81 G, soit 0,9 Go en NVFP4 contre 1,8 Go en int8 par jeton à b=1 ; +0,9 Go à 1 050-1 450 Go/s = **+0,6 à +0,9 ms sur un pas de 2,75 ms** (363 t/s). Prédiction poste7 : **b=1 −15 à −25 % → sous llama.cpp (340,1)** ; b=12 ± 3 % (le pas y est dominé par les experts). Issue qui gênerait poste7 : b=1 ≥ 356 → la lecture n'est pas le poste à b=1, et la règle « 4 bits ou rien au décodage » tombe.

Scellé **P2-déc** (un seuil par grandeur, harnais égal, ABAB contre le défaut classé, carte libre) : b=1 **t/s ≥ 0,98 × 363,3 = 356** ET J brut ≤ 1,02 × ; b=12 t/s ≥ 0,98 × 1 361 = 1 334 ET J net ≤ 1,02 ×. Tenu → **P2 au défaut** (`PREFILL_INT8=cublas` par défaut, alias catalogue Coder → i8c, cellules remesurées). Faux → P2 = **régime nommé « acvram P2 (i8c, cublas) »** dans le comparatif avec ses cinq cellules ; le défaut classé garde le décodage ; chantier **C11 : double disposition des projections** (int8 pour le prefill M > 16, NVFP4 pour le GEMV de décodage, +1,8 Go de VRAM seulement — pas les +14,5 Go des experts), PPL jugée sur le chemin servi (prefill i8c + décodage NVFP4 : `ppl` et `ppl-decode-kv` ≤ 1,020).

## Ordre

* **poste2** — après la porte W8r : **P2-déc** (20 min) : cellules b=1 et b=12 du converti i8c, `PREFILL_INT8=cublas`, ABAB contre le défaut, seuils § 2 ; verdict `revue/verdict-p2-decodage-19-09.md`.
* **chef** — comparatif : ligne « acvram P2 (i8c, cublas) » = PPL 1,0094 · prefill 18 850 · b=1/b=12 « à mesurer (P2-déc) » ; `ETAT` : P2 4/4 tenues, défaut conditionnel à P2-déc ; `REPRISE` § 2 : une ligne.
* **poste1** — rien de plus ; C11 s'ouvre seulement si P2-déc rend faux.
