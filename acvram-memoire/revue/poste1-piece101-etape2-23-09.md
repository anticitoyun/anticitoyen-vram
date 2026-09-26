# Pièce 101, étape 2 — Marlin dense porté (échelle par colonne) : équivalence TENUE, port fidèle, mais débit Engine PERDU aux godets 2-16 — 23/09 (poste1)

* **instrument** : `scratchpad/poste1-p101-23-09/prise-etape2.sh`, prise unique :
  * tests sur carte ;
  * `banc-proj-nvfp4.py` (fidélité du port) ;
  * `vram-marlin.py` ;
  * `capture-godets.py` à godets 1/2/4/8/16, avec PROJ_MARLIN=0 puis 1 ;
  * `equiv-bras.py` et `equiv-compare.py` : critère (a)+(b)+(c) tranché par le chef.
* **commit** : a6d68583 ; .so Marlin 47ac501c (cache isolé).
* **régime** : -lgc 2700 pendant le banc seulement ; seul llama-server 4627 au début et à la fin ; prise de 19:24:32 à 19:27:14.
* **scellé** : `scelle-etape2.md` et son addendum (a6d68583), écrits avant.
* **alias** : `Qwen3-Coder-30B-A3B-nvfp4-qkv-alpha2-22-09`, celui du scellé ; l'alias retenu par la 100 B (alphaqkv-23-09) est venu après.
* **mesuré** :

| # | contrôle | prédit | mesuré | |
|---|---|---|---|---|
| 1 | tests sur carte (test_marlin_dense + port existant) | 8/8 + verts | 43 passés, 11 ignorés, 0 échec. **Le décompte par fichier n'a pas été relevé** : que les 8 tests du dense aient tourné plutôt que d'être ignorés n'est pas prouvé | à confirmer |
| 2 | fidélité du port contre le Marlin vLLM (M 1-12) | ±5 % | qkv 7,13-7,76 contre 7,08-7,76 ; o 8,60-9,26 contre 8,61-9,62 (M=12 : −3,7 %) | **tenu** |
| 3 | capture, 5 godets | 5/5 ok | 5/5 ok, sans repli | tenu |
| 4 | VRAM de la 2ᵉ disposition | 500-520 Mo | **492,7 Mo** ; libre après : 9 884 / 32 111 Mo | à −1,5 % |
| (c) | b=1 au bit, C1 sur 5 invites | au bit | **au bit 5/5** | **tenu** |
| (b) | KL appariée C3/C5/C6 | tenu, ≤ 0,02 | **tenu 15/15** ; le pire est invite 3 C3, 0,1097 contre 0,0731, sous la marge de 0,05 | tenu |
| — | rejeu au bit dans chaque bras | oui | oui | valide |

* **Débit en capture (ms/pas, Engine, horloge libre, PROJ_MARLIN=0 → 1)** : godet 1 3,126 → 3,130 · **godet 2 4,595 → 5,747 (+25 %)** · godet 4 5,691 → 6,425 (+13 %) · godet 8 6,809 → 7,490 (+10 %) · godet 16 8,882 → 9,697 (+9 %).
* **verdict** :
  1. L'équivalence est **tenue** : (c) au bit à b=1, (b) KL appariée 15/15, (a) couvert par le test 1 si le point 1 est confirmé.
  2. Le port est fidèle.
  3. **« Débit ≥ neutre à chaque godet » : FAUX**, de +9 à +25 % sur le pas en Engine. La prédiction du scellé est réfutée.
* **Cause, lue dans les compteurs (hypothèse, non prouvée)** :
  * La passe d'équivalence compte **190** appels `marlin_dense` par passe de décodage, contre les **96** attendus (qkv + o × 48). Le chemin opt-in attrape donc **tous** les linéaires NVFP4 denses de 2 à 16 lignes, et pas seulement les projections d'attention mesurées au banc.
  * Parmi eux sans doute le routeur (N = 128) et d'autres formes étroites, où le Marlin a un plancher d'environ 7 µs, bien au-dessus de leur chemin actuel.
  * +1,15 ms/pas au godet 2, c'est ≈ +24 µs par couche, soit l'ordre de deux appels Marlin inutiles par couche.
  * Le banc (−4 µs/couche sur qkv+o) ne contredit pas ce débit. Il ne mesurait que ces deux formes.
* **Correctif proposé (une prise de ≈ 3 min, au chef)** :
  * restreindre `_marlin_dense` aux formes mesurées, par une éligibilité N ≥ 1 024 et K ≥ 1 024, ou par le nom (`self_attn`) ;
  * compter les appels par forme ;
  * rejouer la capture à PROJ_MARLIN=0/1 et relever les tests par fichier.
  * Prédit : 96 appels par passe, pas au godet 2 ≤ témoin − 0,15 ms.
* L'opt-in reste **désactivé par défaut** ; rien n'est servi.

## Prise corrective (feu du chef) — 19 h 29, commit 9cbbf58a, scellé `scelle-correctif.md`
* **correctif** : `_marlin_dense` n'accepte plus que N ≥ 1 024 et K ≥ 1 024 (`ACVRAM_PROJ_MARLIN_MIN_NK`) ; compte par forme. Prise de 70 s, seul 4627 au début et à la fin.
* **mesuré** :
  * `test_marlin_dense.py` relevé un par un : **8/8 passés**, ce qui lève le point 1 de l'étape 2.
  * Appels par passe (b=4, eager) : **96**, soit 47 × q [4 096 × 2 048], 48 × o [2 048 × 4 096] et 1 × qkv [5 120 × 2 048].
  * ms/pas, PROJ_MARLIN 0 → 1, même prise : godet 2 4,620 → 4,695 (+1,6 %) · godet 4 5,699 → 5,533 (−2,9 %) · godet 8 6,813 → 6,616 (−2,9 %) · godet 16 8,859 → 8,685 (−2,0 %).
* **verdict** : **les deux seuils sont TENUS** (96 appels ; ≤ témoin + 3 % à chaque godet).
  * L'hypothèse est confirmée : la perte de l'étape 2 venait des linéaires étroits pris à tort.
  * Le gain est modeste : −2 à −3 % du pas aux godets 4-16, neutre au godet 2.
* **Fait nouveau, qui explique la modestie du gain** : sur cet alias (alpha2-22-09), q/k/v ne sont **pas empilés** dans 47 couches sur 48.
  * Seul `q` [4 096] passe au Marlin ; k et v (N = 512, sous le seuil) restent sur l'ancien chemin, en deux appels.
  * Le banc supposait un qkv empilé [5 120].
  * Cause probable : des scalers AWQ différents entre q, k et v, qui bloquent la fusion (`layers.py`, « scalers differents entre projections »).
  * L'alias A de la 100 B (alpha COMMUN q/k/v) devrait empiler q/k/v, donc passer par le chemin à échelle par colonne et rendre le gain du banc. À vérifier par le même compte (on attend 48 × 5 120 × 2 048) avant la cellule HTTP de poste2.
* L'opt-in reste à 0 par défaut jusqu'à la décision du chef.
