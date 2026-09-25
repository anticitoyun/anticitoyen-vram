# Pièce 197 — verdict (poste5, 25/09) : test_multi_marlin_146 compte les octets demandés (suite de la 191)

Ordre de chef : bead anticitoyen-vram-ed2, même correctif que la 191 pour `tests/test_multi_marlin_146.py:30-34`,
avec ordre rouge avant, vert après, bras cassant ; puis les autres tests qui bornent `memory_allocated` sous 1 Mio.

## Défaut
Borne `apres − avant < 0,1 · naturelle` = 0,675 Mio, lue par `memory_allocated` (blocs). Les allocations Marlin de
2 Mio (grand pool) prennent un bloc libre de 3 Mio sans le découper (reste ≤ 1 Mio, `kSmallSize`) : +1 Mio compté
selon l'état laissé par les tests précédents. Octets demandés mesurés : −24 600 o (prise diag197).

## Preuves (prise 1, sous verrou) — ordre rouge FABRIQUÉ (`rouge197.py`) : 16 blocs libres de 3 Mio entre des blocs
tenus de 1,25 Mio, posés après les poids naturels. Bras cassant (`casse197.py`) : échelles naturelles gardées (0,75 Mio).
| version | seule | ordre rouge | cassant | rouge + cassant |
|---|---|---|---|---|
| ancienne (`memory_allocated`) | vert | **rouge +1,0 Mio** | rouge +0,7 | — |
| corrigée (`requested_bytes`) | vert | vert | rouge +0,7 | rouge +0,7 |
L'ordre rouge est construit, pas trouvé dans la suite : il reproduit l'état d'allocateur de la 191, le vrai.

## Autres tests (grep `memory_allocated|memory_reserved|mem_get_info` dans tests/)
* `test_pool_dense_exil.py:84-87` : borne BASSE (`≥ 2 × 2 Mio`). L'arrondi par bloc ne fait que gonfler : il ne
  peut pas rendre ce test rouge à tort, et une seule copie (2 Mio, au plus 3 Mio comptés) reste sous 4 Mio, donc il ne
  peut pas non plus le rendre vert à tort. Laissé tel quel.
* `test_graphes_memoire_avant_capture.py`, `test_marge_kv_unique.py`, `test_garde_kv_146.py`, `test_tete_liee_146.py` :
  valeurs posées par monkeypatch, pas de mesure. `conftest.py:213` : lecture de mémoire libre, aucune borne.
