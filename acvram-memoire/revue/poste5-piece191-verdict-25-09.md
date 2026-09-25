# Pièce 191 — verdict (poste5, 25/09) : le test d'octets 179 dépendait de l'ordre par l'ALLOCATEUR, pas par un état du moteur

Ordre de chef : `test_octets_retenus_dans_la_reserve` (179) rouge dans la suite complète, vert seul ; trouver l'état
hérité en fichier:ligne, rendre le test indépendant de l'ordre sans l'affaiblir, prouver par la suite complète sous verrou.
Branche `poste5-191` (depuis origin/main 0c54811f2). Prises : `scratchpad/poste5-p191-25-09/prise{1..5}.{sh,txt}`.

## Reproduction (prise 3)
À b6c0d2c54 (commit de l'échec de la 187), les 67 fichiers de tests jusqu'à 179 inclus, ordre de collecte :
rouge 3/3, `AssertionError: (232718336, 231669760)` — **+1 048 576 o exactement** au-dessus de la borne. À 0c54811f2
(69 fichiers, deux tests de plus avant 179) : vert, et la suite complète aussi (prise 2) — la faute était latente, pas
réparée.

Écartés par le greffon `diag191.py` à l'entrée du test (ordre rouge) : ramasse-miettes (gc.collect = 0 objet, 0 o CUDA),
fils survivants (seul celui de pytest-timeout), `kernels._W_PARTAGES` (None), `_INT8_GEMV_MAX` (80). Aucun des sept
fichiers qui touchent `_DEPAQ_PARTAGE`, `_INT8_GEMV_MAX` ou la portée ne rend le test rouge placé seul devant (prise 1).

## Cause
La borne `ModelSpec.poids_bf16_couche_lineaire_bytes()` (acvram/engine/config.py:404) vaut 231 669 760 o = la somme
EXACTE des cinq poids bf16 (qkv 100 Mio, gate 60, alpha et beta 480 Kio chacun, out 60) : zéro marge, à dessein.
`torch.cuda.memory_allocated` compte les BLOCS de l'allocateur, pas les octets demandés : un bloc libre du grand pool
n'est pas découpé quand le reste fait au plus 1 Mio (`kSmallSize = 1048576`, c10/core/AllocatorConfig.h:21 dans le
torch 2.14 du venv) et l'allocation garde le bloc entier. Ce reste dépend des segments laissés en cache par les tests
précédents : c'est l'état hérité. Il n'est pas dans le code du dépôt ; c'est la disposition des segments de l'allocateur,
qui dépend de l'ordre.

## Correctif (tests/test_depaq_int8_179.py)
`retenu = dedans − après` sur `torch.cuda.memory_stats()["requested_bytes.all.current"]` (octets demandés), mêmes points
de mesure que la version 614bc3e46 (sortie de la portée).

## Preuves (prise 4, sous verrou)
| bras | ordre | résultat |
|---|---|---|
| corrigé | rouge de b6c0d2c54 (67 fichiers) | 324 verts |
| corrigé + 512 o retenus dans la portée (`casse191.py`) | idem | rouge `(231670272, 231669760)` |
| corrigé, seul | HEAD | vert |
| corrigé + 512 o, seul | HEAD | rouge `(231670272, 231669760)` |

Retenu demandé = 231 669 760 = la borne, au bit : 512 o de plus suffisent à le faire rougir. Le test n'est pas affaibli.
Suite complète à 586db124f sous verrou (prise 5, 12 h 19-12 h 26) : **2 874 verts, 0 rouge**, 179 PASSED.

## Hors 179, sur la même branche (ordres de chef)
* 5132e274e — cliquet `PLAFOND_CHEMINS` 2 176 → 277 (purge) : rouge dans la suite de la prise 2, 5/5 après.
* 194c1cbd6 — témoin `.pt` de `test_les_fichiers_binaires_suivis_sont_ecartes` fabriqué dans un dépôt git jetable ;
  bras cassant (`_EXTENSIONS_BINAIRES` vidé) rouge `:316`.

## Reste
`tests/test_multi_marlin_146.py:30-34` mesure aussi par `memory_allocated`, marge 0,1 · naturelle ≈ 0,63 Mo < 1 Mio :
même défaut latent → bead `anticitoyen-vram-ed2`.

LEÇON : un test de mémoire à borne exacte mesure les octets demandés (`requested_bytes`), jamais les blocs
(`memory_allocated`) : l'écart atteint 1 Mio par allocation et dépend de l'ordre des tests.
