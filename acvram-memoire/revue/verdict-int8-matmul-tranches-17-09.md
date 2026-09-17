# Correctif — repli GEMM d'`int8_matmul` par tranches : la tête de Gemma-4-31B déquantifiée d'un coup (5,25 Gio) tombait en OOM au premier préfill (poste3 0cf7fe6)

## Bogue : fichier:ligne

`acvram/kernels/__init__.py` `int8_matmul` : le noyau int8 n'est pris que
pour `n ≤ gemv_threshold` (80) ; au-delà, le repli déquantifiait la matrice
ENTIÈRE (`int8_dequant(t, …)` puis `F.linear`). Sur la tête de Gemma-4-31B
(262 144 × 5 376), `ext.int8_dequant` passe par un intermédiaire fp32 :
262 144 × 5 376 × 4 = **5,25 Gio** d'un coup, exactement l'allocation qui
échouait, après un chargement juste (exil 10/60, prédiction poste7 tenue).

## Correctif (poste4)

Quand `lignes × K × (4 + itemsize)` dépasse `_DEQUANT_TRANCHE_MAX`
(`ACVRAM_DEQUANT_TRANCHE_MAX`, défaut 256 Mio, hors régime), la matrice est
déquantifiée par tranches de lignes de sortie (multiples de 64) et chaque
tranche donne ses colonnes de la sortie. Même arithmétique : chaque tranche
est la même matrice restreinte, la sortie est identique au bit. Pic borné
par le plafond au lieu de la taille de la matrice ; sur la tête de Gemma :
≈ 21 tranches de 12 480 lignes.

Juge : `tests/test_int8_matmul_tranches.py` — 1 000 lignes en tranches de
128 (7 × 128 + 104, comptées), sortie `torch.equal` à la matrice entière ;
bras qui doit différer : une tranche mise à zéro se voit. Suite 780 passed.

Non fait : compter la tête dans la réserve de préfill (`_reserve_prefill`
ajoute déjà la plus grosse matrice déquantifiée des COUCHES) — avec les
tranches, le pic de la tête est 256 Mio, sous la marge.
