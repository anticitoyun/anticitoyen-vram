# Prédiction scellée — témoin CPU du multiplicateur ulp (eager vs sdpa)

poste1, 15/09/2026, avant mesure. poste7 §3 : le multiplicateur « 2 ulp »
est négocié, pas mesuré — à recaler sur un témoin référence-contre-
elle-même (deux implémentations LÉGITIMES de la même référence), pas
nous. Premier témoin, à sec (pas de carte) : HF bf16, `attn_implementation`
`eager` vs `sdpa`, CPU, mêmes 16 positions (`mini-hf`, mêmes jetons que
l'équivalence GLM).

## Prédiction

Le plancher (max sur les 16 positions du delta/ulp, échelle =
max_j|logit_eager,j| de chaque ligne) est > 0 ulp — deux ordres de
sommation différents (eager boucle explicite, sdpa noyau fusionné) ne
peuvent pas être bit-exacts en bf16 — mais reste MODESTE, probablement
sous 4-6 ulp, cohérent avec l'ordre de grandeur déjà observé côté acvram
(0,77-3,45 ulp sur les 15 positions non-ex-aequo).

**Seuil de réfutation** : si le témoin ne dépasse JAMAIS 1 ulp sur aucune
des 16 positions, l'instrument est aveugle (poste7 : « le témoin doit
lui-même dépasser 1 ulp quelque part, sinon il ne prouve rien ») — il
faudrait un témoin plus sensible (GPU vs CPU, ou un vrai token différent)
avant de recaler quoi que ce soit sur ce seul résultat.

## MESURÉ : CONFIRMÉ, prédiction dans la fourchette basse

HF bf16 CPU, `mini-hf`, mêmes 16 jetons, `attn_implementation="eager"`
vs `"sdpa"` — les deux tournent sans erreur (l'instrument n'est PAS
aveugle : top-1 identique aux 16 positions, mais delta non nul dès
qu'un ordre d'accumulation change).

| position | delta | delta/ulp |
|---|---|---|
| 0 | 0,0000 | 0,00 |
| 1 | 0,1875 | 1,50 |
| 2 | 0,1250 | 1,00 |
| 3 | 0,2500 | **4,00** |
| 4 | 0,1875 | 3,00 |
| 5 | 0,1250 | 1,00 |
| 6 | 0,1250 | 1,00 |
| 7 | 0,1406 | 1,12 |
| 8 | 0,1875 | 1,50 |
| 9 | 0,2500 | 2,00 |
| 10 | 0,1250 | 1,00 |
| 11 | 0,1250 | 1,00 |
| 12 | 0,1250 | 1,00 |
| 13 | 0,2500 | 1,00 |
| 14 | 0,2500 | 2,00 |
| 15 | 0,1328 | 2,12 |

**Plancher témoin = 4,00 ulp** (position 3) — dans la fourchette prédite
(4-6), au bas de la fourchette. Deux implémentations LÉGITIMES de la
MÊME référence HF (boucle explicite vs noyau fusionné, bf16, CPU)
divergent déjà de 4 ulp sur ce jeu de 16 jetons — **avant même de
comparer à acvram**.

**Conséquence directe (poste7 §3, "seuil = plancher témoin + 1 ulp")** :
le multiplicateur correct est **5 ulp**, pas 2. Recalculé avec ce seuil
sur `mini-acvram4` vs HF (revue/equivalence-glm-14-09.md) : les positions
3, 4, 5 (2,05-2,15 ulp) et 15 (3,45 ulp) passent TOUTES sous 5 ulp — seule
la position 0 (14,52 ulp, l'ex-aequo réel) reste en échec sur le critère
delta, à traiter par la voie ex-aequo séparément. **Nouveau total : 15/16
passent**, le seul point encore ouvert étant la position 0.

Pas encore fait (poste7 demandait aussi un témoin GPU vs CPU, 5 min de
carte, après poste3 dans la file) : ce second témoin pourrait confirmer
ou dépasser ce plancher de 4 ulp — le seuil 5 ulp reste provisoire tant
que ce second témoin n'est pas mesuré.
