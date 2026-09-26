# Pièce 225 (poste3, 26/09, ordre chef) : relecture à sec de poste1-221 (GEMV int8 en un lancement)

Instrument : lecture de code seule, sans carte, sur `origin/poste1-221` (f5bca6fce, HEAD b35fddb8a
au moment de la relecture). Fichiers touchés : `acvram/kernels/acvram_kernels.cu` (+42/−4),
`acvram/regime.py` (+3), `acvram/cli.py` (+1), `tests/test_int8_gemv_fusion_221.py` (nouveau, 90 l.).

## (1) Ordre d'accumulation — même que la boucle hôte ?

**Oui, par construction, pas seulement au bit mesuré.** Le noyau (`int8_gemv_kernel`,
`acvram_kernels.cu:576-666`) garde EXACTEMENT la même boucle sur K et la même réduction par warp/bloc
qu'avant la 221 pour chaque sortie (row, n) réelle. Ce qui change n'est QUE l'indexation des blocs :

* `acvram_kernels.cu:588-596` : `bx = blockIdx.x ; tr = bx % ntr ; bx /= ntr ; row0 = bx * ROWS`.
  Avant la 221, `tr` (le numéro de tranche) était choisi par un LANCEMENT séparé de la boucle hôte
  (`for base in range(0, Ntot, tranche)`, `acvram_kernels.cu:1913` et suivants) ; la 221 le décode
  depuis un seul `blockIdx.x` plat. C'est un changement d'ORDONNANCEMENT des blocs (indice de tranche
  le plus rapide, pour que les blocs d'un même groupe de lignes W restent voisins dans le temps de
  lancement — le levier L2), jamais un changement de calcul : chaque bloc CUDA est indépendant, ne
  partage aucun état avec un autre bloc, et écrit dans une zone de `y` qui lui est exclusive
  (`y + tr*NV*M`, ligne 592). L'ordre de lancement des blocs ne peut donc pas changer la somme
  flottante d'UNE sortie donnée — cette somme ne dépend que du code À L'INTÉRIEUR du bloc, inchangé.
* Les deux seules lignes touchées dans le corps du calcul (`acvram_kernels.cu:668`, `685`) ajoutent
  `min(n, nloc - 1)` à la lecture et `&& n < nloc` à l'écriture — des GARDES pour la tranche
  partielle (voir (3)), pas une modification du calcul des lanes réelles (n < nloc) : pour celles-ci,
  `min(n, nloc-1) == n` (identité) et la garde d'écriture est vraie, donc le chemin exécuté est
  identique à avant.

**Conclusion (1) : au bit garanti par construction pour toute sortie réelle**, la mesure au bit du
test n'est qu'une confirmation empirique de ce que la lecture montre déjà.

## (2) Les tests peuvent-ils rendre FAUX ?

Analyse à sec (pas de carte disponible pour rejouer — à confirmer par poste1/poste6 si un doute
persiste) :

* **Ordre des tranches inversé** (ou tout mélange tr↔offset) : SERAIT capturé par
  `test_fusion_au_bit_de_la_boucle` sur tout N multi-tranches de `NS` (17, 31, 78, 80 aux deux
  tranches testées) — un bloc écrirait la mauvaise portion de `y` (`y + tr'*NV*M` au lieu de
  `y + tr*NV*M`), sortie différente de la boucle témoin, hash différent → ROUGE. **Confiance haute**,
  pas seulement le mot du docstring : la logique d'offset est directe (ligne 592-593), un mélange y
  laisse une trace arithmétique visible.
* **Garde d'écriture retirée** (`&& n < nloc` supprimée, ligne 685) : sur la DERNIÈRE tranche d'un N
  non multiple de la tranche, `y` est alloué à exactement `Ntot` lignes (`acvram_kernels.cu:1483`,
  `torch::empty({N, M}, ...)` avec `N` = `Ntot` RÉEL, jamais arrondi à `ntr*tranche`) — écrire les
  lanes de bourrage (n ≥ nloc) écrirait HORS BORNES de `y`. Deux issues possibles, toutes deux
  ROUGES : accès mémoire illégal (le sous-processus enfant plante, `r.returncode != 0`, capturé par
  `assert r.returncode == 0` dans `_empreintes`) ou corruption silencieuse d'un tenseur voisin
  (hash différent). **Confiance haute** que cette faute est attrapée, par un mécanisme ou l'autre.
* **Garde de LECTURE retirée** (`min(n, nloc-1)` supprimée, ligne 668) : ici la conclusion est
  différente. `x` est aussi dimensionné à `Ntot` lignes réelles ; sans la garde, les lanes de
  bourrage (n ≥ nloc) liraient `x` au-delà de la tranche courante, potentiellement hors des bornes
  allouées. MAIS ces lanes ne sont jamais ÉCRITES (la garde d'écriture, elle, reste en place dans ce
  scénario) : si la lecture hors bornes ne déclenche pas un accès mémoire illégal (probable sur un
  GPU moderne si elle reste dans la même allocation contiguë élargie par l'allocateur CUDA), le
  résultat garbage est calculé puis JETÉ sans jamais atteindre `y` — le hash final resterait
  IDENTIQUE. **Cette faute précise n'est pas garantie d'être attrapée** par
  `test_fusion_au_bit_de_la_boucle` : c'est un TROU du test, pas seulement une supposition à vérifier
  sur carte. Elle mériterait un test dédié (lire `nloc` et `n` fabriqués, ou un ASAN/compute-sanitizer
  sur ce chemin précis) plutôt qu'un rejeu au bit qui ne verra rien.
* **Tranche partielle entièrement ignorée** (ex. `ntr = Ntot / tranche` au lieu de
  `(Ntot + tranche - 1) / tranche`, ligne 1904) : les sorties de la dernière tranche réelle (n ≥
  ntr_faux * NV) ne seraient JAMAIS écrites — `y` vient de `torch::empty` (non initialisé), le hash
  différerait presque certainement de la boucle témoin. **Confiance haute**, attrapée.

## (3) Bornes demandées : n = 1, 6, 7, 78, 80, 81, dernière tranche partielle

`NS = (1, 2, 5, 6, 7, 8, 12, 13, 16, 17, 31, 78, 80)` dans `tests/test_int8_gemv_fusion_221.py:20`.

| n demandé | présent dans NS | tranche=6 : exact/partiel | tranche=16 : exact/partiel |
|---|---|---|---|
| 1 | oui | 1 tranche (partiel, nloc=1) | 1 tranche (partiel, nloc=1) |
| 6 | oui | EXACT (1 tranche pleine) | 1 tranche (partiel, nloc=6) |
| 7 | oui | 2 tranches, partiel (nloc=1) | 1 tranche (partiel, nloc=7) |
| 78 | oui | **EXACT** (13 tranches pleines, 0 lane de bourrage) | 5 tranches, partiel (nloc=14) |
| 80 | oui | 14 tranches, partiel (nloc=2) | **EXACT** (5 tranches pleines) |
| **81** | **ABSENT** | 14 tranches, partiel (nloc=3) | 6 tranches, partiel (**nloc=1**, 15 lanes de bourrage) |

**Trou trouvé** : n = 81 n'est testé à AUCUNE des deux tranches. Combiné (78 exact à tranche=6,
partiel à tranche=16 ; 80 exact à tranche=16, partiel à tranche=6), le couple 78/80 couvre déjà les
cas exact ET partiel aux deux tranches — mais 81 à tranche=16 est le cas le PLUS adversarial de
tous ceux demandés : `nloc = 1`, 15 lanes de bourrage sur 16, le maximum de lecture/écriture
clampée du jeu de bornes. Aucun n testé n'atteint `nloc = 1` à tranche=16 sauf n=1 lui-même (mais
alors il n'y a qu'UNE tranche, pas de deuxième tranche réelle à côté pour vérifier que l'offset
`tr*NV*K`/`tr*NV*M` de la tranche suivante ne mord pas sur le bourrage de la précédente). **81
manque et couvrirait un cas que ni 78 ni 80 ne couvrent** (nloc=1 avec ntr>1) : à ajouter.

## (4) Variable et témoin déclarés

* `acvram/cli.py:188` : `"ACVRAM_INT8_GEMV_BOUCLE"` ajoutée à `VARIABLES_LUES`, commentaire pointant
  `acvram_kernels.cu, 221`.
* `acvram/regime.py` (bloc `VARIABLES`) : `Variable("INT8_GEMV_BOUCLE", "", None, None, ...)` avec
  la description complète (1 = témoin, comportement par défaut, référence au .cu). **Présent aux deux
  endroits attendus.**

## Verdict

Le mécanisme central (indexation des blocs par arithmétique plutôt que par lancements séparés) est
au bit PAR CONSTRUCTION, pas seulement par la mesure — la lecture du code suffit à l'établir, la
prise 221 le confirme. Deux réserves nommées, à transmettre à poste1 :
1. **n = 81 absent du jeu de bornes** — à ajouter, c'est le cas `nloc = 1, ntr > 1` le plus serré.
2. **Le retrait de la garde de LECTURE (`min(n, nloc-1)`) seule n'est pas garanti d'être détecté**
   par le test au bit actuel (lecture hors bornes silencieuse, résultat jeté avant d'atteindre `y`) —
   contrairement au retrait de la garde d'ÉCRITURE, qui lui l'est avec une confiance haute (accès
   hors bornes de `y`, alloué exactement à `Ntot` lignes réelles, `acvram_kernels.cu:1483`). Un test
   dédié à cette garde précise serait plus sûr qu'un rejeu au bit qui peut rester vert malgré la faute.
