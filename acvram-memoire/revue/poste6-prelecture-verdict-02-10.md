# B2 — prélecture des fragments au chargement : à froid 27,2 s → 4,9 s pour 17,6 Go (5,5 ×), à chaud +0,05 s, tenseurs identiques à l'octet ; défaut ACTIF

instrument : `scratchpad/poste6-076/mesure-chargement.py` (lecteur d'acvram `_ShardReader`, chaque tenseur lu puis jeté, ABBA) et `mesure-disques.py` (modes `partamp3`, `readahead`) ; diagnostics, pas des instruments de cellule
commit : poste6-gemma-anneau d4ddd2ce1 (code + tests, isolé : se reprend seul par `git cherry-pick`), scellé 552076856 poussé avant la première mesure
régime : sans carte (`CUDA_VISIBLE_DEVICES=""`), sans sudo ; cache de pages évincé par `posix_fadvise DONTNEED`, 0,0 % résident vérifié par `mincore` avant chaque bras ; NVMe FN960 ; carte tenue en `service` par poste2 (llama.cpp), autres lecteurs du disque comptés par bras : 0 % ; charge 2,3-2,7
scellé : `poste6-prelecture-scelle-02-10.md` (P1-P5)
mesuré : 4 bras à froid + 4 à chaud sur le modèle entier (Coder-30B nvfp4, 17,6 Go, 56 117 tenseurs), 7 lots de débit
verdict : **P3 TENU — 27,2 / 27,1 s sans, 4,9 / 4,9 s avec (0,65 → 3,6 Go/s, 22 s gagnées)** ; **P4 tenu** (à chaud 1,18 s contre 1,24 s : +0,05 s) ; **P5 tenu** (identité à l'octet, test) ; P1 FAUX par excès de prudence (8 fils : 3,45 Go/s pour 1,5-2,1 prédits) ; P2 tenu, un cheveu au-dessus (3,31-3,54 pour 2,5-3,2). Défaut : `ACVRAM_PRELECTURE=1`.
durée : 11 min du scellé au commit du code (11 h 18 → 11 h 29, horodatage git), dont 117 s de suite ciblée ; 0 min de carte

## Prédit / mesuré

| | prédit | mesuré | |
|---|---|---|---|
| P1 `pread` tamponné à froid, médiane de 3 : 2 / 4 / 8 / 16 fils | 1,2-1,6 / 1,3-1,9 / 1,5-2,1 / ≤ +15 % | 0,79 / 2,57 / **3,45** / 3,70 Go/s (bloc 16 Mio) ; 8 fils en bloc de 1 Mio : 2,08 | FAUX : sous-estimé à 4 fils et plus, surestimé à 2 |
| P2 `readahead(2)` par pas, jusqu'à 100 % en cache | 2,5-3,2 Go/s | pas 256 Kio : **3,31** (1 fil), 3,36 (4 fils) ; pas 128 Kio : 3,54 | tenu, au-dessus de la fourchette |
| P3 lecteur d'acvram à froid, modèle entier | sans 25-29 s ; avec ≤ 12 s | sans **27,2 / 27,1 s** ; avec **4,9 / 4,9 s** (prélecture finie en 4,8-4,9 s) | tenu ; seuils du défaut (≥ 1,5 ×, ≥ 5 s) : 5,5 ×, 22 s |
| P4 à chaud | surcoût ≤ 0,3 s | 1,17 / 1,19 s sans ; 1,24 / 1,23 s avec (prélecture finie en 0,23 s) | tenu : +0,05 s |
| P5 identité | tenseurs égaux, sha256 inchangés | `tests/test_prelecture_b2.py` : égaux à l'octet, fragments inchangés | tenu |

## Ce qui a été choisi, et pourquoi

`readahead(2)` par pas de 128 Kio, UN fil, plutôt que 8 fils de `pread` : même débit (3,3-3,6 contre 3,45 Go/s), aucune
copie vers le processus, un seul fil à côté du chargeur. Le pas vaut `read_ahead_kb` par défaut : le noyau borne chaque
appel à une E/S (c'est pourquoi `fadvise WILLNEED` sur 4 Go ne lisait rien dans le verdict 076-3), un pas plus grand
sauterait des plages sur un disque réglé plus bas. Le fil tourne pendant le chargement ; le chargeur trouve ses pages en
cache ou en vol. Coupée d'elle-même quand les fragments dépassent 80 % de la RAM disponible.

## Corrections au verdict 076-3, dites

* Les passes uniques de lecture tamponnée parallèle que j'y donnais (1,46 / 1,20 / 1,80 Go/s) étaient 2 × trop basses à
  8 fils : la médiane de 3 rend 3,45. Deux causes relevées depuis : la campagne de poste2 charge un modèle llama.cpp
  depuis ce même NVMe toutes les quelques minutes, et le débit d'UN flux tamponné varie de 0,37 à 0,87 Go/s d'un lot à
  l'autre sans autre lecteur. Le script compte désormais les octets lus par d'autres pendant chaque commande.
* Plafond du disque : 3,6-3,7 Go/s (O_DIRECT 8 et 16 fils rejoués : 3,63 / 3,73), non 3,2.
* « B2 : 1,80 Go/s, ≈ 18 s gagnées » devient 3,6 Go/s, 22 s : B2 fait jeu égal avec B1 (O_DIRECT) sans en payer le prix.

## Non mesuré

Le temps de chargement RÉEL (copie vers la carte, quantification de la tête, capture) : à ma fenêtre de carte, en lisant
l'horodatage du service avec `ACVRAM_PRELECTURE=0` puis `1`, cache évincé. Disque dur et USB : un fil de prélecture à
côté du chargeur peut y coûter des déplacements de tête — aucun converti acvram n'y vit, non mesuré.
