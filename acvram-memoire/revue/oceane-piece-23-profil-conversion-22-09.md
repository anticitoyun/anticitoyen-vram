# Pièce 23 — profil de la conversion : la recherche d échelle pèse 99 %, la vectoriser ne rend rien, et 85 % de cette recherche ne sert à rien (22/09, Océane, à sec)

## 0. Ce que les journaux existants ne permettent pas
Le journal par tenseur (`convert._journal_tenseurs`, `convert.py:680-699`) n est
actif que sur processeur ou sous `ACVRAM_JOURNAL_TENSEURS=1`
(`convert.py:686-690`), et aucune conversion de carte du 21-22/09 ne l a activé.
Le seul journal complet retrouvé — 30B-VL,
`manon-w-21-09/scratchpad/qvl30b-reconv-identite-22-09.log` — donne la durée
totale (**475,4 s**, 18 529 tenseurs, soit 25,7 ms par tenseur) et des lignes
de progression **sans horodatage**. Le profil par phase ne s en reconstruit
pas ; il se mesure. Instrument : `outils/profil-conversion.py`.

## 1. Mesure (processeur, 2 cœurs, nice 19, formes réelles du Coder-30B)
| forme | × | recherche | quant max6 | quant 4sur6 | sérialisation | part recherche |
|---|---|---|---|---|---|---|
| [768, 2048] | 12 288 | **697,7 ms** | 3,1 | 7,0 (+3,9) | 0,6 | 99 % |
| [2048, 768] | 6 144 | **277,5 ms** | 3,3 | 8,8 (+5,5) | 0,2 | 99 % |
| [2048, 4096] | 48 | 4 689 ms | 23,2 | 78,4 (+55,2) | 0,7 | 99 % |
| [4096, 2048] | 48 | 2 514 ms | 19,5 | 62,1 (+42,6) | 0,9 | 99 % |
Non mesuré, jamais extrapolé : `lm_head` [151 936, 2048] (1 tenseur).
**Parts : recherche 99 %, quantification 1 %, sérialisation 0 %.**
**Four Over Six coûte +0,81 %** du temps total — le second candidat par bloc
est ~2× la quantification, et la quantification est 1 % : le débat sur son prix
de conversion est clos, il n en a pas.
Contrôle intégré : l extrapolation par lignes a été REFUSÉE par son propre
garde (la moitié des lignes coûte 0,8 fois le tout, pas 0,5, sur processeur) —
les quatre formes sont donc mesurées entières, et `lm_head` est nommé plutôt
qu estimé.

## 2. Vectoriser la grille : RÉFUTÉ avant d écrire le code
Prototype à sec (même arithmétique, 21 candidats empilés en [21·N, K] et une
seule quantification contre 21) :
`[768, 2048]` boucle 329,4 ms contre lot 382,0 — **−16 %** ; `[2048, 768]`
252,1 contre 381,5 — **−51 %**. Le travail arithmétique est identique et la
matérialisation du lot coûte la bande passante en plus. La vectorisation n est
pas le levier ; elle aurait été écrite pour rien.

## 3. Le levier réel, trouvé en cherchant le premier : 85 % de la recherche est prouvée sans effet
`search_channel_scales` (`calibrate.py:322-327`) prend `act = ones` quand
`stats is None`. Or `s = act^alpha / mean(act^alpha)` vaut **1 pour tout
alpha** : les 21 évaluations de la grille sont alors **identiques**, et leur
résultat est connu d avance — l identité (`allclose(best_scale, identity)` à la
fin). Aucun court-circuit n existait : `quantize_with_calibration:595-598`
appelle la recherche dès que `use_awq`, sans regarder `stats`.
Sur le 30B-VL du 22/09, **5 235 experts sur 6 144 sont sans statistique** : avec
trois projections chacun, ~85 % des tenseurs quantifiés parcouraient 21
quantifications complètes pour rien.
Correctif (même commit) : une évaluation au lieu de n_grid+1 quand la
magnitude par canal est constante, journal rempli de la même valeur pour que
rien ne change en aval. **Au bit** : c est la même évaluation, celle d alpha = 0.
Gain attendu sur une conversion dominée par des experts non routés : la part
« recherche » de ces tenseurs divisée par 21, soit **jusqu à ×5 à ×6 sur le
temps total** d une conversion comme celle du 30B-VL (475 s), et rien du tout
sur une conversion dont tous les experts sont calibrés — les deux régimes
doivent être distingués dans la mesure de Manon.

## 4. Extrapolation au 119B, et ce qu elle vaut
Le 119B n est pas sur ce disque : pas de manifeste, donc pas de comptage réel.
Ce qui est solide, c est le rapport de proportion — la recherche est ~21 fois
la quantification **par construction** (n_grid + 1 quant/déquant contre un),
indépendamment de la machine : la part de 99 % mesurée ici tient partout où les
deux phases s exécutent au même endroit. L extrapolation en secondes, elle, ne
tient pas : la mesure est CPU à 2 cœurs, la conversion réelle est sur carte
(25,7 ms/tenseur mesurés contre ~700 ms ici, soit ~27×). Le chiffre à publier
pour le 119B est donc un RAPPORT, pas une durée, tant que personne n a converti
un 119B avec `ACVRAM_JOURNAL_TENSEURS=1`.

## 5. Ce qui rendrait faux
* une conversion réelle où la part « recherche » tombe sous 50 % → la mesure
  CPU ne représente pas le chemin carte, et le § 3 perd sa portée (le gain
  resterait, mais plus petit) ;
* une sortie différente d un bit entre avant et après le raccourci → le
  critère `act` constant est mal posé (test `test_grille_plate_awq.py`) ;
* un gain mesuré < ×2 sur une conversion majoritairement non calibrée → le
  temps n était pas là où la mesure CPU le place.
Tests à sec : `tests/test_grille_plate_awq.py`, 4 verts — égalité au bit avec
l évaluation alpha = 0 calculée à la main, magnitude constante non unitaire,
refus du raccourci dès que la magnitude varie, et gain ≥ 5× vérifié.
