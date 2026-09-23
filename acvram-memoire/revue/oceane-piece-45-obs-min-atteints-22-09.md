# Pièce 45 — le garde obs-min comptait la mauvaise population : facteur sur les experts ATTEINTS, non-atteints renvoyés à la 27 (22/09, Océane, à sec)

Ce qui s est passé : le garde a refusé la reconversion de Manon — 3 522 experts
sur 5 369 sous 512 observations, facteur annoncé **×512**. Ce facteur venait de
`obs_min / minimum`, et le minimum valait 0 ou 1 : il était fixé par des
experts que le corpus **n atteint pas**, pas par un manque de volume. Aucun
corpus raisonnable n aurait levé le refus, et la conversion refusée n avait
rien de mauvais.

## 1. Les trois populations, désormais séparées (`quant/collect.py`)
* **jamais routés** (0 jeton) et **quasi jamais** (1 à 7) : sous
  `MIN_ECHANTILLONS_AWQ = 8` (`convert.py:1699`), ces experts ne reçoivent
  **aucune** échelle AWQ, quelle que soit la taille du corpus. Comptés à part,
  publiés (`jamais_routes`, `quasi_jamais_routes`, `part_non_atteints`) et
  renvoyés à la **pièce 27** : c est du routage, pas du volume.
* **atteints** (≥ 8) : eux seuls portent le jugement. Critère = leur **p10**
  ≥ `obs_min`, facteur = `obs_min / p10_atteints`.
`SEUIL_EXPERT_ROUTE = 8` doit rester égal à `MIN_ECHANTILLONS_AWQ` — un test
lit `convert.py` pour le vérifier, sinon le rapport compterait « atteint » un
expert que la conversion laisse à l identité, et le facteur mentirait encore.

## 2. Ce que le rapport imprime maintenant
`jetons_pour_p10` (le total qu il faudrait) et `minutes_pour_p10` (au rythme
mesuré de la collecte qui vient d avoir lieu — le chronomètre est dans
`collect_activation_stats`). Le refus dit la commande à recopier :
`--corpus-jetons N`, nouvelle option qui fixe le TOTAL et dérive `--calib-seqs`
de `--calib-len` (défaut inchangé : 32 × 512 = 16 384 jetons).

## 3. Corpus MoE par défaut : ce qu il faut pour décider, que je n ai pas
Relever le défaut demande une mesure que personne n a encore : le temps de
collecte est linéaire en jetons, mais son coefficient dépend du modèle. La
règle du dépôt est de ne pas changer un défaut de conversion sans mesure ; je
ne l ai donc **pas** changé. La décision se prend avec un seul chiffre, celui
que la prochaine conversion imprimera : **si `minutes_pour_p10` ≤ 30 sur le
30B, le défaut MoE peut passer à `jetons_pour_p10`** (arrondi au multiple de
`calib_len` supérieur) ; au-delà, c est le corpus qu il faut changer, pas sa
taille — un corpus qui route mieux vaut mieux qu un corpus quatre fois plus
long (les non-atteints, eux, ne bougeront pas).
Repère disponible : 25(a) a mesuré « 3 collectes de 100 / 1 000 / 10 000 jetons
en 2 à 6 min chacune » sur le 30B, soit un ordre de grandeur de ~1 min pour
2 000 jetons — à ce rythme, ×4 (65 536 jetons) coûterait ~30 min de collecte
seule. C est un repère, pas une mesure : le chiffre qui décide est celui que le
rapport imprimera.

## 4. Ce qui rend « faux »
* le facteur redevient ≥ 100 alors que la médiane des atteints est haute → la
  population « atteints » est mal définie (relire `SEUIL_EXPERT_ROUTE`) ;
* `suffisant` vrai alors que des experts atteints restent sous 512 → le p10 est
  trop indulgent, il faudrait un quantile plus bas (p05) ou le minimum des
  atteints ;
* `minutes_pour_p10` s écarte d un facteur 2 du temps réellement constaté à la
  relance → l extrapolation linéaire est fausse, et c est elle qu il faut
  corriger avant de toucher au défaut.
Tests à sec : `tests/test_obs_min_experts.py`, 9 verts — dont le cas exact du
22/09 (30 experts quasi jamais routés + 20 atteints à 600 : `suffisant` vrai au
lieu d un refus ×512), les jetons et minutes nécessaires, et l égalité des deux
seuils lue dans `convert.py`.
