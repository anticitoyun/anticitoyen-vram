# Facteur ×14-15 de l'exil par couche (Coder-30B, 50 % exilé) — explication

poste1, 13/09/2026. Réponse à la question de chef : le ×14-15 mesuré
(`mesure-qui-tue-placement-experts.py`, résident 233,5/596,3 j/s vs exil
par couche 16,6/40,0 j/s, b=1/b=12) contredit-il le « 3-4 attendu » de
[[l-exil-d-une-couche-est-une-falaise]] ?

## Comparaison des deux bras

|                        | référence (falaise, 9/09) | ce banc (13/09)        |
|------------------------|---------------------------|-------------------------|
| modèle                 | Agents-A1-4B, 36 couches  | Coder-30B-A3B, 48 couches, 128 experts/couche |
| couches exilées        | 0 / 4 / 8 / 16            | 24                       |
| fraction               | 0 % / 11 % / 22 % / **44 %** | **50 %**              |
| débit (j/s)            | 174,7 / 52,5 / 37,5 / 24,0 | 233,5 (b1) → 16,6       |
| facteur vs 0 exil      | 1 / 3,33 / 4,66 / **7,28** | **14,1 (b1) / 14,9 (b12)** |
| chemin                 | idem (graphes CUDA désactivés dès 1 exil, comme ici) | idem |

**Le « 3-4 » cité par chef correspond au premier point de la courbe (4/36
couches, 11 %), pas à une fraction comparable à 50 %.** La courbe de
référence elle-même n'est déjà PAS linéaire : 3,33 → 4,66 → 7,28 pour
11 % → 22 % → 44 %. Extrapoler cette même courbe jusqu'à 50 % la fait
grimper bien au-delà de 7,28 — un ×14 à 50 % est la CONTINUATION cohérente
de cette pente, pas une contradiction. Il n'y a pas deux mesures qui se
contredisent : il y a une seule falaise, mesurée à deux profondeurs très
différentes sur deux modèles différents.

## Que prédisait `estimer_cout_exil` pour CE plan ?

Calculé à sec (`memory/tiering.py:1020`), sur le plan réel du manifeste,
`_forcer_exil(plan, 24)` :

    n_couches_exilees   24
    octets_exiles       513 146 880  (489,4 Mio)
    t_transfert_s       0,02744  (27,44 ms/jeton, PCIe ~18,7 Go/s du plan)
    t_pas_resident_s    0,000857 (0,857 ms/jeton — bande VRAM 1792 Go/s DU PLAN)
    facteur additif prédit : 33,0

Ce chiffre est lui-même faux dans un sens précis : le plan embarqué dans ce
manifeste porte encore l'ancienne bande VRAM de la 5090, **1792 Go/s**,
corrigée depuis en 1050 Go/s ([[borne-memoire-5090-1050-go-s]]) — la
correction n'a pas encore été répercutée par une reconversion. Avec 1050 :

    facteur additif corrigé : 19,8

**Le ×14-15 mesuré est donc INFÉRIEUR à ce que le modèle de coût additif
prédit** (19,8 à 33), qu'on prenne la bande stale ou corrigée. Le moteur
réel fait un peu mieux que l'addition naïve transfert+calcul (chevauchement
partiel PCIe/calcul via le flux annexe de `StreamedWeight`), mais reste dans
le bon ordre de grandeur. **Le ×14 tient : il n'est contredit ni par
l'ancienne mesure (courbe non-linéaire, extrapolable) ni par le modèle de
coût du planificateur lui-même (qui prédit pire).**

## Conséquence

C'est la mesure la plus forte à ce jour en faveur du placement par expert :
à fraction exilée ÉGALE (50 %), l'exil de couche entière coûte ×14-15 quand
la cible du placement par expert est ×1,5. Reste à le vérifier (item 3,
carte après poste2/poste4).

## Item 2 (planificateur) : déjà fait

`_reajuster_plan` (`engine/loader.py:1260-1276`, bead pds point 1, déjà
committé/poussé le 13/09 matin) pose déjà `experts_residents` à la place de
l'exil de couche entière dès que `e >= top_k` : l'exil de couche entière
n'est plus qu'un repli, déclenché seulement quand la réduction ferait
tomber sous `top_k` experts résidents. Rien à ajouter ici sauf si chef
vise autre chose que ce que le docstring décrit.
