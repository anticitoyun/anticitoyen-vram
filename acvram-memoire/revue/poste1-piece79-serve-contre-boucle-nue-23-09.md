# Pièce 79 — `serve` contre la boucle moteur nue : d'où viennent les 0,35 ms/pas ? (noyaux seuls sous nsys) — 23/09 (poste1)

## Rappel (pièce 77)

Même commit, même alias (`Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c`), même horloge et même ligne de régime, octet pour
octet. `serve` décode b = 12 à **5,97 ms/pas** (`/metrics`), la boucle nue qui pilote `Engine` comme le client de
cellule à **6,34-6,35 ms/pas** (pas de décodage seuls). `frontiere-pas` rejoue le graphe en 6,305 ms. Trois
hypothèses : **(i)** l'état du processus (fil moteur sous `serve`), qui ferait des trous entre nœuds ; **(ii)** des
graphes différents (13 captures sous `serve` contre 8) ; **(iii)** la mémoire (adresses, pool).

## Protocole (≤ 3 min de carte, une prise)

Deux bras sous `nsys profile -t cuda --cuda-graph-trace=node`, même alias, même client, fenêtre de 5 s (celle de
la p59) :

* **S** : `serve` + `banc-llamacpp-16-09.py decode` ;
* **N** : `hors-noyaux-p77.py nue`, sans le crochet `keep_graph`, qui n'y servait qu'à compter les nœuds.

Lecture : `familles-comparees.py` en mode acvram-contre-acvram (fenêtres de 48 marqueurs, décodage établi,
**durées de noyaux seules** par famille), plus le mur par pas de chaque trace.

## Prédiction et issues, écrites AVANT la prise

| issue | condition (ms/pas) | ce qu'elle rend |
|---|---|---|
| **K — les noyaux diffèrent** | noyaux(N) − noyaux(S) ≥ 0,25 | (ii) ou (iii) : le même moteur n'exécute pas le même travail carte hors de `serve`. La famille qui porte l'écart nomme laquelle (graphes/godets si c'est la glue ou l'attention, mémoire si c'est le MoE ou les projections, à octets égaux) |
| **T — les trous diffèrent** | \|Δ noyaux\| ≤ 0,08 et (mur − noyaux)(N) − (mur − noyaux)(S) ≥ 0,25 | (i) : même travail, mais la boucle nue laisse la carte attendre ; c'est l'hôte |
| **R — nsys efface l'écart** | \|Δ mur\| ≤ 0,10 sous nsys | l'écart dépend d'un régime que nsys écrase ; aucune conclusion, je le dis |

* **prédiction nominale : K**, avec un écart porté par moe_gemm et l'attention. Raison : la frontière de la
  boucle nue est déjà mesurée à 30 µs (77), trop peu pour cacher 0,35 ms de trous.
* **ce qui me gênerait : T.** J'ai écrit à la 77 que la frontière de la boucle nue était saine (30 µs) : T dirait
  que les trous sont ailleurs, entre les nœuds du graphe, là où `frontiere-pas` ne regarde pas.
* **si « oui, les bancs Engine sont pessimistes »** (K ou T avec un écart ≥ 0,25 ms/pas) : liste des chiffres publiés
  qui en dépendaient, tirée du dépôt (README, ETAT, verdicts), pas de mémoire.
