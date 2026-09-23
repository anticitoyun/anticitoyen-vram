# Pièce 79 — `serve` contre la boucle moteur nue : d'où viennent les 0,35 ms/pas ? (noyaux seuls sous nsys) — 23/09 (Océane)

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

## Verdict — 23/09 09 h 5x (Océane)

* **instrument** : `scratchpad/oceane-p79-23-09/prise.sh` (S : `nsys launch/start/stop` autour du client ; N : `nsys profile` de la boucle nue, sans `keep_graph`) ; lecture sur le **plus long train continu de décodage** de chaque trace (coupé à toute fenêtre > 1,5 × la médiane : préfill, fin de lot, changement de godet de blocs), `familles-train.json`
* **commit** : fusion de main du 23/09 09 h 5x dans `oceane-alpha-experts` ; alias `Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c`
* **régime** : -lgc 2700, horloge 2 654 (client) ; sous nsys : servi 1 796,4 t/s, nu 1 758,4 ; compute-apps début = fin = llama-server 4627
* **scellé** : K si Δ noyaux ≥ 0,25 ; T si \|Δ noyaux\| ≤ 0,08 et Δ trous ≥ 0,25 ; R si \|Δ mur\| ≤ 0,10 (commit précédent)
* **mesuré** (ms/pas, train de **509 pas** des deux côtés, même position dans le lot de 1 024) :

| | serve (S) | boucle nue (N) | N − S |
|---|---|---|---|
| mur | 6,129 | 6,429 | **+0,300** |
| noyaux | 5,838 | 6,140 | **+0,302** |
| **Marlin MoE** | **2,877** (144 × 19,98 µs = **59,9 µs/couche**) | **3,176** (144 × 22,06 µs = 66,2 µs/couche) | **+0,299 (+10,4 %)** |
| toutes les autres familles | 2,961 | 2,964 | +0,003 (aucune au-delà de ±0,005) |
| mur − noyaux (trous) | 0,291 | 0,289 | −0,002 |

  Le mur est stable par tiers du train (S 6,121 · 6,122 · 6,144 ; N 6,424 · 6,438 · 6,426) : ce n'est pas une dérive.
* **verdict : K**, et plus étroit que prévu. Hors de `serve`, le moteur exécute **le même graphe, avec les mêmes
  lancements et les mêmes trous** ; seul le **noyau Marlin MoE** y est plus lent de 10,4 %. **(i) est réfutée** (trous
  égaux à 2 µs près). La moitié « attention » de ma prédiction est fausse (+0,002) : l'écart de +0,10 vu d'abord sur
  toutes les fenêtres venait du contexte (chauffe et passes courtes du client), et la lecture par train l'a retiré.
* **durée** : prévue ≤ 3 min ; tenue **1 min 51** (`carte.sh`, tenue=111 s), après les prises de Manon

## Ce que cela veut dire, et ce qui reste ouvert

Un Marlin 10 % plus rapide sur le même graphe, c'est soit **moins d'octets lus**, soit **les mêmes octets mieux
servis** :

* **(ii′) le travail diffère** : les jetons générés, donc le routage. Même invite en ids, glouton des deux côtés,
  mais le chemin HTTP (`/v1/completions`) peut changer la suite (jeton de début, traitement des ids, arrêt).
  D'autres jetons donnent d'autres experts distincts, donc un autre nombre d'octets de MoE. C'est l'hypothèse la
  plus probable : seul le Marlin bouge, et le Marlin est la seule famille dont le coût dépend du **nombre
  d'experts distincts**, les autres ne dépendant que de b et du contexte.
* **(iii) la mémoire** : mêmes experts, mais placement ou état de la L2 différents.

**Test qui départage, ≤ 2 min** : le crochet de la 74 (experts distincts par appel à M = 12) posé dans les deux
chemins, `serve` et boucle nue. S'il y a moins d'experts distincts sous `serve`, c'est (ii′) ; à nombre égal, c'est
(iii).

## « Nos bancs Engine sont-ils pessimistes d'environ 6 % ? »

**Oui, de 4,9 % dans ce régime** (0,300 sur 6,129 ms/pas), et **uniquement par le Marlin MoE**. Si (ii′) se
confirme, il faudra dire « le banc et le service ne décodent pas le même texte » plutôt que « le banc ment », et
décider lequel des deux textes représente le service.

Chiffres qui en dépendaient, relevés dans le dépôt :

* **README : aucun.** Toutes les cellules publiées (b=12 1 831,7, b=1 310,8 et 283,6, énergie) sont servies en HTTP
  par le client de cellule.
* **Décompositions internes, à requalifier** :
  * p73 : « notre Marlin à 1,37 To/s en service, 64,6 µs/couche » venait de `frontiere-pas`, donc d'Engine direct.
    **Sous `serve`, 59,9 µs/couche**, soit ≈ 1,48 To/s sur les 88,6 Mo/couche de la 73.
  * p75 : « notre service colle au banc (64,6 contre 66,8) » est faux pour la même raison. En service, notre Marlin
    **passe sous le banc** (59,9 contre 66,8 pour A3), **comme celui de vLLM** (56,9 contre 62,4) : les deux moteurs
    se comportent de la même façon.
  * p76 : table des familles (trace p73, Engine direct, autre alias) ; son MoE +0,316 devient ≈ +0,15 (2,877 contre
    2,729, alias et commit à la réserve près).
  * p77 : (d) ≥ 0,215 et « servi contre boucle nue » : l'anomalie est désormais localisée.
* **Seuils choisis par ABBA en Engine direct, à relire** : p65, `MOE_TENSOR_MIN_T = 8` (« chaîne p65, frontière
  200 pas »). La comparaison était relative, dans le même harnais. Mais si le biais vient du routage (ii′), il pèse
  sur les deux bras MoE de façon inégale : le GEMV par paire et la GEMM groupée ne dépendent pas de la même façon du
  nombre d'experts distincts.
* **Cellules `certifie-b12`** (inventaire du 18/09) : absolues, Engine direct, pessimistes du même ordre si elles
  sont citées quelque part comme chiffre de service.

## Pour la question du Marlin (Jerome, recherche de Laurine)

« Le nôtre fait l'inverse de vLLM » (62,2 au banc, 64,6 en service) était un **artefact d'instrument** : le 64,6 est
un chiffre d'Engine direct. Sous `serve`, notre Marlin fait **59,9 µs/couche**, sous le banc, comme le leur. À même
régime, nous sommes à 59,9 en 3 lancements contre 56,9 en 2 pour eux ; la fusion w13 (−4,6 µs/couche au banc de la
75) nous mettrait à ≈ 55, à égalité avec vLLM ou devant.

## Addendum — le test (ii′) contre (iii), 23/09 10 h 0x

* **instrument** : `ACVRAM_TRACE_ROUTAGE_PT` dans les deux chemins, **en eager** (l'instrument n'écrit qu'hors graphe, `moe.py:1182` ; le routage ne dépend que des jetons et des mêmes noyaux) ; `scratchpad/oceane-p80-23-09/` (prise.sh, prise-serve.sh, comparaison-*.json)
* **mesuré** (lot de 1 024 jetons, M = 12, 48 960 appels comparés) :
  * **experts distincts par appel : serve 30,16, boucle nue 30,92 (−2,5 % sous serve)** ; par tiers du lot, serve 30,32 · 30,20 · 29,97, nue 31,64 · 30,89 · 30,24 ;
  * **routages identiques appel par appel : 0**, même au premier pas. Couche par couche, sur tout le lot, **51,7 %** des routages par jeton sont communs aux deux chemins.
* **lecture** : sous `serve`, les douze requêtes arrivent échelonnées par HTTP (33 pas avec préfill, contre 12-13
  dans la boucle nue). La composition du lot diffère donc à chaque pas, le glouton n'est pas invariant au lot, et
  les suites divergent. **Les deux chemins ne décodent pas le même texte : (ii′) est vraie**, mais elle ne pèse
  presque rien. Un Marlin borné par la bande suit le nombre d'experts distincts : −2,5 % d'experts rendent ≈ −2,5 %,
  pas −10,4 %. **Environ un quart de l'écart vient du texte ; les trois quarts restent inexpliqués**, compatibles
  avec (iii) (mémoire), non démontrés.
* **réserve** : les comptes viennent de passes en eager, qui sont d'autres tirages que les passes en graphe de la 79 ;
  l'ordre de grandeur est le seul transport légitime.
* **durée** : prévue ≤ 2 min ; tenue **tenue=142s tenue=92s ** (deux prises : le premier bras `serve` n'a pas écrit sa trace, SIGTERM
  sur le groupe ne déclenche pas `atexit` ; rejoué avec SIGINT). Dépassement déclaré.
* **conséquence pratique, inchangée** : juger sur le **servi** ce qui touche au MoE, comme l'ordre de la 82 le prévoit.
  Les bancs Engine direct surestiment le Marlin MoE de jusqu'à 10 % à b = 12.
