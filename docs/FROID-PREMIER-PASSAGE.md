# D'où vient le coût du premier passage

Mesuré le 8 septembre 2026 sur `Qwen3-30B-A3B-abliterated-erotic-AWQ`, cinq
passages, même session, même serveur, seule la capture de graphes change.

|  | passage 1 | passages 2-5 | écart |
|---|---|---|---|
| avec graphes — débit | 166,7 | 180,3 · 180,3 · 179,8 · 180,3 | **−7,5 %** |
| avec graphes — TTFT | **181 ms** | 42 · 40 · 39 · 40 ms | **×4,5** |
| sans graphes — débit | 64,0 | 66,6 · 65,3 · 65,9 · 66,5 | **−3,2 %** |
| sans graphes — TTFT | **1000 ms** | 53 · 56 · 51 · 52 ms | **×19** |

## Ce que ça établit

**Les graphes CUDA valent un facteur 2,7 en débit** : 180,3 contre 66,1 au
régime établi. C'est le même mécanisme que le facteur 4 de
`docs/SEGMENTS-EXTENSIBLES.md`, atteint par un chemin indépendant.

**La capture des graphes n'est pas la cause du froid.** L'hypothèse était
qu'elle est paresseuse et se paie à la première requête. Trois faits la
réfutent :

1. le journal du serveur l'annonce comme faite **au chargement** — « graphes
   CUDA : actifs (decodage), 5 godets capturés **d'avance** » ;
2. le coût du premier passage subsiste **sans graphes du tout** : 1000 ms de
   TTFT contre 52 ms ensuite ;
3. il est **plus lourd** sans graphes qu'avec — 1000 ms contre 181 ms. Si la
   capture était le coût, l'ordre serait inverse.

**Il y a deux coûts distincts, et non un seul.** Le débit du banc est calculé
entre le premier et le dernier jeton : il **exclut** le TTFT. Le premier
passage paie donc séparément un surcoût *avant* la génération (+141 ms avec
graphes, +948 ms sans) et un surcoût *pendant* (−7,5 % et −3,2 % de débit).
Aucune explication unique ne couvre les deux.

## Ce qui reste ouvert

Ce qui coûte une seconde au premier préremplissage sans graphes n'est pas
identifié. Candidats non départagés : première allocation du cache KV,
première mise en service des poids exilés, premier passage sur les experts,
initialisation d'un chemin eager qui n'existe pas quand les graphes portent le
décodage. La mesure qui trancherait : instrumenter le premier préremplissage
et relever où passe la seconde, plutôt que de la déduire d'un débit.

Ce qui est acquis pour le protocole : **le passage de chauffe jeté reste
obligatoire**, et il l'est pour deux raisons indépendantes, pas une.
