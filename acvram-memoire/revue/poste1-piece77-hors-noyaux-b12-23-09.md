# Pièce 77 — décomposer nos 0,842 ms/pas « hors noyaux » à b = 12, sans nsys — 23/09 (poste1)

## D'abord un défaut de la 76, trouvé en préparant celle-ci

Les 0,842 ms/pas de la 76 soustraient au pas **servi** de la 64 (6,551 ms : client de cellule, lots de 1 024 jetons,
contexte 256 → 1 280) les noyaux de **notre trace p73** (`frontiere-pas 12 60` : 60 pas, contexte 256 → 316). Le côté
vLLM de la 76 venait, lui, du client de cellule (1 024 jetons). **Les deux soustractions ne portent donc pas sur le
même contexte.** Notre attention (0,435 ms/pas à ~290 jetons) grandit avec le contexte : une partie des « 0,842 hors
noyaux » peut être du noyau d'attention que la trace courte ne voyait pas. Cette part (d) s'ajoute aux trois de
l'ordre et je la prédis comme les autres.

## Protocole (quatre bras, aucun sous nsys, même commit, -lgc 2700, une chaîne)

* **(a) nœuds × latence** — `outils/gpu/mesure/hors-noyaux-p77.py temoin` : graphe de N noyaux Triton vides,
  N ∈ {1, 64, 256, 800}, 200 rejeux, pente en µs/nœud. Le **compte exact des nœuds** du graphe de décodage b = 12
  est lu sur le graphe lui-même (`CUDAGraph(keep_graph=True).raw_cuda_graph()` → `cudaGraphGetNodes`, par type).
  **Limite écrite d'avance** : la pente d'un noyau vide contient aussi sa propre exécution minimale, que les vrais
  noyaux remplacent par leur durée. Témoin × nœuds est donc un **majorant** de (a). La lecture directe est
  G − K : rejeu du graphe chronométré par événements, sans nsys, moins la somme des noyaux de la p73. Ce dernier
  chiffre est au même contexte court, mais d'un autre commit ; je le déclare.
* **(b) frontière entre deux pas** — `frontiere-pas.py 12 1024` (perf_counter hôte et événements carte autour du
  rejeu, de l'échantillon, de `_consommer`, du trou carte) sur tout le lot de 1 024 pas, sans nsys.
* **(c) couche service** — `serve` + client de cellule (`banc-llamacpp-16-09.py decode`, b = 12, 1 024 jetons,
  fenêtre 20 s, comme la 64) contre **`hors-noyaux-p77.py nue`** : le même moteur construit comme `serve` le
  construit (Engine, `demarrer_service`, ngram par défaut, gc gelé), les **mêmes ids d'invite**, les mêmes lots,
  la même fenêtre, sans HTTP. (c) = pas servi − pas nu.
* **(d) attention au contexte long** — graphe médian de (b) sur 1 024 pas (contexte médian ≈ 768) moins le graphe
  des 60 premiers pas (contexte ≈ 290).

## Prédictions et ce qui réfuterait chacune, écrites AVANT toute prise

| part | prédit (ms/pas) | réfuté si | raison de la prédiction |
|---|---|---|---|
| (a) dans le graphe, entre nœuds (G − K) | **0,25-0,35** | < 0,15 ou > 0,50 | p73 sous nsys : 6,046 − 5,709 = 0,337 ms pour ≈ 800 nœuds, 0,42 µs/nœud |
| (a′) témoin × nœuds (majorant) | pente **0,8-1,5 µs/nœud**, nœuds **790-820** | pente < 0,4 (le majorant passerait sous G − K : instrument faux) | noyau vide ≈ 1 µs d'exécution minimale + lancement |
| (b) frontière (pas − rejeu) | **0,02-0,06** | > 0,15 | p73 : trou carte 23 µs + échantillon 4 µs ; la préparation est recouverte par le pipeline |
| (c) service (servi − nu) | **0,20-0,40 — plus grosse part** | < 0,10 | 12 flux SSE, un événement par jeton et par flux, boucle asyncio dans le processus du moteur (GIL) |
| (d) attention, contexte 768 contre 290 | **0,10-0,20** | > 0,30 : alors la 76 surestimait nettement le hors-noyaux | 12 × ctx × 1 Kio de KV int8 par couche : 3,6 → 9,5 Mo lus, noyau à 9 µs aujourd'hui borné par la latence |
| prefill et fin de lot amortis (nu − frontiere) | 0,02-0,05 | > 0,10 | invites identiques d'un lot à l'autre (cache de préfixe), lots sans queue (ignore_eos) |

* **Fermeture** : noyaux courts 5,709 + (d) + (a) + (b) + lots + (c) doit retomber sur le pas servi mesuré dans la
  même chaîne à ±0,10 ms. Sinon, une part manque et je la nomme au lieu de répartir le reste.
* **Ce qui me gênerait** : (d) ≥ 0,30. La 76 aurait alors appelé « hors noyaux » ce qui était surtout de
  l'attention, et le levier n° 1 que j'y ai écrit tomberait.
* **Pièce visée par la plus grosse part** : (c) → couche service (émission SSE groupée par pas, boucle moteur
  hors du fil asyncio) ; (a) → fusion de nœuds, Q(15) ; (b) → frontière ; (d) → noyau d'attention à contexte long.
