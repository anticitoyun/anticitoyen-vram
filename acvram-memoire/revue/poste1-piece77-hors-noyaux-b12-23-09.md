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

## Verdict — 23/09 09 h 1x (poste1)

* **instrument** : `outils/gpu/mesure/hors-noyaux-p77.py` (témoin, boucle nue, compte des nœuds), `frontiere-pas.py 12 1024` (sans nsys) et `12 60` (sous nsys, **noyaux seuls**), `serve` + `banc-llamacpp-16-09.py decode` (comme la 64), `/metrics` en fin de fenêtre ; chaînes `scratchpad/poste1-p77-23-09/prise{1..5}.sh`
* **commit** : 2fe9b7c9 ; alias **`Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c`** (celui de la 64) dans tous les bras
* **régime** : -lgc 2700, horloge médiane sous charge 2 640-2 661 MHz dans tous les bras (servi 2 642 et 2 661, nu 2 647), bridage puissance nominal, ACVRAM_CPUS 0-15 ; compute-apps début = fin = llama-server 4627
* **scellé** : table « Prédictions » ci-dessus, commitée avant la première prise (ce fichier, commit précédent)
* **mesuré** :

| grandeur | valeur | source |
|---|---|---|
| **servi**, pas moyen | **1 940,5 puis 1 995,5 t/s → 6,18 puis 6,01 ms/pas** ; `/metrics` : 60 780 jetons de décodage en 5 090 pas (11,94 par pas), **5,97 ms/pas de décodage**, 0 jeton spéculé | `banc-servi*.log`, `metrics-fin.txt` |
| **boucle nue**, même moteur, mêmes lots | **1 836,5 · 1 837,0 (KV hôte 8 Gio) · 1 848,0 t/s → 6,49-6,53 ms/pas** ; pas de décodage seuls 6,34-6,35 ms | `nue*.json` |
| frontiere-pas 1 024 pas, sans nsys | pas 6 335 µs (médiane) / 6 491 (moyenne) ; rejeu 6 305 / 6 275 ; trou carte 24 / 212 | `frontiere-1024.json` |
| noyaux seuls, 60 pas sous nsys | **5,831 ms/pas** (798 lancements) ; rejeu du graphe sous nsys 6 091 µs | `familles60.json`, `frontiere-60-nsys.json` |
| nœuds du graphe b = 12 (godet 16) | **789** = 784 noyaux + 3 memsets + 2 copies (godet 1 : 597) | `nue.json` |
| témoin, noyaux Triton vides | **0,391 µs/nœud** (1 → 5,5 µs ; 800 → 320 µs) | `temoin.json` |

* **verdict par part** :
  * **(a) nœuds × latence : 789 × 0,391 = 0,31 ms/pas** (majorant). La lecture directe sous nsys donne 6,091 − 5,831 = **0,26**. Prédiction (a) 0,25-0,35 **tenue**. Pente prédite 0,8-1,5 µs **réfutée** : 0,39 µs, un noyau vide ne coûte presque rien au-delà de son lancement, donc le majorant est serré.
  * **(b) frontière : 30 µs en médiane** (6 335 − 6 305), prédite 0,02-0,06, **tenue**. La moyenne (0,215) est portée par des trous rares (trou carte moyen 212 µs contre 24 en médiane).
  * **(d) contexte long : ≥ 0,215 ms/pas** (rejeu 6 305 sans nsys à 1 024 pas, contre 6 091 sous nsys à 60 pas ; nsys gonfle le second, donc c'est un minorant). Prédit 0,10-0,20 : **dépassé, sans atteindre le seuil de réfutation de 0,30 dans ce que l'on sait borner**.
  * **(c) couche service : NÉGATIVE, de −0,33 à −0,52 ms/pas.** Le service est **plus rapide** que la boucle moteur nue, au même commit, sur le même alias et à la même horloge, deux fois côté servi et trois fois côté nu. Prédiction 0,20-0,40 **réfutée** (< 0,10), dans le sens que je n'avais pas nommé.
* **durée** : prévue ≤ 10 min en deux prises ; tenue **5 min 35 en cinq prises** (52 + 80 + 44 + 93 + 66 s, `carte.sh`). Les prises 3 à 5 n'étaient pas prévues : un bras raté (`functools.partial` sur `CUDAGraph`), puis deux contrôles pour départager l'anomalie de (c).

## Ce que la 77 renverse

1. **La prémisse « 0,842 ms/pas hors noyaux » de la 76 est fausse (erratum 76).** Elle soustrayait au pas servi de la
   64 (6,551 ms, alias `qkvo-i8c`, contexte long, commit de 06 h) les noyaux d'**un autre alias** (`…-nvfp4`, p73) sur
   **un contexte court**. Au même alias, au même commit, dans la même heure : pas servi **6,01** contre noyaux courts
   **5,83**. Tout ce qui n'est pas noyau court tient dans **≤ 0,18 ms/pas**, contexte compris. Ce reste est plus petit
   que celui de vLLM (0,477 à la 76), pas plus gros.
2. **Notre servi du jour est à 1 940-1 995 t/s** ; la 64 donnait 1 829-1 834 ce matin (vLLM 1 970-2 028). Je ne
   revendique **aucune** parité : ce n'est ni la même séance ni un ABBA, et mes deux bras servis s'écartent déjà de
   2,8 % entre eux. **Il faut rejouer l'ABBA de la 64 aujourd'hui** (poste2, même chaîne) avant de rouvrir la question
   de la porte b=12.
3. **Anomalie d'instrument, à fort enjeu** : le même moteur, piloté hors de `serve` (`frontiere-pas`, boucle nue,
   donc probablement `certifie-b12` et toutes nos traces « Engine direct »), rejoue le même graphe b=16 **0,35 ms/pas
   plus lentement** que sous `serve` : rejeu à 6,31 ms, alors que `serve` décode à 5,97 ms par pas. Lignes de régime
   identiques octet pour octet, horloge identique, KV hôte écarté (1 837 contre 1 848). Ce n'est pas le godet
   (le Coder est dense, godet 16 des deux côtés, `graphs.py:68`). **Non expliqué.** Hypothèses, par ordre de
   probabilité :
   * (i) l'état du processus : moteur dans un fil sous `serve`, fil principal ailleurs ;
   * (ii) les graphes eux-mêmes : 13 captures sous `serve` contre 8 dans la boucle nue — les lots qui se remplissent
     par HTTP passent par d'autres godets ;
   * (iii) la mémoire (adresses et pool des graphes).
   Test qui départage, ≤ 3 min : nsys **noyaux seuls** sur `serve` contre la boucle nue. Si les noyaux diffèrent,
   c'est (ii) ou (iii) ; s'ils sont égaux, c'est un trou entre nœuds, donc (i).

## Plus grosse part et pièce qui la vise

* Parmi (a), (b), (c) telles que posées, **la plus grosse est (a), 0,26-0,31 ms/pas**, structurelle (789 nœuds). Elle
  relève de la fusion de nœuds (Q(15)). **Gain prédit modeste** : 0,33 µs par nœud retiré ; une fusion réaliste
  (rope + écriture KV, normes dans leurs voisins, ≈ 100-150 nœuds) rend **0,03-0,05 ms/pas** (0,5-0,8 %). vLLM a un
  nombre de lancements du même ordre (≈ 830 par pas à la 76), donc ce n'est pas là qu'est l'écart.
* **Ce qui mérite la carte avant tout** : (1) l'ABBA de la 64 rejoué aujourd'hui (le vrai écart restant) ; (2) le
  test qui départage l'anomalie (3). Si l'état de `serve` rend 0,35 ms, nos bancs « Engine direct » sous-estiment
  le service d'environ 6 % depuis leur création, et plusieurs seuils écrits sur eux sont à relire.
