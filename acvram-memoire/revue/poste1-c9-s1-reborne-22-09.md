# C9-S1 reborné — cache d experts PCIe, hôte exclu, h_pin = 0,733 mesuré sur le proxy (22/09, poste1, à sec)

* sources : `verdict-c9-m-hote-22-09` (hôte : 0,855 ms/expert = 16,6 Go/s, arrêt (a) — S2/S3 fermées sans noyau AVX-512), `verdict-c9-m-trace-22-09` (proxy Coder-30B top-8/128, C = 52 = 41 % : **h_pin 0,733** (couches 0,56-0,80), h_lru 0,869, uniforme 0,41), `verdict-c9-m0-19-09` (PCIe H2D épinglé 22,6 Go/s), `chantier-c9-m3-19-09` (branche `poste1-c9-m3-cache-experts` 009bc039 : `CacheExpertsCouche`, créneaux + tables d adresses, manquants lus **zéro-copie par UVA** sous `pin`, 12 tests à sec, jamais exécuté sur carte), `poste1-c9-conception-21-09` (volumes : expert 14,16 Mo, 144/jeton = 2,04 Go, résidents 4,66 Go, C ≈ 52/couche)
* ce qui change : (1) le calcul hôte est hors jeu → un expert froid **traverse le bus**, 14,16 Mo ; (2) h n est plus supposé, il est mesuré à 0,733 sur un proxy — au-dessus de la parité (0,55) prédite par poste7, mais sur un routage top-8 d un autre modèle.

## 1. Borne de débit b=1 (ms/jeton = 2,04 Go × (1 − h) ÷ débit du bus + carte)
Deux façons de faire traverser un expert froid, toutes deux dans le module M3 :
* **zéro-copie (`pin`)** : le GEMV lit la ligne hôte épinglée à travers le bus pendant le pas — **capturable** (aucune décision hôte dans le pas), mais un noyau qui lit par UVA n atteint pas le débit d une copie : **15-19 Go/s** attendus (à mesurer : c est le chiffre qui manque, § 3) ;
* **copie (`aveugle`/LRU)** : `cudaMemcpyAsync` épinglé à 22,6 Go/s, mais exige les ids routés sur l hôte à chaque couche → incapturable (M4, graphes par segment) ; h_lru 0,869 > h_pin 0,733 sur le proxy : 2 × moins d octets, mais un chantier de plus.
| stratégie | h | Mo/jeton | ms bus | + carte (3 ms) + eager (0 sous graphe, ~8 hors) | j/s |
|---|---|---|---|---|---|
| S0 tout bus, zéro-copie | 0 | 2 040 | 107-136 | 110-139 | **7-9** |
| **S1 pin, zéro-copie, graphes** | **0,733** | 545 | 29-36 | **32-39** | **26-31** |
| S1 pin, zéro-copie | 0,65 (119B moins concentré ?) | 714 | 38-48 | 41-51 | 20-24 |
| S1 pin, zéro-copie | 0,80 (proxy couche max) | 408 | 21-27 | 24-30 | 33-42 |
| S1 bis LRU copie 22,6 Go/s, sans graphes | 0,869 | 267 | 11,8 | 11,8 + 3 + 8 eager + 3,6 sync | **38** |
Barre : llama.cpp **24,2 j/s** (experts en RAM). **À h = 0,733, S1 fait 26-31 j/s : +7 à +28 % — la parité est passée, pas la marge de sécurité.** Le terme qui décide est le débit zéro-copie (15 contre 19 Go/s = 26 contre 31 j/s), puis h sur le vrai 119B. J/jeton : bus saturé, carte presque oisive (≈ 120-150 W × 35 ms ≈ **4-5 J/jeton**, contre 0,69 J carte seule pour llama.cpp, processeur non compté) : S1 gagne en vitesse, pas en énergie — à écrire tel quel dans la fiche.

## 2. Condition « vaut la peine » (écrite avant toute prise)
* **Vaut la peine** : S1 pin b=1 **≥ 30 j/s** (barre + 25 %) sous graphes, ids au bit contre le tout-VRAM sur un proxy à C forcé. Il faut pour cela **h_pin(119B, C = 52) ≥ 0,72 ET zéro-copie ≥ 17 Go/s**, ou h ≥ 0,78 à 15 Go/s.
* **Ne vaut pas la peine** — arrêt de C9, écrit avant : (a) zéro-copie mesuré < 13 Go/s (S1 ≤ 24 j/s même à h = 0,75 : parité pour six commits) ; (b) h_pin(119B) < 0,62 à C = 52 (S1 ≤ 22 j/s) ; (c) les deux à la limite (h < 0,68 et bus < 16) → S1 bis LRU seul dépasserait, avec M4 en plus : hors devis.
* Alarme : proxy 0,733 ≠ 119B — top-8 sur 48 couches contre top-4 sur 36 : moins d experts par jeton = moins de recouvrement entre jetons voisins, h attendu **plus bas** de 0,05-0,10 ; le h du 119B se mesure sur le 119B, pas ailleurs (commit 3).

## 3. Ce qui manque au chantier pour un premier run (commits nommés, dans l ordre — 5 + 1)
0. **M-UVA (poste2, ≤ 5 min, 0 code : `tests/test_placement_par_expert.py:133` rejoué en chronométrant)** : débit d un GEMV Coder lisant 8 experts de 2,65 Mo sur l hôte épinglé par les tables (`nvfp4_gemv_grouped_gateup_table`), 200 pas → **Go/s zéro-copie** ; prédit 15-19 ; c est (a) du § 2, il tranche AVANT les commits 1-2.
1. `c9-spec-mistral4` : `ModelSpec` depuis `params.json` (dim, MLA, MoE 128/4/partagé, tekken), `llama_4_scaling` refusé nommément (texte ≤ 8 192) ; test à sec sur le `params.json` réel.
2. `c9-noms-consolidated` : table `layers.N.attention.wq_a/wq_b/wkv_a_with_mqa/wkv_b/wo`, `experts.N.w1/w3/w2`, `shared_experts`, `gate`, normes, `tok_embeddings`/`output` → noms acvram ; import direct du nvfp4 compressed-tensors (`nvfp4_direct`, 1/weight_global_scale) ; test : déquantification == compressed-tensors au bit sur un expert réel (déjà fait dans `c9-m-hote`, à déplacer en test).
3. `c9-charge` : chargement texte seul, **tous les experts exilés par expert** sur le chemin existant (`streamed` par expert, loader.py:470-513, tables `_tables_adresses`) = S0 zéro-copie ; premier jeton ; **trace M1 du 119B** (`c9-m-trace --model`, 5 000 jetons) → h_pin(52) réel → **décision § 2 (b)**. Cellule S0 attendue 7-9 j/s : ce n est pas un chiffre à publier, c est la preuve que le chemin sert.
4. `c9-cache-pin` : `CacheExpertsCouche` par couche (`pin_depuis_trace`, C = 52 depuis le budget réel du manifeste, résidents chargés une fois, tables → créneau ou adresse hôte), branché aux points 1 et 4 de `chantier-c9-m3` (loader.py:495-513 ; préfill = tout par UVA) ; **ids au bit** contre le tout-VRAM sur Coder à C forcé (test carte existant :133 étendu) ; `ACVRAM_CACHE_EXPERTS=pin|non` dans `regime.VARIABLES`, ligne `cache_experts=pin(C=52,h=…)` avec le h **mesuré en service** (compteurs `succes/demandes` du module, jamais celui de la trace).
5. `c9-cellule-s1` (poste2) : b=1 contre llama.cpp, même harnais, ctx 2 048, éco 2 700, deux passes ; J/jeton au compteur.
6. (après, seulement si § 2 tient et que h_lru − h_pin > 0,10 sur le 119B) : `c9-lru-m4` — décision différée en fin de pas + graphes par segment ; hors devis ici.
Coût : 0 = 5 min carte ; 1-2 ≈ une session à sec ; 3 ≈ une session + 2 prises ≤ 30 min ; 4 ≈ deux sessions + 1 prise.

## 4. Prédiction
Sur le 119B, S1 pin sous graphes : **h_pin 0,66 ± 0,06, zéro-copie 17 ± 2 Go/s → 24-31 j/s, médiane 27** — au-dessus de la barre, sous la condition « vaut la peine » à la médiane : **le chantier se décide sur M-UVA (5 min) et sur la trace 119B (commit 3), pas avant, et pas sur le proxy**. Réfuté : S1 < 24 j/s mesuré (on ferme, la fiche dit « parité, énergie ×6 ») ; alarme : > 42 j/s (h ou bus hors de tout ce qui a été mesuré : relire l en-tête).
