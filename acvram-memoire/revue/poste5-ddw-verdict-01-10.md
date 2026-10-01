# ddw verdict — décodage KDA b=1, tuile S[h] en mémoire partagée : au bit, noyau 10,74 → 7,29 µs/couche (−32 %), mur b=1 −2,27 % (bande −2,4 à −4,1 : EN DEÇÀ, non FAUX) (poste5, mesureuse, 01/10)

* scellé : `poste5-ddw-scelle-01-10.md` (e1828a66d, avant code et mesure). Noyau B 4 à 7 µs/couche (FAUX > 8) ; mur b=1
  −0,08 à −0,14 ms, −2,4 à −4,1 % (FAUX si gain < 0,05 ms) ; b=12 0 ± 0,5 % ; base A (≈ 11 µs, inférée) à mesurer.
* commits : A = 84123994d (fusion 61w, ancêtre d'origin/main) ; B = 1a4157856 (poste5-ddw). Arbres figés, acvram de
  chaque bras contrôlé (ACVRAM_ARBRE + garde). .so précompilés, aucun JIT : A 98af5877cb43721c (source de 84123994d,
  sha256 vérifiée), B 535cd9a984565f86 — le chemin chargé est imprimé par chaque bras (PRECOMPILE …).
* instruments : inchangés. `outils/gpu/mesure/nsys-kda-p81.sh` de CHAQUE arbre (64 pas par plage, --cuda-graph-trace=node),
  puis `kda-61w-abba.sh` (étape 0 + A1 B1 B2 A2, témoins A1=A2 et B1=B2). Lanceur unique hors dépôt
  (scratchpad de session, prise-ddw.sh). Sorties : `scratchpad/poste5-ddw-01-10/` du worktree, non commitées (trace 45 Mo).
  La ligne « prédit » imprimée par `comparer` est celle de 61w : les seuils de ddw sont appliqués ici.
* régime : Kimi-Linear-35B-kda-nvfp4, NOMINAL (graphes on, hybrides ≤ 12, 0 exilée), glouton, invites synthétiques de
  256, 5090 seule. nvidia-smi au début (12:06:41) et à la fin (12:11:37) : seul llama-server 4436, sur la 3080 Ti
  (bus 02:00.0). Pause e50.2 posée à 11:44:35, carte rendue à 11:46:21, retirée à 12:12:12. Deux faux départs (12:06,
  7 s chacun) : la garde d'import a refusé ACVRAM_ARBRE hérité, puis un cwd dans l'arbre principal. Aucune mesure n'en
  est sortie, et leurs dossiers ont été effacés avant la relance.

## Mesuré

* étape 0 (arbre B, sur carte) : `tests/test_kda_etat_kv.py` **8 passed** (D=64 sur 64 pas, D=128 sur 64 et 512 pas
  chaînés, contre le témoin `kda_decode_vk`, bras cassant).
* équivalence : témoins A1=A2 et B1=B2 au jeton près ; **A et B IDENTIQUES sur 2 048 jetons**, b=1 et b=12 (12 séquences).
* nsys, `kda_decode_kernel<128>` dans la plage `p81:decode_b1` (64 pas × 20 couches = 1 280 lancements) :

  | | A (61w) | B (ddw) | Δ |
  |---|---|---|---|
  | noyau, total sur la plage | 13,753 ms | 9,326 ms | −32,2 % |
  | par couche | **10,74 µs** | **7,29 µs** | −3,46 µs |
  | débit (4,19 Mo lus et écrits) | 390 Go/s, 26 % du plancher | 575 Go/s, 38 % | |
  | GPU par pas, b=1 | 3,2314 ms | 3,1584 ms | −0,073 ms |
  | KDA, b=12 (fla, non touché) | 0,3503 ms/pas | 0,3504 ms/pas | 0 |
  | préfill 8 k (non touché) | 1 394,27 ms | 1 394,99 ms | +0,05 % |

* mur par pas (médiane de 6 répétitions de 128 pas par chemin, ABBA) :

  | | A | B | Δ | scellé | lecture |
  |---|---|---|---|---|---|
  | b=1 | 3,383 ms | 3,306 ms | **−0,077 ms, −2,27 %** | −2,4 à −4,1 % (FAUX si gain < 0,05 ms) | **en deçà de la bande, non FAUX** |
  | b=12 | 8,660 ms | 8,647 ms | −0,15 % | 0 ± 0,5 % | dans la bande |

  Étendue du témoin A : 0,6 % (b=1), 3,6 % (b=12) ; étendue de B à b=1 : 0,4 %.

## Verdict

* **La base inférée est confirmée** : 10,74 µs mesurés contre ≈ 11 inférés par poste1.
* **Le noyau B est à 7,29 µs, juste au-dessus de la bande de 4 à 7** et sous le falsificateur de 8. Le gain au mur
  (−0,077 ms) est cohérent avec le gain GPU du noyau (−0,069 ms par pas sur 20 couches), qui ne se perd donc pas.
  Il est sous la bande (−2,27 % contre −2,4 %) et au-dessus du falsificateur de 0,05 ms.
* **Lecture : gain réel, au bit, plus faible que prédit.** La prédiction surestimait ce que 32 SM tirent de la mémoire :
  575 Go/s, soit 38 % du plancher, contre 40 à 70 % prédits. C'est exactement l'issue nommée au scellé (« 32 SM ne
  suffisent pas à tirer la bande, le noyau plafonne à 7-8 µs »).
* **Défaut servable** : sortie identique au jeton, aucun chemin modifié hors b=1 (b=12 et préfill inchangés au bruit près).

## Suite (levier nommé au scellé, non engagé)

Il reste 7,29 − 2,76 = 4,5 µs par couche au-dessus du plancher, soit ≤ 0,09 ms par pas, ≤ 2,7 % du mur b=1. Deux voies :
(a) 4 blocs par tête, en grappes, avec DSMEM pour les sommes de bloc dans l'ordre du témoin (128 SM au lieu de 32 ; le
support des grappes sur sm_120 est à vérifier) ; (b) le lancement dépendant programmatique (PDL), pour précharger S
pendant les projections. Toutes deux sont à sceller avant tout code ; décision chef.
