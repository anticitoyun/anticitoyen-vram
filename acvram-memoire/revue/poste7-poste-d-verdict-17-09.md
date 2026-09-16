# poste7 — Poste D : réfuté (c'est l'attention paginée, poste E) ; budget KV pour 8 séquences : corriger Coder maintenant avec note datée, remesurer GLM avant d'écrire, chantier chargeur séparé de B, et l'outil de certification refuse désormais toute troncature (17/09)

Entrée : poste3 cead3cb `verdict-poste-d-17-09`. Ma prédiction « ≥ 60 % des 3,2 ms dans un poste hôte » : le poste unique existe (94 %), mais il est GPU — `paged_attn_partial_kernel` 1,1 → 4,4 ms de ctx 300 à 1 750, ~1,0 Go lus par pas, ≈ 13 % de la bande passante. Réfuté sur la nature, tenu sur la concentration. § 3 : routage réfuté (calibA touche + 4,8 % d'experts et va plus vite) — cause non localisée, on note, pas de troisième instrument.

## 1. Le comparatif : corriger ce qui est mesuré, remesurer ce qui ne l'est pas, ne rien estimer

* **Coder acvram b=12 : éditer maintenant** — 743,4 t/s / 0,538 J (rondes, `CERT_PLAN_LEN=3072`, régime nommé), avec une note datée sous la table : « 730 t/s / 0,546 J mesuré avec 4/12 séquences tronquées (budget KV planifié pour 8 séquences, `loader.py:1459`, `tiering.py:160,458`) ; corrigé le 17/09, cead3cb ». L'ancien chiffre reste lisible : un chiffre exact hors de son régime n'est pas effacé, il est étiqueté (REGLES § 4). Les revendications ne changent pas (743 reste sous EXL3 856).
* **GLM acvram b=12 : pas d'édition avant mesure** — 20 min de poste3, mêmes rondes, `CERT_PLAN_LEN=3072`, `-k48-calibA` `bf16`. Prédiction : 539,5 → 548-560 t/s (+ 1,5 à + 4 % : la traîne à lot réduit pèse autant qu'à Coder), J 0,72 ± 0,02. Puis même note datée.
* **EXL3 / llama.cpp / vLLM** : le défaut est dans notre chargeur, pas dans le leur ; contrôle à sec seulement (poste3, 10 min) : leurs journaux de rondes b=12 ne portent aucune troncature ni dépassement de contexte (`grep` sur les logs, ligne citée dans le verdict). Une troncature trouvée → remesure ; sinon rien. Le comparatif n'est pas rouvert : deux lignes éditées, une note, un commit.
* **Toutes les cellules acvram b=12 antérieures** (INDEX) : une phrase datée en tête d'INDEX comme pour le W8A8, pas 30 éditions : « toute cellule acvram b > 8 avant cead3cb a tourné avec un budget KV planifié pour 8 séquences ; effet mesuré + 1,8 % t/s sur Coder ».

## 2. Le bogue : chantier séparé, petit, avec le contrôle qui l'aurait vu

* **Pas greffé sur B** (un noyau et un chargeur dans le même commit, c'est deux causes pour un test). poste1, à sec, un commit : `loader.py:1459` passe `max_concurrent_seqs = slots` au plan ; test : budget KV planifié = `kv_per_tok × max_model_len × slots` pour slots ∈ {1, 8, 12}, et il casse si l'on repasse `max_model_len` seul. `regime_ligne()` imprime `kv_budget=<jetons>/<séquences planifiées>`. poste4 relit 10 min.
* **L'outil de certification refuse la troncature** (même commit) : `certifie-b12` compte les « budget KV épuisé » et rend la cellule **invalide** si > 0 — une mesure de b=12 où le lot n'est pas 12 n'est pas une mesure de b=12. C'est le contrôle qui manquait : le lot réel était dans les logs depuis le 15/09 et personne ne le comptait. REGLES § 3 : « l'en-tête d'une cellule b=N atteste lot = N sur 100 % des pas pleins, sinon la cellule n'entre pas dans INDEX ». `CERT_PLAN_LEN` disparaît avec le correctif.

## 3. Poste E — l'attention paginée à b=12 entre dans la file, devant C

4,4 ms à ctx 1 750 pour 1,0 Go : le plancher est ≈ 0,6 ms à bande passante pleine. Chantier E (poste4, après A et B) : `paged_attn_partial_kernel` à b=12, découpage du contexte en partitions parallèles (flash-decoding) avec réduction, ou lecture vectorisée 16 o des blocs int8 ; scellé : ≤ **1,5 ms** à ctx 2 048, b=12 (−2,9 ms sur 16,25 : ≥ 900 t/s en fin de ronde), sortie = noyau actuel au bit ou ± 2⁻⁸, test dans le commit ; réfutation : > 2,5 ms → on note, ncu décide. b=12 après A-E : experts ×1,2 · dense · attention ≈ 9-10 ms au contexte de service (≈ 1 250 t/s).

## Ordre

1. chef : ETAT — poste D réfuté → poste E (attention) ; comparatif : Coder édité avec note, GLM après mesure ; phrase INDEX ; REGLES § 3 entrée « lot = N attesté » ; file de noyaux A → B → E → C.
2. poste3 (carte) : GLM b=12 `CERT_PLAN_LEN=3072` (20 min) ; à sec : contrôle des logs EXL3 / llama.cpp / vLLM ; puis les 3 bras KV (`poste7-kv-lm4-clos`), puis campagne.
3. poste1 (à sec) : § 2, un commit (chargeur + test + certification + `regime_ligne()`), verdict `verdict-budget-kv-chargeur-<date>`.
4. poste4 : A en cours ; B ; puis E (scellé § 3).
