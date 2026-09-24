# Verdict — 146 : défauts du DÉFAUT servi autour du cache KV — 24/09 (poste1)

Prises poste1-p146 (788423a8), p146b (ac33eb75), p146c (2e4ff7a3), p146e (f63e3fb4), garde rejouée dans p152 (99be0f97) ;
scellé `scratchpad/poste1-p146-24-09/scelle.md` + addenda b, c, d, e, e bis, tous écrits avant leur prise.
Les trois premières prises ont été journalisées sous le nom « env » : c'était une faute d'appel, corrigée depuis (`ACVRAM_NOM=… outils/carte.sh`).

| # | défaut | fichier:ligne (avant) | correctif | preuve |
|---|---|---|---|---|
| 1 | séquence tronquée par KV épuisé JAMAIS livrée (pipeline : chemin par défaut ; spéculatif) : requête HTTP jamais close | pipeline.py:118, :150 ; runner.py:1795 (sortie jetée) ; `idle` runner.py:2120 | `_epuisees` livrées par `step` dans le pas même | 16 verts ; cassant 4cfdf310 ROUGE ; **bout en bout** (uvicorn réel, graphes + pipeline, KV 4 blocs) ROUGE sur 4cfdf310, vert corrigé |
| 2 | KV à 6 % de la VRAM dès qu'UNE séquence tient | tiering.py:896 (faisabilité), :966-973 (`_rang` sans KV) | départage par min(KV, demande) après exil et débit ; `_plan_kv_maximal` sans exil | qwen32 14 256 → **20 480** (A et B) ; Qwen3.8 inchangé |
| 2b | le KV agrandi faisait EXILER 57 poids (gemma31, DÉGRADÉ) | loader.py `_reajuster_plan` | `kv_min` : le KV au-dessus d'une séquence cède avant tout poids | test + cassant ; gemma31 0 exil |
| 2c | tête liée de gemma laissée en bf16 (VRAM mesurée sans rendre la réserve de l'allocateur), puis fp32 5,25 Gio → OOM | loader.py `_tete_liee` | `empty_cache()` avant la mesure | test + cassant ; chauffe 8×2 560 sans OOM (sur 4cfdf310 la même chauffe passait : régression de la 146, corrigée) |
| 2d | embed compté deux fois dans la borne KV | loader.py `_borner_kv_par_la_vram` | `embed_charge` | test + cassant ; gemma31 11 152 → **13 408** (ma prédiction ≈ 16 700 était FAUSSE : c'est désormais la marge de 7 % de `_reajuster_plan` qui borne) |
| 3 | B alloue +2,11 Gio (Qwen3.8), +1,33 (gemma31) | gemm_dense_etroit.py:300 (`MultiProjection._qw/_bs`) | la passe Marlin retire les MultiProjection des poids convertis | inventaire : B − A = −0,07 Gio |
| a | garde Marlin comparée à la demande ; puis référence replanifiée après embed (501 blocs contre 697) | loader.py `_verifier_memoire_marlin` | capacité de B ≥ capacité du défaut, planifiée et bornée au même instant ; `plus_gros` hors réserve | GARDE kv = ref : 838 / 1 280 / 1 280, ACCEPTÉ |
| b | capacité sous la demande silencieuse | runner.py ligne de régime | `kv_sous_demande=capacité/demande` | chauffe gemma31 : `kv_sous_demande=11152/20480` |

Chauffe finale 8 × 2 560 au défaut (p146e) : qwen32 20 480 (pic 25,17 Gio), Qwen3.8 20 480 (20,56), gemma31 13 408 (26,96),
8/8 finies, 0 tronquée, 0 exil, NOMINAL. Reste (bd) : deux marges KV, 5 % et 7 %. Aucun refus nouveau au défaut (décision
utilisateur) ; la capacité sous la demande est nommée.

**Addendum 35df825c (avant push)** : la suite complète a trouvé une régression de 2a24e6e5 (bissection sur test_exil). Sans
max_model_len annoncé, le départage portait le KV au pire cas (Qwen3-14B : 14,46 Gio), et un SECOND chargement dans le même
processus ne voyait plus que 1,1 Gio : refus (test_exil_equivalence_gpu ×2, test_lancements_par_pas_b12 ; même risque
pour le brouillon de --speculative draft). Corrigé : départage seulement si max_model_len est annoncé
(`PlannerOptions.kv_jusqu_a_la_demande`, loader._replanifier), comme le veut la docstring de _replanifier (sans annonce,
comportement précédent à l'identique) ; test ajouté. Suite complète : 5 échecs, tous présents sur 38b6acdb (confiés à
poste3) ; aucun propre à la branche.
