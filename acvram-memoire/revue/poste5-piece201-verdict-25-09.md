# Pièce 201 — verdict (poste5, 25/09) : un modèle qui ne tient pas n'est plus chargé en silence

Ordre de chef : Qwen3.8-27B-nvfp4-attn-gdn-i8c chargeait (0/64 couches exilées, kv 2 560/1) puis tombait en OOM au
premier pas ; le planificateur doit exiler, baisser le KV ou refuser, jamais charger ce qui ne tiendra pas. Base :
poste4-153c (réserve de préfill i8c + gate_up, 153c) fusionnée avec origin/main. Branche `poste5-201`.
Prises : `scratchpad/poste5-p201-25-09/prise{1..12}` ; journaux de service et logits hors git.

## Trois causes, trois correctifs
1. **Poids chargés après la borne du KV, invisibles au planificateur** (9311d8458). `_octets_reels`
   (acvram/engine/loader.py) ne comptait que `model.layers.*`, `embed_tokens`, `lm_head`. La tour de vision (runner,
   `TourVision.depuis_dossier`, 878,8 Mio) et les têtes MTP (`_charger_mtp`, 343,5 Mio) arrivent après
   `_borner_kv_avec_exil`. `_octets_annexes` compte par EXCLUSION tout tenseur hors couches/embed/tête, moins les blocs
   que leur chargeur refuse par déclaration (`vision.tour_declaree`, `mtp.est_tenseur_mtp` sous `ACVRAM_MTP=non`), dans la
   borne ET dans `_reajuster_plan`.
2. **Copie int8 signée gardée à vie** (86d2bb940). `kernels._i8c_poids` (acvram/kernels/__init__.py:850 avant la pièce)
   gardait sur chaque poids par canal une copie de sa taille dès le premier préfill cuBLAS : 308 poids, **6,84 Gio**
   sur l'i8c, fabriqués pendant `warm_graphs`. Prouvé (prise 5) : même chargement, `ACVRAM_PREFILL_INT8=bf16` → warm vert,
   1,96 Gio consommés au lieu de > 8,8. Correctif (feu chef, option B) : `_i8c_eligible` garde le seul booléen ; la
   copie est fabriquée par appel ou une fois par portée `depaquetage_partage` (179), puis rendue.
3. **La tranche nvfp4 de la 153 changeait le préfill servi** (d5b9e1c45, décision chef). Les replis « naturel » et
   `_marlin_seul` tranchaient tout poids > 44,7 M éléments (`_DEQUANT_TRANCHE_MAX` 256 Mio / 6 o), projections servies
   comprises. Seuil `_tranche_copie` : copie fp32 entière > 1 Gio (`ACVRAM_TRANCHE_COPIE_MIN`, 0 = témoin) — les têtes
   à vocabulaire étendu, en PPL.

## Preuves
| contrôle | résultat | cassant |
|---|---|---|
| `test_annexes_201` (bloc inconnu compté, refus déclarés, exil par une tour de 2 Gio, témoin sans tour) | verts | annexes à 0 → `[oui]` rouge « chargé sans exil » |
| `test_i8c_transitoire_201` au bit contre la copie persistante, aléatoire ET q_proj réel de l'i8c | verts | — |
| idem, octets demandés revenus aux seules sorties (hors portée et après portée) | verts | copie regardée → rouge |
| `test_tranche_copie_201` (projections servies non tranchées, têtes tranchées) | verts | seuil 0 → tout tranché |
| diag201 mixte, logits fp32 sha256 (prises 7b, 9, 10) | C1 0ef818ec… partout ; C3 5218cec8… = main f9c7936df | seuil 0 → C3 f58eec4d… |
| attribution C3 (prise 9) | F = main 5218… ; L = main + 153c f58e… ; 201 avant (3) f58e… | (B) au bit : L = 201 |
| tests ciblés (prise 10) | 159 verts | |

Le mixte ne prend jamais le chemin i8c (poids d'origine fp8 → préfill bf16, compteurs cublas/i8c vides) : (B) y est
inerte ; son au bit sur modèle entier ne se juge que sur l'i8c (unitaire sur q_proj réel ; la copie persistante ne
tient pas en mémoire sur ce modèle, c'est la faute corrigée).

## Capacité KV annoncée (prises 2 et 4, B = 8, avant 9878736d6 / après 9311d8458)
| modèle | CTX | avant | après | cause |
|---|---|---|---|---|
| Qwen3.8-27B-nvfp4 | 32 768 | 128 960 | 119 440 (−7,4 %) | MTP 299,7 Mio |
| gemma-4-31B-it-nvfp4-vision | 8 192 | 10 848 | 8 528 (−21,4 %) | vision + MTP 1 098,2 Mio |
| i8c | 32 768 | OOM au warm | 9/64 couches exilées, chargé | vision + MTP 1 222,3 Mio |
C'est le prix d'un compte juste (CHANGELOG).

## Coût servi (prises 8, 8b ; A = main 51d728b64, B = 201 86d2bb940, -lgc 2700, serveur neuf par passe, 0 passe nulle)
| alias | passes | TTFT solo L = 78 | banc chat b=8 t/s | J/jeton net |
|---|---|---|---|---|
| mixte | A B B A B A A B | +0,04 % | 426,05 → 426,05 (0,00 %) | +0,28 % |
| i8c | A B B A (2 + 2) | +0,20 % | −0,21 % | −0,06 % |
Dans le bruit. TTFT L = 512 : « aucun jeton reçu » aux deux bras (outil ou modèle, pas la 201) — préfill 8 × 512 mesuré
hors service (diag201 : 1 430 ms au seuil, = main).

## Critère de poste4 (ABBA b=1/b=8, certifie CERT_PUR, i8c, graphes=on)
Avant la 201 : 4/4 rouge (OOM au warm). À 86d2bb940 (prise 3b) : **4/4 vert**, graphes=on, 0 OOM, sans exil ;
b=1 70,51 / 70,50 t/s, b=8 507,85 / 507,75 t/s. À d5b9e1c45 (seuil de tranche, prise 12) : **4/4 vert**, graphes=on,
0 OOM ; b=1 70,51 / 70,52, b=8 508,59 / 508,41. PPL de poste4 (i8c, fenêtre 2048/2048, wiki-gptq) : **7,2157**, 8 188
jetons, 0 OOM — la même valeur qu'à 13 h (la tête reste tranchée au-delà de 1 Gio).

## Suite complète (prise 13, après fusion d'origin/main 3b65d8f7c, sous verrou)
3 971 verts, 1 rouge : `test_racine_modeles` — mon test i8c codait en dur le chemin du parc ; corrigé (alias ou
`ACVRAM_MODELE_I8C`), prise 14 : garde verte, test i8c 3/3 avec le modèle posé (le réel est sauté sans lui).

## Incident
15:39:51 : déconnexion USB de sde (`/mnt/2TO_2023_980PRO`, porte l'i8c), ext4 arrêté puis remonté à 15:42:16. Relecture
intégrale (prise 11) : 5 safetensors, 2 207 tenseurs, en-têtes et bornes corrects, manifeste complet — INTACT. Aucune
empreinte d'origine n'existait ; celles relevées (prise11.txt) servent désormais de référence.

LEÇONS : (1) un poids « jamais servi » se vérifie au préfill réel à plusieurs M, pas au décodage (la 153 disait n ≤ 12) ;
(2) une mémoire gardée sur un tenseur au premier appel est une allocation après la borne — la chercher au warm ;
(3) un préfill de diagnostic ne garde pas les logits de toutes les positions en fp32 (3,79 Gio) : empreinte par tranches.
