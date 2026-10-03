# g6r, stabilité du régime à 65 536 : scellé AVANT le code (poste6, 02/10 13 h 1x, ordre chef)

Constat (carte, `poste6-g6r-plongements-verdict-carte-02-10.md`) : le régime alterne d'un chargement à l'autre — NOMINAL
(pic de préfill Marlin 4,09 Gio pour 3,35 d'activations réservées, +11 Kio/jeton enregistrés), puis DÉGRADÉ à 2 MLP exilés
(pic 1,93, excès 0), puis NOMINAL. Ordre : ne pas compter les tampons denses dans la réserve quand le plan n'exile rien ;
prédiction et seuils avant ; test qui casse si l'alternance revient.

## Lecture (fichier:ligne, e3c5d15b7+)

* `_reserve_prefill` (`loader.py`) ajoute TOUJOURS `_DENSE_SLOTS` × la plus grosse couche (4 × 0,26 Gio = 1,02-1,22 Gio sur
  gemma) : la place du pool de tampons des poids denses EXILÉS (`_pool_dense`, `loader.py:339`). Sans poids exilé, ce pool
  n'est jamais créé — mais sa réserve peut à elle seule déclencher l'exil qui le rend nécessaire. C'est ce qui arrive au
  second chargement : 3,35 → 2,95 + 0,74 d'excès + 1,22 de tampons, 2 MLP sortent pour 0,36 Gio.
* La chauffe compare le pic (tout le transitoire) à `activations_prefill_bytes` seul ; la réserve du plan, elle, portait
  3,35 + 0,12 (Marlin) + 1,22 (tampons) = 4,69 Gio ≥ 4,09 : au premier chargement c'est la réserve des tampons, inutilisée,
  qui couvrait le pic du chemin Marlin.

## Correctif

Le plan garde sa règle (réserve complète) tant qu'il n'exile rien — tout plan qui tenait reste identique. S'il exile des
MLP, la même chaîne est rejouée SANS la réserve des tampons denses (`plan.sans_tampons_denses`) et retenue seulement si
elle n'exile AUCUN poids (alors le pool n'existera pas) ; au moindre exil pendant ce rejeu, la réserve complète revient.
Ordre des essais : KV plein, KV plein sans tampons, anneau, anneau sans tampons ; le premier sans exil gagne.

## Prédictions

| | prédit | faux si |
|---|---|---|
| T1 à sec, 65 536, excès 0 (1er chargement) | plan IDENTIQUE à celui d'aujourd'hui : anneau, table hôte, 0 exilé, KV 5,45 Gio, 4 096 blocs, plafond 5 120 | un champ diffère |
| T2 à sec, 65 536, excès 11 Kio/jeton (2e chargement) | **0 MLP exilé** (2 sur carte aujourd'hui), anneau, table hôte, mêmes KV et blocs ; plafond 4 096 | ≥ 1 exilé |
| T3 à sec, excès 22 Kio/jeton (double du mesuré) | 0 exilé (marge restante ≈ 0,5 Gio après l'excès mesuré : 22 peut échouer — annoncé : 0 à 2 exilés) | — (non décisif) |
| T4 témoins 4 096 / 8 192 / 16 384 / 27 648 / 41 984, excès 0 | résumés IDENTIQUES à 376a4fe99 | un champ diffère |
| T5 `ACVRAM_KV_ANNEAU=0`, 65 536 | refus, comme avant | autre chose |
| **Carte, deux chargements consécutifs à 65 536** | **même régime (NOMINAL, 0/60) aux deux ; mêmes 32 ids (sha 5aabb6d5… attendu, celui du NOMINAL d'aujourd'hui) ; pic de chauffe ≤ réserve de préfill du plan aux deux** (1er : 4,09 ≤ 3,35 + 0,12 + … ; à lire sur le journal) | régime différent, ids différents, ou pic > réservé |

Seuil du verdict carte (chef) : les trois conditions aux DEUX chargements ; une seule manquante = non tenu.
Issue qui me gênerait : au second chargement la réserve sans tampons (2,95 + 0,74 + 0,12 = 3,81 Gio) passe SOUS le pic
mesuré (4,09) — « pic ≤ réservé » serait alors faux par 0,28 Gio même en NOMINAL stable ; je le lirai tel quel.

## Résultats à sec (02/10 13 h 2x, APRÈS le scellé ; rien ci-dessus n'a été retouché)

| | prédit | à sec | |
|---|---|---|---|
| T1 65 536, excès 0 | plan identique | identique (0 exilé, KV 5,45 Gio, 4 096 blocs, plafond 5 120, réserve complète 4,69 Gio) | tenu |
| T2 65 536, excès 11 Kio/jeton | 0 exilé, plafond 4 096 | **0 exilé**, mêmes KV et blocs, réserve sans tampons 4,66 Gio ; plafond **6 144**, non 4 096 | tenu sur l'exil ; détail du plafond FAUX |
| T3 excès 22 | 0 à 2 exilés | 0 exilé, plafond 4 096 | — |
| T4 témoins | identiques | 4 096 / 8 192 / 27 648 / 41 984 identiques ; **16 384 identique seulement après réordonnancement** | tenu au 2e essai, dit |
| T5 anneau interdit | refus | refus | tenu |

Écart au correctif scellé, dit : j'avais écrit l'ordre « KV plein, KV plein sans tampons, anneau, anneau sans tampons ».
À 16 384 le KV plein sans tampons tenait sans exil (table en RAM hôte, KV 8,1 Gio) et remplaçait l'anneau qui tenait déjà :
un plan qui tenait changeait. Ordre retenu : KV plein, anneau, puis les deux sans tampons — la réserve allégée n'entre en jeu
que si aucun plan à réserve complète ne tient sans exil.
Tests : `test_le_regime_ne_change_pas_au_second_chargement` (cherche sur la réplique l'excès qui fait exiler la réserve
complète, puis exige 0 exilé et le même plan avec le correctif) et `test_un_exil_rend_la_reserve_des_tampons` ; cassure
vérifiée sur copie (réserve toujours comptée : 1 rouge). Lot ciblé 8 fichiers : 49 passés, joué de 13:18:08 à 13:18:20, entre
la rendue de poste5 (`mesure`, 13:18:07) et la prise d'poste1 (`service`).
Aveu : mes rejeux de simulation (≈ 40 s de processeur, un cœur, nice 19) ont tourné pendant la prise `mesure` poste5-zzs
(13:11-13:18) ; aucun pytest.

Reste carte : `scratchpad/poste6-g6r/carte-g6r-2.sh` (deux chargements consécutifs à 65 536, puis ids) — seuil : même régime,
mêmes ids, pic ≤ réservé, aux DEUX chargements.
