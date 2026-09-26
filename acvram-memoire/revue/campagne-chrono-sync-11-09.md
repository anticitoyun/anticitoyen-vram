# Campagne ACVRAM_CHRONO_SYNC — résultats et verdicts

poste3 (`63`), 11/09/2026. GLM-4.7-Grande-Heretic-42B, `slots=12`, `ctx=max_model_len=2048`, 15 tours par bras.

## Données brutes (médiane des médianes, 15 tours)

| bras | sync | replay_ms | pas_total_ms | prefill_ms | vram_Gio | tok/s |
|------|------|----------:|-------------:|-----------:|---------:|------:|
| A1   | non  |     0,017 |       18,829 |       73,2 | 25,3301  | 54,65 |
| A2   | non  |     0,016 |       18,819 |       73,0 | 25,3301  | 54,76 |
| B1   | oui  |    18,482 |       18,791 |       73,1 | 25,3301  | 54,69 |
| B2   | oui  |    18,590 |       18,879 |       73,2 | 25,3301  | 54,60 |

Fichiers : `sync-A1.json`, `sync-A2.json`, `sync-B1.json`, `sync-B2.json` (même répertoire).

## Verdicts

**`replay` mesure le lancement async sans sync (0,017 ms), l'exécution GPU avec (18,5 ms = 98 % du pas) : l'instrument était faux de ×1 137 — c'est le chiffre à publier pour le coût d'un rejeu.**

**`prefill_seconds` est réhabilitée : +0,07 ms sous sync (+0,09 %) confirme qu'une synchronisation existait déjà dans la fenêtre (`.tolist()` de `_emit`) ; la condamnation par lecture du 10/09 était fausse, la mesure tranche.**

## Prédiction 4 (verdict séparé)

La question `slots=4 vs slots=12` avec `reset_peak_memory_stats` est tranchée
dans [`prediction-4-vram-slots.md`](prediction-4-vram-slots.md) : **CONFIRMÉE**,
25,3301 Gio identique au bit près pour les quatre bras (C1/C2 slots=4, A1/A2 slots=12).

## CORRECTION — 13/09/2026 (bead anticitoyen-vram-x0s)

**« slots=12 » ci-dessus dimensionnait ACVRAM_HYBRID_SLOTS, mais le décodage
réel s'est fait à `b_reel=1`.** `outils/verif-max-graphs.py` appelle
`engine.generate()` avec **une seule invite par tour** : le lot concurrent
n'a jamais dépassé 1 séquence, quel que soit le réglage `SLOTS`. Un bug
distinct (`GraphRunner.run()` comparait `bucket_batch(b_reel)` à `max_slots`
au lieu de `b_reel`, cf. `part-reelle-mla-sous-graphes-13-09.md`) aurait de
toute façon empêché tout lot de 9 à 12 séquences réelles de passer par le
graphe — corrigé le 13/09 (`godet_hybride`).

`replay = 18,5 ms = 98 % du pas` **reste exact**, mais décrit le coût d'un
rejeu à **une** séquence, pas à douze concurrentes. Mesuré depuis, à
`b_reel=12` réellement concurrent et corrigé : `replay_median ≈ 91,9 ms`,
`pas_total_median ≈ 92,8 ms` (**toujours ~99 % du pas** — le rapport se
maintient, seule l'échelle absolue change avec la taille réelle du lot).
Débit agrégé : **×1,328** contre douze générations séquentielles à une
séquence (89,0 → 118,2 tok/s) — la mesure de concurrence qu'aucune campagne
n'avait faite avant le 13/09.
