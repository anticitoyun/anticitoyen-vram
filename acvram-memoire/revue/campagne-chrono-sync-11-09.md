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

## Prédiction 4 (non testée ici)

La question `slots=4 vs slots=12` avec `reset_peak_memory_stats` est distincte
et non tranchée par ce bras — les 25,33 Gio A/B sont identiques car sync
n'affecte pas l'allocation. À mesurer séparément (2 exemplaires `slots=4`).
