# Scellé : témoin reprise de gemma-4-31B en KV bf16, court et long (poste6, 01/10, écrit AVANT la prise, ordre chef : 10 min après poste2 275)

Question : le 0,144 du témoin reprise de gemma (court, int8, g9m) vient-il du format int8 ou du noyau (longueur d'appel), comme Devstral
(int8 0,004 → bf16 0,022 : noyau) ? Script `travail/poste6-g9m/scratchpad/poste6-g9m-bf16.sh` (branche poste6-g9m, `ACVRAM_HYBRIDE_PAR_RECURRENCE=1`
pour que le cache serve, `ACVRAM_KV_FORMAT=bf16`, REQUETES=2 sur A1), chaînes court (7 953, ctx 10 240) et long (17 859, ctx 20 480).

| # | grandeur | prédit | FAUX si |
|---|---|---|---|
| K1 | court bf16 admis : KV 10 240 × 60 couches × (8 têtes KV × 256) × 2 × 2 o ≈ 4,9 Gio (int8 : 2,5) + 19 Gio de poids → plus de MLP exilés (8 → ~20) mais admis | admis, 15-30 MLP exilés | refus (budget) — alors la prise long tombe aussi, dit |
| K2 | long bf16 (KV ≈ 9,8 Gio) à 20 480 | **refus probable** ou 45-60 exilés avec graphes off ; si refus, « fenêtre qui tient » dite | — (issue nommée : pas de long bf16 sur cette carte) |
| K3 | témoin A1/A2 | au bit | Δ ≠ 0 |
| K4 | témoin reprise court bf16 (Δ pos 0, ids) | **0,02-0,08**, ids 32/32 — le noyau porte l'essentiel, comme Devstral (0,022) ; l'int8 double au plus | ≤ 0,01 : l'int8 était la cause du 0,144 ; ≥ 0,144 : le format ne compte pas du tout |
| K5 | morceaux B/A1 court bf16 | 0,03-0,09 (0,088 en int8) ; critère 2 × K4 : tenu si K4 ≥ 0,045 | > 0,2 |
| K6 | long bf16 si admis : témoin reprise | 0,01-0,05 | — |
| K7 | durée | ≤ 10 min (6 prises, ou 3 si long refusé) | > 12 |
Issues : (a) K1 refus → rien d'exploitable, le dire, proposer ctx 8 192 (7 953 + 32 tient) ; (b) K4 ≤ 0,01 → l'int8 est la cause sur gemma
(fenêtre glissante × 50 couches amplifie la relecture quantifiée), contrairement à Devstral — à nommer, pas à lisser ; (c) exil différent entre
bras d'une chaîne → rejeu.
