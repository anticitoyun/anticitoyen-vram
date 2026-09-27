# Pièce 269 b — guet d'admission (`ACVRAM_ADMISSION_GUET=1`) : verdict — P1, P2, P3 TENUS, défaut 1 proposé

poste6, 26/09/2026, 14 h 15 → 14 h 21, sous carte.sh (ACVRAM_NOM=poste6-269b), code mesuré **6a5df2a12** (origin/poste6-269b =
main fc10e6eaa + 269 b), régime plein (chef : 97 % pour 98,7 % de cible). Scellé : `poste6-piece269b-a-sec-26-09.md` § 3,
inchangé. Instruments 262 (client + trace, perf_counter commun), fenêtre 5 ms des deux côtés, serveur neuf par bras,
**A0 B1 B1 A0 B1 A0** — 3 bras par côté, 7 tours à 12 + 5 solo chacun (21 tours à 12 et 15 solo par côté, au lieu des 14 + 10
du scellé : plus, pas moins). Carte : llama-server (PID 4219, tiers) absent au début, présent à la fin — hors verrou, hors mesure.

## Résultats (par requête : TTFT client, envoi → réponse ; par tour : trace serveur)

| | A = guet 0 (défaut) | B = guet 1 | scellé | verdict |
|---|---|---|---|---|
| tours à 12 en UN pas | 15/21 (5, 4, 6) | **21/21** (7, 7, 7) | B ≥ 12/14 | **P1 TENU** |
| TTFT p50 par requête | 251,6 ms | **247,4 ms** | B ≤ 250 | **P2 TENU** |
| p95 par tour, médiane / max | 256,2 / 271,5 | **247,8 / 263,4** | B ≤ A | tenu |
| max par tour, médiane / max | 256,3 / 272,7 | **248,0 / 263,5** | B ≤ A | tenu |
| mur médian des tours à 12 | 256,5 · 261,8 · 247,9 | 247,8 · 248,1 · 248,8 | — | −8,7 ms sur la médiane des 3 |
| solo, médiane de 15 tours | 38,4 ms | 37,7 ms | B = A ± 1 | **P3 TENU** (−0,7) |
| fenêtre des tours solo (trace) | 0,00 (12/15), 0,01 (3/15) | 0,00 (12/15), 0,01 (3/15) | 0,00 chez B | tenu : mêmes valeurs qu'A, 0,01 ms = le coût du contrôle lui-même, aucune attente |
| fenêtre des tours à 12 (trace), médiane / max | 9,94 / 10,88 (5,5 sur les tours à 2 pas) | 10,34 / 12,85 | — | le guet tient la fenêtre ouverte jusqu'à la 12e |

Issue défavorable nommée (fenêtre ≈ 20 ms puis pas de 1) : **0 tour sur 21** chez B (max 12,85 ms, tous à un pas de 12).
Issue « qui me gênerait » (P1 tenu, P2 faux) : non — p50 247,4 sous 250, marge 2,6 ms ; le plancher du pas de 12 (225-247 ms)
est bien ce que B atteint, à 0,4 ms près du meilleur pas d'A.

Ce que dit A, au passage : après la 268, A est déjà à 251,6 de p50 (262,5 dans la 269) et 15/21 à un pas (8/14) — le reste,
6/21 tours à deux pas [1, 11], est exactement le cas que le guet ferme (réveil du fil avec 1 en file, les 11 autres dans le
gabarit hors boucle), et B n'en montre aucun.

## Décision proposée (chef)

Scellé tenu sur les trois portes → **`ACVRAM_ADMISSION_GUET=1` au défaut** (test cassant « défaut 1 » dans
`tests/test_admission_guet_269b.py`, ligne CHANGELOG 0.7.3, régime), l'opt-out restant `=0`. Ce verdict ne dit rien sur :
b < 12 (non mesuré, le guet ne change rien à 1 en entrée : fenêtre fermée comme avant), les images (gabarit long, issue
nommée, 0 cas ici faute d'images), un autre modèle que le Qwen3-Coder-30B-A3B nvfp4.

Fichiers : `scratchpad/poste6-p269b-26-09/client-{12,1}-{A0,B1}-*.{json,analyse.txt}` (26), traces et journaux serveur
`~/.cache/acvram/dumps-269b/` (hors git), sortie de prise `p269b` dans le carnet.
