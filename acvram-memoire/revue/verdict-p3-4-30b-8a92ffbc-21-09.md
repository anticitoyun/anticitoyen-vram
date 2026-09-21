# P3 (4) duel Qwen3-VL-30B sur 8a92ffbc (correctif admission +16 blocs) — 21/09

* instrument : `scratchpad/mm-qvl-20-09/chaine-p3-4.sh` (fichier suivi)
* commit : 8a92ffbc (worktree poste2-8a92ffbc-21-09, .venv reconstruit à l'instant par `install.sh`)
* régime : NOMINAL, graphes=on(hybrides≤4), kv=int8, pipeline=1, vision=bf16(eager), deepstack=3, mrope=[24,20,20]
* scellé : TTFT ≤ 0,094 s (incertain), J ≤ 40 (tenu, référence antérieure) — non atteint, voir verdict
* mesuré : rien côté TTFT/J — le serveur a chargé, chauffé (4096/4096 jetons, 4,8 s) et servi (`acvram servi 13:14:58`, healthcheck /v1/models répondu) **sans le crash de la 1re passe** (`verdict-p3-4-30b-21-09.md` : `RuntimeError: The size of tensor a (256) must match the size of tensor b (257)` à la capture de graphe) — **le correctif d'admission +16 blocs (8a92ffbc) semble tenir cette fois-ci**, à confirmer par une mesure complète.
* verdict : **ÉCHEC, harnais cassé avant la mesure** — `temoin-llamacpp-vision.py:86` : `images[0]` sur une liste vide (`IndexError`). Cause : `scratchpad/corpus-prive/images-20/` ne contient que les `.tsv`/`.sha256` (versionnés) dans ce worktree fraîchement créé — les 20 `.png` eux-mêmes ne sont pas suivis par git (scratchpad exclu) et n'existent dans AUCUN worktree actuellement accessible (cherché dans tous les `travail/poste2*`, y compris `poste2.casse-21-09`). Rien sur la carte : `-lgc` posé puis `-rgc` relâché proprement (225→2992 MHz), échec avant tout décodage.
* durée : 218 s (13:11:21–13:14:59), prévu ≤ 30 min, tenue

## Suite
Corpus d'images à reconstituer avant de pouvoir rejouer ce bras (20 `.png`, `scratchpad/corpus-prive/images-20/`, sha256 attendus dans `images-20.sha256`) — hors mon domaine de mesure, à qui détient une copie ou peut les régénérer. Une fois le corpus restauré, ce bras devrait aboutir : rien d'autre ne bloque (serveur sain, capture réussie).
