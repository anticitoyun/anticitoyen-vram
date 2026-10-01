# campagne-275 — verdict : DEUX moteurs distincts, 7/7 et 5/5 PAS additionnés (poste2, 01/10, corrigé après revue chef)

Règle scellée le 26/09 (`poste2-piece275-scelle-26-09.md`) : TENU par modèle si McNemar (p > 0,05)
sur les 4 tâches ET PPL dans la référence ± 1 %.

**Correction (chef, 01/10) : ma première version additionnait 7+5 = « TENU 12/12 » sans vérifier
que les deux lots avaient tourné sur le même moteur.** Ce n'est pas le cas — voir ci-dessous.

## Origine en deux temps (incident documenté, `poste2.md` 01/10)

7/12 résultats viennent de `campagne8.log` (18:38:38-21:29:51), commit `79bd73086` — preuve : la
ligne même qui a interrompu la suite, `REFUS : HEAD 3d74dc3eb != 79bd73086` (campagne8.log:164,
174), confirme que `ATTENDU=79bd73086` pour toute la série (cohérent avec `notes-lancement.md`
entrée 8, décision reçue 17:27:42 sur ce commit). Coder-30B 4/4 + mixte-27B 3/4 (professional_law,
gsm8k, high_school_mathematics) + PPL des 3 modèles.

5/12 viennent du rejeu de ce matin (07:08-07:47, copie figée `poste2-275-figee`), commit
`972642262` (outil corrigé `9726422`) : mixte-27B/college_computer_science + les 4 tâches de
Qwen3-4B. Script : `scratchpad/poste2-rejeu275-01-10/rejeu5.sh`, log complet au même endroit.

**`git diff --stat 79bd73086 972642262 -- acvram/` n'est PAS vide** :

```
acvram/__init__.py        |  2 +-
acvram/cli.py             |  8 +++--
acvram/engine/loader.py   | 91 ++++++++++++++++++++++++++++++++++++++++-------
acvram/quant/calibrate.py | 17 ++++++++-
acvram/quant/convert.py   | 12 +++++--
acvram/quant/nvfp4.py     | 79 ++++++++++++++++++++++++++++++++++++++--
acvram/server/chat.py     | 23 +++++++++++-
7 files changed, 208 insertions(+), 24 deletions(-)
```

6 commits entre les deux (`11de7765e`, `06e578c17`, `87d8bfe0a`, `16158bad0`, `3159e69fe`,
`2f3044f57` — kv31b réserve de préfill, refus KV nommé, 0.7.16, arguments d'outil chat/completions,
`--echelle` convert, ScaleSweep nvfp4). `loader.py`/`chat.py` sont sur le chemin de service réel :
**les deux lots N'ONT PAS tourné sur le même moteur. Pas de 12/12.**

## Lot A — commit `79bd73086` (campagne8.log, 18:38-21:29), 7/7 tenu

| modèle | tâche | mcnemar_p | tenu |
|---|---|---|---|
| Coder-30B | professional_law | 1,0 | 1 |
| Coder-30B | gsm8k | 1,0 | 1 |
| Coder-30B | high_school_mathematics | 1,0 | 1 |
| Coder-30B | college_computer_science | 1,0 | 1 |
| mixte-27B | professional_law | 1,0 | 1 |
| mixte-27B | gsm8k | 1,0 | 1 |
| mixte-27B | high_school_mathematics | 1,0 | 1 |

PPL : Coder-30B 9,2721/9,2721 (0,0) tenu=1 ; mixte-27B 6,5048/6,5048 (0,0) tenu=1 (les 2 seuls
modèles complets sur ce lot — 4B n'a que sa PPL ici, 10,9174/10,9174, tenu=1, ses 4 tâches
manquent toujours sur CE commit).

**Coder-30B : TENU (4/4 + PPL), sur `79bd73086`.** mixte-27B incomplet sur ce commit (3/4, pas de
verdict modèle possible). 4B incomplet (PPL seule).

## Lot B — commit `972642262` (rejeu5.log, 07:08-07:47), 5/5 tenu

| modèle | tâche | mcnemar_p | tenu |
|---|---|---|---|
| mixte-27B | college_computer_science | 1,0 | 1 |
| 4B | professional_law | 1,0 | 1 |
| 4B | gsm8k | 1,0 | 1 |
| 4B | high_school_mathematics | 1,0 | 1 |
| 4B | college_computer_science | 1,0 | 1 |

Ni mixte-27B (3/4 manquant sur ce commit, pas rejoué) ni 4B (PPL manquante sur ce commit, pas
rejouée) n'ont leurs 4 tâches + PPL réunies sur le MÊME commit — aucun verdict modèle complet
possible ici non plus.

## 01/10 10h32 : complément mixte-27B/4B FINI — les 3 modèles verdictables, sur 2 commits

Rejeu de ce matin (07:08-07:12, `rejeu5.log`) + complément de 09:11-10:32 (`completer2.log`,
après le TIMEOUT du 08:42 corrigé par `ACVRAM_DUREE_MAX=3600` sur `professional_law` et le
script sans `set -e`) : mixte-27B et Qwen3-4B ont maintenant chacun leurs 4 tâches + PPL
réunies sur le MÊME commit `972642262` :

| modèle | tâche | mcnemar_p | tenu | source |
|---|---|---|---|---|
| mixte-27B | college_computer_science | 1,0 | 1 | rejeu5.log 07:08 |
| mixte-27B | professional_law | 1,0 | 1 | completer2.log 09:13-09:57 |
| mixte-27B | gsm8k | 1,0 | 1 | completer2.log 09:57 |
| mixte-27B | high_school_mathematics | 1,0 | 1 | completer2.log 10:12 |
| mixte-27B | PPL | — | 1 (6,5048/6,5048, écart 0,0) | completer2.log 09:11 |
| 4B | professional_law | 1,0 | 1 | rejeu5.log |
| 4B | gsm8k | 1,0 | 1 | rejeu5.log |
| 4B | high_school_mathematics | 1,0 | 1 | rejeu5.log |
| 4B | college_computer_science | 1,0 | 1 | rejeu5.log 07:47 |
| 4B | PPL | — | 1 (10,9174/10,9174, écart 0,0) | completer2.log 10:32 |

## Verdict

**Coder-30B : TENU sur `79bd73086`** (4/4 + PPL, un seul moteur, le seul commit où il est
complet).
**mixte-27B : TENU sur `972642262`** (4/4 + PPL, un seul moteur — complet depuis 10:32).
**Qwen3-4B : TENU sur `972642262`** (4/4 + PPL, un seul moteur — complet depuis 10:32).

Les 3 modèles sont TENU, mais **sur 2 commits différents** (`79bd73086` pour Coder-30B,
`972642262` pour mixte-27B et 4B) — toujours PAS « 12/12 sur un seul moteur » au sens strict de
la règle du 26/09, puisque le diff `acvram/` entre les deux commits n'est pas vide. C'est un
résultat suffisant pour clore la campagne (chaque modèle a un verdict propre, traçable, sur un
commit nommé) mais pas la preuve qu'un SEUL moteur passe les 3 — cette preuve demanderait de
rejouer Coder-30B sur `972642262` (ou les deux autres sur `79bd73086`), hors mandat de ce
rejeu (décision Coder-30B/revalidation laissée à chef si besoin).

**Isolation du worktree vérifiée (piège trouvé par poste1, 01/10)** : `prise-tache-275.sh:6`
(`cd "$(dirname "$0")/../.."`) place le cwd sur le worktree AVANT tout `$PY -m acvram.cli` —
Python ajoute le cwd en tête de `sys.path` pour `-m`, donc `import acvram` résout dans le
worktree, jamais dans l'arbre principal, même si `$PY` pointe sur le venv partagé. Vérifié
empiriquement : `cd poste2-275-figee && $PY -c "import acvram; print(acvram.__file__)"` →
`.../poste2-275-figee/acvram/__init__.py`. Même mécanisme pour le rejeu de 07:08 (`rejeu5.sh`,
copie conforme de `prise-tache-275.sh`) et pour le complément en cours (`completer-mixte-4b.sh`,
PPL copiée verbatim de `qualite.sh:36-41`, même `cd` hérité).

**FAIT** (01/10 10h32) : mixte-27B et 4B rejoués en entier sous `972642262` (voir section
ci-dessus). Les 3 modèles sont chacun TENU sur un commit unique, propre. Carte rendue propre
(verrou vide, vérifié). Campagne-275 close de mon côté.
