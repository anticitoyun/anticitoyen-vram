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

## Verdict

**Coder-30B seul est TENU, et seulement sur `79bd73086`** (4/4 + PPL, un seul moteur).
**mixte-27B et 4B ne sont PAS verdictables** : leurs résultats sont répartis sur deux moteurs
différents (`79bd73086` / `972642262`, diff non vide sur `acvram/`, voir ci-dessus) — 7/7 et 5/5
chacun TENU pris séparément, mais aucun des deux modèles n'a ses 4 tâches + PPL sur un seul et
même commit. Pas de « 12/12 », pas de verdict mixte-27B ni 4B pour l'instant.

**Isolation du worktree vérifiée (piège trouvé par poste1, 01/10)** : `prise-tache-275.sh:6`
(`cd "$(dirname "$0")/../.."`) place le cwd sur le worktree AVANT tout `$PY -m acvram.cli` —
Python ajoute le cwd en tête de `sys.path` pour `-m`, donc `import acvram` résout dans le
worktree, jamais dans l'arbre principal, même si `$PY` pointe sur le venv partagé. Vérifié
empiriquement : `cd poste2-275-figee && $PY -c "import acvram; print(acvram.__file__)"` →
`.../poste2-275-figee/acvram/__init__.py`. Même mécanisme pour le rejeu de 07:08 (`rejeu5.sh`,
copie conforme de `prise-tache-275.sh`) et pour le complément en cours (`completer-mixte-4b.sh`,
PPL copiée verbatim de `qualite.sh:36-41`, même `cd` hérité).

**Reste à faire** (décision chef) : rejouer mixte-27B (3/4 déjà connues sous `79bd73086`,
college_computer_science sous `972642262` — soit rejouer les 4 sous le MÊME commit, soit accepter
qu'une revue nomme les 6 commits intermédiaires comme sans effet sur ces 3 modèles avant de les
additionner) et les 4 tâches de 4B pareillement. Carte rendue propre (verrou vide, vérifié).
