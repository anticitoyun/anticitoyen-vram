# campagne-275 — verdict : TENU 12/12 (3 modèles coordonnés, poste2, 01/10)

Règle scellée le 26/09 (`poste2-piece275-scelle-26-09.md`) : TENU par modèle si McNemar (p > 0,05)
sur les 4 tâches ET PPL dans la référence ± 1 %. Campagne entière TENU si les 3 modèles le sont.

## Origine en deux temps (incident documenté, `poste2.md` 01/10)

7/12 résultats viennent de `campagne8.log` (18:38:38-21:29:51, avant que mon `git merge` ne casse
HEAD sous le process détaché) : Coder-30B 4/4 + mixte-27B 3/4 (professional_law, gsm8k,
high_school_mathematics) + PPL des 3 modèles. 5/12 viennent du rejeu de ce matin (07:08-07:47, copie
figée `poste2-275-figee`, HEAD `972642262`, outil corrigé `9726422`) : mixte-27B/
college_computer_science + les 4 tâches de Qwen3-4B. Script : `scratchpad/poste2-rejeu275-01-10/rejeu5.sh`,
log complet au même endroit.

## Les 12 résultats

| modèle | tâche | mcnemar_p | tenu | source |
|---|---|---|---|---|
| Coder-30B | professional_law | 1,0 | 1 | campagne8.log 20:01 |
| Coder-30B | gsm8k | 1,0 | 1 | campagne8.log 20:02 |
| Coder-30B | high_school_mathematics | 1,0 | 1 | campagne8.log 20:04 |
| Coder-30B | college_computer_science | 1,0 | 1 | campagne8.log 20:07 |
| mixte-27B | professional_law | 1,0 | 1 | campagne8.log 20:09 |
| mixte-27B | gsm8k | 1,0 | 1 | campagne8.log 20:54 |
| mixte-27B | high_school_mathematics | 1,0 | 1 | campagne8.log 21:09 |
| mixte-27B | college_computer_science | 1,0 | 1 | rejeu5.log 07:08-07:12 |
| 4B | professional_law | 1,0 | 1 | rejeu5.log |
| 4B | gsm8k | 1,0 | 1 | rejeu5.log |
| 4B | high_school_mathematics | 1,0 | 1 | rejeu5.log |
| 4B | college_computer_science | 1,0 | 1 | rejeu5.log 07:47 |

PPL (les 3, écart relatif) : Coder-30B 9,2721/9,2721 (0,0) ; mixte-27B 6,5048/6,5048 (0,0) ; 4B
10,9174/10,9174 (0,0). Les trois tenu=1 (campagne8.log).

## Verdict

**TENU, 12/12, 3 modèles.** Aucune tâche en écart, aucun McNemar sous le seuil. Carte rendue propre
(verrou vide, vérifié indépendamment de l'annonce ; aucun processus de mesure résiduel).
