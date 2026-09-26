# Scellé — pièce 229, protocole à trois bras (poste3, 26/09, ordre chef)

Objectif : isoler si la régression Coder-30B-A3B-nvfp4 pur b=8 (mon banc, invites réelles,
be837ca1→main : −11,96 % débit / +32,58 % J) vient de la 209 seule ou d'une autre pièce fusionnée
entre be837ca1 et main (187, 194, 195b, 201, 212, 179 — la 209 n'est pas la seule différence).

Trois bras, MÊME banc (`banc-llamacpp-16-09.py` corrigé par la 227, invites réelles), MÊME
protocole (lots répétés 20 s, `/v1/completions` brut, comme ma 217/229), processus séparés :
* **A** = be837ca1 (worktree `poste3-p217-A`, avant toutes ces pièces)
* **B** = main au défaut (worktree `poste3-p229-B`, `ACVRAM_MARLIN_PAR_LIGNE=1`, la 209 active)
* **C** = main avec `ACVRAM_MARLIN_PAR_LIGNE=0` forcé (même code que B, 209 désactivée)

## Prédictions (avant mesure)

| comparaison | isole | prédiction débit | interprétation si tenu | interprétation si FAUX |
|---|---|---|---|---|
| B vs C | la 209 seule | **C plus rapide de 8 à 15 %** (magnitude proche des −12 % A vs B observés) | la 209 porte l'essentiel de la régression | la 209 n'explique pas (ou peu) l'écart — chercher ailleurs |
| A vs C | tout SAUF la 209 (187/194/195b/201/212/179) | **0 à +8 %** (C ≥ A, les autres pièces sont neutres ou légèrement positives sur cet alias — 195b y est active, vue au journal) | l'écart A↔B vient presque en entier de la 209 | une autre pièce que la 209 régresse aussi cet alias — bissection nécessaire |

**Falsificateur** : si A vs C dépasse ±8 % (dans un sens ou l'autre) ET B vs C reste faible (< 8 %),
la cause principale n'est PAS la 209 — c'est une des cinq autres pièces, à bissecter une par une.
Si les deux comparaisons donnent des écarts proches de zéro alors que A vs B montre bien −12 %,
un artefact du protocole (pas la 209 ni les autres pièces) resterait à chercher.

## Durée prévue

3 bras × 5 paires ABAB (ou ABC direct, à composer) ≈ 25-35 min de carte hors file d'attente.
