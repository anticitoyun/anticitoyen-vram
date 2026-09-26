# Scellé — pièce 229 (poste3, 26/09, ordre chef) : rejeu de la 217 au banc corrigé (227, invites réelles)

A = be837ca1 (worktree `poste3-p217-A`, inchangé depuis la 217). B = origin/main (1f72f910f,
226b + 227 + addendum 217 inclus, `ACVRAM_MARLIN_PAR_LIGNE=1` au défaut). Instrument :
`outils/carte.sh` + `cellule-217.sh` (gabarit inchangé) MAIS avec `scratchpad/banc-llamacpp-16-09.py`
de B (227 : `invite_pour` bascule vers un texte réel tokenisé pour tout modèle MoE en génération
libre). Prises `poste3-p229-*`, `ACVRAM_ATTENTE=5400`.

## Prédictions (avant mesure)

| cellule | débit prédit | J/jeton prédit | source de la prédiction |
|---|---|---|---|
| Coder-30B-A3B-nvfp4 b=8 | **+10 à +14 %** | **−15 à −21 %** | 226 (poste6) : PAR_LIGNE seul, invites réelles, +12,8 % / −18,4 % — bande ±2 pt sur le débit, ±3 pt sur J (composition/jour différents, leçon 190) |
| Coder-30B-A3B-nvfp4 b=1 | **0 à +3 %** | **±3 %** | 209c (poste6) : +0,52 % à b=1 (GEMV naturel domine déjà) ; ma propre 217 b=1 avait donné +0,24 %/−1,10 % avec le banc FAUX — cohérent car b=1 n'est pas MoE-sensible aux invites tirées (une seule séquence par lot, pas de divergence croisée à corriger) |
| Qwen3.8-27B-unsloth-mixte-i8c b=8 | **dans ±2 pt de la 217** (débit 4 à 8 %, J −3,4 à −7,4 %) | idem | mixte n'est PAS un modèle MoE (pas de routage d'experts) — le correctif 227 ne devrait rien y changer ; cellule répétée comme FALSIFICATEUR du correctif lui-même : si elle sort de cette bande, le banc corrigé a introduit un autre biais |

**Falsificateur** : mixte hors de ±2 pt de la 217 (6,02 % débit / −5,44 % J) → le correctif 227 a un
effet secondaire sur un modèle non-MoE, à investiguer avant de faire confiance au reste. Coder b=8
hors de +10-14 % → soit la 226 n'était pas non plus généralisable (invites réelles différentes
d'un banc à l'autre), soit un autre facteur (195b, KV) intervient à ce commit précis de B.

## Durée prévue

3 cellules ABAB×5, comparable à la 217 (~20-25 min chacune hors file d'attente).
