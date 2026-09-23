# Verdict — GLM sans AWQ sur les experts : régime NOMINAL retrouvé, mais PPL RÉFUTÉE

Manon, 15/09 soir. Correctif réel (voir `verdict-glm-regime-degrade-
15-09.md` pour l'erreur écartée avant celui-ci) : AWQ n'est plus jamais
cherché pour un tenseur d'expert MoE. Test d'acceptation Jérôme :
`acvram serve --regime` → nominal ET PPL ≤ ×1,01.

## Résultat

```
régime NOMINAL — graphes=on couches_exilées=0/47 experts_exilés=0/2944
piles_ok=True cartes=['cuda:0'] chemin_moe=mma
```

**Le régime est bien NOMINAL, piles_ok=True.** Un avertissement OOM
transitoire pendant la capture des graphes (83 Mio, sans conséquence —
même famille que P6, marge de warmup fine, pas la même chose).

**PPL 8,3076 — ratio 1,02025 vs bf16 (8,1427). RÉFUTÉ (> 1,01).**

## La tension confirmée, pas contournée

| converti | AWQ experts | régime | PPL | ratio |
|---|---|---|---:|---:|
| avant tout correctif | oui (hétérogène) | DÉGRADÉ | 8,1275 | 0,99813 |
| ce soir | non (jamais) | NOMINAL | 8,3076 | 1,02025 |
| A6 (14/09, `--no-awq` global) | non (partout) | — | 8,4415 | 1,0368 |

AWQ sur les experts vaut **−2,22 % de PPL** (8,3076 → 8,1275) — bien
plus que les « ≤ +0,3 % » prédits par Jérôme avant mesure. Retirer AWQ
des experts répare le régime (piles groupées, graphes, débit) mais
coûte la qualité au-delà du seuil. **Aucune des deux versions ne
satisfait les deux exigences à la fois** : qualité (AWQ) OU vitesse
(sans AWQ), pas les deux, avec les leviers essayés ce soir.

## Ce qui n'a pas été essayé (hors périmètre de ce soir)

- Étendre `_try_build_stacks`/le noyau groupé pour accepter une échelle
  AWQ **partagée** (une seule valeur pour tous les experts d'une pile,
  pas une par expert) — c'est exactement le mécanisme `alpha_commun_
  gate_up`/A7 construit plus tôt ce soir pour q/k/v/o, jamais branché
  sur les experts MoE. Si l'engin acceptait UNE échelle commune
  appliquée à toute la pile (pas une par expert), on garderait le
  bénéfice AWQ sans casser l'homogénéité — mais ça touche le noyau
  groupé lui-même (`model.py`/le `.cu`), pas seulement `convert.py`.
- AWQ restreint aux experts LES PLUS ROUTÉS seulement (garder l'identité
  pour les peu-routés, qui contribuaient peu au gain de −2,22 % de
  toute façon) — nécessiterait la MÊME extension moteur pour tolérer un
  mélange, donc pas un simple changement côté conversion.

## Suite

Rendu à Jérôme/Sage : le converti actuel (sans AWQ sur les experts,
NOMINAL, PPL 1,02025) est-il acceptable en attendant un travail moteur
plus profond (alpha commun sur pile d'experts), ou faut-il revenir à la
version AWQ (DÉGRADÉ, PPL 0,998) et servir en régime dégradé annoncé
comme tel (règle de Sage du 15/09 §2) jusqu'à ce travail ? Je n'ai pas
tranché ce choix moi-même — c'est un arbitrage qualité/vitesse, pas une
question de fait.
