# Verdict — test d'acceptation régime : DÉGRADÉ, cause confirmée (famille A6, pas le MTP)

poste2, 15/09 soir. Test d'acceptation demandé par chef avant de donner
le converti à poste3 : `acvram serve --regime` sur
`GLM-4.7-Flash-srcbf16-nvfp4` (reconverti avec le correctif MTP,
`740c91c`).

## Résultat

```
régime DÉGRADÉ — graphes=off couches_exilées=0/47 experts_exilés=0/2944
piles_ok=False (gate_proj : échelle AWQ posée sur certains experts
seulement (pas tous — repli par expert)) cartes=['cuda:0'] chemin_moe=mma
```

**PPL inchangée : 8,1275 (ratio 0,99813 vs bf16 8,1427) — toujours PASSÉ.**

## Ce que ça tranche

Le correctif MTP (`740c91c`) a fonctionné pour ce qu'il visait : plus
aucune couche ne porte un format anormal (la garde `_verifier_
homogeneite_moe` n'a rien refusé cette fois), et la PPL est identique
au converti d'avant (attendu : le bloc MTP est mort au décodage, son
format n'a jamais eu d'effet sur la qualité).

**Mais `piles_ok=False` persiste, avec un message DIFFÉRENT de celui
que j'avais isolé** : « échelle AWQ posée sur certains experts
seulement » — pas « formats mélangés ». C'est la cause que chef
avait nommée comme réserve possible : la **famille A6** (échelle AWQ
par expert hétérogène), ici via l'AWQ RÉEL activé pour ce converti
(contrairement à A6 du 14/09 qui utilisait `--no-awq`) — probablement
`search_channel_scales` convergeant vers l'échelle identité
(`best_scale = None`, ligne ~287 de `calibrate.py`) sur certains
experts et pas d'autres de la même couche, selon combien chaque expert
est routé pendant la calibration.

## Conséquence

**Le converti n'est PAS prêt pour la prise A de poste3 tel quel** :
qualité validée (PPL), mais débit dégradé (graphes off, boucle par
expert) tant que cette hétérogénéité AWQ n'est pas résolue. Deux
causes désormais distinctes et confirmées séparément sur ce modèle :
1. Format MTP mal routé — RÉSOLU (`740c91c`).
2. Échelle AWQ par expert hétérogène — OUVERT, famille A6, nécessite un
   correctif dans `calibrate.py`/`convert.py` (uniformiser l'échelle
   AWQ par groupe d'experts, ou détecter et forcer une échelle commune
   quand certains experts convergent vers l'identité et d'autres non).

## Suite

Rendu à chef/poste7 : la cause 2 est confirmée, pas seulement
soupçonnée. Pas de prise A avant un correctif ou une décision explicite
de servir en régime dégradé (38,7 ms/pas mesurés par poste3, ~25,9 t/s —
publiable seulement comme « régime dégradé », jamais comme cellule du
duel, par la règle de poste7 du 15/09 §2).
