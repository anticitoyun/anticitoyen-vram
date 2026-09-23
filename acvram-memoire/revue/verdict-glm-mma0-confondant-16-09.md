# Verdict — confondant W4A4/W4A16 levé : BOGUE dans alpha-commun × pile groupée

Manon, 16/09. Ordre Sage §2 (main `211faa7`), transmis par Jérôme : le
confondant jamais nommé entre nos deux PPL GLM de la nuit — 0,998 (DÉGRADÉ,
boucle par expert W4A16) contre 1,020/1,031 (NOMINAL, pile groupée W4A4) —
comparait deux arithmétiques différentes, pas seulement AWQ oui/non. Test :
mêmes deux convertis, `ACVRAM_MOE_MMA=0` (force W4A16 en prefill), même
script `acvram eval`, mêmes fenêtres.

**Table d'issues scellée par Sage AVANT mesure** : alpha-commun ≤ 1,010 ET
sans-AWQ inchangé → **bogue** ; les deux baissent d'autant → W4A4 coûte ;
inchangés → métrique.

## Résultat

```
                    MMA=1 (W4A4, ref. de la nuit)   MMA=0 (W4A16)      delta
sans-AWQ (aa963a5)  ratio 1,02025  (ppl 8,3076)     1,022388 (8,325)   +0,0021
alpha-commun        ratio 1,03086  (ppl 8,394)      1,008879 (8,215)   −0,0220
```

**alpha-commun tombe à 1,008879 (≤ 1,010) ; sans-AWQ reste inchangé
(+0,0021, dans le bruit). C'est exactement l'issue « bogue ».**

## Lecture

Le mécanisme `_precalculer_alpha_commun_experts` (`convert.py`, commit
`908e926`) lui-même n'est probablement PAS en cause : sous W4A16 (boucle
par expert, MMA=0), le converti alpha-commun approche le seuil de qualité
attendu — l'échelle forcée par paire gate/up n'est donc pas intrinsèquement
mauvaise. C'est l'INTERACTION avec la pile groupée W4A4 (`_try_build_stacks`
/ le noyau MMA groupé, Laurine) qui dégrade spécifiquement les experts
utilisant une échelle AWQ **partagée** par construction — hypothèse à
vérifier côté noyau : la table `[E,K]` construite à partir d'un alpha
commun (identique pour gate et up d'un même expert, potentiellement
identique ENTRE PLUSIEURS experts si `alpha_commun_gate_up` a convergé
vers la même valeur) pourrait heurter un chemin du noyau groupé qui suppose
une variation par expert (dé-duplication, cache, ou une hypothèse sur la
distribution des échelles). Le sans-AWQ (aucune échelle réelle nulle part,
toujours identité) ne rencontre jamais ce chemin — cohérent avec son
insensibilité au commutateur.

## Suite

Rendu à Jérôme/Sage : bogue à investiguer côté moteur (Laurine), pas côté
`convert.py`. Pas de reconversion "alpha-commun" à retenir pour un duel
avant résolution — le go de Jérôme pour la reconversion suivante utilise de
toute façon l'AWQ indépendant par tenseur (gate≠up, patch `7c3698d`), donc
ce mécanisme alpha-commun est déjà en voie d'être remplacé.
