# L'échelle d'activation se convertissait à chaque appel

`ChannelScaler.apply` faisait `x / self.scale.to(x.dtype)`. Les échelles sont
stockées en **fp16**, les activations sont en **bf16** : la conversion est
réelle, elle lance un noyau, et elle avait lieu **à chaque appel de chaque
projection**.

## Ce qui a nommé la cause — trois faits, aucun chronomètre

    337 noyaux `unrolled_elementwise` par pas, sous ncu
    337 = 7 projections x 48 couches + lm_head, exactement
    calibrate.py:145 : `self.scale.to(x.dtype)`
    verifie sur le modele charge : scale fp16, activation bf16, 336 modules

Un **compte** de noyaux ne dépend d'aucun instrument de temps — `ncu` sérialise
les nœuds d'un graphe, mais il ne les invente ni ne les fusionne. La ligne de
code et le dtype ne dépendent de rien du tout.

Le bf16, qui n'a pas d'échelle, ne lance aucun de ces noyaux : **107 noyaux
élémentaires par pas contre 923**.

## Le correctif

Cache par dtype dans `ChannelScaler`, conversion faite une fois. **Pas de
conversion au placement** : `to(device)` ne connaît pas le dtype d'exécution,
et rien ne garantit qu'un modèle n'en emploie qu'un. `to()` rend un objet neuf,
donc un cache neuf — une échelle mise en cache pour un périphérique ne doit
jamais servir sur un autre.

Numériquement identique **par construction** : c'est la même conversion, faite
une fois.

## PORTÉE : 8 modèles sur 110

**Ce correctif ne rend quelque chose que là où il y a des échelles à
convertir.** Sur les 102 modèles du parc qui n'en portent aucune, il n'a rien à
mettre en cache : **gain nul**.

    modeles portant au moins une act_scale : 8 sur 110  (7 %)
    modele mesure : Qwen2.5-Coder-14B-pur-nvfp4, 336 act_scale — le MAXIMUM du parc

**Le chiffre ci-dessous vaut donc sur 7 % des modèles, et il a été mesuré sur
celui où il est maximal.** Ce n'est pas une diminution du résultat — le
correctif reste juste et gratuit — c'est sa portée, et elle doit accompagner le
nombre partout où il est cité.

## Le gain, mesuré hors profileur

    sans cache   80,26 pas/s   12,46 ms par pas
    avec cache   86,61 pas/s   11,55 ms
    gain         +7,91 %        0,91 ms
                 sur les modeles portant des act_scale (8 sur 110), nul ailleurs

    48 jetons identiques en greedy
    temoin ACVRAM_SCALER_SANS_CACHE, avec garde verifiant qu il coupe

Le nvfp4 passe de **×2,31 à ×2,49** le bf16, même modèle, même carte.

## Ce que cela apprend sur `ncu`

    temps annonce sous ncu   1,321 ms
    gain mesure hors ncu     0,91 ms
    -> surestimation de 45 % sur ces noyaux courts

**`ncu` désignait le bon coupable avec le mauvais chiffre.** Sur des noyaux de
quelques microsecondes, ses temps sont à diviser par environ 1,45 avant d'en
tirer une prédiction — et jamais à additionner pour fermer un budget.

## Une erreur de vérification, dans le même travail

J'ai d'abord annoncé le scale « stocké en float32 », sur la foi de :

    s = ChannelScaler(torch.randn(5120, dtype=torch.float32), 0)
    print('scale stocke en', s.scale.dtype)

**J'ai lu le dtype d'un objet que je venais de fabriquer**, et je l'ai rapporté
comme celui du modèle. Le dtype réel est fp16 — vérifié sur les objets chargés,
336 modules, ce qui concorde avec la lecture du disque. Conséquence : la perte
de précision est de **3 bits de mantisse, pas 16**, et l'arbitrage « diviser en
fp32 » se pose dans des termes bien plus modestes.

## Conditions de la mesure, et ce qui ne se soustrait pas

    +7,91 % : mediane de 5 passages de 200 pas, apres 3 de rodage
              Qwen2.5-Coder-14B-pur-nvfp4, graphes actifs, aucun poids en flux
              temoin ACVRAM_SCALER_SANS_CACHE, avec garde verifiant qu'il coupe
              (dispersion non relevee — a fournir a la reprise)

**Les deux termes du +7,91 % viennent de la même série**, donc le gain tient.
**Mais il ne se soustrait pas des mesures antérieures** :

    76,82 pas/s (13,02 ms)   AVEC chauffe de 180 s
    80,26 pas/s (12,46 ms)   SANS chauffe, temoin coupe
    86,61 pas/s (11,55 ms)   SANS chauffe, cache actif

La dérive thermique vaut jusqu'à 8,78 W entre carte froide et palier, et elle
se traduit en débit. **Comparer 13,02 à 11,55 mêlerait le correctif et le
régime thermique.**

## Ce qui reste

**Le budget de 2,45 ms est périmé par ce correctif et doit être re-dérivé.**
Une soustraction naïve donnerait 1,38 ms d'inexpliqué, mais elle mêlerait deux
régimes thermiques : le chiffre est **indicatif, pas établi**, et le dossier
doit porter *« budget à re-dériver »* plutôt qu'un nombre. La mesure juste est
une reprise du nvfp4 corrigé **avec chauffe**, dans les conditions de la
référence. Les 384 divisions restantes en
sont le premier suspect, mais **replier l'échelle dans les poids est proscrit** :
le poids a été quantifié *après* mise à l'échelle, le replier après coup
quantifierait autre chose que ce qui a été optimisé.
