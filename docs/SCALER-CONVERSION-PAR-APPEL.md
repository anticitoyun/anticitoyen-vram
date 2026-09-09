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

## Le gain, mesuré hors profileur

    sans cache   80,26 pas/s   12,46 ms par pas
    avec cache   86,61 pas/s   11,55 ms
    gain         +7,91 %        0,91 ms

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

## Ce qui reste

Sur les 2,45 ms de surcoût nvfp4, **0,91 est pris, 1,54 reste inexpliqué** — et
ne se répartit pas entre les causes trouvées. Les 384 divisions restantes en
sont le premier suspect, mais **replier l'échelle dans les poids est proscrit** :
le poids a été quantifié *après* mise à l'échelle, le replier après coup
quantifierait autre chose que ce qui a été optimisé.
