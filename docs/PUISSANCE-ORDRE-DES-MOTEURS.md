# La dérive thermique biaise l'ordre des moteurs

Mesuré le 9 septembre 2026, RTX 5090, `Qwen2.5-Coder-14B-bf16-pur`. Douze
passages **identiques** de 200 pas de décodage, un seul chargement, graphes
actifs, aucun poids en flux, puissance échantillonnée à 50 Hz par `pynvml`.

## La question posée : quel écart est résoluble ?

    ecart-type    1,30 W
    etendue       4,01 W
    dispersion    0,42 %

**Un écart de 14 W entre moteurs vaut 3,5 fois l'étendue totale : il est
résoluble.** Sans ce chiffre, comparer −4,7 % et −3,9 % de jetons par kilojoule
n'avait pas de sens.

## Ce que la mesure a sorti en plus : une dérive, pas du bruit

    passage 1     306,54 W   45 degC
    passage 6     308,33 W   47 degC
    passage 12    310,55 W   50 degC

    trois premiers 306,88 W  ->  trois derniers 309,89 W   (+3,01 W)

La puissance croît à chaque passage sans jamais redescendre, en parallèle de la
température. **L'étendue de 4,01 W est presque entièrement cette dérive.**

**Conséquence : une campagne mesure un moteur puis l'autre sur la même carte,
donc le second est mesuré chaud.** Le biais va toujours à son détriment. Comme
acvram passe en premier depuis le début, **notre avantage énergétique publié est
un plancher, pas un plafond.**

## Alterner ne corrige rien

Le biais est proportionnel à l'écart des **barycentres temporels** des deux
moteurs. Moments d'ordre 1 et 2 des positions, écart B − A :

    plan                       n   ordre 1   ordre 2
    A B (actuel)               2      1,00      1,00
    A B A B A B (alterne)      6      1,00      5,00   <- inchange, 3x le prix
    A A B B                    4      2,00     10,00   <- le pire
    A B B A                    4      0,00     -2,00   <- ordre 1 annule
    A B B A A B B A            8      0,00     -2,00   <- rien de plus
    A B B A B A A B            8      0,00      0,00   <- Thue-Morse

**L'alternance répartit du bruit ; elle ne corrige pas une tendance.** Chaque
paire y laisse B après A, donc le biais survit intact pour trois fois le prix.

**`A B B A` égalise les barycentres** et annule une dérive linéaire, pour deux
chargements de plus que l'actuel — environ une minute par moteur, pas dix.

**Doubler `ABBA` n'ajoute aucun ordre** et dégrade l'ordre 3 (de −9 à −21). Pour
annuler le terme quadratique il faut **inverser** la seconde moitié : la
séquence de Thue-Morse `A B B A B A A B`.

## Limite, à déclarer plutôt qu'à cacher

Ces plans annulent un biais **polynomial et déterministe**. Ils ne font rien
contre un **saut** — un ventilateur qui accélère, une bascule d'état d'horloge,
un autre processus qui prend la carte. Douze passages sur cinq degrés ne
prouvent pas qu'il n'y a pas de palier plus loin.

**Le plan réduit le résidu ; il ne dispense pas de le mesurer et de l'écrire.**
Un biais déclaré est utilisable, un biais tu ne l'est pas.

## Piège de dispositif rencontré

Onze passages sur vingt-trois se terminaient sur un **EOS** avant les 200 pas.
Leur moyenne de puissance portait alors sur un régime partiellement **oisif** —
d'où des débits de 560 pas/s, qui auraient pu passer pour une découverte plutôt
que pour un défaut de garde. Compter les pas réellement faits, et rejeter le
passage sinon.
