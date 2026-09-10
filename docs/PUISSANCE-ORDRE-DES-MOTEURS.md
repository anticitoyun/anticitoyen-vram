# La dérive thermique biaise l'ordre des moteurs

> **Note de renommage (9 septembre 2026)** : le modèle témoin cité ici sous
> `Qwen2.5-Coder-14B-bf16-pur` a été renommé **`Qwen2.5-Coder-14B-pur-bf16`**
> lors du rangement de `models_acvram`. Contenu identique ; seul le chemin
> change. Les chiffres publiés sous l'ancien nom restent valides.


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

## La dérive plafonne : trois minutes de chauffe suffisent

Quarante-cinq passages, même dispositif :

    passage    W      degC    ecart au palier
          1   308,61    49      -8,78
          6   310,92    50      -6,47
         12   313,62    52      -3,77
         21   315,63    53      -1,76
         30   317,52    54      +0,13     <- palier
         45   317,48    54      +0,09

    palier 317,39 W — les douze derniers passages tiennent dans 0,45 W

**Ce n'est ni linéaire ni polynomial : c'est une chauffe exponentielle amortie,
et elle atteint son palier.**

    a moins de 2,0 W du palier   passage 21   ~2 min de charge
    a moins de 1,0 W             passage 28   ~3 min
    a moins de 0,5 W             passage 30   ~3 min

**Trois minutes de décodage à vide avant chaque moteur, et le biais tombe sous
0,5 W** — sans chargement supplémentaire, sans séquence à retenir, et en
traitant la cause au lieu de la compenser.

**L'amplitude totale est de 8,78 W**, entre le premier passage et le palier. Nos
campagnes attribuent 14 W d'écart aux moteurs : **si le premier démarre froid et
le second à chaud, jusqu'à 8 W des 14 sont thermiques.** Ce n'est pas une
correction de second ordre, c'est possiblement le terme principal.

**Au palier, l'étendue tombe à 0,45 W** : un écart de 14 W vaut alors trente
fois la dispersion, et même 1 W serait mesurable.

### Protocole retenu

    3 minutes de decodage a vide AVANT chaque moteur mesure
    relever la temperature au debut ET a la fin de chaque moteur
    -> le tableau porte la preuve que les deux etaient au meme palier

Réserves : ce palier vaut pour **ce modèle, ce débit, cette température
ambiante** — un modèle plus gourmand chauffera plus haut et plus longtemps.

## Les plans d'ordre, et pourquoi ils ne servent plus

Ce qui suit reste vrai et a été vérifié, mais **la chauffe traite la cause**, ce
que ces plans ne font que compenser — et ils ne compensent qu'un biais
*polynomial*, ce qu'une exponentielle amortie n'est pas.

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
séquence de Thue-Morse `A B B A B A A B`, vérifiée nulle aux ordres 1 et 2.

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

## La limite de puissance n'a aucune prise sur notre régime

Mesuré le 9 septembre 2026, RTX 5090 :

    limite 500 W    plancher reglable 400 W    defaut 600 W

    GEMM 8192 bf16 (calcul pur)   moyenne 477,6 W   max 510,1 W   56 degC
    decodage au palier            moyenne ~317 W                  54 degC
    lecture memoire soutenue      moyenne 274,0 W                 44 degC

**Le décodage tire 317 W au plus, et le plancher réglable est 400 W :
`nvidia-smi -pl` ne peut rien écrêter sur notre régime**, quelle que soit la
valeur choisie. Le décodage est mémoire-borné.

**Un GEMM dense, lui, tire 478 W** — la limite y aurait prise. Mais le prefill
est bref devant le décodage dans une génération, et c'est le décodage qui porte
l'énergie.

**Ce qui agirait sur notre charge est le sous-voltage** — la courbe
tension/fréquence, qui réduit la consommation à fréquence égale même quand la
carte n'est pas limitée en puissance. Ce n'est pas une commande : sous Linux,
`nvidia-settings` demande X11 et les outils tiers touchent des interfaces non
supportées.

**Méthode, à garder** : la conclusion « la carte n'atteint jamais le plancher »
allait être écrite à partir du seul décodage. Elle était vraie pour ce régime et
fausse comme énoncé général — le GEMM dense la réfute. **Une réfutation tirée
d'un seul régime ne vaut que pour lui.**

**Réserve sur le protocole de chauffe** : le palier de 317 W a été mesuré sur du
**décodage pur**. Le prefill chauffe davantage (56 °C contre 54 pour une durée
bien plus courte). Une chauffe qui reproduit le régime réel — prefills compris —
est donc nécessaire ; celle du banc le fait, parce qu'elle appelle la même
fonction que la mesure.
