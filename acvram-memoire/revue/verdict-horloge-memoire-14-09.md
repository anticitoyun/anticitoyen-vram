# Verdict — l'horloge mémoire n'est PAS réglable sur cette carte (item 1)

poste3, 14/09/2026. Suite à
[`protocole-horloge-memoire-decodage-14-09.md`](protocole-horloge-memoire-decodage-14-09.md) —
campagne complète (8 cellules, 5 min 36 s), close par une preuve trouvée
en la dépouillant, pas avant.

## La preuve, lue directement

    nvidia-smi -i 0 -q -d SUPPORTED_CLOCKS

    Supported Clocks
        Memory                                         : 14001 MHz
            Graphics                                   : 3090 MHz
            Graphics                                   : 3082 MHz
            ... (dizaines de paliers graphiques)

**Une seule horloge mémoire supportée : 14 001 MHz.** Contrairement à
l'horloge SM (des dizaines de paliers discrets, confirmés par la campagne
du 13/09 qui atteignait bien 2400/2100/1800/1500 MHz), **la mémoire de
cette carte n'a AUCUN autre palier à sélectionner.** `-lmc` n'a rien à
verrouiller en dessous du maximum — ce n'est pas une limite logicielle
contournable, c'est l'absence totale d'un second point de fonctionnement.

## Ce que la campagne a révélé — les 8 cellules ont tourné avant que j'aie pu l'arrêter

Le script (5 min 36 s au total) a fini sa boucle avant que je puisse
intervenir sur la base des premières lignes de journal — **les 8 cellules
sont là, et confirment le constat de façon complète plutôt que partielle** :

    palier demandé          horloge mémoire réelle   tok/s   J/jeton net
    mem-10000, sm-défaut    14001 MHz                502,8   0,6646
    mem-12000, sm-défaut    13801 MHz                504,5   0,6651
    mem-14000, sm-défaut    13801 MHz                504,1   0,6615
    mem-stock, sm-défaut    (aucun verrou)           590,5   0,5689
    mem-10000, sm-2100      13801 / SM 2992          469,8   0,4859
    mem-12000, sm-2100      13801 / SM 2992          470,3   0,4836
    mem-14000, sm-2100      14001 / SM 2992          470,6   0,4831
    mem-stock, sm-2100      (mém. libre) / SM 2092   469,8   0,4844

**Aucune des trois demandes (10000/12000/14000) n'a jamais fait baisser
l'horloge mémoire réelle en dessous de 13 801 MHz** — cohérent avec la
liste `SUPPORTED_CLOCKS` : le pilote accepte la commande, ne peut
matériellement pas l'honorer, et ne le dit pas. Exactement le même
comportement que `-lmc 15000` documenté hier (accepté, écrêté à 13 801 sans
message). **Les quatre lignes "mem-X, sm-2100" sont quasi identiques entre
elles** (469,8-470,6 t/s, 0,4831-0,4859 J/jeton) — la variation résiduelle
est du bruit de mesure, pas un effet de l'horloge mémoire annoncée. Elles
recoupent d'ailleurs raisonnablement la campagne SM seule du 13/09 (2100 MHz
y donnait 452,1 t/s / 0,485 J — régime proche, écart plausible par la
fenêtre de repos alors trop courte, corrigée depuis à 30 s).

**Aucune des huit lignes ne teste une horloge mémoire réduite.** Elles ne
sont pas fausses en tant que telles, mais elles ne répondent PAS à la
question posée par le duel énergie (vLLM 0,202 J/jeton) — c'est le même
levier SM déjà mesuré le 13/09, pas un nouveau.

## Conséquence pour la source du duck.ai (item 1, tour de chef)

Le rapport GitHub cité (`-lmc 15000` = -26 % W en path-tracing) décrit
**un mécanisme qui n'existe pas sur cette carte, ou plus sur ce pilote** —
soit leur GPU (non précisé dans la veille) expose plusieurs paliers mémoire
et le nôtre n'en a qu'un, soit le pilote 595.91.07 a retiré cette capacité
pour GDDR7/Blackwell. **Ce levier est fermé, pas seulement inefficace** —
à la différence du balayage SM du 13/09 qui donnait un vrai compromis
(mesurable, quantifié), ici il n'y a rien à mesurer.

## Réserve

Je n'ai testé qu'`-lmc <val>,<val>` (min=max identiques, comme demandé). Un
couple `min≠max` (une plage plutôt qu'une valeur fixe) n'a pas été essayé —
peu probable que ça change la conclusion vu qu'une seule valeur est même
listée comme supportée, mais je ne l'affirme pas sans l'avoir vérifié.

## Anomalie notée, sans lien avec l'horloge

Deux OOM transitoires (`free: 31457280` octets ≈ 30 Mio) pendant cette
campagne, sur un modèle qui avait tourné sans problème la veille (mêmes
paramètres) — la carte était partagée avec poste2/une autre session à ce
moment (contention VRAM, pas un défaut du script). Sans suite ici.

## bd

Rien à créer/fermer — pas de bead pour un levier qui n'existe pas sur le
matériel. Le duel énergie garde son écart réel (0,601 vs 0,202 J/jeton) ;
ni l'horloge SM (13/09, aucun palier viable) ni l'horloge mémoire
(aujourd'hui, aucun palier du tout) ne l'expliquent ou ne le referment.
