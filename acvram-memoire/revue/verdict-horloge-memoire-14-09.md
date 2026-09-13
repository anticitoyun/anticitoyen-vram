# Verdict — l'horloge mémoire n'est PAS réglable sur cette carte (item 1)

poste3, 14/09/2026. Suite à
[`protocole-horloge-memoire-decodage-14-09.md`](protocole-horloge-memoire-decodage-14-09.md) —
campagne lancée, interrompue par la preuve elle-même avant d'aller plus
loin.

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

## Ce que la campagne a quand même révélé, avant l'arrêt

Trois cellules mesurées avant que je m'arrête : `-lmc 10000,10000` a été
**accepté sans erreur** à chaque fois, mais l'horloge relevée après coup a
toujours été **13 801 ou 14 001 MHz** — jamais 10 000. Cohérent avec la
liste ci-dessus : le pilote accepte la commande, ne peut matériellement pas
l'honorer, et ne le dit pas. Exactement le même comportement que
`-lmc 15000` documenté hier (accepté, écrêté à 13 801 sans message).

**Aucune des trois mesures collectées (`mem-10000_sm-defaut`,
`mem-10000_sm-2100`, `mem-stock_sm-2100`) ne teste une horloge mémoire
réduite** — toutes tournent à l'horloge mémoire native. Elles ne sont pas
fausses en tant que telles (le témoin SM seul, `mem-stock_sm-2100`, recoupe
d'ailleurs raisonnablement la campagne SM du 13/09 — 2100 MHz y donnait
452,1 t/s / 0,485 J, ici 469,8 t/s / 0,484 J, régime proche), mais elles ne
répondent PAS à la question posée par le duel énergie (vLLM 0,202 J/jeton).
**Je n'ai pas continué le balayage** (12000, 8000, les combinaisons
restantes) : chaque cellule suivante aurait reproduit exactement le même
non-résultat, pour le coût d'un rechargement de modèle à chaque fois.

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
