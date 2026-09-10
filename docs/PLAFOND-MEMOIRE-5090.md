# Le plafond de la 5090 en lecture : 1050 Go/s, pas 1792

Mesuré le 9 septembre 2026. Rotation de jeux pour déborder le L2 de 96 Mio, et
**une** synchronisation pour soixante appels — jamais une par appel.

    charge                             Go/s   % du pic annonce
    lecture pure (sum) 80 Mio          1025      57,2 %
    lecture pure       200 Mio         1055      58,8 %
    copie 80 Mio (lit + ecrit)         1280      71,5 %

    notre GEMV nvfp4                    991      55,3 %
    cuBLAS bf16                        1056      58,9 %
    pic annonce                        1792     100,0 %

**Une lecture pure plafonne exactement là où cuBLAS plafonne.** Ce n'est donc
pas une limite d'implémentation : c'est la bande passante réelle de la carte
**en lecture seule**. La copie, qui lit *et* écrit, monte à 1280 — le pic annoncé
suppose un trafic mixte, une lecture seule ne sature pas les contrôleurs.

**Notre GEMV nvfp4 est à 94 % du plafond atteignable**, pas à 55 % d'un pic
inaccessible.

Trois mesures indépendantes, trois dispositifs différents, même borne :

    GEMV nvfp4 isole          991 Go/s
    decodage en service       995 Go/s   (26,09 Gio en 28,14 ms)
    lecture pure             1025 Go/s

## Ce que cela ferme

    lire plus vite      6 % restants, contre une borne materielle
    ordonnancer mieux   2,1 % de temps mort (nsys)
    fusionner plus      +2,6 % pris, dentelure au-dela
    noyau nvfp4         +6,9 % pris par le double tampon

**Il ne reste que « lire moins d'octets ».** Et le chiffrage en devient sûr : à
1050 Go/s, un modèle de N Gio par pas ne descend pas sous `N × 1,0737 / 1050`
secondes. Le format ne peut pas contredire cette borne — seul le nombre
d'octets y entre.

## Deux dispositifs faux corrigés en chemin

**Le cache L2.** Une première mesure donnait **2429 Go/s pour cuBLAS**, soit 36 %
au-dessus du pic matériel. Un tenseur de 73 Mio répété reste dans les 96 Mio de
L2 : on mesurait le cache. *Un chiffre qui dépasse une borne physique est le
meilleur détecteur de dispositif faux* — il se dénonce seul.

**Le chronomètre.** Une synchronisation après **chaque** appel ajoutait
**9,0 µs** (écart-type 0,47 sur quatre formes couvrant un facteur 5 d'octets).
Un coût **fixe**, donc la sonde et non le noyau : il pesait 35 % sur un appel de
27 µs et 9 % sur un de 90, d'où une fausse « latence pire sur les petites
matrices » — la forme du défaut de la sonde, prise pour celle du noyau.

Ce second défaut ne dépassait aucune borne et ne se dénonçait pas. Il a fallu
comparer à un dispositif indépendant : l'écart constant en microsecondes,
et non proportionnel aux octets, désignait le coût par appel.
