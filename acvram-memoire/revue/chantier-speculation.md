# Chantier de la spéculation — écrit avant toute mesure

Trois motifs indépendants y mènent : l'occupation, l'énergie, et l'amortissement de la
projection finale. Mais **le chiffre qui commande les trois est déjà mesuré, et il est
faible.** C'est le premier résultat de ce chantier, et il tient sans carte.

## 1. Le verdict d'origine — il n'est pas « non rentable », et il n'est pas au bon régime

`ETABLI.md:454`, mesuré le 8/09 à 21h28 sur **`nemo-12b-thinking-exl3`**, deux familles
de prompts, température 0, cinq passages le premier jeté :

    regime / famille    t/s      jetons/kJ    proposes   par pas
    ngram / repetitif   96,74      320,7         500       1,163
    none  / repetitif   93,61      314,0           0       1,000
    ngram / prose       93,95      318,8         669       1,105
    none  / prose       92,55      314,3           0       1,000

    debit   x1,0334 et x1,0151      energie   x1,0213 et x1,0142
    ecart contre dispersion : 33,2 et 28,0 sigma

**Donc `ngram` gagne, et il gagne aussi en énergie** — contre une prédiction écrite
d'avance qui attendait l'inverse. Le classement « non rentable sous graphes » ne
correspond à aucune mesure que je retrouve dans le dossier : ce que je trouve est un
gain faible mais net, à 28-33 σ. **Le verdict à corriger n'est pas le signe, c'est le
régime.**

Ce que la mesure ne déclare pas, et qui la rend inapplicable au service : **ni le lot,
ni l'état des graphes**. Le modèle n'est pas non plus l'un des nôtres — `exl3`, pas un
converti `nvfp4`. Trois régimes non appariés au service réel.

## 2. Le taux d'acceptation est déjà là : 1,105 à 1,163 jeton par pas

C'est la quantité qui commande les trois motifs, et elle n'est pas à mesurer : elle est
publiée dans la même table. **`tokens_per_step` vaut 1,163 en répétitif et 1,105 en
prose** — soit 0,16 et 0,10 jeton gagné par pas, très loin du 2,71 obtenu sur une invite
artificielle de capitales, et très loin des 2 à 3 qu'une spéculation efficace suppose.

**Les trois motifs, chiffrés avec ce taux et non avec un taux espéré :**

    amortissement de la tete   la tete est lue une fois par PAS, pas par jeton.
                               A 1,163 jeton/pas l'economie vaut 1 - 1/1,163 = 14 %
                               de la tete. Or la tete pese ~1,4 % du pas sur un
                               converti nvfp4 (et non 5,6 %, voir la rectification).
                               Gain : ~0,2 % du pas. NEGLIGEABLE.

    occupation                 la grille est (BQ x q_len, HQ, C) : verifier q_len = 2
                               double le nombre de blocs. Reel, mais LE LOT FAIT DEJA
                               MIEUX — a BQ = 12 la carte est remplie sans speculer.

    energie                    MESURE : +2,1 % et +1,4 %. C'est le seul des trois qui
                               soit deja chiffre de bout en bout, et c'est petit.

**Conclusion provisoire, et elle tempère la convergence** : trois chemins indépendants
mènent bien au même chantier, mais **ils mènent à un petit gain** au taux d'acceptation
que nous mesurons. La convergence dit que le sujet est réel ; elle ne dit pas qu'il est
rentable. Confondre les deux serait prendre un faisceau d'indices pour une amplitude.

## 3. Ce qui rendrait le chantier rentable, et c'est une autre question

Le gain est une fonction croissante du taux d'acceptation. **La question utile n'est
donc pas « faut-il spéculer » mais « peut-on faire monter le taux »** :

- `ngram` propose depuis le texte déjà produit : il paie sur le répétitif (1,163) et
  peu sur la prose (1,105). C'est conforme à sa nature, pas un défaut de réglage.
- `mtp` — une tête de prédiction multi-jetons — est l'autre voie, déjà nommée dans le
  dossier (`revue/protocole-ngram-mtp.md`) et jamais mesurée. **C'est elle qu'il faut
  mesurer, pas `ngram` à nouveau.**
- Un taux de 2,0 changerait l'échelle : l'économie de tête passerait à 50 %, et le pas
  amorti à deux jetons. À 1,15, non.

## 4. Protocole, si le chantier est retenu — les trois motifs séparés

Ils ne se mesurent pas de la même façon et un montage qui les mélange ne tranche rien.

    A. TAUX D'ACCEPTATION — premier, et seul indispensable.
       tokens_per_step sur NOS modeles converti nvfp4, ngram et mtp, deux familles
       (repetitif, prose) et une troisieme reelle (code). Aucune mesure de temps.
       Critere : si le taux reste sous 1,3 sur nos modeles, les deux motifs
       suivants sont sans objet et le chantier s'arrete ici.

    B. OCCUPATION — seulement si A depasse 1,3.
       blocs = BQ x q_len x HQ x C, compte exact, pas une mesure. A comparer a ce
       que le LOT donne deja : si BQ = 12 remplit la carte, la speculation
       n'apporte pas d'occupation, elle apporte des jetons.

    C. ENERGIE ET DEBIT — a BQ realiste, graphes DECLARES, sur un converti.
       Refaire la mesure du 8/09 dans le regime du service, puisque l'originale
       ne declare ni le lot ni les graphes. ABBA, dispersion publiee, jetons/kJ
       et t/s dans la meme manche.

## 5. Les issues, écrites d'avance

    taux < 1,3 sur nos modeles          le chantier s'arrete. Les trois motifs
                                        etaient reels et le gain ne l'est pas.
    taux >= 1,3 avec mtp seulement      c'est mtp qu'il faut implanter, et ngram
                                        reste ce qu'il est : un gain de 1 a 3 %
    taux >= 2 avec l'un des deux        l'echelle change, l'amortissement de la
                                        tete devient reel et le chantier est
                                        prioritaire
    gain de debit NEGATIF a BQ = 12     la verification consomme l'occupation que
                                        le lot utilisait deja — le meme mecanisme
                                        de vagues qui menace PA_WARPS = 16

## 6. Ce que je ne prétends pas

Le taux de 1,105-1,163 est mesuré sur **`nemo-12b-thinking-exl3`**, pas sur nos
convertis, et le taux d'acceptation dépend du modèle et du domaine. Le transporter
serait exactement la faute que je viens de commettre deux fois aujourd'hui. Il vaut
comme **ordre de grandeur et comme borne basse plausible**, pas comme prédiction — et
c'est pourquoi l'étape A ne mesure rien d'autre que lui.
