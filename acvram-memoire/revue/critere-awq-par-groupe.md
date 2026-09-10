# Critère d'acceptation : AWQ par groupe (fusion q/k/v et gate/up en nvfp4)

Écrit par poste1 le 9/09/2026, **avant toute mesure**. Trois branches :
échelle par projection (actuel, fusion impossible), échelle par groupe (fusion
possible), sans AWQ (fusion possible).

## L'unité de décision

Nous visons les **jetons/kJ à qualité donnée**. La qualité n'est donc pas un
terme d'échange : c'est une **barrière**. Un gain de débit payé par une perte de
qualité résoluble n'est pas un gain, et ne se compense par aucun chiffre
d'énergie.

Ordre imposé : la qualité barre d'abord, le débit décide ensuite.

## Étape 0 — recevabilité, avant toute conversion

Borne arithmétique du gain, calculée depuis le temps de pas actuel et les
144 GEMV supprimés par pas. **Si la borne haute est inférieure au seuil de
résolution du banc, la branche est abandonnée sans être convertie.**

    seuil = 2,8 x sigma_rel x racine(2/n)
    avec sigma_rel ~ 0,2 % et n = 4  ->  environ 0,4 %

Ne pas transporter ici les 7,8 µs/module de la fusion bf16 : deux points, même
famille, régime de graphes non isolé — la constante n'est pas déclarée
prédictive. La borne se calcule sur le temps de pas mesuré, pas sur elle.

## Étape 1 — filtre SNR (une conversion, pas de campagne)

Convertir **deux** variantes : par groupe, et sans AWQ. La seconde est gratuite
en protocole et dit si l'AWQ par groupe apporte quelque chose : si son SNR est
équivalent, le vrai choix est « par projection ou rien », et la branche du
milieu disparaît.

**Seuil, sur le tenseur le PIRE, jamais sur la moyenne** — `k` et `v` font
1024 lignes contre 5120 pour `q`, une échelle de groupe est dominée par `q`
à 71 % et les deux petites projections peuvent perdre bien plus que la moyenne :

> le SNR **minimum sur l'ensemble des tenseurs** sous échelle de groupe doit
> rester supérieur ou égal au SNR minimum actuel.

**Asymétrie à poser avant de lire** : ce filtre peut **rejeter**, il ne peut
jamais **adopter**. Un bon SNR ne prouve pas que la perplexité tient. Passer
l'étape 1 ne fait qu'autoriser l'étape 2.

## Étape 2 — perplexité, barrière de qualité

Trois points, **une variable chacun** :

    A  par projection, non fusionne     (reference actuelle)
    B  par groupe,     non fusionne     (effet de l'ECHELLE seule)
    C  par groupe,     fusionne         (effet de la FUSION seule, B -> C)

Sans B, une hausse de perplexité serait attribuée à l'échelle alors que le
chemin fusionné change l'ordre des sommes et modifie la sortie par lui-même.

Cadrage obligatoire : comparaison au **cumul**, `min_context` apparié, corpus au
sha enregistré, et **référence bf16 d'origine** — jamais une source déjà
quantifiée.

**Seuil, ancré sur notre propre budget déjà accepté** plutôt que sur une
constante inventée :

> ΔPPL(B contre A) ≤ **20 %** de ΔPPL(nvfp4 contre bf16)

Autrement dit : la dégradation ajoutée par l'échelle de groupe doit rester
petite devant celle que nous avons déjà consentie en passant au nvfp4. Si cette
dernière n'est pas mesurée, elle l'est d'abord — sans elle le seuil n'existe pas.

## Étape 3 — débit et énergie

Campagne du banc seulement si 0, 1 et 2 sont franchies. Décision sur les
jetons/kJ, la qualité étant déjà barrée. Gain retenu s'il dépasse le seuil de
résolution de l'étape 0, avec sa dispersion par passage.

## Ce qui fait échouer l'adoption

Une seule de ces conditions suffit : borne haute sous 0,4 % · un tenseur sous le
SNR minimum actuel · ΔPPL au-delà des 20 % du budget · gain de débit non
résoluble. Aucune ne se rachète par une autre.
