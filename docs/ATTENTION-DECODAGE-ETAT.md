# Le noyau d'attention au décodage : où nous en sommes

État au 9 septembre 2026, fin de soirée. **Écrit pour être repris sans le
contexte de la session**, et surtout pour que rien de faux ne survive.

## Ce qui est mesuré et tient

    lanceur CPU        0,1 % du temps      evenements CUDA, montage independant
    graphes CUDA       x2,17 sur un MoE    temps GPU identique avec et sans
    pente de contexte  -12,2 %             350 -> 3000 mots, dispersion 0,2 %
    llama.cpp          x1,96 au decodage   meme source, ABBA, protocole ecrit

## Ce qui a été retiré, et pourquoi

    46,5 us de cout fixe dans le noyau     RETIRE
    49,5 us par appel                      RETIRE
    94 % de la duree independante du travail  RETIRE
    l attention a 28,7-36,5 % du pas       A REVERIFIER (meme origine)

Tous venaient d'un micro-banc qui enchaînait **300 appels sur les mêmes
tampons de sortie**. Le GPU les sérialise : chaque appel attend le précédent.
**Le chiffre était une latence de file, pas un coût de travail.**

Ce qui l'a démasqué : la bisection (`ACVRAM_PA_ETAPE`). Un noyau qui écrit
trois flottants et sort coûte **le même temps** que le noyau complet —
49,4 contre 49,5 µs. Aucune étape coupable, donc le temps n'était pas dans le
corps du noyau.

## Ce qui n'est PAS réfuté, contrairement à ce qui a été annoncé

    sous-parallelisation (0,38 bloc par SM)   NON TESTEE
    allocations par appel                     NON TESTEE
    lancement sous graphes                    NON TESTEE

Le balayage de grille qui les « réfutait » rend les mêmes chiffres au dixième
près que la bisection à l'étape 0 : **même montage, même latence**. Une grille
multipliée par huit ne pouvait pas faire bouger un chiffre qui ne mesurait pas
le travail.

**Une réfutation fondée sur une absence de variation exige le contrôle qu'on
applique partout ailleurs : quelque chose doit changer quand on bouge le
paramètre.** Un temps invariant sur un facteur 64 de grille est aussi
compatible avec une grille *inerte* qu'avec un noyau insensible.

## L'outillage est prêt, c'est le protocole qui manque

* `ACVRAM_PA_ETAPE` (0-3) — sortie anticipée du noyau, sorties neutres écrites
  et dépendantes de `sq`/`sm` pour que rien ne soit supprimé ;
* `ACVRAM_PA_CHUNK` — taille de tranche à l'exécution, donc grille variable
  sans recompiler ;
* le contrôle d'empreinte — le binaire doit porter le sha du source, sinon
  refus : `ccache` rendait des objets périmés sans que rien ne le dise.

**Ce qu'il faut écrire : un protocole qui n'enchaîne pas des appels partageant
leurs tampons.** Tampons distincts par appel, ou appels réellement
indépendants — sinon on remesure la file.

## Un candidat à vérifier en premier

**Un parcours de taille fixe** : un noyau qui balaierait toute la table de
pages, ou une boucle bornée par une capacité plutôt que par l'occupation
réelle. Cela expliquerait à la fois un temps indépendant du travail et une
insensibilité à la grille, chaque bloc refaisant le même balayage. **La
bisection le désignera immédiatement, une fois le protocole réparé.**
