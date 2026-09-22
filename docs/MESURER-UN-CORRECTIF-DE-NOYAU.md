# Mesurer un correctif de noyau sans se tromper trois fois

Le 9/09/2026, le double tampon du GEMV nvfp4 a produit **trois** chiffres
avant le bon : +5,2 %, +0,9 %, puis +6,9 %. Les deux premiers étaient faux
pour deux raisons différentes, et aucune n'était visible dans le résultat.

## Le vrai chiffre

| forme   | avant (n=3)  | après (n=5)   | gain    | seuil |
|---------|--------------|---------------|---------|-------|
| qkv     |  835,1 ± 4,1 |  930,5 ± 0,5  | +11,4 % | 0,81 %|
| o       |  830,7 ± 4,6 |  890,8 ± 0,2  |  +7,2 % | 0,89 %|
| gate_up |  972,7 ± 0,2 | 1046,6 ± 1,3  |  +7,6 % | 0,22 %|
| down    |  970,9 ± 0,6 |  983,3 ± 0,7  |  +1,3 % | 0,15 %|
| **moyenne** |          |               | **+6,9 %** | |

Seuil de détection : 2,8 · σ · √(2/n). Les sorties des deux versions sont
identiques **bit à bit**, écart maximal exactement 0.

## Erreur 1 — une mesure contre une mesure

Le premier chiffre, +5,2 %, comparait une exécution unique de chaque côté.
Deux séries du **même** code diffèrent de 0,9 % selon l'ordre — assez pour
déplacer le résultat de plusieurs points. Une mesure unique n'a pas de barre
d'erreur, donc elle n'a pas de seuil, donc elle ne peut rien trancher.

## Erreur 2 — le correctif comparé à lui-même

Le second chiffre, +0,9 %, venait de cinq passages contre cinq. Mais la
commande de restauration n'avait rien restauré :

    git stash push -- acvram/kernels/acvram_kernels.cu

s'exécutait sur un arbre **propre**, le correctif étant déjà commité. Rien à
remiser, donc rien de remisé, donc le fichier inchangé — et les cinq passages
« d'avant » ont recompilé le code corrigé.

**Un stash n'a de sens que s'il y a quelque chose à remiser.** Restaurer une
version se fait par :

    git checkout <rev> -- <fichier>

dont l'effet se **vérifie** avant de mesurer :

    grep -c "Double tampon" acvram/kernels/acvram_kernels.cu   # 0 avant, 1 après

## Ce qui rend l'erreur 2 dangereuse

Le résultat était **plausible** : gain homogène de +0,7 à +0,9 % sur quatre
formes, écart-type minuscule, tout au-dessus du seuil de détection. Un tableau
propre. Rien dans le chiffre ne disait qu'il mesurait une dérive entre séries
plutôt qu'un correctif.

*Le plausible passe le contrôle que l'aberrant ne passerait pas.* Un chiffre
absurde se fait attraper ; un chiffre crédible s'installe. La parade n'est pas
de relire le résultat mais de **vérifier l'état du dispositif avant de le
lancer**, par une propriété observable — ici, une occurrence dans un fichier.

## Recette

1. `git checkout <rev> -- <fichier>`, jamais `git stash` pour ça.
2. Vérifier par `grep` que le fichier est bien celui qu'on croit.
3. Cinq passages de chaque côté, jamais un seul.
4. Comparer au seuil 2,8 · σ · √(2/n).
5. Vérifier que les sorties sont inchangées, bit à bit quand l'arithmétique
   ne doit pas bouger.
6. Écrire le message de commit dans un **fichier** et le passer par
   `git commit -F` : `-m` avec des accents graves fait exécuter leur contenu
   par bash, qui les remplace par du vide. Deux passages ont ainsi disparu du
   message de `9660af3`, comme d'un commit antérieur du même dossier.
