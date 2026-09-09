# Barrière de qualité : protocole, écrit avant la première mesure

Six correctifs ont atterri le 9 septembre 2026. **Aucun n'a été confronté à
l'étalon.** Cinq se disent numériquement neutres, un change les bits par
construction. Ce document fixe ce qui sera mesuré, comment, et **comment on
constate que l'instrument pouvait rendre autre chose**.

## L'étalon

    corpus   /mnt/AI_GENERATOR/corpus/wiki.test.raw
    sha256   173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08
    taille   1 290 590 octets

**Le sha est vérifié avant chaque mesure**, pas supposé. Un corpus différent
rendrait une perplexité différente sans que rien ne le signale.

## Le cadrage

    acvram eval <modele> --corpus wiki.test.raw \
        --window 512 --stride 512 --min-context 256 --max-tokens 16384

**32 fenêtres**, tranché par poste1. Le cadrage fait le chiffre : le même
modèle sur le même corpus donne **7,233** à `min_context` 256 et **9,525** à 0.

**Un modèle par processus.** `acvram eval` chargeait N modèles sans libérer le
premier ; le second était alors mesuré sur les restes — constaté ici même, OOM
à 111 Mio libres sur 31,36 Gio après 32 MLP exilés.

## Le témoin d'instrument — sans lui, une réussite ne démontre rien

**Un dispositif qui rend « conforme » à tous les coups ne mesure peut-être
rien.** Avant de conclure quoi que ce soit, la barrière doit montrer qu'elle
sait dire non :

    temoin NEGATIF   le meme modele, deux executions   ->  ecart nul attendu
    temoin POSITIF   bf16 contre nvfp4, meme source    ->  ecart NON NUL attendu

Le second est une différence **connue et de sens connu** : la quantification
dégrade. **Si la barrière ne distingue pas bf16 de nvfp4, elle ne mesure pas la
qualité** et son verdict sur les correctifs ne vaut rien.

## Le seuil

    ΔPPL / PPL <= 0,1 %

Ancré et non posé : la divergence d'ordre de sommation vaut 5,4e-3 sur des
activations, et une perplexité l'amortit d'au moins un ordre de grandeur.

**Ne pas utiliser le ± de l'estimation.** Il mesure la dispersion **entre
morceaux du corpus**, commune aux deux exécutions puisque c'est le même corpus
dans le même ordre : **elle s'annule dans la différence.** La prendre pour
l'erreur de la différence serait la propriété voisine sur le dernier contrôle
de la journée.

## Les classes

    classe A   identique au bit pres   ->  PPL strictement egale, tout ecart = defaut
               f543e9d, eb4c099, 04d0da7, cache d echelle
    classe B   bits differents         ->  residu numerique attendu
               095b5ef (ordre des sommes), 225aa5f (float32)

**L'ensemble hérite de la classe la plus faible.**

## L'« avant »

**Produit par le code, jamais repris d'un journal.** Un `git stash push` sur
arbre propre ne remise rien et laisse mesurer un correctif contre lui-même —
cinq passages plausibles, +0,9 % annoncé là où le vrai gain valait +6,9 %.

**Vérification de l'état du dispositif avant chaque mesure**, par une propriété
observable du binaire, pas par la confiance dans la commande qui l'a produit.


## Ce que la carte borne : les bf16 de 14B ne s'étalonnent pas contre eux-mêmes

Première exécution, `Qwen2.5-Coder-14B-pur-bf16` :

    carte 30,85 Gio libres sur 31,36 au depart
    plan reajuste : 5 MLP de plus en RAM hote (27,3 Gio pour 29,6 libres)
    arene epinglee : 1,98 Gio
    OutOfMemoryError : 55,88 Mio libres, il manquait 136 Mio

**Ce n'est pas un incident, c'est une borne.** Un bf16 de cette taille exile
déjà cinq MLP au chargement et ne laisse pas de quoi évaluer. **La classe de
modèles que cette barrière peut étalonner contre leur propre bf16 s'arrête donc
en dessous de 14 milliards de paramètres sur cette carte.**

Conséquence pratique : le témoin positif se fabrique sur un modèle **petit**,
converti deux fois depuis la même source.

## État de la barrière au 9 septembre 2026, 20 h

    temoin NEGATIF   PASSE   nvfp4 deux fois : 7,270 et 7,270, deux processus
    temoin POSITIF   ABSENT  le bf16 de 14B ne tient pas a l'evaluation

**La barrière a la reproductibilité et n'a pas la sensibilité.** Les deux sont
nécessaires ; seule la première est acquise. **Statut : suspendue faute de
témoin**, et non « passée ».

Le couple à fabriquer : `Qwen3-0.6B`, même source GGUF, converti en nvfp4
(existe) et en bf16 (à produire). Écart attendu **connu de signe** — le bf16
doit être meilleur, le nvfp4 étant une approximation du même modèle. Deux
chiffres identiques sur ce couple signifieraient un instrument aveugle.
