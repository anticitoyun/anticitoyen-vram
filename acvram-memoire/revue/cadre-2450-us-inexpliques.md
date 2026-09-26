# Cadre écrit d'avance : les 2,45 ms inexpliquées du surcoût nvfp4

Écrit par poste1 le 9/09/2026, **avant l'ouverture du profileur**. Objet : 86 %
du surcoût nvfp4 (2,45 ms sur 2,85, pas de 13,02 ms) sans cause attribuée.

## Règle zéro — l'instrument avant la mesure

**`ncu` ne peut pas fermer un budget de temps qu'il déforme.** Il sérialise les
noyaux : les bornes inter-noyaux y sont gonflées, et nous avons déjà retiré un
titre — « 30 % de gisement » — construit sur cet artefact (8,62 ms / 832 noyaux
= 10,36 µs par borne, incompatible avec un rejeu de graphe à 1-3 µs).

`ncu` reste le bon outil pour les **compteurs par noyau** — occupation, trafic,
instructions. Il n'est pas l'outil d'une **comptabilité de temps**.

Premier instrument : non sérialisant — `nsys`, ou une simple somme d'événements
CUDA par noyau. `ncu` vient **après**, sur les noyaux que la comptabilité aura
désignés, et jamais pour établir un total.

## Étape A — inventaire avant profilage (gratuit)

Différence d'ensembles : quels noyaux existent dans le chemin nvfp4 et pas dans
le chemin de référence, et l'inverse. C'est une **liste**, pas un profil, et elle
se lit sans mesurer.

Le nvfp4 lit environ quatre fois moins d'octets que le bf16. **S'il est plus
lent, le surcoût est dans du calcul ajouté, pas dans la mémoire** — donc dans
des noyaux *présents en plus*, ou dans des noyaux de même nom mais de forme
différente. L'inventaire oriente toute la suite pour zéro mesure, et il borne le
nombre de suspects avant qu'on en rencontre un.

## Étape B — fermer le budget AVANT d'expliquer une part

**Aucune hypothèse n'est examinée tant que la comptabilité ne se referme pas.**
Table obligatoire : somme des temps de noyaux + somme des intervalles = 2,45 ms
à la résolution près. Tant qu'elle ne boucle pas, il manque un poste, et une
cause trouvée dans un budget ouvert est une cause parmi d'autres non recensées.

C'est ce qui empêche de trouver 0,3 ms et de l'appeler « la » cause.

## Étape C — liste de suspects close avant ouverture

Écrire la liste des causes candidates **avant** de profiler, avec pour chacune
la part de 2,45 ms qu'elle prédit. Puis :

* une cause n'est retenue que si la retirer ou la modifier déplace le nombre
  **de la quantité prédite** — le signe ne suffit pas ;
* une cause rencontrée pendant le profilage et absente de la liste est **ajoutée
  à la liste puis éprouvée comme les autres**. L'ordre de découverte ne donne
  aucun crédit.

Cette dernière ligne est la garde contre l'erreur que poste2 vient d'identifier
chez elle : attribuer un écart mesuré à la piste en cours. Sur un écart six fois
plus gros, la même erreur coûte six fois plus.

## Étape D — ce qui clôt

La recherche est close quand la table de l'étape B boucle et que chaque poste au
dessus de la résolution porte une cause éprouvée à l'étape C. Un reste
inexpliqué est **publié comme tel, chiffré**, jamais réparti entre les causes
trouvées.

## Ce qui ne compte pas comme réponse

* Un mécanisme plausible sans prédiction quantitative préalable.
* Une cause établie sur un dispositif dont on n'a pas montré qu'il ne la produit
  pas — voir les deux graphes dégénérés du même jour, qui ont fabriqué une
  anomalie dans chaque direction.
* Un chiffre transporté d'un autre modèle, d'une autre taille ou d'un autre
  régime : la performance est **dentelée** en taille, et une valeur prise à une
  forme n'a pas cours à une autre.
