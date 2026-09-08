# Protocole du microbanc GEMV — q3n contre NVFP4, à formes égales

Écrit **avant** toute mesure, le 8 septembre 2026. Il vaut pour la comparaison
de deux noyaux GEMV lisant des poids de formats différents, et il est conçu
pour qu'un gain annoncé soit défendable ou explicitement refusé.

Le chiffre attendu, posé d'avance : q3n à 3,25 bits par poids contre NVFP4 à
4,5 bits effectifs (4 bits + une échelle FP8 par bloc de 16) lit **27,8 %
d'octets en moins**. Contre 4,25 bits effectifs (échelle par bloc de 32), ce
serait 23,5 % ; contre 4 bits nus, 18,8 %. **Ce que « 28 % » suppose doit être
écrit à côté du chiffre**, sinon on comparera un gain mesuré à une prédiction
dont personne ne connaît l'hypothèse.

---

## 1. Ce que le microbanc doit décider

Un GEMV de décodage lit toute la matrice et fait une seule passe : il est
borné par la mémoire, sauf si le décodage des bits coûte plus que la lecture.
La question est donc unique et binaire :

> Le temps suit-il les octets lus, ou le décodage prend-il le dessus ?

Elle se tranche par une grandeur, et une seule : **la bande passante atteinte**
par chaque noyau, octets lus divisés par temps. Le temps seul ne la tranche
pas.

- q3n atteint la même bande passante que NVFP4 → le temps suit les octets, le
  gain doit valoir 27,8 %, le format tient sa promesse ;
- q3n atteint une bande passante plus basse → le décodage des bits coûte, et
  l'écart entre les deux bandes passantes chiffre exactement ce coût ;
- q3n atteint une bande passante plus HAUTE → on ne mesure pas ce qu'on croit :
  très probablement le cache L2 (voir §4), pas la mémoire.

---

## 2. Formes à couvrir

Les formes réelles du modèle d'abord, les formes synthétiques ensuite.

1. **Les projections réelles** de Coder-Next, par expert : les trois formes
   `(N, K)` telles qu'elles sortent du manifeste, pas des tailles rondes
   choisies pour faire joli. C'est ce qui sera exécuté.
2. **Deux voisines de chaque forme réelle** : `K` et `K+1` blocs. Un noyau qui
   ne gère pas la queue paie ou se trompe seulement sur la seconde.
3. **Une forme volontairement non alignée** : `K` non multiple de la taille de
   bloc du format. Si le noyau la refuse, c'est un résultat ; s'il l'accepte en
   silence avec un résultat faux, c'est un défaut que la vitesse aurait caché.
4. **Une forme très petite** (une seule vague de blocs) et **une très grande**
   (au-delà du L2). La première mesure le coût de lancement, la seconde la
   mémoire. Sans les deux, un gain moyen ne se décompose pas.

Les deux noyaux voient **exactement** les mêmes formes, les mêmes types
d'entrée et de sortie, la même disposition en mémoire.

---

## 3. La justesse avant la vitesse

Aucun temps n'est publié pour un noyau dont la sortie n'a pas été vérifiée sur
la même forme, dans la même exécution :

- écart absolu maximal et écart quadratique moyen contre une référence en
  simple précision, sur des poids **réels** et non aléatoires — un format à
  échelle par bloc se comporte autrement sur une distribution réelle ;
- le nombre de valeurs saturées, s'il y en a. Un noyau qui écrête gagne du
  temps et perd de l'information.

Un noyau rapide et faux ne se compare à rien.

---

## 4. Le piège qui rend tout microbanc caméra faux : le cache

Relancer mille fois le même GEMV sur le même tampon mesure la bande passante
du L2, pas celle de la mémoire — et le format le plus compact gagne alors
mécaniquement, puisqu'il tient mieux en cache. C'est le résultat « trop beau »
type.

Deux parades, à appliquer ensemble :

- **faire tourner les tampons** : allouer plusieurs jeux de poids dont la
  somme dépasse largement le L2 de la carte, et prendre le suivant à chaque
  lancement ;
- **vérifier le total lu** : la bande passante atteinte ne doit pas dépasser
  la bande passante mémoire mesurée de la carte. Si elle la dépasse, on lit du
  cache et la mesure est nulle — le microbanc doit le dire lui-même, pas
  laisser le lecteur s'en apercevoir.

---

## 5. Chauffe, horloges, et la mesure elle-même

- **Chauffe** : lancements jetés jusqu'à ce que l'horloge SM cesse de monter,
  et pas un nombre fixe décidé à l'avance. Le premier lancement d'une carte
  froide est plus lent, systématiquement.
- **Horloges relevées pendant chaque bloc de mesure**, minimum et maximum
  (`outils/energie.py` les donne). **Deux noyaux comparés à des horloges
  différentes ne se comparent pas** : c'est le gouverneur qu'on mesure. Si
  l'écart d'horloge entre les deux séries dépasse 5 %, la comparaison est
  refusée et refaite, ou les horloges sont verrouillées — en disant que le
  verrouillage change le régime.
- **Chronométrage par événements CUDA**, autour du noyau seul, jamais par
  horloge murale autour d'un appel Python.
- **Répétitions** : cinq blocs de quarante lancements par forme. On publie la
  **médiane des moyennes de blocs**, avec le minimum et le maximum des blocs.
  Le minimum est gardé à part : sur une machine calme, le bruit d'un
  microbanc est unilatéral (préemption, rampe d'horloge, il n'ajoute que du
  temps), donc le minimum approche le coût intrinsèque. Mais il ne sert
  **jamais** de base à la comparaison publiée : le même estimateur biaisé qui
  a failli donner une victoire au moteur le plus dispersé le 8 septembre.

---

## 6. Critère de publication

Le même que partout ailleurs dans ce projet :

> **L'écart annoncé doit valoir au moins trois fois la dispersion** des blocs,
> la plus grande des deux séries.

En dessous, on écrit « non conclusif » et on ne choisit pas de plan sur ce
chiffre. Un gain de 28 % avec 3 % de dispersion est acquis ; un gain de 5 %
avec 4 % de dispersion n'existe pas.

Et la comparaison à la prédiction est publiée elle aussi :

- gain mesuré ≈ 28 % → borné par la mémoire, format tenu ;
- gain mesuré nettement inférieur → **le décodage coûte**, et la bande
  passante atteinte dit combien ;
- gain mesuré supérieur → on mesure le cache, retour au §4.

---

## 7. Ce que ce microbanc ne dit pas

Il mesure un noyau, pas un moteur. Un GEMV plus rapide de 28 % ne donne pas
28 % de jetons en plus : la chaîne de décodage sérialise routage, transfert et
calcul, et le profil du bus du 8 septembre a montré des cartes inoccupées une
seconde sur deux pendant le décodage. **Le gain de bout en bout se mesure au
banc, sur le service, et nulle part ailleurs.** Écrire le gain de noyau comme
un gain de moteur serait la même faute que d'appeler « rafale » une fenêtre qui
contient surtout autre chose.
