# Où nous sommes, mesuré — 10/09/2026

poste4. **Premier chiffre que quelqu'un d'extérieur peut refaire**, protocole
et biais déclarés.

## Le résultat

> À douze séquences concurrentes, à source et cache égaux : **1,82× le débit**
> de llama.cpp et **1,36× ses jetons par kilojoule**, corrigés d'un biais de
> taille de 5 % en notre faveur (brut : 1,91× et 1,43×). **TTFT 2,29× plus
> lent.** Notre dispersion est de 22 à 31 % contre 1,5 % chez eux. **À une
> séquence, non départagé** — l'écart de 8 % est très inférieur aux
> dispersions.

## Le protocole, pour qu'il soit refaisable

Qwen3-Coder-30B-A3B-Instruct, **source Q4_K_M des deux côtés**, cache 8 bits des
deux côtés (`q8_0` chez eux, `int8` chez nous), contexte 8192, **spéculation
coupée des deux côtés**, ABBA par processus, invite de ~350 mots de texte réel
tirée du corpus, **différente par essai et la même des deux côtés**.

    corpus    /mnt/AI_GENERATOR/corpus/wiki.test.raw   sha256 173c87a53759e020
    binaire   llama-server CUDA de Jan b9967/linux-cuda-13, /props relevé
    poids     acvram 16,5 Go · gguf 17,3 Go
    outil     outils/duel-moteurs.sh, CONC=1 ou 12

## Les mesures brutes

    b = 12      débit agrégé   disp     TTFT mur   jetons/kJ   W médian
    acvram        564,45      30,8 %     1,465 s     1224       461,1
    llamacpp      302,87       1,5 %     0,642 s      864       350,4
    llamacpp      302,49       1,6 %     0,642 s      862       351,0
    acvram        592,64      21,6 %     1,468 s     1233       480,6

    b = 1        débit    disp      prefill   jetons/kJ
    acvram      424,79   38,2 %     3685,5      1297
    llamacpp    400,47  416,5 %     4914,7      1432
    llamacpp    379,24   12,2 %     6373,3      1161
    acvram      419,54   40,3 %     3676,2      1296

## Trois réserves, dont une qui limite la correction elle-même

**1. La correction de taille est approximative, et dans un sens qu'il faut
dire.** Elle suppose le débit inversement proportionnel aux octets lus. Or ce
modèle est un **MoE** : au décodage, seuls ~3 B des 30 B sont actifs par jeton.
**Le rapport 16,5/17,3 porte sur le poids TOTAL, pas sur les octets réellement
lus par jeton**, et rien ne dit que les deux rapports coïncident — la
quantification ne tombe pas au même endroit des deux côtés. La correction va
dans le bon sens ; sa valeur exacte n'est pas établie. **1,82× est donc à lire
comme « environ 1,8 », pas comme trois chiffres significatifs.**

**2. Notre dispersion est un RÉSULTAT, pas une précaution.** 22 à 31 % contre
1,5 % : **quinze à vingt fois leur variabilité**. Un moteur qui varie de 30 %
d'une manche à l'autre est difficile à déployer quel que soit son débit médian.
Deux candidats déjà nommés, aucun instruit :

- **`MAX_GRAPHS = 16` sans éviction** — les seize premières clés
  `(b, ql, nblk, lb)` gagnent, et lesquelles gagnent dépend de l'ordre d'arrivée
  des requêtes ;
- **le seuil des créneaux à 4**, qui fait basculer le chemin selon la
  concurrence instantanée.

Les deux produisent le même symptôme : **un débit qui dépend de l'histoire de la
session.**

**3. Le TTFT est le seul axe où nous sommes derrière, et le plus visible pour un
utilisateur.** 1,465 s contre 0,642 s à douze séquences. Nous avons retiré
aujourd'hui un backend qui prétendait servir le prefill et perdait partout ; il
reste que **notre prefill est lent et que personne ne l'a instruit**. C'est
probablement le meilleur rapport valeur/effort qui reste.

## Ce que cette note NE dit pas

Le duel du 9/09 donnait 1,96× de retard au décodage et 1,46× en jetons/kJ. Les
facteurs d'aujourd'hui sont presque exactement inversés, et il serait tentant
d'écrire que le retard est devenu une avance. **C'est faux : le duel du 9/09
était à UNE séquence**, le mode concurrent n'existant pas avant aujourd'hui.
Ce qui se compare à son chiffre est le bras à une séquence, et **il est
indécidable**.
