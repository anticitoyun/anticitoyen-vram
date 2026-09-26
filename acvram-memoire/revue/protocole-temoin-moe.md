# Témoin MoE bf16 — protocole soumis à validation des trois

Soumis par chef le 8 septembre 2026. **À valider par poste1, poste2 et
poste4 avant tout lancement.** Ne rien lancer avant leurs trois réponses et
les corrections qui en découlent.

## La contrainte qui commande tout le reste

L'utilisateur pose une règle nouvelle et dure : **un test ne doit absolument
jamais bloquer le PC.** Il doit pouvoir s'en servir — souris, vidéo — pendant
que la mesure tourne.

Ce n'est pas une préférence, c'est un critère d'acceptation. **Le test est
sacrifiable, la machine ne l'est pas.** Un test tué est un incident mineur ;
une machine paralysée a coûté la soirée du 8 septembre.

## Pourquoi le témoin, tel quel, viole cette règle par construction

Trois tentatives, trois échecs, et la cause est maintenant connue :

| tentative | issue |
|---|---|
| 18h41 | OOM CUDA — 27,94 Gio alloués, 2,55 Gio réservés non alloués, 20 Mio manquants |
| 19h10 | **le même, au bit près** |
| 19h20 | arrêté par l'utilisateur : machine inutilisable |

Le modèle pèse 57 Go en bf16. À 30 couches exilées, ce sont 33,75 Gio de poids
en mémoire hôte — **et 67,5 Gio réellement verrouillés**, parce que
`StreamedWeight` épingle la source *et* le tampon (voir
`revue/patch-double-epinglage.md`). La mémoire épinglée n'est ni évinçable ni
swappable : le noyau doit chasser tout le reste, y compris le bureau.

**Donc relancer le témoin sans rien changer reproduirait exactement l'incident.**

## Ce que je propose — trois barrières indépendantes

L'idée directrice : ne pas *espérer* que la mesure se tienne, mais **rendre
impossible qu'elle nuise**. Chacune des trois barrières suffit à protéger la
machine ; les trois ensemble couvrent le cas où l'une échoue en silence — ce
qui est arrivé neuf fois ce soir.

### Barrière 1 — réduire l'empreinte à sa moitié : le correctif (a)

`self.host = _decouper(self.plat, self.decoupe)` après l'emballage. Supprime la
seconde copie épinglée. **67,5 → 33,75 Gio.** Détail et prédiction dans
`revue/patch-double-epinglage.md`.

**(b) n'est PAS dans ce lot** : il ne porte pas le même risque, et les mélanger
nous empêcherait de dire lequel a agi.

### Barrière 2 — un plafond mémoire que le noyau fait respecter

    systemd-run --user --scope --unit=temoin-moe \
      -p MemoryMax=45G -p MemoryHigh=40G -p MemorySwapMax=0 \
      -p CPUWeight=20 -p IOWeight=20 \
      nice -n 19 ionice -c 3 <la commande d'éval>

**`MemorySwapMax=0` n'est pas décoratif : sans lui la borne fabrique le mal
qu'elle prétend éviter.** poste1 l'a éprouvé — avec `MemoryMax=200M` seul, son
programme d'essai a alloué **4 Go sans broncher**, le noyau respectant la borne
de RAM en poussant le surplus **en swap**. Avec `MemorySwapMax=0`, le même
programme est tué net. C'est ce qu'il nous faut : **la victime est toujours la
mesure, jamais le bureau.**

Il évite aussi ce que poste2 a vu venir : le cgroup compte le page cache, et lire
57 Go de safetensors en fait entrer des dizaines de gigaoctets. Avec 33,75 Gio
épinglés **irrécupérables**, `MemoryHigh=40` laisserait moins de 4 Go de jeu et
déclencherait une réclamation permanente — **un test qui rame des heures sans
mourir et sans que rien ne dise pourquoi**. Mieux vaut qu'il meure proprement.

`CPUWeight` et `IOWeight` bas, `nice 19`, `ionice -c 3` : le test ne prend le
processeur et le disque que lorsque personne d'autre n'en veut.

**`--unit=temoin-moe` nomme le scope d'avance**, ce qui rend la barrière 3
déterministe (voir plus bas).

**Relever `memory.peak` du scope à la fin, tué ou non.** C'est ce qui dira si 45
était le bon chiffre, au lieu de le reconduire d'une mesure à l'autre.

### Barrière 2 bis — tokensave sous plafond, et non arrêté

L'utilisateur veut que **tokensave soit utilisé** ; c'est aux tests d'en tenir
compte. Or `tokensave serve` est monté à **18,5 Go en quelques minutes** ce
soir (8,4 → 10 → 17,8 → 18,5), à ~2 Go/min, et c'est lui qui a chassé 8 Go en
swap dont la machine ne s'est pas remise pendant une heure.

**On ne l'arrête donc plus : on le borne.**

    systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0 \
      -- tokensave serve

**Borne d'accident, pas rationnement** — correction d'poste1, et elle est juste.
J'avais écrit `MemoryMax=8G -p MemoryHigh=6G` sur un processus qui en prenait
**29,3**. Deux fautes :
**8 Gio est un chiffre inventé**, et notre propre règle dit qu'un chiffre qui
arbitre doit être mesuré. Une indexation qui a besoin de 29 Gio n'en tiendra pas
29 dans 8 parce qu'on le lui interdit — elle sera tuée, puis retuée au
redémarrage, **à la même étape**.
**`MemoryHigh` n'arrête pas, il étrangle** : au-delà, le noyau entre en
récupération permanente. On obtiendrait un serveur qui ne meurt pas mais ne
répond plus — plus difficile à diagnostiquer qu'une mort franche.
Et surtout : **l'utilisateur a demandé qu'il soit utilisé.** Une borne qui le
rend inutilisable exécute la lettre de sa demande — la machine ne bloque plus —
contre son intention. La propriété visée est « la machine reste utilisable » ;
celle obtenue aurait été « l'outil ne marche plus ». **Encore un contrôle qui
garantit la propriété voisine.**
24 Gio sur 94 ne paralyse rien et laisse l'outil travailler. Le chiffre juste se
relèvera (`MemoryPeak` du scope sur une indexation complète) ; d'ici là il est
haut à dessein.
**Fait à ne pas imputer au cgroup** : tokensave était déjà en `CONNECT_TIMEOUT`
dans la session d'poste1 **avant** toute borne. S'il meurt maintenant, vérifier
avant de conclure.

### Budget mémoire, posé explicitement

| poste | réservé |
|---|---|
| système et bureau | ~5 Go |
| navigateur | ~8 Go |
| tokensave (borné à l'accident) | ≤ 24 Go, tué au-delà |
| **le test** | **≤ 40 Go, tué à 45** |
| marge non allouée | ~12 Go |

Total engagé : ~82 Go sur 93,98 **dans le pire cas simultané** — que rien ne
rend probable : tokensave au repos pèse quelques centaines de Mio, et sa borne
est là pour l'accident, pas pour son régime normal. La marge de 28 Go est ce qui garantit que
l'utilisateur garde sa machine — **elle n'est pas là pour être consommée.**

### Barrière 3 — un chien de garde qui tue le test, pas la machine

**Il tue le scope, pas un PID** — correction de poste2, et c'est le point dur :

    systemctl --user stop temoin-moe.scope

`systemd-run --scope` met le test dans un **cgroup**, pas dans un processus. Le
PID que le shell relève est celui de `systemd-run` ; l'éval a ses propres
enfants. **Tuer un PID peut laisser des orphelins qui gardent la mémoire
épinglée** — c'est ce que `tuer_orphelin()` du banc a dû rattraper après le
6 septembre. Arrêter le scope nommé est déterministe : aucun PID à relever,
impossible de viser à côté. C'est la seule forme qui respecte « jamais par
motif » sans lui substituer un PID qui n'est pas celui du travail.

Toutes les 5 secondes :

* **`Unevictable` > 45 Gio → tuer.** Seuil dérivé de notre propre prédiction et
  non inventé : si le correctif (a) a agi, l'épinglage plafonne vers 33,75 Gio.
  **Au-delà, c'est que (a) n'a pas agi**, et il faut tuer *avant* l'incident.
  Aucune des trois barrières ne testait jusqu'ici que la première avait
  réellement agi ; celle-ci le fait, en vol.
* `MemAvailable` < **12 Go** → tuer. **Mais c'est le compteur faible** :
  il annonçait 83 Go libres pendant que la machine était paralysée. Surveiller
  la conséquence sans surveiller la cause reconduirait le neuvième indicateur
  voisin dans la barrière censée nous en protéger — d'où `Unevictable` au-dessus.
* swap-in (`vmstat`, dernière ligne, colonne `si`) non nul **deux relevés de
  suite** → tuer. Deux, parce qu'un pic isolé peut venir d'un autre processus,
  et qu'un chien de garde qui tue au premier frisson finit désactivé.
* le test dépasse 4 h → tuer.

**Un chien de garde absent ressemble en tout point à un chien de garde qui
veille** (poste2). C'est la seule des trois barrières qui peut mentir en silence
total. Donc : il écrit une ligne **au démarrage**, puis un **battement daté** à
chaque tour ; **deux tours sans battement valent absence de garde**, et le
lancement est refusé si la première ligne ne paraît pas.

Le journal dit **pourquoi** il a tué. Un test tué sans motif écrit est un test
qu'on relancera à l'identique.

## Ce que le protocole mesure, et ce qu'on prédit

Perplexité de `qwen3-coder-30b-bf16` sur `wiki.test.raw`, fenêtre 512, stride
512, `min_context` 256, contre l'étalon **9,7380 ± 0,0779** (Q4_K_M du même
modèle, mesuré par poste1 avec llama.cpp e34f042).

**Le nombre de fenêtres n'est pas présumé** : il sera lu dans le journal, pas
recopié de Coder-Next. 584 vaut pour les modèles à vocabulaire 151 936 ; il a
déjà été donné à tort au 12B, qui en fait 580.

**Prédictions, écrites avant :**

* **≈ 9,74** → le chemin MoE est propre. Le résidu de 0,367 nat vient du format
  ou d'ailleurs, mais pas du MoE.
* **nettement au-dessus, disons > 11** → le chemin MoE porte un défaut propre,
  indépendant du scoring (identique), de la normalisation (corrigée), de la
  quantification (bf16) et de l'exil (**établi sans effet** : A₁=A₂=B=58,815).

**Réserve à inscrire à côté du chiffre** : exil forcé à 30 couches au lieu des
27 choisies par le plan, pour contourner une fragmentation de 2,55 Gio.
Placement non choisi par le planificateur.

## Réserve à lever AVANT le lancement — confiée à poste2

**La mémoire épinglée par CUDA est-elle comptée par le cgroup ?**
`cudaHostAlloc` passe par le pilote, et selon le chemin la comptabilité peut
échapper au contrôleur. **Si elle échappe, la borne ne protège de rien pour
notre cas précis** — celui-là même qui a paralysé la machine.

S'éprouve sans mesurer quoi que ce soit d'utile : charger un modèle exilé sous
une borne délibérément trop basse et voir s'il meurt. S'il ne meurt pas, on le
sait **avant** d'en avoir besoin.

## Ce que je demande à chacune

**poste1** — le cadrage et l'étalon. Le nombre de fenêtres attendu pour ce
modèle est-il bien 584, et le sais-tu ou le supposes-tu ? Une prédiction de plus
à ajouter avant la mesure ?

**poste2** — les barrières. Le plafond de 45 Go est-il au bon endroit ? Le chien
de garde peut-il tuer la mauvaise chose, ou échouer à tuer ? Et surtout : **ta
règle « le calcul doit devenir un relevé »** s'applique-t-elle ici — faut-il
relever `Unevictable` pendant cette mesure, ou est-ce mélanger deux tests ?

**poste4** — la relecture des trois barrières comme mécanismes qui peuvent
échouer en silence. Chacune peut-elle *sembler* agir sans agir ? C'est ton
audit appliqué à un cas concret : le protocole prétend qu'un plafond, un chien
de garde et un correctif protègent la machine — lequel des trois peut mentir ?

## Ordre d'exécution, une fois validé

1. correctif (a) écrit, `pytest -q` passé, **non poussé** avant la mesure ;
2. relevé de l'état machine : `MemAvailable`, swap occupé, swap-in (dernière
   ligne de `vmstat`), VRAM libre, charge — **refus de lancer si le swap-in
   n'est pas nul** ;
3. lancement sous cgroup avec chien de garde armé ;
4. relevé du même état après ;
5. le chiffre, sa réserve, et ce qui n'est pas mesuré.
