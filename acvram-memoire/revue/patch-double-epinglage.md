# Double épinglage — proposition soumise à relecture

Non écrit dans l'arbre. Cible : `acvram/engine/layers.py`.

## Le fait

    66  self.host = {k: (v.pin_memory() if not v.is_pinned() else v) ...}
    74  self.plat, self.decoupe = _emballer(self.host)
    44    plat = torch.empty(max(off, 1), dtype=torch.uint8).pin_memory()
    46    plat[o:o + n].view(dt).view(forme).copy_(host[k])

Deux tampons épinglés contenant la même chose. Le DMA ne part que de `plat`
(lignes 113 et 119) ; `self.host` ne sert plus qu'au repli sans GPU (125) et à
`nbytes` (142). Relevé de tous les usages fait par chef et **refait
indépendamment par poste2** : rien n'écrit dans `self.host` en aval.

À 30 couches de 1,125 Gio : **67,5 Gio verrouillés au lieu de 33,75**, sur
93 Go de RAM. La mémoire épinglée n'est ni évinçable ni swappable : le noyau
doit chasser tout le reste. C'est ce qui a paralysé la machine le 8 septembre.

Et le commentaire du champ, trois lignes plus haut, dit déjà ce que le
correctif rétablit : « Le dictionnaire `host` reste **la vue** par clé. »

## (a) — supprimer la seconde copie

```python
    self.plat, self.decoupe = _emballer(self.host)
    # `host` redevient ce que la ligne au-dessus annonce : des vues sur le
    # tampon plat, et non une seconde copie épinglée du même contenu. Les
    # deux usages survivants (repli sans GPU, nbytes) n'ont besoin ni de
    # copie ni d'épinglage ; le DMA, lui, part de `plat`.
    self.host = _decouper(self.plat, self.decoupe)
```

Vérifié par poste1 : `_decouper` (ligne 51) rend `plat[o:o+n].view(dt).view(forme)`
— tranche contiguë puis vues, **aucune copie** ; et `_emballer` aligne déjà les
décalages sur 256 octets, ce que sa docstring dit être « pour les copies **et
les vues** ». La contrainte d'alignement de `.view(dt)` est donc respectée.

**Prédiction, écrite avant la mesure — et elle porte sur DEUX compteurs, dont un
seul répond** (précision d'poste1, sans laquelle la prédiction serait lue comme
fausse) :

* **`Unevictable` doit tomber de ~67,5 à ~33,75 Gio.** C'est le compteur qui
  décide. Si le chiffre ne bouge pas, (a) n'a pas agi et rien ne continue.
* **`MemAvailable` ne bougera PAS de 33,75 Gio**, parce que (a) supprime une
  copie *épinglée*, pas une *occupation* : la source mmap qui reste n'était pas
  comptée comme prise. Qui contrôlerait par `MemAvailable` conclurait que le
  correctif n'a rien fait.

Deux compteurs voisins, un seul répond à la question — la même famille que tout
le reste du dossier.

## (b) — ne pas épingler la source, SÉPARÉMENT et après

```python
    self.host = dict(host_tensors)     # ligne 66 : plus de pin_memory()
```

`_emballer` fait une copie **CPU→CPU** ; l'épinglage ne sert qu'au DMA, qui part
de `plat`. La ligne 66 est donc du travail pur perte.

**Le gain est acquis pour ce modèle-ci, et poste1 l'a prouvé par la capacité
plutôt que par la lecture.** `_resolved_weight` appelle `_rehydrate(self.qweight,
tensors)` à **chaque** forward : `self.qweight` sert de gabarit et n'est donc
jamais libéré. La mémoire hôte porte en permanence la **source** plus
`self.host` plus `self.plat` — trois copies, pas deux. Or 3 × 33,75 = 101,25 Gio
sur une machine qui en a 93,98, et **le chargement a bien eu lieu** : l'échec
était côté carte, pas côté RAM. **Donc la source n'occupe pas de RAM anonyme :
c'est un mmap.** Ce que (b) supposait est établi.

**Le gain dépend néanmoins du format en général, et il faut le dire** : `StreamedWeight`
reçoit `dict(self.qweight.state_dict())`. Ces tenseurs ne sont des vues du mmap
safetensors que si **rien ne les a matérialisés** entre le `safe_open` et là.

* modèle chargé tel quel dans son format de fichier → source vraiment mmap,
  pages **propres et évincibles**, le noyau les jette sans rien écrire ;
* modèle requantifié au chargement → tenseurs anonymes, seulement **swappables**.
  Gain réel — le noyau retrouve une porte de sortie — mais pas celui annoncé.

Le témoin bf16 est dans le premier cas : poids bf16 lus d'un fichier bf16,
aucune requantification.

**À ne PAS fusionner avec (a)** : les deux ne portent pas le même risque, et les
mélanger nous rendrait incapables de dire lequel a agi.

## (c) — deux corrections de poste2, corrigées par poste1

**`nbytes` (142) sous-estime**, et le remède que j'avais écrit reproduisait la
faute. Il somme les tenseurs sans l'alignement à 256 octets appliqué par
`_emballer` — d'autant plus faux que les tenseurs sont petits et nombreux, trois
par poids en NVFP4 — et il alimente les chiffres publiés de `cli.py:412`,
`evaluate.py:161`, `bench.py:327`. J'avais proposé « recalculer depuis
`decoupe` » : **`decoupe` contient `n`, la taille NON paddée**, donc le calcul
reproduirait exactement la sous-estimation. Le seul chiffre exact est
**`self.plat.numel()`** — par construction la mémoire réellement réservée,
padding compris. Une ligne, qui ne peut pas dériver.

**Et ma correction du repli sans GPU ne faisait rien du tout.** J'avais écrit
« ne rien épingler quand `device.type != \"cuda\"` » au niveau de la ligne 125.
Or ce chemin est `wait()`, qui se contente de `return self.host` : **il
n'épingle rien**. Le seul point d'épinglage est la **ligne 44**,
`torch.empty(...).pin_memory()`, appelée inconditionnellement depuis
`_emballer`. C'est elle qu'il faut conditionner.

Trouvé par poste1, et c'est le motif que j'avais confié à poste4 — *un
correctif qui peut ne rien faire* — commis dans le patch qui corrige le
neuvième indicateur voisin. Il serait passé : il est plausible, et personne
n'aurait revérifié une ligne « appliquée ».

## Contrôle exigé — le calcul doit devenir un relevé

Condition posée par poste2, et elle vaut contre l'auteur du patch : les
« 67,5 Gio au lieu de 33,75 » sont un **calcul**, pas une mesure.

Relever `Unevictable` (`/proc/meminfo`) et `VmLck` (`/proc/<pid>/status`)
**avant chargement, après chargement, après déchargement**. Le retour à la
valeur initiale au déchargement est ce qui prouve qu'on mesure la bonne chose.

**Préalable gratuit, exigé par poste1** : vérifier qu'`Unevictable` réagit *du
tout* au chargement d'un modèle exilé. Plancher de repos sur cette machine :
**0,93 Gio**. S'il ne bouge pas, le relevé ne prouve rien, ni dans un sens ni
dans l'autre.

`Mlocked` est **exclu** : 132 kB contre 960 664 kB pour `Unevictable` sur cette
machine — il ne voit pas l'épinglage du pilote CUDA. `Unevictable` décide,
`VmLck` attribue ; si `VmLck` reste à zéro pendant qu'`Unevictable` monte, cela
dit que CUDA n'épingle pas par `mlock()`, pas que rien n'est verrouillé.

## À écrire au moment du correctif

Il ne touche aucun chiffre de qualité — il déplace de la mémoire, il ne change
pas une valeur. Mais il change la pression mémoire, donc **aucun débit mesuré
avant n'est comparable à un débit mesuré après**.
