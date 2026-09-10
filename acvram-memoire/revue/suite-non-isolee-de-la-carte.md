# « Suite verte » ne veut rien dire si on ne dit pas ce qui tournait à côté

Deux épreuves ont échoué le 10/09 pendant qu'une autre session occupait la
5090, et **les deux repassent seules**. Ce n'est donc pas un défaut de `main` —
mais ce n'est pas non plus un incident sans conséquence : **la formule « 474
passés » que nous mettons dans nos messages de commit affirme moins que ce
qu'elle semble dire.**

```
17h55  test_awq_does_not_degrade_the_model_end_to_end   ECHEC pendant « poste3-slots8 »
       le meme fichier seul, apres                      5 passes
15h50  test_le_gemv_nvfp4_ne_part_pas_en_emulation      ECHEC pendant mon A/B d'ordre
       le meme fichier seul, apres                      6 passes
```

## Le mécanisme n'est pas celui que je croyais

J'ai d'abord soupçonné le planificateur : `auto_plan` lit la mémoire libre, donc
sous contention il placerait autrement et le test assertant sur son résultat
échouerait. **C'est faux, et c'est une bonne nouvelle sur notre conception** :

```python
def target_rig():
    return load_profile("rig-14900k-5090-3080ti")
```

Le rig est **synthétique** — un profil déclaré, pas une lecture de la machine.
Le planificateur est donc éprouvé contre une machine fixe, et son résultat ne
dépend pas de ce qui tourne à côté. **L'exposition n'est pas là.**

Elle est dans ce que les épreuves font **ensuite** : `test_awq…` appelle
`load_model` et charge réellement sur la carte ; `test_le_gemv…` appelle
`get_extension()`, qui a besoin d'un contexte CUDA. Avec 27,8 Go des 32,6
occupés par une autre session, le chargement échoue ou se replie.

## Ce que cela implique pour le correctif

**Un marqueur qui exclurait ces épreuves d'une exécution concurrente réduirait
la couverture en silence** — et une couverture qui se réduit toute seule quand
la machine est occupée est exactement le genre de garde qui ne peut plus rendre
« faux » au moment où on en a le plus besoin.

Ces épreuves **utilisent réellement la carte**. La seule correction juste est
donc qu'elles **prennent le verrou** `carte.sh`, comme n'importe quelle mesure.
Le coût est qu'une suite complète devient sérialisée avec les manches — ce qui
est la vérité de ce qu'elle fait.

**26 fichiers de test sur l'arbre touchent la carte** ; douze portent déjà un
`skipif` sur son absence. **Un `skipif` protège d'une absence, pas d'une
contention** — la carte est là, elle est simplement prise.

## Conduite, en attendant le correctif

1. **ne pas lancer la suite complète pendant qu'une carte est occupée** ;
2. si on la lance quand même, **publier ce qui tournait à côté avec le
   résultat** — « vert sur machine calme » et « vert en contention » ne sont
   pas la même affirmation ;
3. et devant un rouge, **relancer le fichier seul avant de conclure**. Deux
   fois sur deux aujourd'hui, c'était la contention.

**Ce qui reste vrai de nos messages de commit** : les suites vertes annoncées
ce matin et cet après-midi l'étaient sur machine calme, sauf mention contraire.
Ce qui est faux, c'est de présenter le nombre sans son contexte — et je l'ai
fait une dizaine de fois aujourd'hui.

---

## CE QUI N'EST PAS ÉTABLI, et c'est le morceau qui manquait

**Nous accusons deux tests sur la foi de deux échecs observés.** Relevé par
chef, et c'est la bonne objection : deux observations ne font pas une
fragilité établie, et **sérialiser la suite entière pour une fragilité non
prouvée serait payer un prix réel contre un défaut supposé.**

Ce qui est observé :

```
17h55  test_awq_does_not_degrade_the_model_end_to_end   ECHEC pendant « poste3-slots8 »
15h50  test_le_gemv_nvfp4_ne_part_pas_en_emulation      ECHEC pendant mon A/B d'ordre
       les deux repassent seules
```

Ce qui n'est **pas** observé : que ces échecs soient **reproductibles**, ni
qu'ils soient **causés** par la contention plutôt que corrélés avec elle. Une
coïncidence deux fois de suite reste une coïncidence tant qu'on ne la provoque
pas.

### Le protocole qui l'établirait, ou le réfuterait

**Faire tomber les tests exprès, sous une charge GPU connue.** Trois conditions,
et la troisième est celle qui décide :

```
1. machine calme, carte a 16 Mio          les deux tests doivent PASSER
2. charge GPU connue et tenue             les deux doivent ECHOUER, a volonte
3. la meme charge, tests sous carte.sh    les deux doivent PASSER a nouveau
```

La charge doit être **connue et reproductible** — pas « une autre session
tournait », mais un occupant délibéré dont on choisit les octets. Un simple
allocateur qui réserve N gibioctets et les tient suffit, et il vaut mieux qu'un
modèle : on choisit N, donc on peut chercher le **seuil** à partir duquel chaque
test tombe. Ce seuil est le vrai résultat : il dit si le test est fragile ou
simplement gourmand.

**Et si un test qu'on croit fragile résiste, c'est un résultat aussi** — il
sort de la liste, et la sérialisation coûte moins.

### Ce que la condition 2 doit distinguer

Les deux échecs n'ont **pas le même mécanisme**, donc le seuil ne sera
probablement pas le même :

```
test_awq...      charge un modele reel     tombe quand la VRAM libre manque
test_le_gemv...  demande get_extension()   tombe si le contexte CUDA echoue
```

Le second peut tomber pour une raison qui n'est pas la mémoire — compilation,
cache de noyaux, contexte. **Le confondre avec le premier ferait attribuer à la
mémoire un défaut de contexte**, et le correctif par verrou marcherait sans
qu'on sache pourquoi.
