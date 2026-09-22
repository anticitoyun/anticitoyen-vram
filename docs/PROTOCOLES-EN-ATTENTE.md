# Deux protocoles prêts, en attente du feu vert de l'utilisateur

Écrits le 8 septembre 2026, machine fermée. **Rien n'est lancé** : les mesures
sont suspendues par l'utilisateur, et aucun accord entre sessions ne lève cette
décision. Ces protocoles partent tels quels quand il rouvre.

Chacun nomme sa variable unique, sa prédiction écrite d'avance, et **le
contrôle qui dit si l'instrument a fait ce qu'on lui a demandé**. Ce dernier
point n'est pas une précaution de style : le 8 septembre, `--passages` était
accepté sans effet, `ACVRAM_EXIL_COUCHES=18` a été ignoré en silence, et le
test d'exil aurait été validé sur trois exécutions identiques de la même chose
sans une ligne de journal exigée d'avance.

---

## 1. Le verrouillage mémoire : transformer un calcul en relevé

**Ce qu'on veut établir.** La double épinglure de `StreamedWeight` verrouille
deux fois les poids exilés — `self.host` épinglé ligne 66, puis recopié dans
`plat` également épinglé ligne 44. Le chiffre avancé, **67,5 Gio au lieu de
33,75**, est aujourd'hui un **calcul**. Il doit devenir un **relevé**, parce
qu'un chiffre qui justifie un correctif est un chiffre qui décide.

**Compteur.** `Unevictable` de `/proc/meminfo` pour décider, `VmLck` du
processus pour attribuer. **Pas `Mlocked`** : relevé sur cette machine au
repos, `Unevictable` 949 Mio contre `Mlocked` 132 kio — `Mlocked` ne compte que
le `mlock()` classique, et la mémoire épinglée par le pilote CUDA ne passe pas
par là. Et **pas `VmSwap`**, qui mesure ce qui a été chassé : il reste à zéro
sur une machine calme, avec ou sans correctif.

**Trois relevés, et le troisième est le contrôle** : avant chargement, après
chargement, **après déchargement**. Le retour à la valeur initiale prouve qu'on
mesure son propre effet et non une dérive du système. S'il ne revient pas, le
relevé ne dit rien de la double épinglure.

**Prédiction écrite d'avance.** Avant correctif, `Unevictable` doit monter
d'environ **deux fois** la taille des poids exilés. Après correctif, d'environ
**une fois**. Si la montée vaut déjà une fois avant correctif, il n'y a pas de
double épinglure et le correctif ne sert à rien — issue nommée d'avance.

**Un seul correctif à la fois**, réserve d'Océane, retenue : les vues sur
`plat` d'abord, mesurées ; le non-épinglage sans carte ensuite. Mélangés, on ne
saurait pas lequel a agi.

---

## 2. `ngram` contre `mtp` : notre mode de spéculation n'a jamais été choisi

**Ce qu'on veut établir.** `acvram serve --speculative` a pour défaut `ngram`
(`cli.py:585`) et le lanceur ne passait **jamais** l'option. Donc tous nos
débits publiés — et **toutes nos comparaisons aux concurrents** — sont dans un
mode que personne n'a choisi, sur des modèles dont nous chargeons et
quantifions une tête MTP jamais utilisée. Nous n'avons jamais publié dans quel
mode nos chiffres étaient pris.

**Variable unique.** `SPECULATIF=ngram` puis `SPECULATIF=mtp`, sur le même
modèle, la même session, le même plafond de puissance, le même filtre. Le
lanceur porte cette variable depuis le 8 septembre, neutre par défaut.

**Modèle.** Un modèle qui **a** une tête MTP — à vérifier avant, pas pendant.

**Le contrôle que j'avais écrit visait un piège qui n'existe pas.** J'avais
prévu que `mtp` puisse être refusé en silence et qu'on mesure alors `ngram`
contre `ngram`. C'est faux, et Océane l'a relevé : `cli.py:381-384` fait
`return 2` — **le serveur ne démarre pas du tout**. Il n'y aurait donc aucune
mesure, pas une mesure trompeuse. Un contrôle qui cherche un piège absent est
pire que pas de contrôle : il rassure.

Vérifié dans le code : le mode qui substitue en silence est **`auto`**
(lignes 367-369), mais il **réassigne** `args.speculative` avant que le journal
ne l'imprime (ligne 416). L'annonce est donc fidèle, y compris derrière `auto`.

**Le vrai risque est la variable unique, et il est dans `k`.** `--spec-k` vaut
**4 par défaut pour tous les modes** (ligne 592). Or `k` est le nombre de
jetons proposés par pas, et le `k` optimal diffère par nature : un n-gramme
propose mal et loin, une tête MTP propose bien et court. Une comparaison à `k`
fixe ne dit donc pas lequel est meilleur — elle dit **lequel est meilleur à
k=4**, et publier l'un pour l'autre serait transporter une conclusion hors de
ses conditions, le piège central de ce dossier.

**Le contrôle, proposé par Océane et retenu** : le journal porte déjà
`speculation : <mode>, k=<n>` (ligne 416-417). Les deux lignes des deux
mesures **ne doivent différer que par le mot du mode**. Si `k` diffère, la
mesure n'existe pas. Vérifiable sur la sortie, sans instrumentation.

**Extension à faire si `mtp` gagne** : refaire les deux modes à `k` = 2 et 8.
Si `ngram` l'emporte à un autre `k`, la conclusion n'est pas « `mtp` est
meilleur » mais « chacun a son `k` », et c'est le réglage qu'il faut publier,
pas le vainqueur.

**Grandeurs relevées.** `t_s_passages` et `ttft_passages` — les deux, parce que
la spéculation agit sur la génération et non sur le préremplissage : un gain
qui apparaîtrait aussi dans le TTFT désignerait autre chose qu'elle.

**Cinq passages, premier jeté.** Établi le 8 septembre : le premier passage est
systématiquement plus lent, et la dispersion tombe de ~10 % à moins de 0,6 %
quand on l'écarte.

**Prédiction écrite d'avance, à k=4.** `mtp` doit être plus rapide que `ngram` sur du
texte qui ne recopie pas son entrée, et l'écart doit être plus grand que la
dispersion sans le premier passage — sans quoi il ne conclut rien. Si `mtp` est
plus **lent**, la tête MTP que nous transportons depuis toujours est un coût
net, et c'est un résultat aussi utile que l'inverse.
