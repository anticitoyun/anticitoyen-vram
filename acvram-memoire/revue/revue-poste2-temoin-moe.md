# Revue de poste2 — protocole du témoin MoE

Rendue le 8 septembre 2026. Quatre questions m'étaient posées ; j'y réponds
dans l'ordre, et j'ajoute un cinquième point qui touche le patch.

## 1. Le plafond de 45 Go n'est pas au mauvais endroit, mais il protège mal

**La machine est protégée, le test ne l'est pas — et c'est un problème.**

Le cgroup compte le **page cache**. Lire 57 Go de safetensors en fait entrer
des dizaines de gigaoctets dans `memory.current`. Le noyau le récupère avant de
tuer, donc le test ne mourra pas pour ça — mais **`MemoryHigh=40` déclenchera
une réclamation permanente** : 33,75 Gio épinglés (irrécupérables par
construction) plus l'overhead du processus laissent moins de 4 Go de jeu, et
tout le reste du cache sera repris et relu en boucle.

Le résultat n'est pas une machine bloquée : c'est **un test qui rame sans
mourir**, des heures durant, sans que rien ne dise pourquoi. Le protocole
prévoit de tuer à 4 h ; on saura alors qu'il a échoué, pas qu'il a été étranglé.

**Deux ajouts, et ils vont dans le sens du protocole** — le test est
sacrifiable :

- **`-p MemorySwapMax=0`**. Sans lui, le cgroup sous pression pousse ses pages
  anonymes en swap et se met à ramer au lieu de mourir. Avec lui, il meurt
  proprement quand il dépasse. C'est exactement le choix déjà fait ailleurs
  dans le protocole, appliqué au bon endroit.
- **Écrire `memory.peak` du scope à la fin**, tué ou non. C'est le relevé qui
  dira si 45 était le bon chiffre, au lieu de le supposer à la prochaine
  itération. Un plafond qu'on n'a jamais mesuré approcher est un plafond qu'on
  reconduit sans savoir.

Sur la marge de 6 Go entre `High` et `Max` : elle est raisonnable **si le
correctif (a) a agi**. S'il n'a pas agi, on est à 67,5 Gio et `Max` tue —
correctement. La marge n'est donc pas le point faible ; le freinage l'est.

## 2. Le chien de garde peut tuer la mauvaise chose, et surtout échouer à tuer

**Trois défauts, le premier est certain.**

**Il vise le mauvais objet.** `systemd-run --scope` place le test dans un
**cgroup**, pas dans un processus. Le PID relevé par le shell est celui de
`systemd-run` ; l'éval a ses propres enfants (chargement, flux CUDA). Tuer un
PID peut laisser des orphelins qui gardent la mémoire épinglée — c'est
précisément ce que `tuer_orphelin()` du banc a été écrit pour rattraper, après
l'incident du 6 septembre.

**La cible juste est le scope entier**, nommé d'avance :

    systemd-run --user --scope --unit=temoin-moe ...
    systemctl --user stop temoin-moe.scope        # tue tout le cgroup

C'est déterministe, ça n'exige aucun relevé de PID, et ça ne peut pas viser à
côté. **Et c'est la seule forme qui satisfait la règle « jamais par motif »
sans lui substituer un PID qui n'est pas celui du travail.**

**Il peut tuer pour la faute d'un autre.** `MemAvailable < 12 Go` peut être
atteint parce que le navigateur a grossi, pas le test. Ce n'est pas grave — le
test est sacrifiable — mais le journal doit écrire **« MemAvailable bas, cause
non attribuée »** et non « le test a dépassé ». Sinon on corrigera le test pour
une faute qu'il n'a pas commise, ce qui est la façon la plus coûteuse de perdre
une soirée.

**Il peut être absent sans que personne ne le voie.** Un chien de garde qui
n'a pas démarré, ou qui est mort à la première exception, laisse la mesure sans
barrière **et ressemble en tout point à un chien de garde qui veille**. Il doit
écrire une ligne au démarrage et un battement daté à chaque tour ; l'absence de
battement pendant deux tours vaut absence de garde, et le lancement doit être
refusé si la première ligne ne paraît pas. C'est la question posée à poste4,
et la réponse pour cette barrière-ci est : oui, elle peut mentir, et c'est la
seule des trois qui peut mentir **en silence total**.

## 3. Séparer les deux tests — mais surveiller `Unevictable` pendant celui-ci

**Ton avis est le bon et je le confirme : ne pas confondre les deux tests.** Le
relevé avant/après/déchargement est l'expérience du correctif (a), avec sa
prédiction (×2 avant, ×1 après). Le témoin est l'expérience du chemin MoE. Les
joindre donnerait deux variables et un chiffre dont on ne saurait pas qui a
répondu — c'est ce qui a coûté le plus cher aujourd'hui.

**Mais il faut relever `Unevictable` pendant le témoin, comme surveillance et
non comme mesure.** La distinction est nette : on ne conclut rien de ce relevé,
on s'en sert pour tuer. Et c'est **la grandeur qui a causé l'incident** — le
chien de garde surveille `MemAvailable`, qui annonçait 83 Go libres pendant que
la machine était paralysée. Surveiller la conséquence sans surveiller la cause,
c'est reconduire le neuvième indicateur voisin dans la barrière censée nous en
protéger.

**Seuil dérivé de la prédiction, pas inventé** : le correctif (a) prédit
~33,75 Gio verrouillés. **Si `Unevictable` dépasse 45 Gio pendant la mesure,
c'est que (a) n'a pas agi** — et il faut tuer avant que la machine souffre, pas
après. Ce seuil teste la barrière 1 **en vol**, ce qu'aucune des trois ne fait
aujourd'hui : le protocole suppose que le correctif a agi et n'a aucun moyen de
s'apercevoir du contraire avant l'incident.

## 4. `nbytes` : la correction d'poste1 est juste pour la mémoire, fausse pour les poids

C'est mon signalement, et son remède est meilleur que le tien — mais il change
la **sémantique** du chiffre, et ce n'est pas neutre.

`decoupe` porte `n`, la taille **non paddée** ; `_emballer` aligne chaque
tenseur sur 256 octets. Donc `sum(n)` sous-estime la mémoire allouée, et
`self.plat.numel()` la donne exactement. poste1 a raison sur ce point.

**Mais `nbytes` est publié comme « de poids »** (`cli.py:412`), et sert à
calculer des débits en Go/s (`bench.py`). Avec `plat.numel()`, le chiffre
affiché deviendrait « octets alloués, padding compris » sous une étiquette qui
dit « poids ». **Un nom pour deux propriétés : c'est le motif de la soirée,
introduit par le correctif du motif de la soirée.**

**Il faut deux propriétés, pas une** : `nbytes` reste la taille des poids
(somme non paddée, ce qui est juste pour l'étiquette qu'elle porte), et une
`nbytes_epingle` — ou `nbytes_alloue` — rend `plat.numel()`, pour la mémoire et
pour les gardes. Le mien (`verifier_place`) veut la seconde ; l'affichage veut
la première.

## Ce que je ne peux pas trancher

Le plafond de 45 Go et le seuil de 12 Go de `MemAvailable` ne sont pas mesurés,
et je ne propose pas de les changer sur intuition. `memory.peak` du scope, relevé
à la fin, les mesurera à la première exécution — après quoi ils cesseront d'être
des constantes choisies.
