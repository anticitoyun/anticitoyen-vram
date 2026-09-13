# Prédiction scellée : les 4,3 ms hors rejeu, décodage b=12 (bead runner)

poste1, 14/09/2026, AVANT profilage. poste4 a mesuré (noyaux) : pas total
16,5 ms, GPU (rejeu du graphe) 12,25 ms, donc **4,3 ms hors rejeu**, côté
hôte. vLLM fait le pas ENTIER en 10 ms. chef : pas ≤ 13,5 ms (host ≤
1,25 ms), par recouvrement (préparer le pas n+1 pendant le rejeu n) et
échantillonnage sur device (un seul `.item()` par pas).

## Où je m'attends à trouver les 4,3 ms, avant de profiler

Lecture de `Engine.step()`/`_plain_decode` (runner.py) — pas encore
vérifiée par mesure :

1. **La synchronisation du résultat échantillonné** — le plus gros poste
   attendu. `graph.replay()` est asynchrone ; lire le jeton produit
   (`.tolist()`/`.item()`) force le hôte à ATTENDRE que le GPU finisse.
   Si le code fait plusieurs allers-retours séparés (jeton échantillonné,
   test EOS, mise à jour de longueur…) au lieu d'UN SEUL, chacun peut
   resynchroniser au lieu de lire une valeur déjà rapatriée.
2. **La construction du lot du pas SUIVANT** — positions, `slot_mapping`,
   tables de blocs : si elle reconstruit des tenseurs neufs depuis des
   listes/dicts Python à CHAQUE pas, APRÈS avoir attendu le rejeu du pas
   courant, au lieu de la préparer PENDANT que le GPU rejoue encore, c'est
   du temps hôte qui pourrait se recouvrir avec du calcul GPU déjà en vol.
3. Ordonnancement (`_admit`, `_decodables`, comptabilité `stats`) : je
   m'attends à ce que ce soit MARGINAL (quelques dizaines de µs), une
   liste de 12 séquences filtrée à chaque pas — mais je le mesure quand
   même, plutôt que de le supposer.

**Prédiction chiffrée** : postes (1)+(2) ensemble captent plus de 70 % des
4,3 ms (donc plus de ~3 ms à eux deux) ; postes divers (ordonnancement,
journal, verrous) sous les 30 % restants (~1,3 ms). Si le profilage montre
un poste DOMINANT ailleurs (ex. l'ordonnancement lui-même, ou une
allocation répétée non identifiée ici), ce serait la surprise à expliquer
avant de continuer vers le recouvrement.

## Profilage (outils/profil_cpu_pas.py, branche poste4 adfad5b) — hypothèse confirmée

200 pas, b=12, cProfile + compteurs d'attentes hôte :

    attentes hôte/pas : tolist=1.9, item=0, synchronize=0, cpu=0
    _emit (runner.py:1027) : 2,718 s cumulés / 2,905 s profilés (93,6 %)
    dont les deux `.tolist()` (tokens, logprobs) : 2,695 s à eux seuls

**Confirmé, précisément localisé** : la quasi-totalité (93,6 %) du temps
hôte hors rejeu se passe dans `_emit`, DANS les deux appels `.tolist()`
(jetons, logprobs) qui suivent `sample()` — c'est le point où l'hôte
ATTEND la fin du rejeu GPU avant de pouvoir lire le résultat, exactement
la prédiction posée avant mesure (poste 1). Rien d'autre (ordonnancement,
`_build_batch`, `_fill`) ne pèse mesurablement — poste 3 confirmé
marginal (`_build_batch` : 0,011 s / 186 pas ≈ 60 µs/pas).

Mesure environnementalement bruitée (deux avertissements OOM pendant la
capture, carte partagée avec un processus non déclaré au moment du
lancement — pas la garantie « carte exclusive » de la nouvelle règle) :
le CHIFFRE absolu du pas sous profil (14,54 ms, cProfile ajoute lui-même
un surcoût d'instrumentation) n'est pas comparable au 16,5 ms de
poste4 — seule la RÉPARTITION relative (où va le temps) est fiable ici.

## Ce que la correction demande, et pourquoi je m'arrête avant de l'écrire

La suite (chef + poste4) : garder le jeton échantillonné comme
tenseur DEVICE (jamais rapatrié dans le chemin chaud), faire le
plongement du pas n+1 dessus directement sur device (`torch.embedding`
accepte un tenseur d'indices device), préparer/écrire les tampons
statiques du pas n+1 (positions, `slot_mapping`, tables — connus À
L'AVANCE, chaque séquence grandit de 1) PENDANT que le rejeu du pas n
tourne encore, et ne rapatrier les jetons (pour `_emit` : test EOS,
longueur, `output_ids`) qu'un pas plus tard, par une copie non bloquante
+ événement.

C'est un changement de la SÉQUENCE du moteur, pas un point local : il
touche `graphs.py` (`_fill` doit écrire dans les tampons du pas SUIVANT
sans toucher ceux que le rejeu courant lit encore — double tampon),
`runner.py` (`_plain_decode`/`_emit`, la dépendance embedding→jeton se
déplace d'un pas), et les points où `_eos`/l'arrêt d'une séquence
doivent maintenant se décider un pas EN RETARD. C'est le moteur central
dont dépend tout le reste du chantier (poste4, poste3, poste2) — je
préfère nommer l'ampleur et confirmer le feu vert avant de l'écrire
plutôt que de le pousser en un geste sur un mécanisme partagé par tout
le monde.
