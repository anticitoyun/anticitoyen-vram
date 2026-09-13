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
