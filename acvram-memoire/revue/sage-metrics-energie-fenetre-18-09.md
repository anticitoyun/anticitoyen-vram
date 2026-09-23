# Sage — J/jeton de `/metrics` : contrôle tenu (−0,18 %) ; corriger la fenêtre « depuis le dernier appel » par une fenêtre fixe que la lecture ne consomme pas (18/09)

Entrée : Jérôme — Laure `verdict-controle-gui-energie-18-09` (a2f7d4a) : 0,3794 contre `energie.py` 0,3801 J/jeton, même fenêtre (18 360 jetons, 16,8 s, 72 req), tenu. Limite : delta « depuis le dernier appel `/metrics` », état partagé côté serveur — deux lecteurs se volent le delta.

## 1. Décision : corriger, une heure, à sec (Jérôme)

Une lecture qui **modifie** ce qu'elle mesure est un instrument corrélé à son lecteur (REGLES § 4) : la GUI, un `curl` de session et Prometheus donneraient trois chiffres pour une grandeur — et le mauvais instant d'échantillonnage (un lecteur qui tombe juste après un autre) rend `None` ou un delta de 30 ms, lu comme « rien ne tourne ».

Forme : un anneau d'échantillons `(t, energie_mj, jetons)` alimenté **par le serveur** (tic de 1 s, ou à chaque requête servie — le tic, pour que le repos aussi s'échantillonne), et `/metrics` rend la pente sur une fenêtre fixe, calculée sans rien consommer. Le nom porte le régime : `energie.j_par_jeton_10s`, avec `fenetre_s` et `jetons_fenetre` à côté ; `None` seulement si `jetons_fenetre = 0`. Pas d'identifiant par client (un état par lecteur est le même défaut en plus grand).

Tests, dans le même commit : (1) deux lecteurs concurrents à 50 ms rendent la même valeur au même instant ; (2) un lecteur à 100 ms ne voit jamais `None` sous charge ; (3) bras cassant : l'ancien « delta consommé » ⇒ rouge sur (1). Contrôle sur carte, une fois, dans une fenêtre de Laure déjà prise : `j_par_jeton_10s` ± 10 % de `energie.py` sur 10 s — la prédiction de Jérôme < 3 % tient pour la même fenêtre.

## Ordre

* Jérôme : correctif + 3 tests à sec, commit, `.deb` suivant ; `verdict: revue/<fichier> — (passed, skipped)`. Laure : le contrôle ± 10 % à la prochaine fenêtre qu'elle tient de toute façon, une ligne.
