# Tri — vidage visuel des lignes du haut : corrigé, sans garde automatisée fiable (poste3, 30/09-01/10)

instrument : vraie fenêtre GTK sous Xvfb (`Xvfb :9N`), clic RÉEL `xdotool mousemove … click 1`
(jamais l'appel API `sort_by_column` du crochet `ACVRAM_GUI_TEST`), capture `import -window`.
commit : `HEAD` de `poste3-tri`. régime : aucun (pas de carte). scellé : aucun. mesuré : oui
(reproduit à la main plusieurs fois, avant/après correctif). durée : longue (~2 h, diagnostic).

## Repro confirmée (chef, écran réel `:0`, puis re-confirmée ici sous Xvfb)

Données réelles (`~/TSV`, 266 alias), fenêtre neuve, clic sur l'en-tête « Qualité » : les
lignes du HAUT restent une zone noire, le compteur affiche toujours 266/266, un CLIC dans la
zone ne change PAS la sélection (contrairement à une ligne normale) — testé ici avec 7
positions différentes dans la zone, aucune n'a réagi. Molette bas puis haut : la zone se
redessine correctement et reste correcte ensuite. `_textes_visibles()` (le crochet
`ACVRAM_GUI_TEST=trier:`) ne voit RIEN d'anormal : il lit `Gtk.Label.get_text()`, qui est
correct même quand la ligne n'est plus interactive/dessinée — c'est pour ça que
`tests/test_gui_tri_colonnes.py` (28 cas, données réelles) ne l'a jamais vu.

## Correctif appliqué

`self.sorter_natif.connect("changed", self._sur_tri_natif_change)` (`_construire_liste`,
`fenetre.py`) ; `_sur_tri_natif_change` fait `vue_liste.scroll_to(0, None,
Gtk.ListScrollFlags.NONE)` puis `queue_draw()`, avec garde sur liste vide (le premier tri posé
à la construction, avant `_recharger`, lève sinon `gtk_column_view_scroll_to: assertion
'pos < …' failed`). `queue_draw()` SEUL ne suffisait pas (testé, toujours vide) ; `scroll_to`
répare la vue à chaque fois que reproduit sur données réelles (3 essais indépendants).

## Ce qui N'EST PAS confirmé — à ne pas sur-affirmer

Je n'ai pas isolé la cause EXACTE côté GTK (GtkColumnView/GtkListView, version du système).
Une tentative de repro minimale et automatisable (dépôt synthétique, 266/80 alias, décalage
contrôlé de la ligne sélectionnée) n'a PAS reproduit le même symptôme : elle montre un
comportement de défilement normal (GTK garde la ligne sélectionnée visible, pas forcément en
position 0) — vérifié en remontant : les lignes existent, sont correctement triées, rien n'est
vide ni figé. Le vidage RÉEL ne s'est reproduit QUE sur les données réelles (266 alias avec
tok/s, refus, usage variés sur toutes les colonnes), jamais sur un TSV synthétique réduit à la
seule colonne Qualité — la cause dépend probablement d'un facteur non isolé (richesse des
colonnes, ordre d'arrivée des lignes, minutage de rendu). Le fichier de test écrit d'abord
(comptage de couleurs distinctes) a été retiré : il passait AUSSI bien avec l'ancien code que
le nouveau — un test qui réussirait même si le défaut était présent (REGLES, poste3 §
« jamais ‹ recensement complet › ») n'a rien à faire dans le dépôt.

## Reste (pas une pièce fermée)

**Aucune garde automatisée dans `tests/`** contre une régression de cette correction — à
écrire si une méthode fiable de repro minimal est trouvée (probablement un vrai test visuel,
hors du crochet `ACVRAM_GUI_TEST` existant, avec un jeu de données proche du réel). En
attendant : si le vidage revient, le premier réflexe est de vérifier que
`_sur_tri_natif_change` est toujours branché et que `scroll_to` ne lève pas une exception
avalée silencieusement (aucune trace au journal pendant tout ce diagnostic, GTK semble
avaler les erreurs de ses gestionnaires de signaux).

verdict: acvram-memoire/revue/poste3-tri-vidage-verdict-30-09.md — vidage visuel du tri confirmé sur vrai clic (écran réel ET Xvfb, données réelles), corrigé par `scroll_to(0)+queue_draw` sur le signal `changed` du trieur natif, vérifié à la main 3 fois ; cause GTK exacte non isolée, repro minimale automatisable non trouvée — pas de test de régression écrit, à noter comme dette
