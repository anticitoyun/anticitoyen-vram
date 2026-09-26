# Pièce 238 — verdict (poste2, 25/09, ordre chef) : 10 README traduits au modèle 231, garde chiffres/ancres/liens

* instrument : traduction manuelle de,es,it,pt,nl,ca,ro,da,sv,nb sur le modèle animematrix (231) ;
  `tests/test_readme_chiffres_ancres_238.py` (nouveau — ancres, chiffres, cibles de liens internes
  contre le FR) + `tests/test_readme_traductions.py` rattrapé au nouveau modèle (logo 3 lignes,
  barre de langues par drapeaux 🇫🇷/🇬🇧, jetons CLI comptés sur tout le fichier)
* commit : `1172a2f34` (test cassant seul), `18e27d718` (10 traductions + tests), `dc87185c3`
  (carnet, point de reprise) sur `poste2-238`
* régime : à sec, aucune carte
* scellé : néant (traduction/doc, pas une mesure)
* mesuré : 10/10 langues (de,es,it,pt,nl,ca,ro,da,sv,nb) — logo officiel en tête, barre de langues
  complète, lien de soutien présent, blocs de code identiques en nombre de lignes, jetons CLI
  (`acvram`, `curl`, `install.sh`, drapeaux `--...`, etc.) comptés à l'identique du FR, ancres
  `<a id="...">` identiques, chiffres recopiés à l'identique (multiset, ordre libre dans une même
  phrase), cibles de liens internes alignées (`docs/` → racine près) — 10/10 OK ; suite complète
  rejouée sous `carte.sh` par chef à la fusion (pas de pytest hors verrou de mon côté)
* verdict : 10/10 langues conformes au modèle 231 ; el/ru/uk/tr/eo/id restées en attente (239b),
  pl/cs/hu/fi à la garde de poste3
* durée : session du 25/09, à sec, sans carte

Note tardive (26/09, ordre chef) : cette pièce n'avait pas de note dédiée dans `revue/` — signalé
par ma propre contre-lecture (251) de `docs/notes/v0.7.0.md`, qui la cite. Écrite après coup pour
combler ce trou, contenu conforme au carnet `acvram-memoire/poste2.md` (25/09) et aux deux commits
ci-dessus, aucune mesure nouvelle.
