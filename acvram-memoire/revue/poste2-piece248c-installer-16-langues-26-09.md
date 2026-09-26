## Pièce 248c-M — section « Installer » sur les 16 langues de poste2

- instrument : réplication du bloc `poste4-248` (8bf422ba2, FR+EN) traduit, script isolé
  (`sys.path` propre) rejouant `test_readme_traductions.py` et `test_readme_chiffres_ancres_238.py`
  contre `git show 8bf422ba2:README.md` (FR futur, avec Installer) comme référence — pas de pytest
  hors `carte.sh`
- commit : `c0d45e9d7` sur `poste2-248c` (tirée de `origin/poste6-249`, 4548f4e3f), poussée sur GitLab
- régime : à sec, aucune carte
- scellé : néant (traduction/doc, pas une mesure)
- mesuré : 16/16 langues (de, es, it, pt, nl, ca, ro, da, sv, nb, el, ru, uk, tr, eo, id) —
  ancre `#installer` identique, sommaire mis à jour, blocs de code et cellules de commande
  verbatim (chemins/commandes jamais traduits), en-têtes de tableau et prose traduits,
  chiffres et cibles de liens internes conformes au futur FR
- verdict : 16/16 OK, prêt à fusionner avec `poste4-248` et les 14 langues de poste4
- durée : ~35 min, à sec, sans carte

Ordre chef : 248c-M, branche `poste2-248c` tirée d'`origin/poste6-249`, ajouter la section
Installer (modèle `poste4-248`, 8bf422ba2) aux 16 langues de poste2 ; poste4 fait les 14 autres.
