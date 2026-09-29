# i61 — parc-installer ne crée plus de clés aléatoires quand des clients existent : verdict (poste5, 29/09)

* instrument : tests à sec `tests/test_parc_secrets_clients_i61.py` (poste factice de test_paquet_parc, parc-installer réel en sous-processus)
* commit : poste5-cles2 (voir git log)
* régime : à sec, aucune carte, rien écrit hors tmp
* scellé : les 5 tests doivent casser sur l'ancien parc-installer (vérifié : 5/5 rouges) et passer sur le nouveau (5/5)
* mesuré : 5 verts ; suite parc voisine (test_paquet_parc, test_menus, kimi, classer_dossiers, gui_dossiers, index) 49 verts
* verdict : VRAI
* durée : 0 (à sec)

## Cause (rappel, revue/poste5-cles-verdict-29-09.md)
parc-installer:566-571 (avant) : `ia-secrets.env` absent → `token_hex(8)` par moteur, sans regarder les clients. De
plus, la fusion kimi ajoutait les fournisseurs absents avec la clé PAR DÉFAUT, jamais avec celle des secrets : même
sur un poste neuf, kimi et les serveurs divergeaient.

## Correctif (parc/bin/parc-installer)
* `cles_des_clients` : clés que portent déjà config.toml kimi ([providers.<moteur>].api_key) et
  `<extras.openwebui_dir>/openwebui.env` (URL → moteur par la route /<moteur>/ ou le port), par variable CLE_* ;
  conflits nommés par source, jamais par valeur.
* Les clés sont décidées AVANT la fusion kimi, qui les reçoit pour ses nouveaux fournisseurs. Fichier présent : lui,
  jamais touché. Absent : clés des clients, défaut des lanceurs sinon (ce que servaient les serveurs sans fichier).
  Aléatoires seulement sur un poste sans aucun client, et alors kimi reçoit les mêmes. Deux clients en désaccord :
  REFUS rc 4 avec « À faire », ni kimi ni secrets écrits.
* Contrôle 7 ajouté à l'en-tête du programme.

## Limites
* La base Open WebUI (table `config`, qui prime sur l'env) n'est pas lue : seul openwebui.env l'est. Une base
  divergente de l'env ne serait pas vue (cas du 28/09 : les deux portaient les mêmes clés).
* Les scripts hors dépôt à clé littérale (memoire-consolider, rapport-sante-ia, banc-moteurs, verifier-pile) ne sont pas
  lus ; avec l'option B (clés d'avant rendues), ils sont de nouveau d'accord.
