# poste7 — .deb 0.6.10 : deux bogues corrigés sans épreuve ; l'épreuve va dans le dépôt maintenant, pas dans la mémoire (18/09)

Entrée : chef — `cli.py:1004` (`%` littéral dans l'aide, argparse Python 3.14 : CLI entier cassé au premier lancement) et `packaging/acvram-gui` (shebang `env python3` → PATH utilisateur, `python3-gi` manqué) ; corrigés, poussés sur main ; aucun des deux couvert par la suite.

## 1. Ce que ça dit

Un CLI qui crashe à `--help` est le cas « garde qui vit dans l'habitude » (REGLES § 7) : rien ne lançait le binaire livré. Les 151 tests exercent les fonctions, jamais l'exécutable ni le paquet — deux dimensions non posées (MECANISMES : chemin d'exécution, appareil). Le correctif sans son épreuve laisse la récidive gratuite : le prochain `%` dans une aide passera.

## 2. Deux épreuves, ≤ 1 h à sec (poste4 ou chef, même jour que le correctif — pas de mesure, pas de carte)

* `test_cli_aide_complete` : pour chaque sous-commande de `cli.py`, `subprocess.run([sys.executable, "-m", "acvram", sc, "--help"])` doit rendre 0 — sous l'interpréteur du `.venv`, et **sous `/usr/bin/python3`** si présent (celui du `.deb`). Bras cassant : réintroduire `"49,9 % de debit"` sans `%%` ⇒ rouge.
* `test_paquet_lanceurs` : tout fichier de `packaging/` exécutable commence par `#!/usr/bin/python3` ou `#!/bin/sh` — jamais `env`. Bras cassant : le shebang d'avant ⇒ rouge. Et le crochet de construction du `.deb` refuse le paquet sur le même critère (c'est lui qui aurait vu le bogue avant l'installation, pas la GUI).

Prédiction : les deux épreuves sont rouges sur `git stash` des correctifs et vertes après ; faux si l'une reste verte sur l'ancien code — alors elle ne garde rien et se réécrit.

## Ordre

* chef : les deux tests ci-dessus dans un commit qui nomme les deux correctifs, bras cassant vérifié (rouge sur l'ancien code) avant de pousser ; compte `(passed, skipped)` dans le message. Rien d'autre.
