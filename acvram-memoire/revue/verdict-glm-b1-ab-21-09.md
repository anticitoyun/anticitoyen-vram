# GLM b=1 A/B (paquet installé vs 0d77e742) — 21/09

* instrument : `scratchpad/glm-b1-ab-21-09/chaine.sh` (fichier suivi, ce commit)
* commit : A = paquet installé (dpkg control « 0.6.34 », code réel ≠ 0d77e742 et voisins — vérifié par `cmp` sur `regime.py`, pas pris sur la note) ; B = worktree figé 0d77e742
* régime : B2 seul régime propre atteint — `eco=2700(2692) kv=latent-bf16 mla_glue=2 mla_prep=grille pipeline=1`
* scellé : moyenne(B) ≥ 0,98 × moyenne(A)
* mesuré : A1 ÉCHEC, B1 ÉCHEC, A2 ÉCHEC, B2 TENU (164,2 t/s, 0,9878 J, 224,8 W)
* verdict : **INDÉCIDABLE** — 1 seul point sur 4, pas de moyenne(A) calculable (calcul final `AttributeError` sur `None.group()`, script à corriger). Deux causes distinctes, aucune sur le moteur :
  1. **A1/A2 : le paquet installé n'a pas les sources Marlin** — `acvram/kernels/marlin_port/bindings.cpp` absent de `~/.local/share/acvram/venv/…/site-packages/acvram/kernels/marlin_port/` (seuls `__init__.py` et `disposition.py` présents ; le fichier existe dans l'arbre source `acvram/kernels/marlin_port/bindings.cpp`). **Même défaut que § 1b sur acvram-coder-i8c** (`verdict-1b-precharger-21-09.md`), confirmé une 2e fois sur un modèle différent (GLM) → défaut de *packaging* du .deb installé, reproductible, pas un accident. Le message « disposition Marlin refusée » s'affiche avant le crash : le crash n'est pas sur le chemin retenu mais sur une compilation JIT qui référence ce fichier quand même.
  2. **B1 : mon timeout (300 s, `chaine.sh`) était trop court** pour un premier chargement sur ce worktree (compilation JIT probable, cache froid) — le serveur était encore en train de chauffer/capturer (dernière ligne du log : « mémoire avant capture ») quand `timeout` l'a tué, sans message d'erreur. B2, juste après, a réussi en 63 s (cache chaud). Défaut de mon script, pas du moteur.
* durée : 11:00:19–11:41:45 (41 min), très au-dessus des ≤ 30 min visés — dominé par mes deux boucles d'attente curl (jusqu'à 450 s chacune) sur des serveurs déjà morts (A1, A2)

## Suite
1. Paquet installé cassé pour Marlin sur ≥ 2 modèles (Coder, GLM) — à signaler avant tout `dpkg -i` d'un nouveau candidat : vérifier que le .deb en attente embarque bien `kernels/**/*.cpp`/`*.cu` (probable oubli dans `tools/construire-deb.sh` ou le `MANIFEST`).
2. Rejouer B seul (2-3 passes, cache chaud) pour une moyenne exploitable, avec un timeout ≥ 600 s au premier chargement et un abandon par `kill -0 $SRV` réel (pas par la boucle curl pleine) — chaîne à corriger avant relance.
