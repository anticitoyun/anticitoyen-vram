# Pièce 261 — verdict (poste2, 26/09, ordre chef, reprise de la recherche de poste4) : `outils/panel-taches.sh`, panel mini lm-eval MMLU+GSM8K, testé à sec contre un faux serveur

* **instrument** : `outils/panel-taches.sh` (pilote `lm_eval` en deux invocations séparées —
  `--limit` est global à une invocation et MMLU/GSM8K veulent des limites différentes —
  contre un `acvram serve` déjà lancé, ne lance rien lui-même), `outils/panel-taches-resume.py`
  (fusionne les deux `results_*.json`, score par tâche/moyenne/pire tâche, garde le FILTRE
  déclaré en premier par la config pour une métrique à plusieurs filtres — GSM8K `strict-match`
  avant `flexible-extract`, sinon un des deux se perd au hasard de l'ordre d'itération) ;
  `tests/test_panel_taches_261.py` (bras cassant : un résumé sur résultats vides doit rendre
  faux) ; `scratchpad/poste2-p261-26-09/faux-serveur.py` (contrat `/v1/completions` minimal,
  pour le test à sec, aucun modèle réel).
* **commit** : (à la fusion de cette branche) sur `poste2-261`, tirée d'`origin/poste4-247`
  puis fusionnée avec `origin/main` (0.7.0 livrée, tag `v0.7.0`, 3d2993ccc).
* **régime** : à sec (aucune carte) — le test à sec tourne contre un faux serveur HTTP local,
  jamais un modèle chargé.
* **scellé** : néant (livraison d'outil, pas une mesure de qualité/vitesse).
* **mesuré** : recherche de poste4 reprise telle quelle (rien de commité sur `poste4-261`,
  juste sa synthèse par message) : `/v1/completions` supporte déjà `echo`+`logprobs`
  (`acvram/server/app.py:1171`, via `engine.logprobs_invite`) — compatible avec
  `local-completions` de `lm_eval` pour les tâches `loglikelihood` (MMLU) ; tâches vérifiées
  existantes dans `lm-eval` 0.4.13 : `mmlu_high_school_mathematics`, `mmlu_professional_law`,
  `mmlu_college_computer_science` (multiple_choice), `gsm8k` (generate_until).
  Sous-ensemble et graine FIXÉS dans le script (jamais en argument, pour que P0/P1 tournent
  sur le même tirage) : 100 questions par tâche MMLU, 50 pour GSM8K, graine 1234.
  Test à sec : `lm-eval[api]` (`tenacity`, `tiktoken`) et `transformers` manquaient au premier
  essai (`ModuleNotFoundError`), installés dans `.venv-panel` (CE worktree seulement, jamais
  le `.venv` du dépôt — dépendances d'évaluation séparées de torch/triton du serveur) ; une
  fois installés, exécution bout en bout propre (code 0), résumé correct : 4/4 tâches,
  `gsm8k` prend bien `exact_match,strict-match` après la correction du bogue de fusion de clé
  (1er essai fusionnait `strict-match` et `flexible-extract` sous une seule clé `exact_match`,
  perdant l'un des deux au hasard — corrigé avant la livraison, pas après coup).
* **verdict** : outil livré, plomberie vérifiée à sec (aucune carte engagée) ; prêt pour la
  237c (comparaison P1 contre P0 sur l'alias `Qwen3-Coder-30B-A3B-nvfp4`, sous `carte.sh`,
  `ACVRAM_NOM` posé).
* **durée** : ~1 h à sec (installation venv, lecture du contrat `local-completions`, écriture,
  bogue de fusion de clé trouvé et corrigé avant livraison), aucune carte.

## Reste ouvert (hors périmètre de cette pièce)
* Le test à sec ne couvre pas un VRAI `acvram serve` (aucune carte engagée dans cette pièce) —
  la 237c sera la première mesure réelle contre le panel.
* `.venv-panel` n'est PAS committé (poids/paquets Python, hors dépôt par construction) — à
  recréer sur tout poste qui reprend cette branche (`uv venv .venv-panel --python 3.12 && uv
  pip install --python .venv-panel "lm-eval[api]" transformers`).
