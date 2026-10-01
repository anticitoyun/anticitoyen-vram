# e50.3 § 7 — yaml + qualite-e50.sh + bac à sable écrits, --simule vérifié, --executer NON exercé (poste3, 01/10)

instrument : pytest ciblé, processeur seul, aucune carte (REGLES §3). commit : `HEAD` de
`poste3-e50-3` (sur `poste3-e50-1`, fusionnée — `deriver_capacites` nécessaire à la détection
« thinking »). régime : aucun. scellé : méthode d'poste6 (`poste6-e50-3-methode-qualite-01-10.md`),
non touchée ici. mesuré : oui pour tout ce qui ne dépend pas de `.venv-panel`/un serveur réel
(cassant vérifié par `git stash`/retrait de ligne) ; **rien mesuré pour `--executer`, qui
n'a PAS pu être exercé — `.venv-panel` n'existe pas sur ce poste**. durée : 0 min de carte.

## Ce qui est livré

- `outils/lm_eval_taches/{mmlu_e50_hsm,mmlu_e50_law,mmlu_e50_ccs,gsm8k_e50,humaneval_e50}.yaml`
  — dérivés des yaml 275d (MMLU) et du schéma public lm-eval (GSM8K, HumanEval), 0-shot,
  `max_gen_toks` figé (768 MMLU/GSM8K, 512 HumanEval), filtre `get-answer-v275` repris tel
  quel pour MMLU (validé, voir plus bas). `humaneval_e50.yaml` : `metric_list: []` À DESSEIN —
  voir § 2 bis.
- `outils/bac-a-sable-humaneval.sh` — bwrap (`--unshare-all`, racine en lecture seule, `$HOME`
  démonté PUIS remonté en lecture seule, `/tmp` seul inscriptible, ulimits CPU/mémoire/fichiers/
  processus) ; `PY_BAC_A_SABLE` pour les tests seulement.
- `outils/qualite-e50-humaneval-score.py` — pass@1 calculé À PART de lm-eval (jamais son
  exécuteur `unsafe_code` interne, non vérifiable ici sans le paquet) : reconstruit
  prompt+complétion+test par échantillon journalisé, score via le bac à sable.
- `outils/qualite-e50-bareme.py` — barème § 3 (étoiles ↔ S, gardes par palier, « format »
  prioritaire), fonction pure.
- `outils/qualite-e50.sh <alias> [--executer --je-sais-que-la-carte-est-libre]` — `--simule`
  (défaut) résout l'alias (dossier, détection « thinking » via `deriver_capacites`, e50.1) et
  imprime le plan exact, sans rien lancer. `--executer` est ÉCRIT (serveur neuf, 2 appels
  lm-eval, écriture PAR `parc.py:ecrire_note` SEULEMENT) mais REFUSE explicitement de
  continuer au-delà du plan (`.venv-panel` absent) — honnête plutôt que de prétendre avoir
  vérifié un chemin que je n'ai pas pu exécuter.

## Ce qui N'EST PAS vérifié, dit sans détour

`.venv-panel` (lm-eval) n'existe nulle part sur ce poste — aucune prise n'était de toute façon
demandée. Je n'ai donc PAS pu : lancer un vrai `lm_eval run` avec ces yaml, vérifier que
`gsm8k_e50`/`humaneval_e50` se chargent réellement (schéma dérivé du format public lm-eval,
jamais rejoué contre le paquet installé), ni vérifier le branchement MMLU+GSM8K en un seul
appel ni la boucle serveur de `--executer`. **La pièce suivante (poste2/poste1,
`campagne-qualite-e50.py`) doit d'abord faire passer `qualite-e50.sh --executer` sur UN alias
avant toute campagne — c'est le premier test réel de ce qui est écrit ici.**

## Ce qui EST vérifié, sans lm-eval ni carte

- **Bac à sable** (`tests/test_bac_a_sable_humaneval.py`, 10 cas, bwrap réel installé sur ce
  poste) : les 6 tentatives hostiles de la méthode (réseau local + externe, écriture
  `$HOME`/`/etc`/`/mnt`, lecture d'un secret, boucle, allocation excessive, fork-bomb) ÉCHOUENT
  toutes, un code honnête PASSE. **Cassant** : retrait de `--unshare-all` et
  `--remount-ro "$HOME"` → 3/10 échouent (réseau et écriture `$HOME` réussissent) ; restauré,
  10/10. Piège trouvé en écrivant le test : `--tmpfs "$HOME"` seul est RÉINSCRIPTIBLE (tmpfs
  vide mais accessible en écriture) — il faut `--remount-ro` après, et `.venv-panel` vit SOUS
  `$HOME` (tous les worktrees y sont) : un `--ro-bind` du dépôt doit revenir APRÈS le tmpfs et
  AVANT le remount-ro, sinon bwrap ne peut plus trouver son propre interpréteur.
- **Extraction MMLU** (`tests/test_qualite_e50_extraction.py`, 4 cas) : le filtre
  `get-answer-v275` rejoue À L'IDENTIQUE 20 conclusions RÉELLES de la campagne 275
  (`tests/fixtures/echantillons_mmlu_e50_20.json`, extraites de
  `~/.cache/acvram/qualite-275/`), comparées à ce que lm-eval avait lui-même extrait avec le
  MÊME filtre — 20/20 identiques.
- **Barème** (`tests/test_qualite_e50_bareme.py`, 7 cas) : bornes centrales et de chaque
  palier, garde qui descend d'UNE étoile (jamais plus), absence de garde sous ★★★, « format »
  prioritaire sur le calcul de S, cas E3 de la méthode (Coder-30B).
- **pass@1 HumanEval** (`tests/test_qualite_e50_humaneval_score.py`, 3 cas) : une bonne/une
  fausse, toutes bonnes, toutes fausses — sur un jsonl construit à la main.
- **Script `qualite-e50.sh --simule`** (`tests/test_qualite_e50_script.py`, 6 cas, parc réel) :
  résout un alias « thinking » et un alias non-thinking, refuse un alias inconnu, refuse
  `--executer` sans confirmation, refuse un argument inconnu, pin des limites par tâche
  (30/30/30/40/40 — casse si quelqu'un les change sans y toucher ici).
- **yaml** (`tests/test_qualite_e50_yaml.py`, 5 cas) : les 5 fichiers existent et se parsent,
  portent leur `task:`, pin `num_fewshot`/`max_gen_toks`, filtre MMLU identique au 275d,
  `humaneval_e50` sans métrique exécutrice native (REGLES § 6).

**Total : 35/35 verts**, suite GUI/parc élargie rejouée sans régression.

## Correctif chef (après relecture, chargement réel avec le venv figé de poste2)

Confirmé exactement ce que je craignais, dit plus haut : `gsm8k_e50.yaml` et
`humaneval_e50.yaml` levaient `HfUriError: Repository id must be 'namespace/name', got
'gsm8k'`/`'openai_humaneval'` au premier chargement réel (`travail/poste2-275-figee/.venv-panel`,
le vrai lm-eval de la campagne 275). Corrigé : `dataset_path: openai/gsm8k` et
`openai/openai_humaneval`. `tests/test_qualite_e50_taches_chargent.py` (3 cas, SAUTE si ce venv
précis est absent) charge RÉELLEMENT les 5 tâches (`TaskManager(include_path=...)
.load_task_or_group`, `eval_docs`, `doc_to_text` du 1er document — ex. GSM8K 1 319 docs,
HumanEval 164, MMLU 100-1 534 selon le sujet) et casse sur l'ancien `dataset_path` (rejoué dans
une copie isolée des yaml, vérifié : l'ancien lève `HfUriError`, le nouveau non). 38/38 verts au
total, aucune prise.

verdict: acvram-memoire/revue/poste3-e50-3-mise-en-oeuvre-verdict-01-10.md — 5 yaml e50, qualite-e50.sh (--simule vérifié, --executer écrit mais honnêtement non exercé, .venv-panel absent), bac-a-sable-humaneval.sh (10/10, cassant vérifié, piège $HOME tmpfs réinscriptible trouvé et corrigé), barème/extraction/score pass@1 testés sans lm-eval ; écriture exclusivement par parc.py:ecrire_note ; 35/35, aucune prise
