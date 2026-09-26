# Verdict — budget KV dimensionné pour le lot réel, pas pour 8 par défaut (17/09)

poste1, à sec, ordre poste7 (`revue/poste7-poste-d-verdict-17-09.md` § 2), sur
un bogue trouvé par poste3 (`certifie-b12`, `cead3cb`).

## Le bogue

`Engine._replanifier` (`loader.py:1448`, anciennement ligne 1459) ne
passait que `max_model_len` à `PlannerOptions` — jamais
`max_concurrent_seqs`. Le budget KV (`tiering.py:447`,
`wanted = kv_per_tok × max_model_len × max_concurrent_seqs`) tournait donc
**toujours** pour la valeur par défaut de `PlannerOptions` (8 séquences),
quel que soit le `b`/`--max-batch` réellement demandé — `_replanifier`
rejoue le plan à **chaque** chargement (« sans condition », docstring
existant), donc le défaut s'appliquait systématiquement, pas seulement au
premier chargement. À b=12, 4 séquences sur 12 tronquaient silencieusement
avant `max_tokens` (`Engine._finish_budget_epuise`, budget épuisé) : aucun
compteur ne le signalait, `certifie-b12` mesurait un débit dont le lot réel
était plus petit que celui annoncé.

## Le correctif — un commit

* `loader.py` : `load_model`/`_plan_from_manifest`/`_replanifier` acceptent
  désormais `max_concurrent_seqs` et le passent au planificateur. Absent
  (`None`), on retombe sur ce que le manifeste avait planifié
  (`kv_planned_seqs`, nouveau champ de `Plan`), jamais sur le défaut caché de
  8 — même logique que le traitement déjà en place pour `max_model_len`.
* `cli.py` (`acvram serve`) passe `args.max_batch` ; `bench.py::bench_decode`
  passe `1` (cohérent avec son `max_batch_size=1`).
* `tiering.py` : `Plan.kv_planned_seqs` porte le nombre de séquences pour
  lequel `kv_max_tokens` a été calculé, sérialisé dans le manifeste.
* `runner.py` : `EngineStats.sequences_tronquees_budget` compte les
  troncatures par budget épuisé (incrémenté dans `_finish_budget_epuise`,
  exposé par `to_dict()`) ; `regime_ligne()` imprime désormais
  `kv_budget=<jetons>/<séquences_planifiées>`.
* `scratchpad/certifie-b12-15-09.py` et `certifie-b12-duree-15-09.py`
  (poste3/poste4) : passent `max_concurrent_seqs=B` à `load_model` — le
  contournement `CERT_PLAN_LEN` (gonfler `max_model_len` pour compenser)
  disparaît, comme prévu par poste7. Les deux refusent désormais de publier
  un résultat si `engine.stats.sequences_tronquees_budget > 0` : elles
  écrivent `{"invalide": true, "cause": ...}` et sortent en erreur plutôt
  que d'écrire un débit dont le lot a été rogné en cours de mesure.

## Tests — `tests/test_budget_kv_slots_chargeur.py`, 6 tests

Sur un point de contrôle jouet (fixture `converted` de `conftest.py`,
planifié pour 2 séquences), `detect_rig` monkeypatché sur `target_rig` pour
un rig déterministe : le budget grandit avec le nombre de séquences
demandé (1 < 8 < 12, jamais identique) ; `kv_planned_seqs` porte le slot
demandé ; sans argument, on hérite du manifeste (2), pas du défaut de
`PlannerOptions` (8) ; `regime_ligne()` porte `kv_budget=<jetons>/<seqs>` ;
`sequences_tronquees_budget` s'incrémente et apparaît dans `to_dict()` et
dans `regime_ligne()` quand un allocateur minuscule force l'épuisement.

**Doit casser, vérifié** : `git stash` des 5 fichiers sources (loader,
runner, tiering, cli, bench), les 6 tests échouent immédiatement
(`TypeError: load_model() got an unexpected keyword argument
'max_concurrent_seqs'`) — pas un échec de logique, la preuve que le
paramètre n'existait pas avant ce commit. Stash restauré (`git stash apply
<sha>`, jamais `pop`), puis supprimé.

Suite complète : 617 passed (+6 sur les 611 d'avant ce commit), 1 échec
pré-existant hors périmètre (`test_le_cliquet_des_chemins_absolus_ne_monte_pas`,
fichiers d'autres sessions, déjà signalé le 16/09).

## Pour REGLES § 3 (proposition, à trancher par chef/poste7)

« L'en-tête d'une cellule b=N atteste lot = N sur 100 % des pas pleins,
sinon la cellule n'entre pas dans INDEX » — `sequences_tronquees_budget`
donne l'instrument : tout banc qui compte un débit doit vérifier ce
compteur à zéro avant de publier, pas seulement lire les logs.
