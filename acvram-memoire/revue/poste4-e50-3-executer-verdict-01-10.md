# e50.3 — implémentation de `--executer` (serveur, lm-eval, pass@1, barème, écriture), campagne, estimation (poste4, 01/10, à sec)

Bead : anticitoyen-vram-e50.3 (P2). Méthode et barème déjà scellés par poste6
(`revue/poste6-e50-3-methode-qualite-01-10.md`, non rouverts ici) ; squelette §7.1-7.2 (yaml,
`qualite-e50.sh --simule`, bac à sable, bareme/score) déjà livré par poste3, fusionné. Reste
exactement ce que la note §7 appelait « pièce suivante » : le corps réel de `--executer`, la
campagne (§7.3), un test de bout en bout, l'estimation de temps.

instrument : lecture du code existant (qualite-e50.sh, panel-taches.sh, serveur-bras.sh,
qualite-e50-bareme.py, qualite-e50-humaneval-score.py), parc réel (~/TSV, ~/.kimi-code/config.toml)
commit : fusion sur `poste4-224` depuis `origin/main` d7e7de8ba
régime : à sec (REGLES §3), aucune carte, aucun GPU, aucun lm-eval réel installé sur ce poste
scellé : aucun nouveau seuil — le barème § 3 d'poste6 n'est pas touché
mesuré : rien de qualité réelle (aucun modèle interrogé) ; la PLOMBERIE est testée contre un faux
serveur et un faux lm-eval (ci-dessous)
verdict : **`--executer` était un BOUCHON** (`qualite-e50.sh` sortait `ÉCHEC : --executer n'est
pas encore exercé...` après la résolution de l'interprète lm-eval, rien d'autre) — implémenté,
testé de bout en bout sans carte ni lm-eval réel ; `campagne-qualite-e50.py` (§7.3) écrit ;
estimation affinée : **111 alias non mesurés aujourd'hui** (pas 108 : le parc a bougé depuis le
01/10 matin, chiffre exact utilisé, pas corrigé vers l'ancien), **≈ 11,1 h de carte** en 12 lots
de 10 + témoin
durée : 0 min de carte

## Ce qui était fait avant cette pièce (non refait)

Méthode (§1-6), barème (§3), témoin (§4) : poste6, scellés. Yaml e50 (×5), `qualite-e50.sh
--simule`, `outils/bac-a-sable-humaneval.sh` + son test adversarial (6 évasions + 1 code honnête),
`qualite-e50-bareme.py`, `qualite-e50-humaneval-score.py`, résolution de l'interprète lm-eval
(`$ACVRAM_LMEVAL_PY` > `.venv-panel` > copie figée 275) : poste3, fusionnés, tous testés (41 tests
verts avant cette pièce). `--executer` s'arrêtait net après avoir imprimé l'interprète résolu.

## Ce qui a été ajouté ici

1. **Corps de `--executer`** (`outils/qualite-e50.sh`) : réutilise le harnais
   `outils/gpu/mesure/serveur-bras.sh` (déjà éprouvé contre un faux serveur par
   `tests/test_serveur_bras.py`, ordre chef 01/10 — jamais de `$!` maison) pour lancer et
   arrêter le serveur ; deux appels lm-eval `local-chat-completions` (MMLU 90 + GSM8K 40 en un
   seul ; HumanEval 40 à part, génération seule) ; agrégation MMLU = moyenne des 3 sous-tâches,
   GSM8K = filtre `flexible-extract` (méthode § 2 : le filtre strict punit le format, pas le
   calcul) ; pass@1 HumanEval via `qualite-e50-humaneval-score.py` (bac à sable, déjà testé) ; S
   composite, étoile (`qualite-e50-bareme.py`), écriture `outils/qualite-e50.tsv` + `ecrire_note`
   (refus/tps/usage existants préservés, seule la colonne qualité change).
2. **`outils/campagne-qualite-e50.py`** (§7.3) : `--simule` lit le parc réel (intersection
   `config.toml` ∩ `notes-modeles.tsv` « non mesuré »), classe chaque alias (dense_30b/moe/
   petit_dense/grand_dense_exile) par son nom, prédit la durée, imprime les 12 lots de 10 +
   témoin. `--executer` joue le témoin en tête de chaque lot, garde une **cohérence simple**
   (écart de S au témoin > 0,05 → arrêt) — **PAS le McNemar item par item prescrit § 4** (celui-ci
   exige les échantillons appariés, `panel-taches-resume.py` sait le faire mais reste à brancher
   par qui mesure réellement, poste2 ou poste1, avant de publier une étoile) ; pause coopérative
   (fichier, même convention que `campagne-e50.2-nocturne.py`) ; un alias en échec garde son
   ancienne valeur, jamais un 0 écrit à la place (même règle que e50.2).
3. **Test de bout en bout contre un FAUX serveur ET un faux `lm_eval`**
   (`tests/test_qualite_e50_executer_faux_serveur.py`) : serveur HTTP maison (`/v1/models`,
   `/v1/chat/completions`, complétion fixe), module `lm_eval` injecté par `PYTHONPATH` (répond au
   contrat CLI `python -m lm_eval run --tasks ... --output_path ... --model_args ...`, écrit un
   `results_*.json`/`samples_*.jsonl` de la forme réelle), parc isolé (`ACVRAM_PARC_CONFIG`,
   jamais `~/TSV` ni le config.toml réel). **Exerce toute la plomberie** : lancement/arrêt serveur,
   deux appels lm-eval, agrégation, bac à sable, S composite (vérifié = 0,3333 pour
   mmlu=0,6/gsm8k=0,4/humaneval=0), étoile, ligne TSV, `ecrire_note`. **Ne valide PAS** : le
   comportement réel du paquet `lm_eval` (absent de ce poste, aucun `.venv-panel`), ni
   `acvram-serveur` réel, ni une vraie mesure de qualité — ceci reste À FAIRE à la première
   campagne réelle (poste2/poste1).
4. **Un bogue trouvé et corrigé en écrivant le test** : `printf %.4f` plante sous locale
   `fr_FR.UTF-8` (virgule décimale attendue, `printf: 0.3333...: nombre non valable`) — remplacé
   par un formatage Python (`LC_NUMERIC=C`), jamais `printf %f` dans ce script désormais.
5. **Un risque de sécurité évité sur un test existant** : `test_executer_resout_acvram_lmeval_py_en_priorite`
   (poste3) ne testait que l'impression de l'interprète résolu — avant cette pièce, `--executer`
   s'arrêtait juste après, donc inoffensif ; une fois le corps réel écrit, ce même test aurait
   **lancé pour de vrai `acvram-serveur`** (carte GPU réelle, hors tout verrou `carte.sh`) rien
   que pour vérifier une ligne de texte. Corrigé : `ACVRAM_E50_LANCEUR=/bin/false` ajouté à ce
   test (échoue immédiatement, après que la ligne attendue est déjà imprimée) — signalé ici car
   c'est le genre d'erreur que REGLES cherche à prévenir, trouvée avant qu'elle ne morde.
6. **Trois variables d'environnement de test ajoutées** au script (jamais utilisées hors tests,
   toutes avec un défaut qui reproduit le comportement réel) : `ACVRAM_E50_LANCEUR` (commande de
   lancement du serveur, sinon `acvram-serveur` réel), `ACVRAM_E50_TSV` et `ACVRAM_E50_SCRATCH`
   (chemins de sortie, sinon `outils/qualite-e50.tsv` et `scratchpad/qualite-e50-<alias>/` réels)
   — sans elles, un test aurait écrit dans le dépôt réel suivi par git ou lancé une vraie carte.

## Estimation de temps (affinée, pour poste2, campagne e50.2)

Parc actuel (01/10, intersection menu servi ∩ « non mesuré ») : **111 alias**, pas 108
(poste6 avait écrit son estimation le 01/10 matin sur un menu à 266 alias ; il en sert 335
maintenant — chiffre à jour utilisé, écart signalé, pas lissé). Classement par taille déduite du
nom (dense_30b ≥ 20B non-MoE, moe = A3B/« Flash », petit_dense < 20B, grand_dense_exile ≥ 60B) :

| catégorie | n | min/alias (méthode §2) | total |
|---|---|---|---|
| dense_30b | 32 | 10 | 320 min |
| moe | 26 (+16 `glm-4.7-flash-*` reclassés) = 42 | 3,5 | 147 min |
| petit_dense | 31 (+5 non classées : phi-4, ernie-thinking, nemotron-lightning×2, jan-v2-vl) = 36 | 3 | 108 min |
| grand_dense_exile | 1 | 60 | 60 min |
| témoin × 12 lots | — | 3 | 36 min |
| **total** | **111** | — | **≈ 635 + 36 = 671 min ≈ 11,2 h** |

Recoupe l'estimation grossière d'poste6 (« ≈ 11 h »), maintenant posée sur le compte réel des
111 alias plutôt qu'une moyenne globale à 6 min. **Ce temps sera pris par poste2 dans la campagne
e50.2** (ordre chef) — `outils/campagne-qualite-e50.py --simule` donne le détail lot par lot,
rejouable à tout moment si le parc bouge encore d'ici la mesure réelle.

## Reste

* **McNemar item par item (méthode § 4)** non branché dans `campagne-qualite-e50.py` — garde de
  cohérence simple seulement (écart de S > 0,05). À faire avant toute étoile publiée : brancher
  `panel-taches-resume.py` ou équivalent sur les échantillons appariés du témoin, par qui mesure.
* `--executer` et `campagne-qualite-e50.py --executer` **jamais exercés contre un `acvram-serveur`
  et un `lm_eval` réels** — première validation à faire par poste2/poste1 au premier alias réel,
  avant de lancer les 111.
* Le classement moe/dense par regex sur le nom est une heuristique (ex. `llamacpp-mistral-small-4-119b`,
  `acvram-ornith-1-0-35b-kimi-nvfp4` : noms irréguliers, `\d+b` extrait parfois le mauvais nombre) —
  à vérifier visuellement par qui lance la campagne si un alias semble mal classé avant de faire
  confiance à l'estimation de durée d'un lot précis.

**RESTE** : rien en cours à moi. Prochaine reprise : ordre de chef, ou campagne réelle par
poste2/poste1.
