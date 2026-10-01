# e50.3 — McNemar item par item remplace la garde simple (poste4, 01/10, à sec)

Ordre chef, suite au verdict `poste4-e50-3-executer-verdict-01-10.md` qui signalait la garde
simple (écart de S > 0,05) comme un reste non conforme à la méthode § 4.

instrument : lecture des échantillons `samples_*.jsonl` de lm-eval (`--log_samples`), déjà
produits par `qualite-e50.sh --executer` ; aucune mesure réelle
commit : branche `poste4-224`
régime : à sec, aucune carte
scellé : seuil repris tel quel de la méthode § 4/E1 (« réponses identiques ≥ 90 % »), pas
inventé ici — 10 % de discordance maximum
mesuré : rien de réel ; le module est testé sur des cas construits (6 tests)
verdict : **la garde simple est remplacée.** `outils/qualite-e50-mcnemar.py` calcule un McNemar
exact binomial (b/c discordances, p) sur les items COMMUNS entre deux passes du témoin, fusionnés
sur les 5 tâches (préfixe par tâche, aucune collision de `doc_id`). `temoin_tenu(a, b)` exige
**p > 0,05 ET taux de discordance ≤ 10 %** — les deux conditions sont nécessaires : un cas
construit (15 items justes→faux + 15 faux→justes, même S aux deux passes) donne p ≈ 1 (McNemar
seul ne verrait rien, la symétrie masque le signal directionnel) mais 100 % de discordance —
seul le second critère l'attrape. `tests/test_qualite_e50_mcnemar.py::
test_temoin_tenu_rejette_la_derive_symetrique_que_la_garde_simple_laisserait_passer` CASSE si
quelqu'un retire le critère de taux et ne garde que p (6/6 tests verts).
durée : 0 min de carte

## Câblage

`qualite-e50.sh --executer` imprime désormais, en fin de passe, une ligne `SAMPLES <tâche>:
<chemin>` par tâche (les 4 chemins MMLU/GSM8K du dossier `mmlu_gsm8k/`, le chemin HumanEval déjà
calculé). `campagne-qualite-e50.py` les lit, construit `{tâche: {doc_id: correct}}`
(`corrects_mmlu_gsm8k` lit la métrique déjà posée par lm-eval dans chaque échantillon ;
`corrects_humaneval` rejoue chaque complétion sous bac à sable, par item — jamais nu, § 2 bis),
pose la référence du témoin au premier lot (`~/.cache/acvram/qualite-e50-temoin-reference.json`,
**persiste entre deux lancements** de la campagne, pas seulement dans un process), compare
chaque lot suivant par `temoin_tenu`.

## Ce qui change dans le comportement de la campagne

Avant : « témoin FAUX » = S s'écarte de plus de 0,05 de son premier S. Après : « témoin FAUX » =
McNemar p ≤ 0,05 OU discordance > 10 % — une dérive peut être détectée MÊME SI S ne bouge pas (le
cas construit par le test), ce que l'ancienne garde ne pouvait jamais voir par construction (une
moyenne ne distingue pas « rien n'a changé » de « tout a changé dans les deux sens »).

## Reste

* La référence témoin persistée (`~/.cache/acvram/qualite-e50-temoin-reference.json`) n'est posée
  qu'à la PREMIÈRE exécution réelle de `--executer` sur le témoin — tant qu'elle n'existe pas,
  rien à comparer (premier lot = pose la référence, jamais un verdict TENU/FAUX). La première
  validation réelle (`poste4-e50-3-premiere-validation-01-10.md`) la pose.
* Jamais exercé contre un vrai lm-eval (même limite que `--executer` lui-même) — la forme des
  clés `exact_match` dans `samples_*.jsonl` est supposée d'après la convention documentée de
  lm-eval (chaque sample porte sa métrique calculée), pas vérifiée sur une sortie réelle du
  paquet : à confirmer à la première passe réelle.

**RESTE** : rien à moi. Dépend de la première validation réelle pour poser la référence.
