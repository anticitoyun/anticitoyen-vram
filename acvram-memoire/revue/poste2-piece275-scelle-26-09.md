# Pièce 275 — scellé AVANT mesure (poste2, 26/09, ordre chef) : garde de régression qualité, 3 modèles

Nouvelle semaine, régime plein. Objectif : une commande unique `outils/qualite.sh <modele>
<bras>` qui rend **TENU** ou **FAUX** contre une référence figée, pour toute pièce qui change
la sortie servie (mérite d'être rejouée avant chaque décision de défaut, comme 237b-c-d
l'auraient été plus vite avec cet outil déjà prêt).

## Modèles (coordonnés avec poste1, pièce 274)

| alias | rôle |
|---|---|
| `Qwen3-Coder-30B-A3B-nvfp4-qkvo-i8c` | MoE, servi |
| `Qwen3.8-27B-unsloth-mixte-i8c` | mixte, servi |
| `Qwen3-4B-srcgguf-nvfp4` | dense, servi (source `Qwen3-4B-Instruct-2507`, contexte 262144, rope 5e6) |

## Panel de tâches (repris de la 261b, aucun changement)

4 tâches, mêmes limites/graine que 237d : `gsm8k` (n=250), `mmlu_flan_cot_fewshot_{high_school_mathematics,professional_law,college_computer_science}`
(n=150 chacune, plafonné à la taille réelle du split `test` si plus petit — vu sur
`college_computer_science`, 100 questions). `outils/panel-taches.sh` (261/261b), inchangé.

## PPL

`acvram eval <modele> --corpus wiki-gptq.txt --window 2048 --stride 2048 --min-context 256
--json` — protocole cadré déjà en usage dans plusieurs pièces PPL du dépôt (REGLES §8 : un
chiffre hors de son régime est faux comme décision — `--min-context 0` donne un nombre NON
comparable, facteur ≈ 2 sur wiki.test.raw d'après l'avertissement du CLI lui-même). Corpus et
fenêtre repris À L'IDENTIQUE de cet usage établi, pas réinventés.

## Décision (chef, avant mesure)

**TENU** si, pour le `<bras>` testé contre la référence figée du même `<modele>` :
1. **McNemar (comme 237d) p > 0,05 sur CHAQUE des 4 tâches** — aucune dégradation détectable
   par tâche ; **ET**
2. **PPL du bras dans la référence ± x %**, avec **x = 1 %** (convention du dépôt, seuil
   `≤ 1,010` déjà en usage pour la PPL ailleurs — repris à l'identique, pas réinventé).

**FAUX** sinon (au moins une tâche McNemar p ≤ 0,05, OU PPL hors ± 1 %).

Pas de critère d'IC de récupération façon 237d ici (chef ne l'a pas redemandé pour cette
pièce) : `qualite.sh` est une garde RAPIDE de non-régression, pas un scellé de changement de
défaut — McNemar + PPL suffisent à ce niveau, un scellé spécifique (comme 237d) resterait
nécessaire pour un changement de défaut réel.

## Références figées

Générées UNE FOIS, sur le `<bras>` par défaut actuel de chaque modèle (0.7.3, `main`),
sauvegardées HORS GIT (`~/.cache/acvram/qualite-275/<alias>/` — échantillons `--log_samples`
de `panel-taches.sh` + résultat `acvram eval --json`), sha256 de chaque fichier consigné dans
la note de verdict de cette pièce, PAS recommis à chaque `qualite.sh` (les références ne
changent pas entre deux appels, seul le bras testé tourne à nouveau).

## Prédiction (avant de générer les références)

Aucune prédiction de score à faire ici (générer une référence n'est pas une comparaison) —
seule contrainte vérifiable : chaque référence doit produire un jsonl `--log_samples` non vide
par tâche et un JSON `acvram eval` avec une PPL finie > 0. Falsificateur d'une référence
invalide : fichier vide, PPL infinie/NaN, ou moins de la moitié des questions attendues par
tâche (signe d'un crash partiel côté serveur, comme le `400 max_model_len` déjà rencontré en
261b — corrigé une fois, à surveiller si ça revient sur un autre alias).

## Prises

≤ 30 min chacune, une prise = (un modèle × [panel OU PPL]), sous `outils/carte.sh`,
`ACVRAM_NOM` posé. 3 modèles × 2 mesures = 6 prises pour les références seules.

Écrit et poussé AVANT toute génération de référence.
