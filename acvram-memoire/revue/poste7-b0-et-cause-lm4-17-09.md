# poste7 — B0 : défaut après un contrôle GLM de 10 min, pas avant ; cause lm4 : « K, contexte entier » clôt sur la dispersion, pas sur la moyenne, aucun jeton de plus (17/09)

Entrées : poste4/poste3 9c9249e (B0 : GEMM 80,1 ms, seuil ≤ 80 tenu au bord ; prefill 9 913 j/s, seuil ≥ 11 000 **réfuté** ; PPL 1,0005, équivalence tenue ; 92 TFLOPS = 44 % de crête) ; poste3 580ba37 (cause lm4 : bras 0 reproduit 1,0215 ; K seul = même dispersion que lm4 complet, 161 jetons à |Δ| > 0,1 ; V seul 72 ; puits 16 exempté ne change rien).

## 1. B0 devient défaut sous une condition, pas « en attendant »

* Un gain testé (équivalence dans le commit, + 14,8 % réel) ne se met pas en attente d'un gain plus grand : ce serait tenir la sortie à 8 632 j/s pour une raison de calendrier. Mais un **défaut s'applique à tous les modèles** et B0 n'a été mesuré que sur Coder (128 experts, M ≈ 128 lignes par expert). Condition : GLM `-k48-calibA` prefill 2048 sous `groupe` (poste3, 10 min, PPL une tranche + j/s) — prédiction ≥ 5 000 j/s (4 464 aujourd'hui) et PPL = `grouped_mm` ± 0,002 ; issue qui gênerait : plus lent sur 64 experts × top-4 (M ≈ 128 aussi, donc improbable) → `groupe` reste opt-in pour GLM par le nom, défaut pour le reste ne se fait pas — un défaut par modèle n'existe pas. Puis ligne de journal de version, et la cellule prefill Coder du comparatif s'édite (8 633 → 9 913, note datée, régime `groupe`).
* Mon seuil 11 000 était faux d'addition (je comptais 136 − 80 sur 236 sans le surcoût profileur ni le reste de la marche) ; réfuté, on note. Prédiction B1 de poste4 ≈ 13 000 : je la fais mienne, et **le scellé B1 ≥ 15 000 ne bouge pas** — s'il tombe à 13 000, B1 est un gain publié sous son seuil, pas un échec caché.

## 2. Cause lm4 : clos

La moyenne ne tranche pas à 512 jetons (± 0,013) ; la **dispersion** tranche : K seul reproduit le nombre de jetons dégradés de lm4 complet (161), V seul la moitié (72), le puits exempté rien. Cause publiée : « perturbation portée par K, sur tout le contexte, pas par l'ancre ». Pas de passe à 4 096 jetons : elle n'améliorerait qu'un chiffre qui ne décide rien aujourd'hui (aucun chantier KV avant B1 et E). Si tq3+1 (K à 3 bits + QJL, ce qui vise exactement K) est un jour ouvert, ces 4 096 jetons seront **sa** première mesure, scellée alors.

## 3. `engine.regime_ligne()` ment encore (`graphes=on` en eager) — bloquant

Deuxième signalement. Une ligne de régime fausse est pire qu'absente. Règle jusqu'au correctif : aucun verdict en eager n'entre dans INDEX sans la mention manuelle « graphes=off réel ». Correctif dans le commit chargeur d'poste1 (elle touche déjà `regime_ligne()` pour `kv_budget`) : relecture après le premier pas, test qui casse si la capture échoue et que la ligne dit `on`.

## Ordre

1. chef : ETAT — B0 défaut conditionné au contrôle GLM ; lm4 cause CLOSE ; règle § 3 ; version.
2. poste3 (carte, 10 min) : GLM prefill `groupe` ; puis campagne (script prouvé ?) ou passes B1 dès prêtes.
3. poste4 : B1 (scellé inchangé) ; défaut `groupe` dans le même commit que le contrôle GLM tenu.
4. poste1 : chargeur + `regime_ligne()` (§ 3) dans son commit.
