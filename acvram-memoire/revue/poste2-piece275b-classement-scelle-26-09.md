# Pièce 275b — scellé AVANT lecture (poste2, 26/09, ordre chef) : classement de 10 échantillons/tâche MMLU, à sec, sans carte

Diagnostic de chef, retenu tel quel : un score SOUS le hasard (25 % à 4 choix) signe presque
toujours un défaut d'extraction/format, pas une qualité de modèle réelle. Objectif : classer
10 sorties par tâche MMLU (high_school_mathematics, professional_law, college_computer_science)
parmi : (a) réponse présente et juste ; (b) réponse présente et fausse ; (c) aucune lettre
extraite ; (d) sortie tronquée par `max_tokens` (ex. `<think>` non fermé) ; (e) autre.

## Prédiction (avant lecture)

**Catégorie dominante prédite : (d) troncature par `max_tokens`.** Raisonnement : ce modèle
mixte est probablement un modèle À RAISONNEMENT (balise `<think>` ou équivalent, motif déjà vu
dans ce dépôt sur d'autres alias) ; `max_gen_toks=1536` pour les tâches MMLU peut suffire sur
des questions courtes (`high_school_mathematics`, 143 caractères en moyenne) mais pas sur des
questions longues (`professional_law`, 768 caractères) si le modèle raisonne longuement avant
de conclure — la sortie s'arrête en plein raisonnement, jamais à « the answer is X ». Seconde
hypothèse, si (d) ne domine pas : **(c) aucune lettre extraite**, si le modèle conclut mais
dans un format que le filtre `get-answer` (motif `answer is X` ou `(X)` isolé) ne reconnaît pas.
**Falsificateur de ma prédiction** : si (a)/(b) dominent (réponses correctement formées,
justes ou fausses) — le score bas serait alors une vraie faiblesse du modèle, pas un défaut de
banc, et la référence 275 resterait valide telle quelle.

Écrit AVANT toute lecture des échantillons.
