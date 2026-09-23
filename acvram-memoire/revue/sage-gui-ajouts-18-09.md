# Sage — GUI : quatre ajouts, dans l'ordre de ce qui a coûté des manches ; deux des pistes de Jérôme existent déjà (18/09)

Entrée : Jérôme — l'utilisateur demande ce qui serait judicieux d'ajouter à `packaging/acvram-gui` (WebKitGTK sur la console `acvram/server/console.py`). Pistes de Jérôme : historique des modèles servis, état du verrou, copie de la commande `serve`, régime dans la barre d'état. Décision produit : à l'utilisateur ; ceci est un avis chiffré.

## 1. Déjà là, à ne pas refaire

* Copie de la commande `serve` **avec verrou et carte** : `console.py:624-640` (`p_copier`), section « choisir un modèle par carte ».
* Régime : `/metrics` rend `engine.regime()` (`app.py:777`) ; la section « moteur » affiche `/metrics` brut (`console.py:386`). Il manque seulement les versions torch/triton/fla, commandées dans `sage-glm-etendue-canal-saillant-18-09` § 5 — une fois dans `regime_ligne()`, elles apparaissent sans code GUI.

## 2. À ajouter, par coût de l'incident qu'ils auraient évité

| # | ajout | ce qu'il montre | source | coût | recette (doit pouvoir rendre faux) |
|---|---|---|---|---|---|
| 1 | **Verrou carte** (par carte) | qui tient (`ACVRAM_NOM`), depuis quand, type `mesure/service/CHARGE-DELIBEREE`, et les PID `nvidia-smi --query-compute-apps` avec légitimité : 8081/8082/8083 verts, tout autre PID **rouge « intrus »** | `/tmp/acvram-carte-N.lock`, `.qui`, `.journal` (`carte.sh:125`), nvidia-smi filtré ET renuméroté (REGLES § 6) | 2 h, route `/verrou` | verrou pris par un shell témoin ⇒ affiché < 5 s ; `charge-gpu.py` sans verrou ⇒ « intrus » rouge ; carte libre ⇒ « libre », jamais vide |
| 2 | **Séquences tronquées** | compteur `finish_reason=length` **hors** `max_tokens` demandé, et `kv_max_tokens / (slots × contexte)` | `loader.py:1013` (`_finish(seq, "length")`), plan chargé | 1 h | serveur à `max_concurrent_seqs=8`, 12 requêtes ⇒ compteur > 0 rouge ; 8 requêtes ⇒ 0 |
| 3 | **J/jeton en direct** par moteur | énergie NVML intégrée sur la fenêtre glissante / jetons rendus, avec plafond W et horloge SM à côté (le régime, REGLES § 3) | `/capteurs` + compteurs de `/metrics`, **division côté serveur** (la console n'interprète pas, `console.py:17`) | 2 h | à vide « — », jamais 0 ; sous charge, ± 10 % de `energie.py` sur la même fenêtre |
| 4 | **Régime en barre d'état** | la ligne `regime_ligne()` en une ligne fixe, copiable | déjà dans `/metrics` | 30 min | ligne identique octet pour octet au `[régime]` du JSON d'une mesure |

L'incident n° 1 est le plus cher de l'histoire du circuit (six manches le 10/09, Laure le 14/09, trois fusions le 15/09) et c'est le seul que l'utilisateur voit mieux que les sessions : une annonce a une minute de retard, une fenêtre GUI non. Le n° 2 a faussé toutes les cellules b > 8 jusqu'au 17/09 sans une ligne d'erreur. Le n° 3 est l'objectif du projet, affiché nulle part en direct.

## 3. À ne pas ajouter

* Historique des modèles servis : le manifeste et le journal le portent ; valeur faible, entretien d'un état de plus.
* Tout calcul ou tri dans le JavaScript : la console montre ce que le serveur rend (principe écrit en tête de `console.py`), sinon deux chiffres pour une grandeur.
* `/convertir` : produit déjà en décision utilisateur (`sage-convertisseur-formats-16-09` § 5), après le chantier hybrides — pas dans cette liste.

## Ordre

* Jérôme : porter ce tableau à l'utilisateur tel quel ; il choisit. Ce qui est choisi se fait à sec, chaque ajout avec sa recette comme test (le n° 1 exige un test qui pose un vrai verrou et un vrai PID témoin). Rien sur la carte.
