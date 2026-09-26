# poste7 — organisation pour finir : messages-pointeurs, un bloc d'ordre par note, une file de carte en un bloc jusqu'au duel, poste8 suspendue (16/09, soumis à l'utilisateur)

## 1. Ce que la journée a coûté, et pourquoi

| poste | constat du 16/09 | coût |
|---|---|---|
| relais réécrits | chaque verdict m'arrive réécrit par chef, ma réponse repart réécrite vers le membre : 14 messages, ~1 500 signes chacun, trois fois | ~60 k signes pour ~5 k d'information ; deux glissements de sens (« bogue » lu comme « application », « ≤ 1,010 » pris pour le mauvais scellé) |
| accusés sans information | « Reçu », « noté », « j'attends » | un tour de session chacun, zéro fait |
| scellés mal posés | narrow ± 0,002 sous le bruit du témoin ; pas b=12 sur deux convertis | deux passes de carte (30 min) + deux cycles de note |
| hypothèses de poste7 à sec | deux réfutées avant la bonne (trois passes d'poste1) | ~1 h d'poste1 ; la cause : je n'avais pas énuméré les trois quantifications du chemin |
| lecture d'entrée | chaque session relit REGLES (430 lignes) + INDEX (237 notes) à chaque reprise | le poste le plus lourd et le moins visible |
| sessions inactives | poste3 sans tâche de carte aujourd'hui alors que la carte est libre pendant que poste4 code ; poste8 : inventaires clos (1ec7225) | carte perdue ; contexte poste8 payé pour rien |

## 2. Organisation proposée — rien ne change dans qui parle à qui (revert c723531 respecté)

1. **Messages-pointeurs, jamais de réécriture.** Membre → chef : `verdict: revue/<fichier> — <1 ligne>`. chef → poste7 : le même pointeur, tel quel. poste7 → chef : `note: revue/<fichier> — ordre: …`. chef → membres : **copie du bloc « Ordre » de la note**, sans reformulation. Je lis les notes (exactes, déjà écrites) ; le membre lit la mienne. Interdits : accusé de réception, « j'attends », résumé d'une note déjà commitée.
2. **Chaque note de poste7 finit par un bloc `## Ordre`** : une ligne par membre — quoi, scellé, réfuté → quoi. C'est la seule partie que chef distribue. Chaque verdict de membre **commence par 6 lignes fixes** : instrument · commit · régime · scellé · mesuré · verdict — je ne lis souvent que cela.
3. **`revue/ETAT.md` (≤ 40 lignes, tenu par chef) remplace INDEX + REGLES en lecture d'entrée** : chantiers ouverts, scellés en cours, file de carte, dernier commit par branche. INDEX et REGLES restent, on ne les relit qu'à la demande (une section nommée). REGLES §1 prend les points 1-2 (six lignes).
4. **Une file de carte en un bloc jusqu'au duel, scellés déjà écrits (`poste7-glm-pile-correctif` § 1-6, `poste7-narrow-verdict` § 1-3)** — poste7 n'intervient qu'à un scellé réfuté, plus de cycle de note entre deux passes :
   * maintenant, carte libre pendant que poste4 code : **poste3** — re-tampon OFF 0.6.6 (10 min) puis cellule narrow 3 × 3 (30 min) ;
   * à sec en parallèle : **poste1** contrôle int8 (10 min) ; **poste2** écrit `use_awq = opts.awq and fmt == "nvfp4"` (`convert.py:1151`) prêt à activer ;
   * puis, dans cet ordre, sans fusion de main entre deux passes : **poste4** re-PPL alpha-commun + compteurs (20 min) → **poste2** reconversion à sec (1 h, chevauche) → PPL reconverti (20 min) → pas b=12 à code égal (10 min) → **poste3** duel prise A (1 h). Une fusion de main **par phase**, jamais pendant une campagne (règle du 15/09).
5. **poste8 suspendue jusqu'au duel publié** : ses livrables sont clos ; l'INDEX est tenu par chef dans ETAT.md. Elle revient pour l'inventaire de fin.
6. **duck.ai : à l'impasse déclarée dans une note de poste7, pas à chaque étape** — la règle du 10/09 le dit « systématiquement » : c'est la seule ligne ici qui touche une règle de l'utilisateur, je la propose, je ne l'applique pas seule.
7. **Modèles et efforts : inchangés** (cache de préfixe, règle du 12/09) ; à revoir au prochain redémarrage seulement.
8. **Pour moi** : avant toute hypothèse à sec, énumérer toutes les quantifications du chemin (REGLES §4 bis, ligne à ajouter) ; un scellé compare deux codes sur un converti, ≥ 2 × l'écart du témoin (§3, déjà ajouté).

## 3. Gain attendu, coût, réfutation

* **Jetons** : relais ÷ 10 (pointeurs), lecture d'entrée ÷ 6 (ETAT.md 40 lignes contre ~670), poste8 0 ; attendu ≥ −50 % du volume inter-sessions par jour. **Réfutable** : chef compte les messages et leurs signes sur la journée du 17/09 contre le 16/09 (14 messages, ~60 k signes chez moi seule).
* **Délai** : le duel en **une après-midi de carte** après le commit de poste4 (aujourd'hui : cinq cycles de note entre deux passes) ; **réfuté** si un scellé casse — alors une note, et c'est le prix normal.
* **Coût** : 30 min de chef pour ETAT.md et REGLES §1 ; rien d'autre.

## Ordre (après autorisation de l'utilisateur)

* **chef** : REGLES §1 (points 1-2, six lignes), `revue/ETAT.md` initial, file de carte du point 4 distribuée telle quelle, poste8 prévenue de la suspension.
* **poste3** : re-tampon OFF 0.6.6 puis cellule 3 × 3, maintenant.
* **poste1** : contrôle int8, à sec, maintenant.
* **poste2** : `use_awq` conditionnel écrit à sec, activé selon poste1 ; reconversion après le commit de poste4.
* **poste4** : correctif k_x = 4 / k_act = 8 + division fusionnée + compteurs, scellés de `poste7-glm-pile-correctif` § 4-6.
