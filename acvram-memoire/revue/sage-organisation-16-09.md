# Sage — organisation pour finir : messages-pointeurs, un bloc d'ordre par note, une file de carte en un bloc jusqu'au duel, Katy suspendue (16/09, soumis à l'utilisateur)

## 1. Ce que la journée a coûté, et pourquoi

| poste | constat du 16/09 | coût |
|---|---|---|
| relais réécrits | chaque verdict m'arrive réécrit par Jérôme, ma réponse repart réécrite vers le membre : 14 messages, ~1 500 signes chacun, trois fois | ~60 k signes pour ~5 k d'information ; deux glissements de sens (« bogue » lu comme « application », « ≤ 1,010 » pris pour le mauvais scellé) |
| accusés sans information | « Reçu », « noté », « j'attends » | un tour de session chacun, zéro fait |
| scellés mal posés | narrow ± 0,002 sous le bruit du témoin ; pas b=12 sur deux convertis | deux passes de carte (30 min) + deux cycles de note |
| hypothèses de Sage à sec | deux réfutées avant la bonne (trois passes d'Océane) | ~1 h d'Océane ; la cause : je n'avais pas énuméré les trois quantifications du chemin |
| lecture d'entrée | chaque session relit REGLES (430 lignes) + INDEX (237 notes) à chaque reprise | le poste le plus lourd et le moins visible |
| sessions inactives | Laure sans tâche de carte aujourd'hui alors que la carte est libre pendant que Laurine code ; Katy : inventaires clos (1ec7225) | carte perdue ; contexte Katy payé pour rien |

## 2. Organisation proposée — rien ne change dans qui parle à qui (revert c723531 respecté)

1. **Messages-pointeurs, jamais de réécriture.** Membre → Jérôme : `verdict: revue/<fichier> — <1 ligne>`. Jérôme → Sage : le même pointeur, tel quel. Sage → Jérôme : `note: revue/<fichier> — ordre: …`. Jérôme → membres : **copie du bloc « Ordre » de la note**, sans reformulation. Je lis les notes (exactes, déjà écrites) ; le membre lit la mienne. Interdits : accusé de réception, « j'attends », résumé d'une note déjà commitée.
2. **Chaque note de Sage finit par un bloc `## Ordre`** : une ligne par membre — quoi, scellé, réfuté → quoi. C'est la seule partie que Jérôme distribue. Chaque verdict de membre **commence par 6 lignes fixes** : instrument · commit · régime · scellé · mesuré · verdict — je ne lis souvent que cela.
3. **`revue/ETAT.md` (≤ 40 lignes, tenu par Jérôme) remplace INDEX + REGLES en lecture d'entrée** : chantiers ouverts, scellés en cours, file de carte, dernier commit par branche. INDEX et REGLES restent, on ne les relit qu'à la demande (une section nommée). REGLES §1 prend les points 1-2 (six lignes).
4. **Une file de carte en un bloc jusqu'au duel, scellés déjà écrits (`sage-glm-pile-correctif` § 1-6, `sage-narrow-verdict` § 1-3)** — Sage n'intervient qu'à un scellé réfuté, plus de cycle de note entre deux passes :
   * maintenant, carte libre pendant que Laurine code : **Laure** — re-tampon OFF 0.6.6 (10 min) puis cellule narrow 3 × 3 (30 min) ;
   * à sec en parallèle : **Océane** contrôle int8 (10 min) ; **Manon** écrit `use_awq = opts.awq and fmt == "nvfp4"` (`convert.py:1151`) prêt à activer ;
   * puis, dans cet ordre, sans fusion de main entre deux passes : **Laurine** re-PPL alpha-commun + compteurs (20 min) → **Manon** reconversion à sec (1 h, chevauche) → PPL reconverti (20 min) → pas b=12 à code égal (10 min) → **Laure** duel prise A (1 h). Une fusion de main **par phase**, jamais pendant une campagne (règle du 15/09).
5. **Katy suspendue jusqu'au duel publié** : ses livrables sont clos ; l'INDEX est tenu par Jérôme dans ETAT.md. Elle revient pour l'inventaire de fin.
6. **duck.ai : à l'impasse déclarée dans une note de Sage, pas à chaque étape** — la règle du 10/09 le dit « systématiquement » : c'est la seule ligne ici qui touche une règle de l'utilisateur, je la propose, je ne l'applique pas seule.
7. **Modèles et efforts : inchangés** (cache de préfixe, règle du 12/09) ; à revoir au prochain redémarrage seulement.
8. **Pour moi** : avant toute hypothèse à sec, énumérer toutes les quantifications du chemin (REGLES §4 bis, ligne à ajouter) ; un scellé compare deux codes sur un converti, ≥ 2 × l'écart du témoin (§3, déjà ajouté).

## 3. Gain attendu, coût, réfutation

* **Jetons** : relais ÷ 10 (pointeurs), lecture d'entrée ÷ 6 (ETAT.md 40 lignes contre ~670), Katy 0 ; attendu ≥ −50 % du volume inter-sessions par jour. **Réfutable** : Jérôme compte les messages et leurs signes sur la journée du 17/09 contre le 16/09 (14 messages, ~60 k signes chez moi seule).
* **Délai** : le duel en **une après-midi de carte** après le commit de Laurine (aujourd'hui : cinq cycles de note entre deux passes) ; **réfuté** si un scellé casse — alors une note, et c'est le prix normal.
* **Coût** : 30 min de Jérôme pour ETAT.md et REGLES §1 ; rien d'autre.

## Ordre (après autorisation de l'utilisateur)

* **Jérôme** : REGLES §1 (points 1-2, six lignes), `revue/ETAT.md` initial, file de carte du point 4 distribuée telle quelle, Katy prévenue de la suspension.
* **Laure** : re-tampon OFF 0.6.6 puis cellule 3 × 3, maintenant.
* **Océane** : contrôle int8, à sec, maintenant.
* **Manon** : `use_awq` conditionnel écrit à sec, activé selon Océane ; reconversion après le commit de Laurine.
* **Laurine** : correctif k_x = 4 / k_act = 8 + division fusionnée + compteurs, scellés de `sage-glm-pile-correctif` § 4-6.
