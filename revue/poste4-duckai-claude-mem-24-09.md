# duck.ai 24/09 — claude-mem (thedotmack/claude-mem), avis de chef à contredire ou pas

Question posée telle quelle aux trois modèles de raisonnement (contexte complet : 8 sessions, ~4000
appels/12h, 449$ dont 1,7 G jetons de cache relu ; existant context-mode/ICM/leann/beads/MEMORY.md/
tokensave ; mécanisme claude-mem = résumé par appel claude-haiku-4-5 + service Bun/SQLite/Chroma).
Avis provisoire de chef donné aux trois : NON (appels en plus, doublon, confidentialité).

## GPT-5.6 Luna (raisonnement)
Ne contredit pas franchement. Conditionnel : intégrer seulement si le coût dominant vient de la
réinjection de contexte répétée sur les mêmes dépôts par plusieurs sessions — sinon ça consomme plus
que ça n'économise. Formule le critère de décision comme un ratio : gain = jetons économisés par la
mémoire − jetons dépensés à extraire/injecter ; propose de le mesurer avant/après sur des tâches réelles,
pas seulement en comptage de jetons. Ne traite pas la confidentialité (pas demandé explicitement mais
la question le demandait — absent de la réponse).

## gpt-oss 120B (raisonnement, 1 s de réflexion)
Réponse la plus faible : générique, ignore l'existant nommé dans le contexte (ne mentionne ni context-
mode, ni ICM, ni leann, ni tokensave), aucun chiffre, ne traite pas la confidentialité. Conclut « bonne
stratégie » pour des projets moyens/grands sans jamais comparer à ce qui est déjà en place. À écarter
comme avis fiable sur ce cas précis — n'a pas utilisé le contexte fourni.

## Gemma 4 31B (raisonnement, réponse tronquée en fin de génération)
Va dans le sens de chef sans le nommer : pour un usage interactif (notre cas), demander à l'agent de
tenir lui-même un fichier mémoire (type memory.md/project_state.json, relu en début de session) donne
un résultat équivalent sans couche logicielle externe — c'est essentiellement ce que MEMORY.md + ICM
font déjà. Situe l'intérêt réel de claude-mem dans un pipeline **autonome/orchestré** où on ne veut pas
laisser l'agent décider quand mettre à jour sa mémoire — cas différent du nôtre (sessions interactives
avec crochets). Ne traite pas non plus la confidentialité.

## Contradictions notées
Les trois modèles sont en désaccord entre eux sur l'utilité générale (Luna : conditionnel : gpt-oss :
plutôt oui, sans justification solide ; Gemma4 : plutôt non pour un usage interactif). Aucun des trois
ne chiffre une économie réelle rapportée à notre profil (1,7 G jetons de cache, 449$) — aucun n'a
d'information factuelle sur le taux de compression ou le coût des appels haiku supplémentaires de
claude-mem : à traiter comme absence de donnée, pas comme confirmation. Aucun des trois ne répond à la
question de confidentialité posée (résumés locaux vs tiers) — la question suppose que Chroma/SQLite
tournent en local d'après le dépôt, mais ceci n'est pas vérifié par duck.ai, à vérifier sur le code
source si la question reste ouverte.

## Verdict global (poste4)
Aucun des trois modèles ne contredit l'avis provisoire de chef avec des chiffres qui l'emporteraient.
Le point le plus solide et convergent (Luna + Gemma4, indépendamment) : notre pipeline fait déjà, par
d'autres briques, ce que claude-mem propose (mémoire persistante en fichiers/base + injection ciblée en
début de session) — ajouter claude-mem serait un doublon fonctionnel, sauf besoin d'automatiser
l'extraction sans intervention de l'agent (pas notre cas, sessions interactives à crochets). Avis
provisoire de chef tient, pour raison de doublon fonctionnel — pas d'élément chiffré trouvé qui le
change. Confidentialité : non tranchée par duck.ai, resterait à vérifier dans le code du dépôt si
nécessaire.
