# poste7 — PAUSE du groupe dès que possible (21/09, 08 h 35, utilisateur)

Utilisateur, mot pour mot : « passage en pause du groupe des que possible ».

## Ordre — chef (copie à chacun)

1. **poste2** : une prise en cours se termine (≤ 30 min de verrou, jamais interrompue au milieu d'un bras) ou, si aucun bras n'est lancé, carte rendue tout de suite (`carte.sh` -rgc, `.qui` vide, `nvidia-smi --query-compute-apps` vide relevé) ; verdict de ce qui est fini, sept lignes ; ce qui n'est pas fini = « non joué », pas un chiffre.
2. **poste1, poste9** : commit de ce qui est cohérent (tests processeur au vert ou marqués), rien de partiel sur main ; pointeur d'une ligne à chef.
3. **chef** : fusions, `grep` des marqueurs, push GitLab ; tête PAUSE d'ETAT ≤ 12 000 o : SHA de main, état des 8 ouverts (0.6.34 : quel maillon reste — 4 alias / § 1b / GLM b=1 / dpkg), ordre de reprise = la file de `poste7-s2-k48-feu-vert-21-09` § Ordre + addenda, questions à l'utilisateur (bf16 30B 60 Go, C9 119B, clic GUI). Sessions fermées après la tête.
4. Rien d'autre : pas de nouvelle tâche, pas de mesure « pour finir ».

Réfutation : un commit sur main après la tête PAUSE, ou un PID hors verrou à la fermeture = pause non tenue, à écrire dans ETAT.

## Addendum 08 h 4x — deux faits du `verdict-4-alias-chauffe-clampee-21-09` à mettre en tête de la reprise

1. **GLM k48 ne charge plus** : OOM à la capture des graphes CUDA, avant la chauffe. La capture alloue le KV au ctx **demandé** (32768) : l'ordre est faux — plan → clamp → capture → chauffe (qui confirme), jamais capture avant clamp. poste1, à sec ≤ 30 min, test cassant : ctx demandé > tenu → capture au ctx clampé, chargement réussi. Prédiction : k48 charge et tient 31744.
2. **i8c clampé à 4096** (prédit 15360, hier [1]×N tenait 15360) : la chauffe aléatoire ne coûte pas 4× plus de KV ; hypothèse la plus probable = les essais descendants laissent des fragments (pas d'`empty_cache` + `synchronize` entre deux pas), chaque pas voit moins de libre et le clamp s'effondre. Prédiction : avec libération entre pas (ou recherche depuis le bas), i8c ≥ 12288 ; sinon le mécanisme est autre et se nomme avant toute colonne. Colonne : 4096 est une valeur servie (règle 2b), datée et marquée « clamp sévère, à rejouer » — pas un défaut de la colonne, un défaut du .deb : **0.6.34 ne part pas avec i8c à 4096**. Porte inchangée + un critère : clamp à ≤ 10 % de la chauffe homogène d'hier pour les 4, ou l'écart expliqué.
3. Branche `poste1-commande-unique` (cc47482c) : première fusion à la reprise, après ses 5 tests au trou.
