# poste7 — si l'instrument NVML est en cause, qu'est-ce qui change ? (14/09, écrit AVANT le run 20 s de poste3)

Sources : `outils/gpu/mesure/energie.py:93-97,177-202,241-262` ; `outils/puissance_nvml.py:12-76` (branche poste3) ; `verification-energie-py-14-09.md` ; `audit-a2-duel-vllm-14-09.md:63-66,124-135` ; `energie-brute-faible-lot-14-09.md:14-52` (2fa9fbe) ; `scratchpad/resultat-energie-brute-vllm-14-09.json` ; `nvidia-smi --query-gpu=enforced.power.limit` relevé à l'instant : **400 W** (5090), 275 W (3080 Ti).

## Il n'y a pas un instrument, il y en a deux — et ils ne mesurent pas la même chose

| | A — `energie.py` | B — `puissance_nvml.py` |
|---|---|---|
| source | compteur matériel `nvmlDeviceGetTotalEnergyConsumption` (:93), J = Δcompteur, W = J / durée | `nvidia-smi --query-gpu=power.draw.instant` **en sous-processus** (:12-17), un relevé toutes les 0,1-0,2 s |
| agrégat | intégrale exacte | **médiane** des relevés (:76) ; repos = médiane sur 3 s (:20-29) |
| a servi à | décodage b=12 (audit-a2:124-135), b=1-4 (poste3), horloge SM 13/09 | prefill pp2048 (audit-a2:63-66 : « 17 W / 78 W nets » acvram, « 64 W / 99 W » vLLM) |

**Une médiane de puissance n'est pas une énergie.** Sur une charge par rafales (boucle `generate(max_tokens=1)`, trous hôte), la médiane rend le mode dominant — repos ou pleine charge selon le rapport cyclique — jamais la moyenne. Un écart de +51 % entre A et B sur 2 s est ce qu'on attend de B, pas un défaut de NVML. Et le repos « 17 W » d'acvram vient de B (médiane instantanée, carte froide) : c'est ma prémisse fausse de 2a.

## Prédictions scellées pour le run 20 s

1. **Compteur A vs moyenne des relevés instantanés sur 20 s : accord à ±3 %.** Si l'écart reste > 10 % avec la **moyenne**, NVML est en cause et tout ce qui suit change (§ dernier). Si poste3 compare le compteur à la **médiane**, l'écart ne se referme pas à 20 s tant que la charge est par rafales — et cela ne dit rien de l'instrument.
2. **Le compteur n'est pas quantifié grossièrement** : σ = 0,23 % sur trois fenêtres de 0,9 s (energie-brute:32) — un quantum de 40 J sur 280 J donnerait ±14 %. Réfuté d'avance.
3. **Le vrai défaut des fenêtres courtes est le limiteur, pas le compteur.** Puissance moyenne d'acvram déduite du compteur : 326 / **401 / 426 / 435 W** à b=1-4 sur 0,9-1,5 s, plafond appliqué 400 W. Un compteur juste ne peut dépasser un plafond tenu : donc le plafond n'était **pas encore engagé** — sa réaction est plus lente que la fenêtre. Prédiction : sur 20 s, acvram à b≥2 rend **≤ 400 W** (≈ 385-395), débit b=4 −3 à −6 % (horloge rabattue), J/jeton b=4 dans ±10 % du 0,804 actuel. **Réfutation** : 20 s à > 405 W de moyenne → le limiteur ne tient pas et c'est `enforced.power.limit` qui ment.

## Ce que ça requalifie, résultat du run connu ou non

| chiffre | instrument | fenêtre | statut |
|---|---|---|---|
| duel b=12 : 0,202 vs 0,601 J/jeton net, ×2,97 (audit-a2:130-135) | A | 2,0 s (vLLM) / 3,3 s (acvram) | **ratio tient** (même instrument, même carte, même jour : un biais commun se simplifie) ; régime **transitoire** pour acvram (326-435 W hors plafond), plafond atteint chez vLLM. À refaire à ≥ 20 s (≥ 24 000 jetons par bras, ~1 h poste3) avant d'être le chiffre officiel de la semaine. |
| 2a : b=1 −3,8 %, b=2 −3,1 % brut | A | 0,9-2,0 s | b=1 déjà retiré par poste3 (σ vLLM 2,43 % > marge 1,25 %) ; b=2 tient à la répétition mais **régime transitoire** : à refaire à 20 s, le crossover avec. |
| prefill pp2048 : 78 W vs 99 W nets, repos 17 / 64 W | B (médiane) | ~2 s | **non comparable**, à retirer de toute table ; aucune décision ne repose dessus. Le prefill se juge en j/s (×1,82-2,03, instrument = chronomètre), pas en watts médians. |
| horloge SM 13/09 : J/jeton −15 à −23 % de 2 400 → 1 800 MHz | A | 3-4 s | sens tient ; ampleur ±10 pts tant que non refait à 20 s ; décision utilisateur « non retenu » inchangée. |

## Deux gardes à poser, quel que soit le résultat

* `energie.py:241-262`, `invalidations` : ajouter **`moyenne > plafond` → « fenêtre plus courte que la réaction du limiteur : régime transitoire »** et **`duree < 10 s` → avertissement**. C'est le contrôle qui aurait rendu « faux » sur 401/426/435 W sans qu'on ait à déplier le tableau. Test : une fenêtre de 1 s à pleine charge doit être invalidée, une de 20 s acceptée.
* `puissance_nvml.py:76` : rendre la **moyenne** (et garder la médiane à côté pour voir le rapport cyclique) ; ou retirer B des duels et n'y laisser que A. La médiane reste utile pour le repos.
* Nom du banc porte la fenêtre : `…-20s`. Règle 4 (« chiffre exact hors de son régime »), appliquée à la durée.

## Ordre : inchangé, avec un préalable

1. Énergie instr/octet + J par noyau (poste4 ncu, poste3 NVML) — **J par noyau avec le compteur A, boucle ≥ 20 s par noyau** (mon « 10 s » de la stratégie passe à 20). Grille de lecture inchangée : instr/octet ≥ 3× vLLM → les noyaux ; ≤ 1,5× → ailleurs ; par noyau ~2× et indépendant de b → cohérent avec le rapport de puissance nette 1,25× (b=1) / ~2× (b≥2, plafond touché).
2. 1aj W4A4 projections + `lm_head`. 3. Duel officiel b=12 et 2a à 20 s (poste3, 1 h). 4. 2b MLA, 2c llama.cpp. 5. 0si TMA.

**Si la prédiction 1 tombe** (compteur ≠ moyenne instantanée de > 10 % sur 20 s) : tous les J/jeton absolus sont requalifiés ; les **rapports** du même jour survivent ; J par noyau se fait à ∫instant (25 ms, `nvmlDeviceGetPowerUsage` n'est pas l'instant — il faut le champ `power.draw.instant`, jamais appelé dans le dépôt) ; rien ne change à l'ordre, seul l'instrument de l'étape 1 change.
