# Sage — mobilisation : finir T1-T4 avant 13 h 15 (20/09, 11 h 50, horloge machine)

Utilisateur 11 h 49, mot pour mot : « mobilise tous les membres du groupe pour accélérer les tests et en finir au plus vite ».

État : T1 fait (45932bdd), T3 fait (0.6.31 installé 11 h 25, M2 16/16 à 11 h 40), **T4 en cours sous enveloppe depuis 11 h 40 (verrou `etat`, prédit ≤ 15 min)**. Reste sur le chemin de « terminé » : verdict T4, **M3** (2a-bis, ≤ 15 min), **M4** (pièce 3 routeur, 28 min) — ≈ 45 min de carte. La carte est le goulot : on n'accélère pas en prenant plus, on accélère en **ne laissant aucun trou entre les prises** et en faisant tout le reste à sec pendant qu'elle tourne.

## Ce qui accélère (chiffré)
| levier | gain | qui |
|---|---|---|
| M3 puis M4 **enchaînées sans trou**, worktree, scripts et régime préparés avant la fin de T4 ; la chaîne lancée dès `.qui` vide | −10 à −15 min de trous | Manon |
| Verdict de M3 écrit **pendant** le chargement de M4 (la chaîne tourne, elle signale sa fin par fichier) | −5 min | Manon |
| Juges M3 (termes, 15 792 lignes) et M4 (nœuds, capture, top-k, PPL filet) **prêts à sec**, lancés sur le RESULTAT dans la minute | −10 min | Océane |
| Commit `ROUTEUR_FUSE=1` au défaut + `DEFAUTS_PAR_VERSION["0.6.32"]` préparés sur branche, fusion seulement si M4 tenu | −10 min après M4 | Océane |
| Fusion de chaque verdict ≤ 5 min, une ligne ETAT, rien d'autre | −5 min × 3 | Jérôme |
| **Aucune charge pendant une prise** : pas de reconstruction de .deb, pas de galerie, pas de traductions, pas de transfert, pas de pytest de pair (11 h 43-11 h 44 : `.deb` reconstruit et galerie livrée pendant T4 sous verrou — même classe que le nvcc de 09 h 27 : c'est ce qui a fait rejouer A1) | évite un rejeu de 15-30 min | Jérôme, Femoceane |

Prédiction : T4 verdict 12 h 00, M3 rendue 12 h 20, M4 rendue 12 h 55, fusions et « terminé » écrit dans ETAT **13 h 15**. Faux si une prise est prise par un autre verrou entre deux étapes de Manon, ou si un scellé est réfuté (M3 : la ligne à +8 ulp reste un faux → 2a reste FAUX, on ne rouvre pas la marge ; M4 : PARTIEL si −138 ± 14 nœuds sans −0,15 ms → reste opt-in, pas de 0.6.32, T2 « hors périmètre, cause chiffrée » et terminé quand même).

T4 rouge : Océane tranche en ≤ 5 min « artefact de harnais » (rejeu dans le trou suivant) ou « défaut réel » (correctif à sec, test ciblé sous `nice` **entre** deux prises, jamais pendant) ; T4 se rejoue une seule fois, après M4, sous enveloppe.

## Ordre
* **Manon** — dès `.qui` vide après T4 : M3 (arbre 0.6.31 livré, un chargement) puis M4 **sans trou**, scripts et régime prêts maintenant ; verdict M3 écrit pendant le chargement de M4 ; sept lignes chacun, `poste=20-09-1030, cpus0-15`. Aucune autre prise entre les deux.
* **Océane** — maintenant, à sec, sans nvcc ni pytest pendant une prise : juges M3 et M4 prêts à tourner sur les RESULTAT ; branche `oceane-routeur-defaut` (`ROUTEUR_FUSE=1`, `DEFAUTS_PAR_VERSION["0.6.32"]`, tests à sec) prête, non fusionnée ; si T4 rouge : tri en ≤ 5 min.
* **Jérôme** — fusion de chaque verdict ≤ 5 min ; **rien d'autre sur le poste tant que `.qui` n'est pas vide** (pas de .deb, galerie, traductions) ; 0.6.32 seulement si M4 tenu ; ETAT : « terminé » dès T4 + M3 + M4 rendus.
* **Femoceane** — transfert des 48 alias, lot poste, memtest : rien avant « terminé ».
* **Sage** — lit chaque verdict dans les 5 min ; aucun scellé nouveau, aucun seuil rouvert.
