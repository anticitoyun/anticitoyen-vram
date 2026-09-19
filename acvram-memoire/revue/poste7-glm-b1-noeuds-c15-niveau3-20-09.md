# poste7 — GLM b=1 : ce n'est pas l'hôte (0,64 ms), ce sont 2,3 ms d'espaces entre les 2 163 nœuds du graphe — ma prédiction C16-GLM réfutée, le levier b=1 est le nombre de nœuds ; C15 niveau 3 scellé ; le test « graphes off » du niveau 2 vise le piège JIT-dans-capture (20/09, 05 h 20)

Source : `verdict-glm-b1-osrt-19-09` (poste2 eeda94f, main 28441096 = C14 + C15-1) ; `poste7-glm-cellules-w4a4-hote-20-09` § 1 (prédiction et clause écrites avant) ; `verdict-c15-19-09` (niveau 2 : 1 552 nœuds, déviant au pas 15) ; REGLES § 6 (un noyau JIT dans une capture corrompt silencieusement ; remède : registre + chauffe avant `warm_graphs`, garde en capture).

## 1. Le pas b=1 GLM, décomposé pour de bon
| poste | ms | part |
|---|---|---|
| noyaux (somme nsys) | 6,1-6,3 | 67 % |
| **espaces entre nœuds du graphe** (2 163 lancements × ≈ 1 µs : le rejeu n'efface pas la latence par nœud) | **≈ 2,3** | **25 %** |
| hors GPU (hôte, `osrt` : poll/nanosleep des fils d'attente, rien sur le fil du pas) | 0,64 | 7 % |
| pas | 9,21 | — |

Ma prédiction (« ≥ 2 ms de Python par pas dans les créneaux hybrides ») est **réfutée** ; la clause écrite (« hors GPU ≤ 1 ms → les 3 ms sont des trous dans le graphe ») s'applique : **C16-GLM fermé** avant d'ouvrir. Pour MECANISMES : *la somme des noyaux d'un graphe n'est pas son temps GPU : chaque nœud coûte ≈ 1 µs de latence de rejeu ; à 2 163 nœuds c'est 25 % d'un pas de 9 ms — un pas court se gagne en nœuds autant qu'en noyaux, et un chantier de fusion se scelle sur les deux.* Le budget de parité b=1 (vLLM 5,45 ms) se réécrit : noyaux 6,2 → 5,2 (C15-2 −1,0) et nœuds 2 163 → ≤ 700 (espaces 2,3 → ≤ 0,7) → **≈ 6,6 ms → 150 t/s** ; la parité exige les deux.

## 2. Deux ordres
* **C15 niveau 2, test qui tranche** (poste2, 10 min, après les concurrents) : `=2` **graphes off** contre `=1` graphes off, mêmes entrées, 64 pas, logits par pas. Prédiction : **au bit ou ≤ 1 ulp bf16** — alors la déviation au pas 15 sous graphes est le piège de REGLES § 6 (un noyau Triton du niveau 2 compilé pendant la capture) et le remède est celui de la règle : registre des noyaux du niveau 2, chauffe explicite de chacun avant `warm_graphs`, garde qui lève sur compilation en capture, test GPU cache Triton vidé ; si `=2` off ≠ `=1` off → l'écart est dans l'aiguillage moteur, poste1 le nomme fichier:ligne. Dans les deux cas : aucune fenêtre de qualité avant le correctif.
* **C15 niveau 3** (poste1, fiche après le niveau 2, code demain) : **nœuds par pas ≤ 700** à b=1 GLM (≈ 15 par couche : les élémentaires restants fusionnés dans les noyaux qui les entourent — rmsnorm + résidu dans le prologue des projections, rope + écriture KV dans le noyau MLA, normalisation du routeur dans `route_fusee`), jetons ou logits selon la forme (au bit si seul l'ordre des lancements change, ± 1 ulp bf16 sinon), `ppl-decode-kv` publiée avec son SE, capture 5/5 ; scellé : **espaces ≤ 0,7 ms ET pas b=1 servi ≤ 7,0 ms** (≥ 140 t/s). Réfutation : ≤ 700 nœuds mais espaces > 1,2 ms → la latence n'est pas par nœud mais par dépendance (chaîne sérielle de petits noyaux) : mesurer la profondeur critique avant d'insister.

## Ordre
* **poste2** — concurrents 2 700 (en cours) ; test `=2`/`=1` graphes off (§ 2) ; C1 noyau ; C9.
* **poste1** — remède § 6 prêt pour le niveau 2 selon le résultat du test ; fiche C15 niveau 3 ; C1 noyau.
* **chef** — ETAT : C16-GLM fermé (0,64 ms d'hôte), 2,3 ms d'espaces entre nœuds, C15 niveau 3 scellé, test graphes off ; MECANISMES (ligne § 1) ; INDEX ; commit + push.
