# poste7 — feu vert 0.6.21 (trois bras tenus sur 51f55188, deux arbres refusés avant) ; NARROW_GEMM faux et inerte ; file de poste2 pour la nuit : concurrents sous 2 700 avant toute revendication d'énergie, niveau 2 GLM, C9 (19/09, 21 h 14, heure du commit)

Source : `verdict-verif-eco-defaut-19-09` addendum 21 h 12 (poste2 d8265b8) ; `verdict-narrow-gemm-b12-19-09` ; `verdict-statics-hybrides-19-09` ; `poste7-eco-2700-defaut-19-09` § 2 et addenda ; `poste7-c9-119b-cache-experts-19-09` (M0-M3) ; `verdict-capture-parc-19-09`.

## 1. Tranché
* **0.6.21 : feu vert**, sur 51f55188 seulement — `eco=2700(2692)` carte froide, libre après SIGTERM, `off(2670 : verrou 2700 posé hors processus)`. Les deux arbres précédents auraient livré une lecture fausse (`2700(225 : non pris)`, `2700(1102)`) : la vérification à trois bras a rendu « faux » deux fois pour la bonne raison — c'est ce qu'un contrôle doit faire ; REGLES § 3, une ligne : *un .deb qui touche un réglage de carte se prouve par ses trois bras sur l'arbre livré, jamais par les tests à sec*.
* **NARROW_GEMM=1 à b=12 : faux et inerte** (1,002 ×, 0 `narrow_cuda`) → reste 0, chantier fermé sur ce lot ; le chemin étroit servi est déjà celui de C11 (`gemm_etroit` Triton). Statics hybrides indépendants de b : acquis, rien à faire.

## 2. La revendication d'énergie n'est pas encore comparable : les concurrents à horloge égale
Nos six cellules sont au verrou 2 700 ; celles de llama.cpp et vLLM sont à horloge libre. Le contexte, le harnais et les cartes font partie du régime (REGLES § 4) — **l'horloge aussi**. « Devant llama.cpp et vLLM en énergie à b=12 » ne se publie qu'à horloge égale. Prédiction écrite avant : sous `-lgc 2700`, llama.cpp b=12 **1 040-1 065 t/s · 0,19-0,20 J net** (−8 à −12 % de J, comme nous), vLLM b=12 **1 170-1 200 · 0,18-0,19** ; issue qui me gênerait, et probable : **à horloge égale nous repassons derrière les deux en J à b=12** (0,2005 contre ~0,19), devant en vitesse seulement ; b=1 : llama.cpp 335-340 · ~1,05 (nous 354 · 0,446 : devant reste vrai). Le comparatif porte alors deux lignes par concurrent (libre, 2 700) et la revendication dit exactement à quelle horloge elle est vraie.

## 3. File de poste2 (dans l'ordre, verrou par fenêtre)
1. p2 GLM b=12 et b=1 propres (5 min) dès que l'hôte est sans indexeur (en-tête : trois premiers consommateurs CPU).
2. **Concurrents sous 2 700** (§ 2, 30 min) : llama.cpp b=1 / b=12 / prefill, vLLM b=12 / prefill, harnais égal, `-lgc` sous verrou et rendu après ; verdict avec les deux lignes ; chef réécrit la revendication ensuite, pas avant.
3. Niveau 2 TF32 au décodage GLM (`MLA_CORE_DECODE`, 15 min : `sgemm` b=12 ≤ 0,6 ms, `ppl-decode-kv` corpus préfixé ± 0,001 avec tranches, capture 5/5, pas b=12 13,2 → 12,2 prédit) → défaut si tenu.
4. **C9** : M0 bande PCIe (10 min) → chargement 119B (`ModelSpec` mistral4, table consolidated → HF, exil par couche) → premières cellules b=1 prefill/décodage contre llama.cpp experts en RAM (prédiction `poste7-c9` § arithmétique : 10 j/s sans cache, 18-28 avec) ; M1 trace Coder (1 h) dans les trous.
5. Fenêtres C14 / C15 à leurs commits (15 min chacune, prioritaires sur C9 quand elles arrivent) ; capture godets {1, 2, 8, 16} du parc (30 min) ; vLLM KV fp8 sur `ppl-decode-kv` (10 min) ; C13-c à son commit (2 000 paires).

## Ordre
* **poste2** — la file § 3 telle quelle ; chaque fenêtre avec `eco` et horloge SM moyenne en tête.
* **chef** — dpkg -i 0.6.21 : feu vert donné à l'utilisateur ; revendication d'énergie **suspendue** jusqu'au verdict § 2 (le comparatif dit « à horloge libre chez les concurrents » d'ici là) ; REGLES § 3 (ligne § 1) ; ETAT ; INDEX ; commit + push.
* **poste1** — inchangé : C14, C15, C1 noyau, fiches C17/C5-b/C13-c ; niveau 2 : variable `MLA_CORE_DECODE` nommée si elle ne l'est pas encore (une ligne à poste2).
