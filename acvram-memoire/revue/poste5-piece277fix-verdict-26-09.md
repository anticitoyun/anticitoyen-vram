# 277fix — vider le pas simple en vol avant de spéculer : VERDICT (poste5 26/09 17 h 3x)

Correctif `_pipeline_vider()` (pipeline.py) appelé par `step()` (runner.py:1651) quand un pas simple est en vol au moment de
spéculer. Branche poste5-277fix (8e580ebe2, base main 8d5c5580c). Scellés : 277a-bis (seuils de marge) et 277cm
(`poste5-piece277cm-scelle-26-09.md` sur poste5-277, 35f913760, avant la mesure Coder).

| | mesuré | issue |
|---|---|---|
| **Mixte-i8c (hybride)**, `tests/test_spec_pipeline_277.py` (carte, 16:40-16:43) | ngram k = 4 et k = 1 : sortie gloutonne = none au bit (5 × 32), vidages > 0 ; témoin « spéculer sans vider » : sortie différente | **3/3 verts** |
| **Coder -qkvo-i8c (dense MoE)**, même test | égalité ngram ÉCHOUE (k = 4 et k = 1) ; témoin bloqué au délai (instrument) | voir départage |
| Départage Coder (même générateur, 5 × 32, 17:12-17:16) | base SANS correctif : 3/5 invites ≠ none ; AVEC : 2/5 (0 et 3) ; aucun blocage (≤ 55 pas/invite) | correctif réduit |
| Marges Coder aux 2 divergences restantes (17:27, rejeu eager fidèle) | invite 0, j = 20 : marge **0,0152**, jeton spéculatif = top2 ; invite 3, j = 4 : marge **0,0154**, spéculatif = top2 | **quasi-égalités ≤ 0,5 : écart de forme** (prédit, 60 %) |

## Verdict
* **Le bogue (jetons répétés, marges 5-13) est éteint** sur le mixte (au bit) et sur le Coder (plus aucune divergence nette ;
  les deux qui restent sont à 0,015 de logit, ≈ 1/8 d'ulp bf16 autour de 16 : ordre des sommes de la vérification MoE à q_len 5).
* Le test `test_spec_pipeline_277.py` exige l'égalité EXACTE : il est vrai pour le mixte, faux par construction pour le Coder
  (écart de forme). À décider (chef) : restreindre l'égalité exacte aux alias où elle tient, ou juger par la marge (≤ 0,5)
  au premier jeton divergent comme ici.
* **Coût** : ngram corrigé = 24 à 55 pas pour 32 jetons (base 25-32, none 32) — l'alternance amorce du pipeline / vidage
  annule le recouvrement quand le proposeur hésite. Cahier des charges 277e (chef) : pas de retour du ngram au défaut tant
  qu'il fait plus de pas que none. Q26 (poste4) : ni vLLM ni SGLang ne vident ; ils corrigent l'état hôte optimiste après la
  vérification et ne gardent que le préfixe accepté — la voie propre pour la 277e.

## Entrée en main (décision chef, 0.7.5, hors défaut)
Test en deux volets (`tests/test_spec_pipeline_277.py`) : sur 277fix (ed9b165b5) **3/3 verts** (mixte au bit k = 4 et 1 ;
Coder : divergences toutes top1/top2 à marge ≤ 0,5) ; sur la base 8d5c5580c SANS correctif : **mixte ROUGE 2/2**, **Coder
ROUGE** (invite 0, j = 8, marge 12,19). Avertissement de `serve --speculative ngram` et test 283 réécrits (bogue corrigé,
quasi-égalités < 0,02 sur le Coder, jusqu'à 55 pas pour 32 jetons, pas le défaut). Tests à sec ciblés : 27 passed.
