# poste2-banc — `reasoning` (vLLM ≥ 0.13/0.29) ignoré dans banc-4moteurs.py (poste2, 29/09)

* pièce : P2, sans carte (aucune mesure, correctif + test cassant sur trace synthétique)
* fichier : `outils/gpu/mesure/banc-4moteurs.py`, fonction `generer` (streaming), ligne 599
* commit : branche `poste2-banc`
* ordre : chef, 29/09 — même cause déjà corrigée côté syy/poste5-menus (`test-menus-reels.py`,
  commit `b7d33b814`)

## Cause
`generer()` ne lisait que `delta.get("content") or delta.get("reasoning_content")` sur chaque
morceau du flux SSE. vLLM ≥ 0.13 rend le raisonnement dans le champ `reasoning`
(`protocol.py:71` en 0.29), plus dans `reasoning_content` : un modèle à raisonnement servi par
vLLM 0.29 fait courir des deltas non vides que le banc comptait vides — TTFT et débit faussés
(prise du premier jeton retardée ou jamais atteinte), voire `RuntimeError("aucun jeton reçu")`
si tout le flux n'était que du raisonnement.

## Correctif
Extraction d'une fonction pure `_texte_delta(delta)` (ordre `content`, `reasoning`,
`reasoning_content`, premier champ non vide) appelée aux deux points où le delta était lu.
Aucun accès carte requis pour la tester : trace synthétique du delta JSON, pas de flux HTTP.

## Test cassant
`tests/test_banc_4moteurs_reasoning.py` (4 tests) : `_texte_delta` avec `content` seul,
`reasoning` seul (le champ qui cassait), `reasoning_content` seul (ancien champ), et delta vide.
Vérifié cassant AVANT le correctif (`git stash` sur le fichier de code seul, module sans
`_texte_delta` → `AttributeError` sur les 4 tests) ; vert après (43/43 avec
`test_banc_4moteurs_protocole25.py`, aucune régression sur les fonctions pures existantes).

## Reste
Campagne qualite-275 toujours en attente du signal de chef (edz poste1 + suite 0.7.12).
Rien d'autre en cours de mon côté. Repos.
