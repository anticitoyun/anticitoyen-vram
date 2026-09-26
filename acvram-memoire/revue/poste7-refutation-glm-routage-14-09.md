# poste7 — équivalence GLM-4.7-Flash rouge (8/16 positions) : cause la plus probable, lue dans le code (14/09)

Réfutation rapportée par chef : |Δlogit| 5,07 (seuil 0,05), cos 0,952 (seuil 0,999), **8 positions sur 16**, tenseurs complets, 6-pré fusionné avant (`16bac2f`), donc la détection MLA n'est plus en cause.

## Cause la plus probable : scoring softmax sans biais au lieu de sigmoid + biais

| côté | fichier:ligne | ce qui se passe |
|---|---|---|
| HF `glm4_moe_lite` | `modeling_glm4_moe_lite.py:402-423` | `scores = logits.sigmoid()` **inconditionnel** ; `scores_for_choice = scores + e_score_correction_bias` ; top-k sur `scores_for_choice` ; poids = `scores` (sans biais) ; `norm_topk_prob` ; × `routed_scaling_factor` 1,8. **Sa config n'a pas de clé `scoring_func`** (`configuration_glm4_moe_lite.py` : seulement `routed_scaling_factor`, `norm_topk_prob`) |
| nous | `acvram/engine/config.py:587` | `router_scoring = cfg.get("router_scoring") or cfg.get("scoring_func") or "softmax"` → **softmax** pour GLM-4.7-Flash, puisque la clé est absente |
| nous | `acvram/engine/model.py:1071-1073` | branche softmax : top-k sur `softmax(logits)`, **sans le biais** — le biais est lu (`loader.py:451-459`) mais n'entre que dans la branche sigmoid (`:1067-1070`) |

**Pourquoi 8/16 et pas 16/16** : le biais ne change le top-4 que sur les jetons où il inverse un rang — environ la moitié des positions ; sur les autres, seuls les poids diffèrent (softmax renormalisé contre sigmoid renormalisé), assez pour tomber sous 0,999 mais le test les a peut-être classées vertes selon sa tolérance par position. Un défaut de MLA ou de couche dense toucherait les 16.

## Contrôle qui peut rendre « faux » (poste1, 10 min)

À la couche 1, publier par jeton `topi` acvram et `topk_indices` HF (même invite). **Si les ensembles diffèrent exactement sur les 8 positions rouges → c'est le routage, cause confirmée.** Si les ensembles sont égaux partout → ce n'est pas la sélection : mesurer alors les poids (`topw` vs `topk_weights`) puis la sortie de l'expert partagé ; si tout cela est égal, remonter à l'attention (v_head_dim 256).

## Remède (poste1, même commit que le test)

`router_scoring = "sigmoid"` pour `model_type == "glm4_moe_lite"` à la lecture (`config.py`, avec les autres réglages de famille) — **le nom est ici la source**, puisque HF câble le sigmoid dans la classe et non dans la config. Ne pas déduire « sigmoid » de la présence du biais : `ernie4_5_moe` a un biais **et** un softmax (`config.py:527-534`). Test : config `glm4_moe_lite` minimal → `router_scoring == "sigmoid"` ; casse si le défaut revient. Puis rejouer l'équivalence de poste2 (sur CPU, forcé) : scellé inchangé, ≤ 0,05 et ≥ 0,999 sur 16/16.

Le journal « régime DÉGRADÉ, formats mélangés entre experts » sur bf16 pur (`runner.py:476`) est une seconde question, à ouvrir seulement si l'équivalence reste rouge après le routage.
