# ÉTAT — seule lecture d'entrée (≤ 40 lignes, tenu par chef ; REGLES et INDEX à la demande, par section)

Mis à jour : 16/09 (initial, poste7). Objectif : duel GLM-4.7-Flash NVFP4 b=12 contre vLLM (`poste7-objectif-14-09`, seuil 0,7-1,3× notre débit).

## Chantiers ouverts et scellés

| chantier | qui | scellé | réfuté → | note |
|---|---|---|---|---|
| correctif `nvfp4_quant_act` : k_x = 4, k_act = 8, division s fusionnée, compteurs | poste4 | noyau = référence Python sur x/s et act/s_d ; err pile ≤ 1,1 × err(iv) down ; re-PPL alpha-commun ≤ 1,010 ; B/B∅ ≤ 1,008× ; compteurs 0/0 | > 1,015 → sonde par étape | `poste7-glm-pile-correctif` § 1, 4, 6 |
| contrôle int8 AWQ (à sec) | poste1 | FAIT : 0,982 / 0,983 ≥ 0,9 → retrait des tables int8 | — | `verdict-glm-awq-int8` |
| `use_awq = opts.awq and fmt == "nvfp4"` (à sec) | poste2 | ACTIVÉ (poste1 ≥ 0,9) | — | § 5 |
| reconversion GLM indépendante (après commit poste4) | poste2 | PPL NOMINAL MMA=1 ≤ 1,005 | > 1,010 → métrique W4A4 cb2784b | § 1.6 |
| pas b=12 reconverti, à code égal | poste4 | ≤ 1,012 × `-sansawq` | > 1,03× → diff manifestes | § 5.4 |
| re-tampon 0.6.6 narrow OFF | poste3 | ≈ 14,0 ms / 0,51 J, ABAB | — | `poste7-narrow-verdict` § 1 |
| cellule narrow 3 × 3 (0.6.7) | poste3 | moyenne B/A 1,000 ± 0,002 et par tranche ≤ 1,5 × écart graphes/eager | > 1,002 ou tranche > 1,004 → poste4, M capturé/rejoué | § 3 |
| duel prise A | poste3 | vLLM NVFP4 0,7-1,3× notre débit b=12 | — | `poste7-objectif-14-09` |

## File de carte (un bloc, une fusion de main par phase)

1. poste3 : re-tampon OFF (10 min) → cellule 3 × 3 (30 min) — maintenant, carte libre.
2. poste3 : re-PPL alpha-commun + compteurs (20 min) — après le commit de poste4.
3. poste2 : reconversion à sec (1 h, chevauche 2) → poste3 : PPL reconverti (20 min) ; poste2 en veille si ≤ 1,005.
4. poste3 : pas b=12 à code égal (10 min).
5. poste3 : duel prise A (1 h).

## Dernier commit par branche

main (voir git) · poste4 (correctif en cours) · poste2 cb2784b (hors main) · poste1 dbb84c8 · poste3 — · poste8 1ec7225 (suspendue)

## Suspendu / veille

poste1 en veille (rappelée sur scellé réfuté). Un message par bloc, redémarrage à chaque phase, poste7 sur scellé réfuté ou duel seulement.

poste8 jusqu'au duel publié. duck.ai : à l'impasse déclarée dans une note de poste7 seulement.
