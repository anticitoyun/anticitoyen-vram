# ÉTAT — seule lecture d'entrée (≤ 40 lignes, tenu par chef ; REGLES et INDEX à la demande, par section)

Mis à jour : 16/09 (initial, poste7). Objectif : duel GLM-4.7-Flash NVFP4 b=12 contre vLLM (`poste7-objectif-14-09`, seuil 0,7-1,3× notre débit).

## Chantiers ouverts et scellés

| chantier | qui | scellé | réfuté → | note |
|---|---|---|---|---|
| correctif `nvfp4_quant_act` v2 : échelle globale PAR LIGNE g_r = amax_r/(6×448), épilogue MMA × g_r[r] × gscales[e], division s fusionnée, plus de k | poste4 | re-PPL alpha-commun ≤ 1,010 ; 0 saturé, flush ≤ 0,01 % ; noyau = référence Python ; durée ≤ 1,2 × k=0 (6 s) | > 1,015 → poste1, sonde par étape § 3.4 | `poste7-glm-pile-correctif` § 7 (k(4,8) réfuté 1,098, `verdict-reppl-alpha-commun`) |
| contrôle int8 AWQ (à sec) | poste1 | FAIT : 0,982 / 0,983 ≥ 0,9 → retrait des tables int8 | — | `verdict-glm-awq-int8` |
| `use_awq = opts.awq and fmt == "nvfp4"` (à sec) | poste2 | ACTIVÉ (poste1 ≥ 0,9) | — | § 5 |
| reconversion GLM indépendante (après commit poste4) | poste2 | PPL NOMINAL MMA=1 ≤ 1,005 | > 1,010 → métrique W4A4 cb2784b | § 1.6 |
| pas b=12 reconverti, à code égal | poste4 | ≤ 1,012 × `-sansawq` | > 1,03× → diff manifestes | § 5.4 |
| re-tampon 0.6.6 narrow OFF | poste3 | FAIT : 13,60 ms / 806 t/s / 0,494 J (ON 11,38 / 0,412) | — | `verdict-retampon-cellule-narrow` |
| cellule narrow 3 × 3 (0.6.7) | poste3 | TENU (poste7 § 6) : moyenne 0,9984, aucune tranche > 1,004 → narrow ON par défaut en 0.6.7 ; 0.6.6 reste OFF ; condition par tranche retirée (mal posée) ; −1,2 % non attribué écrit tel quel | — | `verdict-retampon-cellule-narrow` |
| duel prise A | poste3 | vLLM NVFP4 0,7-1,3× notre débit b=12 | — | `poste7-objectif-14-09` |

## File de carte (un bloc, une fusion de main par phase)

1. FAIT (efc0dd3).
2. poste3 : re-PPL alpha-commun + compteurs (20 min) — après le commit de poste4.
3. poste2 : reconversion à sec (1 h, chevauche 2) → poste3 : PPL reconverti (20 min) ; poste2 en veille si ≤ 1,005.
4. poste3 : pas b=12 à code égal (10 min).
5. poste3 : duel prise A (1 h).

## Dernier commit par branche

main (voir git) · poste4 01c48ef fusionné · poste2-awq-independant 33c0a95 fusionné · poste2 cb2784b (réserve) · poste1 dbb84c8 · poste3 — · poste8 1ec7225 (suspendue)

## Bruit de l'instrument

PPL teacher-forcing à 24 k jetons : étendue 0,008 entre tranches → tout scellé futur ± 0,004 minimum, ou 3 tranches.

## Suspendu / veille

poste1 en veille (rappelée sur scellé réfuté). Un message par bloc, redémarrage à chaque phase, poste7 sur scellé réfuté ou duel seulement.

poste8 jusqu'au duel publié. duck.ai : à l'impasse déclarée dans une note de poste7 seulement.
