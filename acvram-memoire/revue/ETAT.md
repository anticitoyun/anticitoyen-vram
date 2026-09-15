# ÉTAT — seule lecture d'entrée (≤ 40 lignes, tenu par chef ; REGLES et INDEX à la demande, par section)

Mis à jour : 16/09 (initial, poste7). Objectif : duel GLM-4.7-Flash NVFP4 b=12 contre vLLM RÉFUTÉ 0,7-1,3× (×4,08, `verdict-duel-glm-prise-a-16-09`) — B réduit à (c) : parité MLA puis table à 5 moteurs, décision utilisateur en attente.

## Chantiers ouverts et scellés

| chantier | qui | scellé | réfuté → | note |
|---|---|---|---|---|
| noyau MLA une passe (cu:4050-4180) : cache latent lu 2×/tête → 1 lecture/pas (20 têtes, mma bf16 M=32, softmax en ligne, split-L) — 2e commit : cache latent fp8 par ligne | poste4 | mla_* ≤ 6 ms b=12 ctx 2048 (réfuté > 10 → ncu) ; pas ≤ 30 ms (réfuté > 36 → nsys) ; top-1 identique, cos ≥ 0,9999, PPL 3 tranches ± 0,004 ; attendu ~0,55-0,6× vLLM, pas la parité | > 10/36 → ncu/nsys | `poste7-duel-verdict`, `poste7-avis-exterieur` |
| résidence dynamique des experts (hors VRAM), après MLA | poste1 (mesure à sec au retour) | fréquence d'activation : ≥ 60 % sur 50 % des experts (réfuté < 60 % → sans objet) | < 60 % → écarté | `poste7-avis-exterieur` |
| noyau v2 par ligne (877169c/1b8ddfe, dans main) | poste4 | compteurs 0/0, 6,3 s, nominal : TENUS ; PPL alpha-commun 1,0277 **RÉFUTÉE** (> 1,015) ; -k48 1,0183 **RÉFUTÉE** (> 1,010) — coût = E2M1 sur les 3 sites, pas les blocs à zéro (flush 0 ne rend que 0,003/0,022) | → poste7 ; sonde par étape poste1 ; cb2784b | `verdict-reppl-sage7` |
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
5. FAIT : duel prise A **RÉFUTÉ** (8d5edb5) — b=12 195 t/s / 1,58 J contre vLLM 796 t/s / 0,445 J (×4,08) ; b=4 ×3,4 ; b=1 ×2,0 ; prefill ×6,1. Lecture : MLA (32,8 ms), pas le NVFP4 — mêmes octets d'experts que Coder-30B. → poste7. ; colonne PPL avec le régime, temps de prefill à part (1 h).

Après le duel : poste1 sonde par étape 3 sites (10 min, choix A8/rotation) ; poste2 cb2784b (nominal MMA=1 ≤ 1,010 ; réfuté → Hadamard par bloc) ; poste4 rien (noyau par ligne reste). Fait à publier : k_x=4 saturant 1,019 < 1,0277 sans saturation.

## Dernier commit par branche

main (voir git) · poste4 01c48ef fusionné · poste2-awq-independant 33c0a95 fusionné · poste2 cb2784b (réserve) · poste1 dbb84c8 · poste3 — · poste8 1ec7225 (suspendue)

## Bruit de l'instrument

PPL teacher-forcing à 24 k jetons : étendue 0,008 entre tranches → tout scellé futur ± 0,004 minimum, ou 3 tranches.

## Suspendu / veille

poste1 en veille (rappelée sur scellé réfuté). Un message par bloc, redémarrage à chaque phase, poste7 sur scellé réfuté ou duel seulement.

poste8 jusqu'au duel publié. duck.ai : à l'impasse déclarée dans une note de poste7 seulement.
