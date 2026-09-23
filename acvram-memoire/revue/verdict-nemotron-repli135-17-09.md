# Verdict — Nemotron-3.5-Lightning-30B-A3B `-nvfp4-calibA-repli135` (Manon, repli accepté tel quel) : PPL **1,2634 × bf16** → **non classé**, chantier qualité Nemotron FERMÉ

instrument : `scratchpad/nemotron-repli135-17-09/chaine.sh` (= montage srcbf16 / officiel / calibA : `ppl-acvram-17-09.py`, 3 tranches `corpus-prive/tranches-glm/tranche{0,1,2}`, × bf16 15,367 / 13,315 / 11,801, géo) ; converti `models_acvram/Nemotron-3.5-Lightning-30B-A3B-nvfp4-calibA-repli135`
commit : 0f05ec5 (arbre gelé, régime classé, graphes on, `GDN=fla`, `DENSE_NVFP4=triton`)
régime : NOMINAL, PPL seule (aucune ronde demandée)
scellé (Sage) : géo ≤ 1,020 classée, sinon « non classé, précision officielle + calibA-repli135 » et chantier FERMÉ ; prédiction 1,018-1,028
mesuré : 18,585 / 17,309 / 15,137 → **1,2094 / 1,2999 / 1,2827**, géo **1,2634** ; tranche 0 : 0 fenêtre explosée, fenêtres 8,8-30,6 (calibA : 10,0-33,8 ; officiel ≈ 1,03 × bf16) — dégradation uniforme, un peu moindre que calibA
verdict : **FAUX — 1,2634 > 1,020, prédiction 1,018-1,028 réfutée de 0,24** ; publication « non classé, précision officielle (1,0304) + calibA-repli135 (1,2634) » ; chantier qualité Nemotron FERMÉ (décision Sage)

## Lecture
- Le repli ramène 1,4301 → 1,2634 : il retire ~40 % de l'écart, pas le défaut. Même signature que calibA (toutes les fenêtres × 1,2-1,3, aucune explosée) : l'échelle calibrée est toujours mal appariée sur toutes les couches, le repli n'en corrige qu'une partie.
- L'officiel non calibré (1,0304) reste la meilleure conversion acvram de ce modèle ; l'écart à vLLM (0,987) n'a pas été récupéré par la calibration.
- Les 5 conversions Nemotron acvram se rangent : officiel 1,0304 < srcbf16 1,0632 < repli135 1,2634 < calibA 1,4301 ; le classement (≤ 1,02) n'a jamais été atteint.

## Addendum (Sage, 18/09) — le converti repli135 était invalide (bogue de plancher côté conversion, pas la calibration) : la mesure 1,2634 reste vraie, la prédiction reste réfutée, mais elle n'engage plus la clôture ; chantier ROUVERT, suite : protocole-nemotron-calibA-AB-17-09.
