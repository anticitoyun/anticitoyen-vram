# Verdict — AWQ non déterministe : GLM `-k48-calibA-verif17-09` (poste2, mêmes options) contre le `-k48-calibA` d'origine

instrument : montage bras A (`ppl-acvram-17-09.py`, 3 tranches `corpus-encode-brut-glm/prive-tr*`, `PPL_PREFIXE="[gMASK]<sop>"`, `premiers_ids_fenetre_0` [154822, 154824, …] vérifiés, 12 fenêtres × 1 024, 24 576 jetons par tranche) × bf16 `ppl-refonte-17-09/glm-prive-tr*-bf16` ; `scratchpad/glm-verif-17-09/`
commit : a9ef9a3 (arbre gelé, régime classé : MOE_MMA=0, DECODE_MMA=0, NARROW_GEMM=1, GDN=fla, DENSE_NVFP4=triton)
régime : les deux convertis remesurés dans la même fenêtre (23:57-23:59), même chaîne, même carte
scellé : |Δ géo| ≤ 0,002 tenu (seuil unique, poste7)
mesuré : origine 1,0093 / 1,0107 / 1,0250 → géo **1,0150** (identique aux 4 décimales au bras A de 13h40 : montage reproductible) · verif17-09 1,0135 / 1,0071 / 1,0193 → géo **1,0133** · Δ = **−0,0017**
verdict : **TENU** — |Δ| = 0,0017 ≤ 0,002 ; aucune marge de remplacement à élargir ; le verdict Nemotron calibA (1,4301) ne se relit pas

## Lecture
- Le Δ géo tient de peu (85 % du seuil), mais les tranches bougent davantage : +0,0042 (tr0), −0,0036 (tr1), −0,0057 (tr2) — la non-déterminisme de l'AWQ se voit à 5 × 10⁻³ par tranche et se compense partiellement dans la géo. Un seuil de remplacement par tranche serait faux ; le géo sur 3 tranches est le bon grain, et 0,002 reste le seuil.
- Le verif est meilleur que l'origine sur 2 tranches sur 3 : les 10,7 % de reconstruction différente ne portent pas de biais dans un sens.
- Contrôle du montage : l'origine remesurée rend 10,7871 / 13,0833 / 16,5876, les mêmes valeurs qu'à 13h40 (`porte-a4-17-09/glm-off-tr*`) — l'instrument est déterministe, la variation mesurée est celle de la conversion seule.
- Incident sans effet sur le verdict : la carte était tenue par un pair (PID 2311419, mesure, 1 h) ; deux tranches ont dépassé l'attente de carte.sh (rc 3) et ont été rejouées après (`rejoue.sh`), dans la même fenêtre que le verif.
