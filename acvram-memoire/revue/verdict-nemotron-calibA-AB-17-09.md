# Verdict — Nemotron-3.5-Lightning-30B-A3B, convertis calibA corrigés (Manon 315decc) : A `-etendue4096-repli8` **1,0950**, B `-sansdown` **1,0273** → aucun ≤ 1,020, chantier qualité Nemotron FERMÉ pour de bon, menus = precision-officielle 1,0304

instrument : `scratchpad/nemotron-calibA-AB-17-09/chaine.sh` (= montage srcbf16 / officiel / calibA / repli135 : `ppl-acvram-17-09.py`, 3 tranches `corpus-prive/tranches-glm/tranche{0,1,2}`, × bf16 15,367 / 13,315 / 11,801, géo) ; les deux convertis mesurés dans la même fenêtre (03:33-03:35)
commit : 20fa8ed (arbre gelé, régime classé, graphes on, `GDN=fla`, `DENSE_NVFP4=triton`)
régime : NOMINAL, PPL seule
scellé (Sage) : seuil unique ≤ 1,020 classé, le plus bas gagne ; prédictions A 1,015-1,030, B 1,020-1,030 ; aucun ⇒ fermé, menus = officiel 1,0304
mesuré : A 16,616 / 14,860 / 12,841 → **1,0813 / 1,1160 / 1,0881**, géo **1,0950** (0 fenêtre explosée, tr0 8,3-26,2) · B 15,797 / 13,756 / 12,046 → **1,0280 / 1,0331 / 1,0208**, géo **1,0273** (0 explosée, tr0 7,9-24,8)
verdict : **FAUX pour le classement** — A 1,0950 > 1,020 (prédiction 1,015-1,030 réfutée de 0,065), B 1,0273 > 1,020 (prédiction 1,020-1,030 tenue, classement non atteint) ; chantier FERMÉ pour de bon ; menus = precision-officielle 1,0304

## Lecture
- B (down_proj identité, SNR 20,8 dB) est la meilleure conversion acvram de Nemotron : 1,0273, sous l'officiel 1,0304 de 0,003 — un gain réel mais sans effet sur le classement, et obtenu en retirant l'AWQ du down_proj, pas en le corrigeant.
- A (AWQ partout, 8/5935 repliés, SNR 23,1 dB) est pire que B de 0,068 malgré un SNR supérieur de 2,3 dB : l'AWQ appliqué au down_proj coûte ~0,07 de PPL — même sens que calibA (1,43) et repli135 (1,26), à échelle réduite. Le SNR sur les poids ne prédit pas la PPL ici (voir verdict calibA) ; le down_proj est le tenseur où la calibration nuit.
- Classement final des 6 conversions acvram : sansdown 1,0273 < officiel 1,0304 < srcbf16 1,0632 < etendue4096-repli8 1,0950 < repli135 1,2634 (invalide) < calibA 1,4301 ; vLLM 0,987 reste hors d'atteinte : l'écart de 0,04 n'est pas dans la calibration.
