# Scellé — point contre NInfer avec les défauts du soir (poste5, 24/09 19 h 5x, AVANT la prise)

Ordre de chef (régime plein). Défauts du soir : Marlin (PROJ_MARLIN=1), 157, fusions 156 c (F2/F4/F5) et F6. Bras C :
défaut + F1 + F3 en opt-in (`ACVRAM_GDN_PORTES_NOYAU=1 ACVRAM_GDN_NORME_FUSEE=1`), ce que rapporterait la décision de
l'utilisateur.

* **instrument** : banc chat de la 102 (`banc-chat-openai.py`, invite fixe, max_tokens 256, fenêtre 20 s, NVML
  `energie.py`), serveur neuf par passe, A C C A A C C A A C (5 passes par bras), -lgc 2700, carte 0 ; même banc que
  NInfer dans la 139 (b=1 74,6 t/s · 4,350 J ; b=8 463,3 · 0,699 J, séance du 24/09 matin, AUTRE séance).
* **cellules** : alias mixte `Qwen3.8-27B-unsloth-mixte-i8c` (converti sous `ACVRAM_HFQUANT_PAR_GROUPE=1`, 139) et
  `Qwen3.8-27B-nvfp4`, b=1 et b=8.
* **validité d'une passe** : `marlin(` dans le journal du serveur (sinon repli : passe NULLE) ; en C, la ligne de régime
  porte les deux drapeaux ; `fenetre_valide` du banc. Ligne de régime relevée à chaque passe.
* **limite nommée d'avance** : l'alias nvfp4 n'est pas le point de contrôle de NInfer (unsloth mixte) : ses chiffres
  contre NInfer comparent deux formats, pas deux moteurs.

## Prédictions (t/s du banc ; J/jeton net)

| alias | b | A prédit | C/A prédit | FAUX si | NInfer |
|---|---|---|---|---|---|
| mixte | 8 | 305-320 t/s (139 bis 296,3 à 27,0 ms/pas ; −1,2 ms de 156 c, −0,3 de F6), 0,86-0,92 J | +2 à +4 % | C/A < +1,4 % | 463,3 / 0,699 |
| mixte | 1 | 59-61 t/s (58,7 ; F6 −0,3 ms), 4,9-5,2 J | +2 à +3,5 % (F3 seule à b=1) | C/A < +1,4 % | 74,6 / 4,350 |
| nvfp4 | 8 | 480-530 t/s (14,1 ms/pas en processus + préfill par lot), 0,50-0,60 J | +4 à +6 % | C/A < +2,8 % | (463,3 / 0,699, autre format) |
| nvfp4 | 1 | 74-79 t/s (12,72 ms/pas), 2,6-3,4 J | +2,5 à +4,5 % | C/A < +1,7 % | (74,6 / 4,350, autre format) |

A hors de sa fourchette de ±8 % : prédiction fausse, écrite comme telle. L'issue qui me gênerait : nvfp4 b=8 sous
NInfer (< 463 t/s) — l'écart processus/service serait plus grand que je ne le crois.
