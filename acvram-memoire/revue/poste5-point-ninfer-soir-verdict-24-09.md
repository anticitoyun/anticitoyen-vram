# Verdict — point contre NInfer avec les défauts du soir (poste5, 24/09) — prédictions de débit TENUES

* **instrument** : `scratchpad/poste5-pninfer-24-09/prise.sh` + `banc-chat-openai.py` (banc de la 102), A C C A A C C A A C,
  serveur neuf par passe, fenêtre 20 s, NVML ; dépouillement `resume.py` → `resume.txt`
* **commit** : 2863e111 (branche poste5) ; défauts : Marlin, 157, 156 c (F2/F4/F5), F6 ; C = défauts + F1 + F3
* **régime** : 40/40 passes avec `marlin(…)` (mixte `seuls=112`, nvfp4 `seuls=305`), `repli_eager=0` au `/metrics` de
  chaque passe, C porte `ACVRAM_GDN_PORTES_NOYAU=1 ACVRAM_GDN_NORME_FUSEE=1` à sa ligne de régime ; 0 passe nulle ;
  -lgc 2700, plafond 400 W, bridage « puissance » signalé sur les 40 fenêtres (comme à la 139) ; cpu-safe 100 ; llama-server
  sur la 3080 Ti tout du long ; repos NVML de 8 s (< 30 s recommandés, ligne de base plus bruitée)
* **scellé** : `revue/poste5-point-ninfer-soir-scelle-24-09.md` (avant la prise)
* **durée** : 20:1x-20:55:12 (`carte.sh`, poste5-point-ninfer)

| alias | b | A t/s (σ) | A J/jeton | C t/s | C/A | prédit C/A | NInfer (139, autre séance) | A / NInfer | C / NInfer |
|---|---|---|---|---|---|---|---|---|---|
| mixte | 8 | 315,5 (1,2) | 0,899 | 324,8 | **+2,94 %** | +2 à +4 | 463,3 · 0,699 | −31,9 % · J +28,6 % | −29,9 % · +28,9 % |
| mixte | 1 | 60,1 (0,0) | 5,124 | 61,7 | **+2,66 %** | +2 à +3,5 | 74,6 · 4,350 | −19,4 % · +17,8 % | −17,3 % · +18,1 % |
| nvfp4 | 8 | 502,9 (2,2) | 0,604 | 531,1 | **+5,61 %** | +4 à +6 | (463,3 · 0,699) | +8,6 % · −13,6 % | +14,6 % · −14,3 % |
| nvfp4 | 1 | 76,3 (0,0) | 3,833 | 78,8 | **+3,28 %** | +2,5 à +4,5 | (74,6 · 4,350) | +2,3 % · −11,9 % | +5,6 % · −11,9 % |

σ = 0,0 à b=1 : le banc arrondit à 0,1 t/s et les 5 passes y tombent sur la même valeur (le Welch t de `resume.txt` n'a
alors aucun sens).

## Lecture

* **Débit : toutes les prédictions tenues**, A dans sa fourchette et C/A dans la sienne pour les 4 cellules.
* **J/jeton** : mixte dans la prédiction (0,899 et 5,124). nvfp4 b=8 0,604, juste au-dessus de 0,50-0,60. **nvfp4 b=1
  FAUX** : 3,83 contre 2,6-3,4 prédits ; la carte tient 370 W à b=1, j'avais supposé moins.
* **F1 + F3 ne font pas baisser le J/jeton** (C/A −0,7 à +0,3 %) : le pas raccourcit, la puissance monte d'autant
  (+9 à +15 W). En régime borné par le plafond de 400 W, ils rapportent du débit, pas de l'énergie.
* **Même point de contrôle que NInfer (mixte)** : l'écart se referme par rapport à la 139 bis (b=8 −36 % → −31,9 % au
  défaut, −29,9 % avec F1/F3), mais il reste large ; le reste est dans les int8 (139 bis).
* **Alias nvfp4** : au-dessus des chiffres NInfer aux deux b, et −12 à −14 % en J/jeton. Ce n'est pas une victoire de
  moteur : NInfer sert l'unsloth mixte (W4A4 + W8A8), nous un nvfp4 W4A16, ailleurs et à une autre séance. Seule
  affirmation permise : sur ce banc, notre meilleur alias Qwen3.8 dépasse le débit que NInfer tire de son checkpoint.
