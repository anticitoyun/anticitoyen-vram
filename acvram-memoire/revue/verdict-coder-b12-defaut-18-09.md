# Verdict — cellule officielle Coder b=12 AU DÉFAUT (main 23903b5 : `GROUPED_RPW=4`, `GROUPED_XREG=down`, aucune variable posée) : **pas nu 9,15 ms (1 312 t/s)** ≤ 9,30 → **TENU** ; bridé 400 W **1 198 t/s, 0,334 J/jeton**

instrument : `scratchpad/coder-defaut-18-09/chaine.sh` — garde « aucune `ACVRAM_GROUPED_*` dans l'environnement » (compte 0 imprimé en tête) ; `certifie-b12-15-09.py` ×2 (bridé, J/jeton, ligne régime) puis `profil-pas-coder-17-09.py … b12` ×2 (40 pas nus) ; `cert-p*.json`, `profil-p*.json`
commit : 7d38733 (laure = main 23903b5), 09:03-09:08, régime classé seul (`NARROW_GEMM=1 MOE_MMA=0 MOE_DECODE_MMA=0`, `HYBRID_SLOTS=12`), graphes on, `chemin_moe=gemv`
régime — preuve du défaut : la ligne `[régime]` des quatre JSON ne nomme QUE `NARROW_GEMM=1 MOE_MMA=0 MOE_DECODE_MMA=0 GDN=fla` (elle ne cite que les écarts au défaut : l'absence de `GROUPED_*` prouve que rien n'a été posé) ; valeurs effectives lues par `acvram.regime.regime_noyaux()` au même arbre, même environnement : `ACVRAM_GROUPED_RPW='4'`, `ACVRAM_GROUPED_XREG='down'`, `hors_defaut={}` ; code : `regime.py:66,69`, `acvram_kernels.cu:1880/2106` (rpw 4 sans variable) et `:2086` (`!e → 1` = down)
scellé (Sage, sage-xreg-down-defaut-18-09) : moyenne des deux pas nus ≤ 9,30 ms tenu ; prédiction 9,15-9,25
mesuré : nu **9,134 / 9,159 ms → 9,15 ms, 1 313,8 / 1 310,1 t/s** ; bridé 399,5 W : pas **10,004 / 10,035 ms, 1 199,6 / 1 195,9 t/s, 0,3332 / 0,3340 J/jeton** ; deux passes concordantes à 0,3 %
verdict : **TENU** — 9,15 ≤ 9,30 (marge 0,15 ms), dans la prédiction ; cellule source pour Katy : **Qwen3-Coder-30B-A3B-nvfp4 b=12 — 1 198 t/s · 0,334 J (bridé 400 W, nu 1 312 t/s, 9,15 ms/pas) · régime classé, défauts RPW=4 + XREG=down** — nu 1 312 (dérive entre fenêtres 2 %, témoins 9,51/9,72) — (remplace la cellule 35d3086 : 1 162 / 0,344 / 1 262)

## Lecture
- Le défaut rend exactement le bras B de l'ABAB (9,165 / 9,193) : aucune variable n'était nécessaire, les deux gestes sont bien en place dans le .cu et dans regime.py, et la mesure ne dépend pas de la main qui les pose.
- Chemin des quatre jours sur ce pas Coder b=12 nu : 10,76 (rpw=1) → 9,51 (rpw=4) → 9,15 (xreg down) : −15 % de ms/pas, −13 % de J/jeton (0,385 → 0,334), sortie identique au bit à chaque étape.
