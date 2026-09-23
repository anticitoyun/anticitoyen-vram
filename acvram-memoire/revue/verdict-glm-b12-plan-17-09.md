# Verdict — GLM `-k48-calibA` b=12 sous `bf16`, budget KV pour 12 séquences : **568,1 t/s, 0,700 J** (contre 539,5 / 0,732 avec 4 séquences tronquées)

instrument : `certifie-b12-15-09.py` rondes ×2, `CERT_PLAN_LEN=3072` (preuve dans `preuve.CERT_PLAN_LEN`) — `scratchpad/poste-d-17-09/glm-rondes-b12-plan3072-p{1,2}.{json,log}`
commit : arbre laure 3725629 (= main) ; protocole `protocole-glm-b12-plan-17-09.md`
régime : `[régime] ACVRAM_NARROW_GEMM=1 ACVRAM_MOE_MMA=0 ACVRAM_MOE_DECODE_MMA=0 extension=oui` ; moteur `NOMINAL graphes=on 0/47 exilées 0/2944 experts piles_ok=True chemin_moe=gemv prefill=bf16` ; SLOTS=12, ctx 2048, invite 256, plan 3 072 × 8 = 24 576 jetons de KV ; une carte, 400 W (bridage puissance pendant la fenêtre, comme toutes les rondes)
scellé : Sage 548-560 ; moi 545-555, 0,72-0,74 J ; > 565 → la traîne pesait plus sur GLM que sur Coder
mesuré : p1 **563,4 t/s, 0,7016 J** (21,30 ms/pas, 2 851 MHz) · p2 **572,7 t/s, 0,6981 J** (20,95 ms, 2 831 MHz) · 1 787 pas × 12 = 21 444 jetons, **aucune ligne « budget KV épuisé »** (4 dans chaque passe d'avant)
verdict : **568,1 t/s, 0,700 J (moyenne des deux passes) — les deux fourchettes sont réfutées par le haut : la troncature coûtait + 5,3 % à GLM (539,5 → 568,1) contre + 1,8 % à Coder (730,4 → 743,4) ; l'énergie par jeton baisse de 4,4 %. Ligne à publier pour la cellule acvram × GLM b=12 : 568,1 t/s · 0,700 J ; le régime nommé est le même, seul le budget KV du plan a été mis au niveau du lot.**
Réserve : l'écart p1/p2 (1,6 %) est l'ordre du bruit thermique des rondes ; la valeur retenue est la moyenne, comme pour Coder.
