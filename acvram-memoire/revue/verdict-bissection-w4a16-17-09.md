# Verdict — bissection du 1 % W4A16 (geste B) : ni déquant, ni GEMM MoE, ni routage, ni tête — le chemin torch de référence est cassé sur ce converti

instrument : `ppl-acvram-17-09.py` (privé, 3 tranches, préfixe, géo ; une seule variable d'environnement par bras) sur `GLM-4.7-Flash-vllm-direct` W4A16 (`ACVRAM_MOE_MMA=0`) ; référence CUDA du même converti = 1,0280 (`passage-direct-17-09`) ; cible vLLM Marlin W4A16 = 1,0164 ; `outils/dequant-trois-references-17-09.py` (Laurine) sous carte — journaux `scratchpad/geste-b-17-09/`
commit : arbre laure 08cd4f9 → 8d9c736 (= main, geste A de Laurine inclus) ; protocole `protocole-bissection-w4a16-17-09.md`
régime : prefill 2 048 jetons ; à cette taille le MoE W4A16 passe déjà par déquant + `torch._grouped_mm` (`_MOE_GEMM_MAX = 48` jetons/expert, `model.py:1497`, 2048 × 8 / 64 ≈ 256), pas par le noyau CUDA — dit avant de lire les bras
scellé (Sage) : (i) ≤ 1,020 → nos noyaux CUDA perdent le 1 % ; (i) ≈ 1,028 → perte en amont ; (ii) témoin inerte attendu 1,028 ; (ii') MoE torch seul
mesuré (× bf16, géo, tranches identiques à la 4e décimale quand « = ») :
    bras                                             ×bf16      / réf. CUDA   lecture
    réf. CUDA W4A16 (déquant + grouped_mm bf16)      1,0280     1,00000
    (i)  ACVRAM_DISABLE_KERNELS=1 (tout torch)       7,6·10⁶    —            **chemin de référence cassé** (PPL ≈ 10⁸ sur les 3 tranches)
    (ii) ACVRAM_MLA_EAGER_TORCH=1                     1,0280     1,00000      inerte en prefill, comme prévu (témoin tenu)
    (ii') ACVRAM_PREFILL_DEQUANT=1 (MoE torch)        1,0280     1,00000      = réf. : le prefill était déjà en déquant torch
    ACVRAM_MOE_ROUTE_PACK=0 (routage torch)           1,0280     1,00000      noyau de routage exact
    ACVRAM_TETE_FP32_ENTREE=1                         1,0280     1,00000      inerte (tête déjà fp32)
    ACVRAM_MOE_GEMM_MAX=10⁹ (GEMM CUDA W4A16 forcé)   1,0264     0,99847      le noyau CUDA fait 0,15 % MIEUX que déquant + grouped_mm
    déquant (Laurine + 3e bras sous carte)            0 ulp      —            noyau `nvfp4_dequant` = vLLM = nvfp4.py, 3 experts, VERDICT TENU
verdict : **le 1 % (1,028 contre 1,016 de Marlin sur les mêmes poids) n'est ni dans la déquantification (0 ulp), ni dans le GEMM MoE (CUDA ou torch : ± 0,15 %), ni dans le routage, ni dans la tête. Tout ce qui a une bascule donne le même chiffre à 10⁻⁴ : la perte est dans ce qui n'en a pas et qui est commun aux deux chemins — la sémantique ou la précision du modèle en amont des noyaux (attention MLA : rope partiel, normes q_a/kv_a, échelle ; formule du routeur GLM : `e_score_correction_bias`, `norm_topk_prob`, `routed_scaling_factor` ; résidu bf16). Pas le lecteur compressed-tensors : les poids sont à 0 ulp. Second résultat : `ACVRAM_DISABLE_KERNELS=1` ne rend pas une référence mais du bruit (PPL 10⁸) sur ce converti — l'un des replis torch (rmsnorm, GEMV int8, dequant nvfp4 côté engine) est faux, à nommer (Laurine) ; tant qu'il l'est, « tout torch » n'est pas un témoin.**

## Vitesses vLLM Marlin W4A16 (`-a16`, en-tête du duel : KV fp8, fenêtre 20 s, invites tirées sans préfixe comme le duel)
    b=1  183,5 t/s  1,705 J brut (1,329 net)   | W4A4 : 153,7  1,97
    b=4  472,7      0,673 (0,530)             | W4A4 : 421,1  0,77
    b=12 858,4      0,397 (0,310)             | W4A4 : 796,3  0,445
    prefill pp2048 18 117 j/s (113 ms)         | W4A4 : 26 732
    → la meilleure qualité de vLLM sur sm_120 est aussi son meilleur décodage (+8 % b=12, −11 % J) ; seul le prefill perd (−32 %). Contre nous (-k48 prise B : 514 t/s, 0,774 J) : vLLM W4A16 ×1,67 à b=12.

## Ce qu'il faudrait pour finir la bissection (Laurine, à sec puis 1 min de carte)
Une sonde couche par couche : mêmes ids, même fenêtre, sortie de chaque bloc (attention, routeur top-k et poids, MoE, résidu) d'acvram contre vLLM W4A16 (hooks) — la première couche où l'écart dépasse le bruit désigne l'opération ; ou un témoin par formule (routeur GLM recodé en torch fp32 depuis `config.json`, comparé au nôtre sur une fenêtre).
