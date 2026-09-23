# Verdict (suite) — repli torch corrigé (Laurine 9b5b428) : le chemin tout-torch rend 1,010 × bf16, MIEUX que Marlin (1,016) et 1,7 % sous nos noyaux (1,028)

instrument : `ppl-acvram-17-09.py` + `PPL_MASQUE_NOYAUX` (cache un noyau à l'extension : `hasattr` → faux, le moteur prend le repli torch de ce noyau seul) ; privé, 3 tranches, préfixe, géo ; `GLM-4.7-Flash-vllm-direct` W4A16 — journaux `scratchpad/geste-b-17-09/`
commit : arbre laure 906eabb (= main 97fba4b, correctif 9b5b428 inclus)
scellé (Jérôme) : `ACVRAM_DISABLE_KERNELS=1` = PPL des noyaux au bruit près (1,028 ± quelques 10⁻³) · lecture Sage (i) : ≤ 1,020 → nos noyaux CUDA perdent le 1 %
mesuré (× bf16 HF, géo ; « / CUDA » = rapport au chemin noyaux 1,0280) :
    bras                                             ×bf16    / CUDA
    tout torch (repli corrigé)                       **1,0104**  0,9830   ← sous Marlin (1,0164) et sous les noyaux
    masque rmsnorm_bf16                              1,0245   0,9967
    masque moe_act + swiglu_bf16 + swiglu2_bf16      1,0260   0,9981
    masque rope_inplace                              1,0280   1,0000   (inerte)
    masque rmsnorm + act + moe_reduce(_trie)         1,0275   0,9995   (non additif : autre chemin torch de glue)
    GEMM CUDA W4A16 forcé (MOE_GEMM_MAX=10⁹)         1,0264   0,9985
    routage torch, tête fp32, MLA torch              1,0280   1,0000
    masque moe_reduce_trie seul / nvfp4_dequant seul crash : appelés sans repli (`model.py:1090`, `kernels/__init__.py:457`) — pas masquables à ce niveau
verdict : **le scellé « tout torch = noyaux ± 10⁻³ » est réfuté dans le bon sens : une fois le repli corrigé, la référence torch rend 1,0104, soit 1,7 % de mieux que nos noyaux et 0,6 % de mieux que Marlin. Lecture (i) de Sage : ce sont bien nos noyaux CUDA qui perdent — et plus que le 1 % : notre chemin torch battrait vLLM. Les noyaux masquables un par un n'expliquent que 0,2 à 0,35 % chacun et ne s'additionnent pas ; le gros (≥ 1,2 %) est dans la partie du chemin MoE prefill qui n'a pas de repli individuel : `nvfp4_dequant` CUDA → `grouped_mm` bf16 (et sa glue `moe_reduce_trie`), contre `dequantize_nvfp4` torch → matmul dans le chemin tout-torch. Prochain geste (Laurine, à sec) : diff des deux chemins MoE prefill — dtype de sortie de la déquant (bf16 vs fp32), dtype du GEMM, ordre pondération/réduction top-k, arrondi du résidu — puis un témoin par différence.**

## Bornes
- La PPL tout-torch (1,0104) est un chiffre de qualité, pas de vitesse : le chemin est 10 à 50 fois plus lent.
- Les masques changent le chemin, pas seulement le noyau : le groupe rmsnorm+act+reduce déclenche une glue torch différente (1,0275 > 1,0245 du seul rmsnorm) — les rapports par masque ne sont pas des contributions additives.
- Tous les bras : ids [154822, 154824, …], 0 fenêtre explosée, mêmes fichiers de poids.
