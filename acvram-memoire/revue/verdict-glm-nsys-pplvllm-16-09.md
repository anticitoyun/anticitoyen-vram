# Verdict — pas b=12 GLM : 15 534 lancements/pas, le MLA est exécuté séquence par séquence (2 845 GEMV int8 à lot 1) ; PPL vLLM GadflyII = 1,056 : le duel comparait des qualités différentes

- **instrument** : nsys `--cuda-graph-trace=node -t cuda,nvtx`, `nsys-rejeu-b12-15-09.py` (50 pas de décodage pur b=12, ctx 256→306, synchronize par pas), analyse `nsys-trous-analyse-15-09.py` (poste `mla` ajouté) ; PPL vLLM `ppl-vllm-glm-16-09.py` (cadrage `acvram eval` : 4 fenêtres 2048/2048, cibles [257:2048], 7 164 notées comme HF) ; sorties `scratchpad/glm-nsys-pplvllm-16-09/`
- **commit** : travail/poste3 **85ff032** (code `acvram/` = main ce71723) ; protocole même commit
- **régime** : `-k48`, prefill W4A16, décodage MMA=1 MIN_T=5 (PREUVE lue), graphes actifs, lot 12 ; vLLM GadflyII NVFP4, KV fp8 (régime du duel) puis KV auto (bf16) en contrôle
- **scellé** (mes prédictions) : pas 58-64 ms, poste mla 50-60 %, trou 1-3 ms (> 5 = lancement) ; PPL vLLM 1,000-1,006
- **mesuré** : pas **43,9 ms** ; postes : elementwise **12,6** (32,7 %), denses **11,5** (29,8 %), mla **6,4** (16,6 %), moe_gemm 5,2, normes 1,7, quant 0,6, route+pack 0,4 ; **trou 5,3 ms** ; **15 534 noyaux/pas**. PPL vLLM **8,5970 = 1,0558** (KV fp8), **8,6666 = 1,0643** (KV bf16)
- **verdict** : mes trois prédictions **réfutées** dans la lettre, et la lecture est plus utile : le poste n'est pas l'attention MLA (6,4 ms) mais **la forme du chemin MLA au décodage — par séquence** (§ 1). Colonne PPL : **vLLM 1,056 contre nous 1,002** — le ×4,08 du duel est acheté avec +5,4 points de PPL (§ 2)

## 1. Ce que les 15 534 lancements disent (en-tête du chantier MLA de poste4)
```
noyau                                   lancements/pas   ms/pas   lecture
int8_gemv_kernel<4,1> (lot 1 !)              2 845          5,6     12 séq × 47 couches × 5 proj. (q_a q_b kv_a kv_b o) : le lot 12 est éclaté en 12 GEMV à lot 1
int8_gemv_kernel<4,12>                         —            2,1     les projections qui, elles, voient le lot 12
gemvx / gemv2T (cuBLAS bf16)              948 + …            3,1     matrices bf16 du MLA (absorption ?) à lot 1
direct_copy / cat (CatArrayBatchedCopy ×3)   2 077 + 2 900     7,1+3   les 12 `torch.cat` par couche (`model.py:1252`) et leurs copies
elementwise binaire / index / add          3 858 + 1 895 + 2 214  8,2   arithmétique de RoPE/scores par séquence
rmsnorm_bf16                                 2 123            1,7     12 × 47 × ~4 : la norme aussi est par séquence
mla_scores_batch + mla_reduce_batch            —              6,4     l'attention elle-même, déjà groupée
nvfp4_gemm_grouped_mma2 (MoE, lot 12)          —              5,2     le MoE : 13 % du pas, il n'est pas le sujet
```
- Coder-30B au même lot : **1 517 lancements, 13,6 ms, trou 0,78**. GLM : ×10 lancements, trou 5,3 ms. Le MoE (5,2 + 0,6 + 0,4 + 0,1 = 6,3 ms) est comparable à Coder ; **tout l'excédent (≈ 37 ms) est la boucle par séquence du MLA** : projections int8 à lot 1 (5,6 ms au lieu de ~0,5 à lot 12 — le `<4,12>` fait 2,1 ms pour la même somme d'octets), copies/`cat` (≈ 10 ms), élémentaires (≈ 8 ms), normes (1,7), trou (5,3).
- **Attendu du chantier** (à sceller par poste7) : projections, normes, RoPE et `cat` au lot 12 → denses 11,5 → ~3 ms, elementwise 12,6 → ~2, normes 1,7 → 0,2, trou 5,3 → ~1 : **pas 43,9 → ~19-22 ms** (vLLM : 15,1). Le contexte compte en plus : 43,9 ms à ctx ≈ 300 contre 56,1 en rondes (256→2048) — +12 ms de KV lu par pas à 2048, à mesurer après le chantier.

## 2. Colonne PPL complète (référence bf16 8,1427, même corpus, même cadrage, 7 164 jetons notés)
```
acvram -k48   prefill W4A16 (régime du duel)     8,1569   1,0018
acvram -k48   prefill W4A4 par ligne             8,2913   1,0183
vLLM GadflyII NVFP4, KV fp8 (régime du duel)     8,5970   1,0558
vLLM GadflyII NVFP4, KV bf16                     8,6666   1,0643
```
Le converti GadflyII (NVFP4 sur tout, sans AWQ ?) perd 5,6 % là où le nôtre perd 0,2 % : le duel n'a pas comparé deux qualités égales. Le débit vLLM reste ×4 ; ce qui est à nous, c'est le MLA par séquence (§ 1), et ce qui est à eux, c'est un converti plus grossier. Un duel apparié demanderait GadflyII sous acvram ou notre -k48 sous vLLM (compressed-tensors) — poste7 tranche.
