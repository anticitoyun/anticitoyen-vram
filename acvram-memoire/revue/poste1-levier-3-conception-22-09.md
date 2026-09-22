# Levier 3 — conception détaillée à sec : ce que la « glue » est vraiment en 0.6.35, et les trois modules qui restent (22/09, poste1 ; relecture poste4)

* correction à ma note H2 (`poste1-h2-graphe-22-09` § 1) : les 433 nœuds de glue et leurs 1,09 ms viennent du nsys du **19/09**, AVANT deux changements de défaut — `ACVRAM_GLUE_COMPACT=1` (C15-3d, `regime.py:170`) et `chemin_moe=mma-a4+pile`. Lu dans l arbre (fichier:ligne) : **289 de ces 433 nœuds n existent plus en service** — `Fill` ×97, `reduce_kernel` fp32 ×96, casts ×96 étaient `torch.zeros` + `y.sum(0)` + `.to(x.dtype)` de `gemm_etroit.py:190-193`, remplacés par `_etroit_reduit_kernel` seul (`gemm_etroit.py:181-188`, `compact=glue_compact("etroit")` à `kernels/__init__.py:1008`) ; l attention compacte écrit `out` en bf16 en un lancement (`attn_paginee.py:283-307`), le `.to(x.dtype)` d `attention.py:398` est un no-op ; rope + kv_write = un noyau (`_rope_kv_fusee`, `attention.py:380-386`) ; le routeur = un noyau (`_routeur_compact`, `model.py:1417`) ; la somme résiduelle est dans `add_norm` (`model.py:2289-2292`). **Le plafond « 0,48 ms » de H2 était faux comme décision** : il comptait des nœuds déjà retirés. poste2 tranche avec `familles-noyaux` ; ci-dessous ce que la lecture prédit qu elle verra, et ce qui reste à fusionner.

## 1. Le pas b=12 lu dans le code (décodage, graphe, `decode_fixed_res`), lancements par couche
| # | noyau | où | famille |
|---|---|---|---|
| 1 | `add_norm` (résidu + rmsnorm, entrée attention) | model.py:2289 → layers.add_norm | normes |
| 2 | qkv `_etroit_reduit_kernel` (int8, split-K réduit dans le noyau) | attention.py:260 `qkv_proj` → kernels/__init__.py:1008 | étroites |
| 3 | `rope_kv` (normes par tête + RoPE + écriture int8 du cache) | attention.py:384 | rope_kv |
| 4 | attention paginée compacte (`_partiel_reduit`) | attention.py:389 → attn_paginee.py:295 | attention |
| 5 | o `_etroit_reduit_kernel` | attention.py:400 | étroites |
| 6 | `add_norm` (résidu + rmsnorm, entrée MoE) | model.py:2292 | normes |
| 7 | routeur compact (GEMV + top-k fusionnés) | model.py:1417 `_routeur_compact` → `route_logits_fusee` | routage |
| 8 | `moe_route_pack` (tri par expert, tuiles, x rassemblé) | model.py:1048-1050, .cu:3655 (1 lancement) | routage |
| 9 | `nvfp4_quant_act(xs)` (amax par ligne + E2M1 + pack) | model.py:1094 | **glue A4** |
| 10-11 | `nvfp4_gemm_grouped_mma` gate, up (même xq : `xq2 = xq` sans up_distinct) | model.py:1117-1118 | experts |
| 12 | `moe_act` (silu·up) | model.py:1119 | experts_glue |
| 13 | `nvfp4_quant_act(act)` | model.py:1120 | **glue A4** |
| 14 | `nvfp4_gemm_grouped_mma` down | model.py:1121 | experts |
| 15 | `moe_reduce_trie` (Σ_k poids × sortie, dans l ordre des jetons) | model.py:1122 | experts_glue |
→ **15 lancements/couche = 720/pas + tête + argmax capturé ≈ 722** (poste4 19/09 en comptait 13 sur le chemin Marlin : cohérent, le chemin mma-a4 en a deux de plus, les quantifications A4). **Prédit pour la table de poste2 : 700-760 lancements/pas, `glue_torch` ≤ 10 lancements et ≤ 0,05 ms, `experts_glue` 96 lancements** (les noyaux du .cu ne sont pas nommés `at::native`). Si elle voit 420-440 comme je l avais écrit, c est que le service n est pas sur mma-a4+pile ou que GLUE_COMPACT est retombé : à lire sur la ligne de régime AVANT tout autre chiffre.

## 2. Ce qui reste fusionnable au bit — trois modules, un épilogue chacun
Coût d un lancement dans un graphe : ≈ 2,5 µs (latence + noyau minuscule, `poste4-scelle-b12` H2) ; la lecture des 15 noyaux ne laisse que des noyaux à calcul réel, sauf trois quantifications/agrégations qui pourraient vivre dans le noyau qui produit leur entrée :
| module | fusion | nœuds retirés/pas | invariant au bit | prédit |
|---|---|---|---|---|
| **B** `moe_act` + `quant_act(act)` | l amax par ligne de `act` [t·k, I] et le pack E2M1 calculés dans l épilogue de `moe_act` (une ligne = un bloc : l amax est un max, associatif, le pack est par élément) | −48 | sortie (aq, asf, gra) == `nvfp4_quant_act(moe_act(g, u))` au bit, entrées réelles, jouet CPU pour la référence, carte pour le noyau | −0,10 à −0,15 ms |
| **C** `moe_route_pack` + `quant_act(xs)` | le pack rassemble déjà x par ligne (.cu:3655) : amax + E2M1 dans le même passage ; `awq_g` (échelle par expert) et `hd_x` (Hadamard) restent des conditions d éligibilité (sans elles d abord ; Coder = sans) | −48 | (xq, xsf, gr) == chemin actuel au bit | −0,10 à −0,15 ms |
| **D** gate + up en une GEMM groupée | même xq, mêmes tuiles, deux piles : un noyau qui écrit g et u (deux sorties) — pas une concaténation de piles (les poids ne bougent pas, disposition unique) | −48 | g, u == deux appels au bit (mêmes MMA par tuile, même ordre d accumulation par sortie) | −0,05 à −0,10 ms (les GEMM ne sont pas à la latence : le gain est le seul lancement) |
Écartés d avance : `moe_reduce_trie` dans l épilogue de down (atomiques : ordre de sommation non reproductible, `_MOE_FUSED_ATOMIQUE` « témoin, non reproductible » model.py:1772) ; `add_norm` dans l épilogue de o/down (le résidu traverse deux noyaux à formes différentes) ; `nvfp4_moe_fused` (`_MOE_DECODE_FUSED=0`, model.py:1770 : opt-in existant jamais promu — à rebancher, ce serait B+D+E en un, mais son équivalence est jugée « contre float64 », pas au bit : hors levier 3).
**Somme B + C + D : −144 lancements ≈ −0,25 à −0,40 ms/pas (−3,7 à −6 %)** ; il en faut 55 µs pour passer devant vLLM après le levier 2. **Réfuté module par module** : `familles-noyaux` A/B ne retire pas les 48 lancements, ou le gain du module < 40 µs (le noyau fusionné a pris ce que le lancement rendait), ou ≠ au bit.

## 3. Ordre et tests (chaque module = un commit : noyau + test au bit + compte de lancements)
1. **poste2 d abord** : table `familles-noyaux` sur 0.6.35 (≤ 5 min) — elle décide du go du module 1 (B) : go ssi lancements/pas ∈ [700, 760] et `experts_glue` + quantifications ≥ 96 lancements ; sinon je relis avant tout noyau.
2. Module B (1 j) : `moe_act` prend `aq/asf/gra` en sortie optionnelle ; test `tests/test_moe_act_quant_fusee.py` : égalité au bit contre `moe_act` puis `nvfp4_quant_act` sur 12 lots réels (t·k ∈ {12, 48, 96}), et **compte de lancements** par `torch.profiler` (CUDA) : −1 par couche, doit rendre faux si l ancien chemin est repris ; ids au bit b=1/b=12 (poste2) ; `familles-noyaux` A/B.
3. Module C (1 j) : même protocole sur `moe_route_pack`.
4. Module D (1-2 j) : après B et C, seulement si l écart à vLLM n est pas déjà comblé (D coûte plus qu il ne rend si B + C suffisent).
5. Chaque module : opt-in d abord (`ACVRAM_MOE_EPILOGUE=b|bc|bcd`, déclaré dans `regime.VARIABLES`, ligne `moe_epilogue=`), défaut après ABBA ≥ +1 % et J ≤ 1,005 × A.

## 4. Prédiction globale et arrêt
Après leviers 1+2 : 1 584 t/s (−0,75 % vLLM). Après B + C : **1 625-1 660 (+1,8 à +4 %)** ; D en plus : +0,7 à +1,5 %. **Arrêt de la piste « frontière/glue »** si la table de poste2 montre ≤ 650 lancements/pas (les fusions sont déjà faites ailleurs) ou si B ne rend pas 40 µs : il ne reste alors que le calcul (mma-a4 à mesurer contre son plancher — jamais fait sur ce chemin, seul Marlin l a été : c est le chiffre à demander ensuite).
