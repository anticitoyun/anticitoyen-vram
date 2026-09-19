# Chantier C15 — glue du pas de décodage GLM b=1 : attribution des 2 622 lancements, 6 retirés par couche au bit (`ACVRAM_MLA_GLUE=1`), 15 de plus par le noyau de préparation (`=2`, numérique du lot) ; le seuil ≤ 18 élémentaires n'est atteint qu'avec le niveau 2 (poste1, 19/09, à sec)

**Objectif** (poste7-glm-decode-budget-c14-c15-19-09 § 2, C15) : les ~1 700 petits noyaux du pas b=1 (2,0 ms, 26 %) — les attribuer ligne par ligne, retirer ce qui ne calcule rien sans changer un bit, nommer ce qui exige un noyau. Aucun GPU ce soir ; le `.cu` n'est pas recompilé.

**Prédiction scellée (poste7, recopiée)** : « glue MLA/MoE par couche : ≤ **18 noyaux/couche** (36 aujourd'hui) ; jetons identiques ; b=1 −0,8 ms au moins ; prédiction −1,0 ms ; réfutation : < 0,5 ms gagnée pour 18 noyaux : le coût n'est pas le lancement, arrêter ». **Comment je compte** : lancements/pas = `Instances / 50` du CSV `glm-decode-b1_cuda_gpu_kern_sum.csv` (poste2, 53acea4) — 2 622 noyaux/pas au total (7,73 ms), dont « élémentaires » = tout sauf GEMV/attention/experts/routeur/reduce/swiglu ; par couche = sur une couche MoE Marlin (33 des 46), le total de la couche est 53, dont 37 élémentaires (le « 36 » de poste7 : 1 700 / 47). **Ma prédiction, avant toute mesure** : `MLA_GLUE=1` : 2 622 → ≤ 2 340 lancements/pas (−6 × 47 − 1), jetons identiques au bit sur 256 pas, −0,3 à −0,5 ms/pas ; `=2` : ≤ 1 650 lancements/pas, élémentaires 18/couche, −1,0 ± 0,3 ms, PPL décodage ± 0,001 (jetons non garantis : q_abs ≤ 1 ulp). Réfutation : jetons différents à `=1` = bogue (retirer) ; `=2` sous −0,5 ms → poste7 a raison, le coût n'est pas le lancement.

## Attribution (b=1, 47 couches = 1 dense + 46 MoE : 33 Marlin, 13 « distinctes » xreg ; chemin = `_la_decode` model.py:2501 → `decode_static` mla.py:610)

| noyau nsys | /pas | /couche | fichier:ligne | pourquoi |
|---|---|---|---|---|
| `int8_gemv<bf16,bf16>` | 233 | 3 + 2 MoE | mla.py:233 (`qa_kv`), :264 (`q_b`), :523 (`_o`) ; model.py:2001 expert partagé (gate_up, down) | 47×3 + 46×2 |
| `rmsnorm_bf16` | 189 | 4 (+1 finale) | model.py:2439 entrée, mla.py:264 q_a_norm, :509 kv_a_norm, model.py:2551 post ; :3007 `norm` | déjà 1 lancement (layers.py:578) |
| `add<bf16>` | 234 | 5 | model.py:2441 et :2554 résidus (2) ; mla.py:300-301 RoPE (sub = add α=−1, add) (2) ; :1882 `y + shared` (1 MoE) | 47×4 + 46 |
| `Mul<bf16>` non vectorisé | 188 | 4 | mla.py:300-301 `x0*c, x1*s, x0*s, x1*c` (vues à pas 2) | RoPE « norm » |
| `direct_copy` → fp32 | 212 | 4,5 | mla.py:619 `q_eff.to(f32)` (47), :521 `v_b.to(f32)` (47) ; model.py:1812 `x.to(f32)` routeur (46) ; :1770 `.to(act.dtype)` après division AWQ (33+13) ; :1762-1764 `x_g/x_u.to(f32)` (13×2) | conversions bf16→fp32 (cast dynamique, 2,8 µs, 0,60 ms : le plus cher) |
| `bfloat16_copy` (fp32→bf16) | 93 | 2 | mla.py:523 `.to(x.dtype)` (47) ; model.py:1770 `act.to(bf16)` (33 Marlin : la gate·up sort en fp32 ; 13 xreg) | |
| `direct_copy` → int64 | 164 | 3-5 | model.py:1684 `tok.long()` (33+26), :1686 `eid.long()` (33+26), :1770 `eid.long()` (46) | int32→int64 pour l'indexation |
| `vectorized_gather` | 164 | 3-5 | model.py:1684 `x[tok]`, :1686 `table[eid]`, :1770 `awq_down[eid]` | 33×3 + 13×5 |
| `Div<bf16>` | 108 | 2-3 | model.py:1686 (33 + 13×2), :1770 (46) = 105 ; 3 hors couche non attribués | échelles AWQ par expert |
| `index_kernel` | 94 | 2 | layers.py:757 `_cos[positions]`, `_sin[positions]` via mla.py:284 | tables RoPE bf16 |
| `Cat` 2-D vect. | 94 | 2 | mla.py:507 `cat([c, k_pe])` redécoupée :508, :512 `k_new` | |
| `Cat` 3-D (2 variantes) + `alignedK` 4-D | 47+47+47 | 3 | mla.py:306 `cat([q_pe, k_pe])`, :511 `cat([q_abs, q_pe])`, :302 `torch.stack((y0,y1))` | |
| `index_copy`, `add_<long>` | 47, 47 | 2 | mla.py:489 `_ecrit_ligne`, :522 `len.add_(1)` | écriture du latent, avance |
| `gemv2T` bf16, `gemvx` f32 (cuBLAS) | 47, 47 | 2 | mla.py:510 einsum k_b, :521 einsum v_b | |
| `dot`+`reduce_1Block` f32, `_route_fusee`, `arange` | 46 ×3 | 3 | model.py:1812 `F.linear` fp32 (M=1 → 2 noyaux cuBLAS), :1868, :1690 `tok_g` | routeur |
| `mla_1p` + `combine`, `moe_reduce`, `swiglu`, Marlin ×2, xreg ×3, `silu`+`Mul<f32>` ×13 | | | mla.py:138 ; model.py:1786, :2002 ; :1706/:1774 ; :1763-1776 ; :1765 | calcul, pas glue |

Total par couche Marlin : 4 (bloc) + 27 (MLA) + 22 (MoE) = 53, dont 37 élémentaires ; xreg : 63.

## Ce qui existe déjà, et pourquoi pas sur MLA
`add_norm` layers.py:585 → `rmsnorm_bf16` avec résidu (.cu:5738, :5751 `bf16(fp32(res)+fp32(y))` = l'addition bf16 de torch, au bit) ; pris seulement par `ACVRamModel._res_differe` model.py:3054, qui exige `type(l) is DecoderLayer` — GLM est en `DecoderLayerGDN` (MLA dans `linear_attn`), donc `decode_fixed` :2439-2441 + `_mlp` :2551-2554 = 2 rmsnorm + 2 add séparés. `int8_matmul_norme` kernels/__init__.py:877 (`Attention.decode_fixed_norme` model.py:384) : réfuté 6dbb1bb (+0,31 ms, norme recalculée par bloc), pas transposé. `mla_prep_batch` .cu:5600 (RoPE + einsum k_b + norme kv_a + cat + fp32 en 1 lancement) et `mla_ecrit_latent` .cu:5285 : chemin `decode_static_batch_complet` mla.py:556, pris à b > 1 seulement (`_MLA_BATCH=2`, model.py:2486) ; à b=1 `decode_static` refait tout en torch. `_v_b32` mla.py:548 : utilisé par le chemin complet, pas par `_sortie_decode`.

## Fait ce soir — `ACVRAM_MLA_GLUE` (mla.py:102, regime.py, garde cli.py), défaut 0 = inchangé
`=1`, mêmes opérations et arrondis : (a) `_v_b32` dans `_sortie_decode` :519 et le repli :632 (−1 copie fp32) ; (b) cat kvp :507 supprimée (−1 cat) ; (c) RoPE écrite en place :297-298 (`torch.sub/add(out=)`, −1 stack) ; (d) demi-tables cos/sin empilées `tables_demi` layers.py:710, une indexation :279 (−1 index) ; (e) `DecoderLayerGDN.decode_fixed_res` model.py:2446 + `_res_differe` :3054 étendu aux couches MLA (−2 add, par `add_norm`, + −1 sur la norme finale). **−6 lancements/couche** : 53 → 47, élémentaires 37 → 31. `=2` : b=1 par `decode_static_batch_complet` (model.py:2492) — −15 lancements MLA (27 → 12) : 53 → 32, élémentaires 31 → **18** = le seuil de poste7, atteint seulement ainsi.

**Preuve à sec** : `tests/test_mla_glue_c15.py` — jouet GLM (2 couches MLA bas rang + RoPE « norm » + MLP, bf16, processeur), 32 pas greedy : jetons, logits et caches latents **égaux au bit** entre `=0` et `=1` ; compteur d'opérations aten hors vues (`TorchDispatchMode`) : **154 → 146 ops/pas, −8 = 4/couche exactement** (`to`, `cat`, `stack`, `index` : ×2) ; témoins qui cassent : cos/sin permutés → diverge ; delta doublé dans `decode_fixed_res` → diverge ; à `=0` ni `tables_demi` ni `decode_fixed_res` ne sont appelées. Rejouer : `CUDA_VISIBLE_DEVICES= ACVRAM_TESTS_PENDANT_MESURE=1 python -m pytest tests/test_mla_glue_c15.py tests/test_regime_noyaux.py tests/test_cadrage_perplexite.py tests/test_mla_norme.py tests/test_mla_decode_batch.py -q -p no:cacheprovider` → 41 passed, 13 skipped (CUDA).

## Reste (noyau CUDA nommé, non écrit) et non vérifié
* Division AWQ dans les GEMV experts (Marlin gate·up / down, xreg) avec l'arrondi bf16 du quotient reproduit : −5 (Marlin) / −8 (xreg) par couche ; `tok`/`eid` en int64 servis par `index_jetons`/`route_fusee` : −2/−3 (glue torch, mais `_forward_grouped` est CUDA seul : pas de preuve à sec possible, non fait).
* Logits du routeur dans `_route_fusee` (Triton) : −3 ; `o_proj` lisant fp32 (`int8_gemv<float,bf16>` non instancié) : −1 ; `mla_ecrit_latent` à b=1 (la longueur doit être copiée avant : gain nul sans noyau) : 0.
* **Non vérifié à sec** : l'égalité au bit de `add_norm` (le .cu, pas le repli torch : à sec le compteur ne voit pas ces −2/couche) ; le niveau 2 (`mla_prep_batch`, cas B=1 ajouté à `test_mla_decode_batch.py::test_module_batch_complet`, saute sans carte) ; toute durée.

## Première fenêtre carte (poste2, 15 min, chaîne nsys de poste2 `chaine-decode-glm.sh`)
1. `test_mla_decode_batch.py` (B=1 complet) puis nsys b=1 sous `ACVRAM_MLA_GLUE=0 | 1 | 2` : lancements/pas par nom (`Instances/50`) — attendus `=1` : `direct_copy`→fp32 212 → 165, `Cat` 2-D 94 → 47, `alignedK` 47 → 0, `index_kernel` 94 → 47, `add<bf16>` 234 → 140, total ≤ 2 340 ; `=2` : ≤ 1 650, plus de `Mul<bf16>` ni de `gemv2T`.
2. 256 pas greedy, invite 256 jetons, `=0` contre `=1` : **jetons identiques** exigés ; `=2` : `ppl-decode-kv` ± 0,001 avec `=0` en témoin.
3. ms/pas (somme des noyaux, ABAB) : seuils ci-dessus ; publier aussi ce qui contredit.
