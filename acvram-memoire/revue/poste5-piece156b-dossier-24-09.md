# Pièce 156 (b) — dossier à sec (poste5, 24/09) : le pas b=8 hors lecture des poids, face à NInfer, et ses fusions

Sans code, sans prise. Source : la trace nsys de la 148 bis (alias mixte `Qwen3.8-27B-unsloth-mixte-i8c`, PROJ_MARLIN,
b=8, 93 fenêtres de pas, durées de NOYAUX) relue par nom de noyau (`scratchpad/poste5-p156-24-09/detail-familles.py`) ;
la trace NInfer de la même pièce, agrégée par nom. fla installé : 0.5.2.

## 1. Où sont passés les « ~13 ms hors poids » (148 : 27 ms au banc, ≈ 14 ms de poids)

| part | ms/pas | statut |
|---|---|---|
| préfill par lot vu au banc chat (banc 27,0 contre processus 21,5) | ≈ 5,5 | 150 : c'est du préfill ; 150 bis : −0,81 s/lot au mixte (opt-in) |
| GEMM bf16 alpha/beta (alias mixte seulement) | 2,85 | pièce 149, reportée |
| **noyaux hors poids du pas** (glue, GDN, normes, copies, attention, rope) | **4,15** | **ce dossier** ; NInfer 1,86 (hors sa tête) |
| trous entre nœuds du graphe (21,5 − 20,87) | ≈ 0,6 | hors champ |

## 2. Les 4,15 ms, noyau par noyau (µs par pas, b=8 ; 48 couches GDN, 16 d'attention)

| noyaux | µs/pas | lancements/pas | site |
|---|---|---|---|
| `fused_recurrent_gated_delta_rule_fwd_kernel` (fla) | 1 085 | 48 (22,6 µs) | `gdn.py:302` ; NInfer `recurrent_batch_update` 18,7 µs, EN PLACE |
| RMSNorm `rmsnorm_bf16_kernel` | 486 | 129 (3,8 µs) | NInfer 1,5 µs par appel, résidu en épilogue |
| memcpy D2D | 421 | 55 (48 = l'état S recopié) | `gdn.py:304` `S.copy_(S_new)` |
| casts fp32 des 4 projections | 287 | 193 | `gdn.py:125` |
| copies élémentaires (repeat_interleave q/k, état de conv, contiguous) | 186 | 190 | `gdn.py:263, 271-272` |
| portes : softplus, exp, neg, sigmoid, add, mul | ≈ 250 | ≈ 290 | `gdn.py:273-274` |
| norme gated : pow, mean, +eps, rsqrt, mul ×3, silu, cast bf16 | ≈ 380 | ≈ 390 | `gdn.py:130-136` |
| conv depthwise + silu + cat | 257 | 144 | `gdn.py:262, 264` |
| additions résiduelles bf16 | 102 | 112 | `couches.py:227, :348` |
| attention paginée + réduction, rope, kv_write, porte sigmoïde | ≈ 450 | ≈ 80 | couches d'attention |

NInfer fait la même couche GDN en quatre noyaux : `gdn_projected_conv` (2,1 µs, conv + portes à la sortie des projections),
`recurrent_batch_update` (18,7 µs, état en place), `gdn_norm_gating` (4,9 µs), `sigmoid_gate_mul` ; nous en lançons ~25.

## 3. Fusions proposées (ordre : gain / risque), fichier:ligne, gain prédit, effet sur la sortie

| # | fusion | site | gain prédit (ms/pas, b=8) | sortie |
|---|---|---|---|---|
| F4 | **État S en place** : appeler le noyau fla avec `ht` = `h0` (le tampon statique) au lieu d'allouer `final_state` puis de le recopier | `gdn.py:304` ; fla `fused_recurrent.py:211` (allocation), `:110` (lecture de h0), `:179` (écriture de ht) | **−0,40** | **AU BIT par construction** : même noyau compilé, seule l'adresse de sortie change. Condition à prouver par lecture et test : chaque programme lit TOUTE sa tuile h0 (l. 110, avant la boucle) avant d'écrire la même tuile (l. 179), et aucun ne lit la tuile d'un autre |
| F5 | **Résidu différé étendu aux couches GDN non MLA** (`decode_fixed_res` existe déjà, générique) | `model.py:433` (`mla_glue` exige `rank`) ; `couches.py:232-252` | −0,09 (96 additions) | **AU BIT** par le contrat d'`add_norm` (`bf16(fp32(res) + fp32(y))`, `couches.py:238-241`). À vérifier : normes de Qwen3.8 bien `RMSNorm` (poids 1+w repliés à la conversion) |
| F1 | **Portes dans le noyau fla** : `use_gate_in_kernel`, `A_log`, `dt_bias`, `use_beta_sigmoid_in_kernel` (présents dans fla 0.5.2) | `gdn.py:273-274` ; fla `fused_recurrent.py:135` | −0,20 à −0,27 | **± ulp fp32** : exp et softplus de Triton (`fla.ops.utils.softplus`) contre `expf`/`log1pf` de torch → KL 2 × max des témoins |
| F2 | **Conv de décodage fusionnée** : projections bf16 lues sans cast, cat + mise à jour de l'état de conv en place + conv 4 prises + silu + découpe q/k/v, sans `repeat_interleave` (fla gère HV ≠ H, `fused_recurrent.py:211`) | `gdn.py:125, 261-272` | −0,55 à −0,65 | **au bit visé** : 4 prises fp32, ordre séquentiel des FMA du noyau torch `conv_depthwise2d_forward_kernel_generic` reproductible ; à prouver par un test au bit contre torch, sinon ± ulp et KL |
| F3 | **Norme gated fusionnée** (RMSNorm sur dv = 128 × silu(z) × w, sortie bf16) | `gdn.py:130-136` | −0,15 à −0,25 | **± ulp fp32** : ordre de la somme sur 128 changé → KL |
| F6 | **RMSNorm plus rapide à M = 8, H = 5 120** (warp par ligne, vectorisé) | `layers.py:587-610`, `acvram_kernels.cu:7557` | −0,25 à −0,30 | au bit SEULEMENT si l'arbre de somme est reproduit (précédent : `rmsnorm_bf16_warp`, « même ordre, au bit », limité à H ≤ 2 048, `layers.py:605-610`) ; sinon ± ulp |
| F0 | GEMM alpha/beta par le chemin étroit (alias à alpha/beta bf16) | pièce 149 | −2,6 (mixte seul) | ± ulp |

**Somme F1-F6 : −1,6 à −2,0 ms/pas**, soit −6 à −8 % du pas b=8 du défaut (25,4 ms) et −7 à −9 % du mixte (21,5 ms).
Dont **au bit par construction ou visé : F4 + F5 (+ F2) = −0,5 à −1,1 ms/pas**, livrables sans KL si le test au bit tient.
Après F1-F6, nos noyaux hors poids passeraient de ≈ 4,15 à ≈ 2,2-2,5 ms, contre 1,86 chez NInfer. Le reste tient à
leur récurrence plus rapide (18,7 contre 22,6 µs) et à l'attention.

## 4. Limites de ce dossier

* La trace est celle de l'alias MIXTE. Le chemin GDN du défaut `Qwen3.8-27B-nvfp4` passe par le même code (`gdn.py`),
  mais ses noyaux n'ont pas été tracés : chaque gain se remesure sur le défaut avant d'être annoncé.
* Tout est sous graphe CUDA : un lancement de moins ne vaut que la durée de son noyau, pas un coût hôte. Les gains
  ci-dessus sont des sommes de durées de noyaux, donc des bornes hautes : un noyau fusionné n'a pas une durée nulle.
  J'ai compté 2-5 µs par couche pour F2 et F3.
* Ordre proposé si chef le veut : F4 et F5 d'abord (au bit, petits, test d'équivalence au bit + bras cassant), puis
  F2 (au bit visé), puis F1/F3/F6 (KL).
