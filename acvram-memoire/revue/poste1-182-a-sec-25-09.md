# 182 (à sec) : glue, normes, copies, attention contre NInfer — 25/09 (poste1)

Sources : trace nsys 173 (`p173_cuda_gpu_trace_nvtx=mesure.csv`, hors git ; b = 8 sous graphes, 50 pas, alias mixte,
invite 512, contexte 572 → 622) ; NInfer `familles-ninfer.txt` et acvram `familles-148.txt` (148 bis, banc chat b = 8,
invite 256) ; la trace NInfer détaillée est perdue, seules ses familles restent. Relus avant : 148 bis, 156b-d (F1-F6), 163.

## Le « +0,9 » est d'avant F1-F6
148 bis : normes / copies / attention / rope = 1,41 contre 0,53 (+0,88). Aujourd'hui (173), les mêmes quatre postes
= **0,99 ms au contexte du banc** (attention prise dans 148 bis, 0,339) : **+0,46 ms**. À contexte 600 (173) : 1,20,
soit +0,67 (l'attention croît avec le contexte, 30,7 → 34,4 µs par appel en 50 pas).

## Ce que nous avons et que NInfer n'a pas (µs par pas, trace 173)
| noyau | appels | µs | µs/appel | source | NInfer |
|---|---|---|---|---|---|
| cast bf16→fp32 de **z** (`direct_copy`, grille 96) | 48 | 96 | 2,0 | `gdn.py:177` | 0 |
| casts bf16→fp32 de **α, β** (grille 1) | 96 | 115 | 1,2 | `gdn.py:180-181` | 0 |
| q `.contiguous()` de la porte d'attention (strided) | 16 | 29 | 1,8 | `attention.py:300` | 0 |
| porte : `sigmoid` + `mul` séparés | 32 | 42 | 1,3 | `attention.py:474` | `sigmoid_gate_mul` 16 appels, 22 |
| `rope_inplace` (normes q/k comprises) + `kv_write_int8` | 32 | 86 | 2,7 | `layers.py:874` | `rope_kv` 16 appels, 37 |
| `paged_attn_partial` + `reduce` (contexte 600) | 32 | 547 | 17,1 | grille (8, 24, 16) | 32 appels, 257 |
| idem au contexte du banc (148 bis) | 32 | 339 | 10,6 | | 257 |
| `rmsnorm_bf16_reg` (8 blocs à b = 8) | 129 | 273 | 2,11 | F6 | 161 appels (+ q/k), 232 |
| échantillonneur (logsumexp, argmax, memcpy) | ~20 | ≈ 45 | | pas | « autres » ≈ 36 |

En notre faveur, pour mémoire : `swiglu_bf16` 64 appels, 60 µs contre `quant_act` nvfp4 + fp8 NInfer, 209 µs (−149).
Le cœur GDN est à parité : conv 107 + récurrence 1 058 + norme à porte 56 = 1 221, contre `gdn` NInfer 1 244. Seule
la récurrence est plus lente par appel (22,0 contre 18,7 µs), mais elle est dans fla, hors glue.

## Les 3 fusions qui rendraient le plus
1. **Casts GDN supprimés** : −144 lancements, **≈ −0,21 ms/pas**. **Au bit** : la conversion bf16→fp32 est exacte, et
   les deux consommateurs castent déjà au chargement : fla `fused_recurrent.py:121,130` (`tl.load(p_beta/p_g).to(tl.float32)`)
   et `_norme_gated_kernel` (`gdn_norme.py:30`).
   (a) α/β (−115 µs) : ne plus appliquer `.to(float32)` sur la voie F1, sans noyau à écrire. Cette partie empiète sur la
   175 (poste6) : si α/β passent par une pile étroite, ces casts suivent.
   (b) z (−96 µs) : `_norme_gated` lit z bf16 par (pas de lot, pas de tête) au lieu de lignes contiguës (z est une vue de
   la pile 176, que l'on ne peut pas remodeler sans copie).
   Prédiction : 19,90 → 19,69 ms/pas (±0,03).
2. **Attention décodage groupée GQA + réduction repliée** : aujourd'hui une case par (séquence, tête q, tranche) ;
   chaque tuile KV est lue par les 6 têtes q de son groupe, et la réduction est un second lancement. Une case par
   (séquence, tête kv, tranche) traiterait les 6 têtes q, et le dernier bloc arrivé ferait la réduction. Gain : −0,08
   au contexte du banc (pour rejoindre NInfer), jusqu'à ≈ −0,25/−0,35 au contexte 600. Cette fourchette est large : le
   coût est latent (≈ 74 ns par jeton de contexte et par appel), pas en octets (≈ 10 Mo de KV par couche ≈ 6,5 µs).
   **Au bit** à deux conditions : chaque tête q garde son ordre d'accumulation jeton par jeton et les mêmes bornes de
   tranche, et la réduction garde la même séquence de combinaison. Changer le découpage casserait le bit.
3. **Porte d'attention** : `sigmoid·mul` en un noyau qui lit la porte en strided, et q lu en strided par `rope_inplace`
   (sans `.contiguous()`). −32 lancements, **≈ −0,05 ms**. **Au bit** si le noyau arrondit sigmoid en bf16 avant le
   produit, comme torch aujourd'hui (deux opérations bf16), et reprend la formule de torch. Sinon ±1 ulp, à juger par KL.

Écartées : rope + kv_write en un noyau (−0,05), déjà réfuté en Triton (f3a : cache corrompu sous graphe,
`attention.py:60`). rmsnorm à plus de blocs par ligne (≤ −0,09) : l'arbre de réduction changerait, donc pas au bit
(F6 l'a gardé exprès).

**Total des trois : ≈ −0,34 ms au contexte du banc (sur +0,46 restants), ≈ −0,5 ms à 600.** Aucun code écrit.
