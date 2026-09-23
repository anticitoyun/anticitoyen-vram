# Conception — cache KV à 4 bits par rotation (TurboQuant) : format, coût par jeton, scellé à ctx ≥ 8k

Laurine, 17/09, à sec, sur ordre sage-plan-completion-comparatif-17-09 § 3 : **note de conception seulement** — aucune implémentation, aucune carte avant les menus. Ce document fixe ce qui serait écrit, ce que ça coûte et ce qui serait mesuré, avec les seuils écrits AVANT toute mesure.

## 1. Ce qu'on a aujourd'hui (fichier:ligne)

| cache | format | bits/élément | où |
|---|---|---|---|
| attention GQA (Coder-30B, Qwen, Llama…) | int8 par vecteur de tête (D = 128), échelle fp16 par (jeton, tête) | 8 + 16/128 = **8,125** | `memory/kvcache.py:239` (`KVCacheConfig.dtype` int8 \| fp8_e4m3 \| fp16 \| bf16), `:353-359` quantification torch, `:389` → `kv_write_int8` (`acvram_kernels.cu:2735-2790`), lecture `paged_attn_partial_kernel` (`:891`) |
| MLA (GLM-4.7-Flash) : latent rang 512 + RoPE 64 | bf16 par défaut ; fp8 E4M3 par ligne `[max_len, W+16]` sous `ACVRAM_MLA_LATENT_FP8=1` (défaut 0) | 16 ou **8 + 32/576** | `engine/mla.py:44,48-70` (`_fp8_quant_rows/_fp8_dequant_rows`), `:352-390` (`new_static`, `_ecrit_ligne`), noyau `mla_ecrit_latent_kernel`, `mla_1p_kernel<…, FP8>` |

Par jeton, toutes couches : Coder-30B-A3B (48 couches × 2 × 4 têtes × 128) = 49 152 éléments → **bf16 96 Kio, int8 48,75 Kio** ; GLM-4.7-Flash (47 couches × 576) = 27 072 éléments → **bf16 52,9 Kio, fp8 27,2 Kio**.

## 2. Le principe (TurboQuant, Zandieh et al. 2025) en trois lignes

1. **Rotation fixe** Π (orthogonale, Hadamard × signes aléatoires de graine fixe, bloc = D) appliquée à chaque vecteur k et v avant écriture : les coordonnées d'un vecteur tourné sont approximativement gaussiennes de variance ‖x‖²/D — les canaux aberrants (puits d'attention, canaux massifs de K) sont étalés sur les D coordonnées ; **une seule échelle par vecteur suffit**, et c'est la norme, pas l'amax.
2. **Quantification scalaire optimale** (Lloyd-Max gaussien) de chaque coordonnée à b bits : 16 centroïdes fixes pour b = 4 (MSE 0,0095 σ², soit 9,7 % RMS), 8 pour b = 3 (0,0345 σ², 18,6 % RMS). Codes 4 bits, table de 16 fp32 en registres à la lecture.
3. **Le produit scalaire ne demande pas de dé-rotation** : ⟨Πq, Πk⟩ = ⟨q, k⟩. Au décodage, on tourne q une fois par pas et par tête (FWHT 128 : 448 add/sub), on lit les clés tournées telles quelles ; pour V, la sortie o' = Σ p_i Π v_i est dé-tournée une fois (Πᵀ o'). Le noyau d'attention ne change que par son chargeur de K/V.

Variante TurboQuant complète : K à 3 bits + **1 bit QJL** (signe de la projection du résidu k − k̂ sur un vecteur gaussien fixe, estimateur non biaisé du produit scalaire) = 4 bits ; V à 4 bits Lloyd-Max (la sortie est une moyenne pondérée, un MSE-optimal suffit). Le papier tient 3,5 bits presque sans perte ; c'est le **bras 2** ici, pas le défaut.

## 3. Format proposé

**Bras 1 (défaut proposé)** — « lm4 » : K et V à 4 bits Lloyd-Max après rotation, échelle fp16 = ‖x‖/√D par (jeton, tête).

```
GQA   kc, vc : [bloc, bs, HKV, D/2] uint8 (deux codes par octet, index pair = quartet bas — même convention que NVFP4)
      ks, vs : [bloc, bs, HKV]      fp16  (inchangé : `k_scale/v_scale`, kvcache.py:342)
      bits/élément = 4 + 16/128 = 4,125     Coder : 24,75 Kio/jeton (÷1,97 contre int8, ÷3,9 contre bf16)
MLA   ligne  : [max_len, 512/2 + 64 + 2 + 2] = 324 → 336 octets alignés
               (latent 512 tourné + 4 bits LM ; RoPE 64 gardé en fp8 E4M3, échelle fp16 ; échelle fp16 du latent)
      bits/élément latent = 4,03           GLM : 15,4 Kio/jeton (÷1,77 contre fp8, ÷3,4 contre bf16)
```

Pourquoi la partie RoPE reste à 8 bits : 64 coordonnées seulement, la rotation Π sur 64 gagne peu, et le score `q_rope · k_rope` porte la position — on ne mélange pas les deux dans un même bloc tourné (le latent absorbé et la partie RoPE ont des échelles différentes, `mla.py _k_b_c`). Sur GQA, les clés sont écrites APRÈS RoPE (`kvcache.py:389` reçoit k tourné par `rope_emb`) : Π après RoPE ne change rien au score, q reçoit RoPE puis Π.

**Bras 2** — « tq3+1 » : K 3 bits LM + 1 bit QJL (plan de bits `kj : [bloc, bs, HKV, D/8]`), V 4 bits LM. Même empreinte (4,125), meilleure fidélité des scores attendue à norme égale, un produit scalaire de plus par clé (D/8 octets lus, popcount).

Ce qui ne change pas : la table de blocs, `BlockAllocator`, `HostKVPool` (dont les transferts hôte↔carte sont divisés par ~2 — l'exil du KV en profite au même titre que la VRAM), le cache de préfixe (mêmes clés de bloc), les graphes (la LUT de 16 valeurs est une constante de compilation, la graine de Π une constante du manifeste — rien d'hôte pendant la capture, leçon `torch.tensor(scalar, device=cuda)` du 16/09).

## 4. Coût par jeton

| poste | où | coût |
|---|---|---|
| écriture (prefill et décodage) | `kv_write_lm4` remplaçant `kv_write_int8` | par vecteur : FWHT 128 (448 add/sub fp32) + norme (128 FMA) + 128 recherches dans 15 seuils (≤ 4 comparaisons par binaire) — **borné par la mémoire**, ~0,3 µs par (jeton, tête) comme le noyau int8 |
| lecture (décodage) | chargeur de `paged_attn_partial_kernel` | **moitié des octets** de l'int8 par clé et valeur ; dépaquetage 2 quartets + LUT en registres : 2 instructions par élément, sous le débit mémoire — le noyau reste borné par la mémoire comme aujourd'hui (revue decodage-faible-lot-etat-de-l-art § 3) |
| rotation de q | avant l'attention, par pas | FWHT 128 par (séquence, tête q) : Coder b = 12 → 12 × 32 × 48 = 18 k FWHT = 8 MFLOP par pas, négligeable (< 5 µs) |
| dé-rotation de o | après l'attention | idem, FWHT 128 par (séquence, tête q) |
| MLA | `mla_ecrit_latent_kernel`, `mla_1p_kernel` | FWHT 512 par jeton à l'écriture (2 304 add/sub) ; q_lat tourné une fois par pas ; lecture des 512 latents à 4 bits + 64 RoPE fp8 |

Octets lus par pas de décodage à ctx = 8 192, b = 12 (toute la fenêtre lue, pas de fenêtre glissante) :

| modèle | bf16 | int8 / fp8 (aujourd'hui) | lm4 |
|---|---|---|---|
| Coder-30B-A3B | 9,66 Go | 4,91 Go | **2,49 Go** |
| GLM-4.7-Flash (MLA) | 5,32 Go | 2,74 Go (fp8, non défaut) / 5,32 (bf16, défaut) | **1,55 Go** |

À 1,79 To/s (5090 bridée, MATERIEL) : Coder 2,7 ms → 1,4 ms de lecture KV par pas ; GLM 3,0 ms (bf16) → 0,9 ms.

## 5. Scellé (avant toute mesure, carte de Laure, après les menus et sur ordre)

Instrument : PPL fenêtre **8 192** (min_context 4 096, préfixe, privé 3 tranches, géo) et pas de décodage b = 12 à contexte **8 192** rempli (`decode-…` avec invites de 8 k, pas de cache de préfixe), sur `Qwen3-Coder-30B-A3B-nvfp4` (int8 aujourd'hui) et `GLM-4.7-Flash-vllm-direct` (bf16 latent aujourd'hui, fp8 en témoin). Trois bras par modèle : aujourd'hui, **lm4**, **tq3+1** ; bf16 en référence pour la PPL.

Prédictions, toutes issues nommées :

1. **Mémoire** (exact, pas une prédiction) : Coder 24,75 Kio/jeton (÷1,97), GLM 15,4 (÷3,4 contre bf16) — vérifié par `KVCacheConfig.bytes_per_block` et `nvidia-smi` à ctx 8 k × 12 slots : Coder 2,32 Gio contre 4,57.
2. **Qualité** : PPL 8 192 lm4 ≤ int8/fp8 **+ 0,003** (Coder) et ≤ bf16 **+ 0,004** (GLM) ; tq3+1 ≤ **+ 0,006**. Au-delà de + 0,010 sur l'un des deux : le bras est **réfuté** à 4 bits pour ce modèle, on ne cherche pas un réglage. Contrôle qui peut dire « faux » : le bras bf16 et le bras int8 doivent d'abord se distinguer de moins de 0,002 (sinon l'instrument ne voit pas 4 bits non plus) ; et un bras **lm2** (2 bits, 34 % RMS) DOIT sortir de la tolérance — un instrument qui ne casse pas à 2 bits ne prouve rien à 4.
3. **Vitesse** : à ctx 8 192, b = 12, pas de décodage **−4 à −8 %** (Coder), **−8 à −12 %** (GLM depuis bf16) ; à ctx ≤ 1 024, **± 1 %** (la lecture KV y est < 5 % du pas — si l'on mesure un gain à 1 k, c'est un artefact). Écart intra-bras attendu ≤ 0,3 ms ; un gain sous 2 × cet écart est « nul ».
4. **Prefill** : ± 2 % (l'écriture est bornée par la mémoire et divisée par deux ; la rotation est gratuite devant les GEMM).

Si (2) tient et (3) ne tient pas : le format vaut pour la **capacité** (contextes ×2 à VRAM égale, ou 2 × plus de slots), pas pour la vitesse — ce qui est déjà le cas d'usage de KVMem/TurboQuant, et se publie tel quel.

## 6. Risques nommés

- **Puits d'attention** (Sage § 12, position 0 dominante) : la ligne 0 a une norme 10-40 × celle des autres ; par vecteur, son échelle est la sienne — pas de canal partagé, pas d'écrasement. À vérifier par le test `test_creneaux_egalent_la_reference_hors_creneau[puits-pos0]` étendu au format.
- **Hybride GQA + MLA + KDA (Kimi-Linear, Qwen3-Next)** : seuls les caches d'attention changent ; les états récurrents (`gdn.py`, `kda.py`, `mamba2.py new_static`) ne sont pas concernés.
- **Le noyau processeur** (`HostKVPool`, décodage exilé) doit lire le même format : le repli torch (`kvcache.py:362-366 _dequantize`) se fait avec la même LUT — test d'équivalence noyau = torch au bit, comme `test_repli_torch_pile_nvfp4`.
- **Graine de Π** : dans le manifeste (`kv_rotation_seed`) ou fixe dans le code ; un cache exporté (`static_export`) ne se relit qu'avec la même — le format porte le régime par le nom (`dtype = "lm4"`), REGLES § 6.
- **L'ordre des mesures** : aucune carte avant la fin des menus (Sage). Ce que ce document permet de faire à sec ensuite : le noyau d'écriture et sa référence torch, la LUT Lloyd-Max et son test (centroïdes contre les valeurs tabulées : 4 bits gaussien {±0,128, ±0,388, ±0,657, ±0,943, ±1,257, ±1,619, ±2,070, ±2,733} σ, MSE 0,00950 — recalculés par itération de Lloyd à sec, 3 bits {±0,245, ±0,756, ±1,344, ±2,152} MSE 0,0346, 2 bits {±0,453, ±1,510} MSE 0,1175), l'équivalence lecture/écriture sur processeur — tout ce qui ne demande pas la 5090.

## 7. Ce qui n'est pas dans cette note

Aucun chiffre mesuré ; aucune ligne de noyau ; pas de choix entre lm4 et tq3+1 — c'est la mesure (5) qui tranche, pas la lecture du papier.
